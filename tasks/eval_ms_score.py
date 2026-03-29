"""
MS Score CLI

Usage:
    PYTHONPATH=. python tasks/eval_ms_score.py \
        --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
        --gpu 2 \
        --sample_size 5000 \
        --output_dir results/vanilla_e32_ep10/ms

    # 모든 variant 비교
    PYTHONPATH=. python tasks/eval_ms_score.py \
        --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
                      models/checkpoints/topk_e32_ep10/best.pt \
                      models/checkpoints/jumprelu_e32_ep10/best.pt \
                      models/checkpoints/msae_e32_ep10/best.pt \
        --gpu 2 --sample_size 5000
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SAEConfig
from src.evaluation.ms_score import compute_ms_score, load_dinov2
from src.extract.activation_store import ActivationStore
from src.models.base_sae import BaseSAE


def parse_args():
    p = argparse.ArgumentParser(description="Compute MS Score for SAE checkpoints")
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="체크포인트 경로 (여러 개 가능)")
    p.add_argument("--data_dir", default="data/activations",
                   help="activation .pt 파일 디렉토리")
    p.add_argument("--gpu", default="0", help="GPU 인덱스 또는 'cpu'")
    p.add_argument("--sample_size", type=int, default=5000,
                   help="샘플링할 토큰 수 (메모리/시간 균형, 기본 5000)")
    p.add_argument("--split", choices=["val", "train", "full"], default="val",
                   help="평가할 데이터 split")
    p.add_argument("--output_dir", default=None,
                   help="결과 저장 디렉토리 (기본: checkpoint 디렉토리/ms)")
    p.add_argument("--dino_variant", default="dinov2_vitb14",
                   help="DINOv2 variant (dinov2_vitb14 / dinov2_vitl14)")
    p.add_argument("--dino_batch", type=int, default=64,
                   help="DINOv2 이미지 인코딩 배치 크기")
    p.add_argument("--sae_batch", type=int, default=512,
                   help="SAE activation 추출 배치 크기")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_activation_samples", type=int, default=10,
                   help="유효한 MS Score 계산에 필요한 최소 활성 샘플 수")
    return p.parse_args()


def run_single(checkpoint: str, args, dino_model, device: str,
               activation_files: list[Path]) -> dict:
    ckpt_path = Path(checkpoint)
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model = BaseSAE.load(str(ckpt_path), device=device)

    result = compute_ms_score(
        sae_model=model,
        activation_files=activation_files,
        dino_model=dino_model,
        device=device,
        sample_size=args.sample_size,
        dino_batch_size=args.dino_batch,
        sae_batch_size=args.sae_batch,
        seed=args.seed,
        min_activation_samples=args.min_activation_samples,
    )
    result["model_type"] = model.cfg.model_type
    result["checkpoint"] = str(ckpt_path)
    return result


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if args.gpu != "cpu" else "cpu"

    # activation 파일 목록
    from dataclasses import replace as dc_replace
    cfg_tmp = SAEConfig()
    cfg_tmp.data_dir = args.data_dir
    store = ActivationStore(cfg_tmp)
    if args.split == "val":
        files = store.val_files
    elif args.split == "train":
        files = store.train_files
    else:
        files = store.train_files + store.val_files
    print(f"Split '{args.split}': {len(files)} activation files")

    # DINOv2 로드 (한번만)
    print(f"\nLoading DINOv2 ({args.dino_variant})...")
    dino_model = load_dinov2(args.dino_variant, device=device)

    results = []
    for ckpt in args.checkpoint:
        result = run_single(ckpt, args, dino_model, device, files)

        # 저장 경로 결정
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = Path(ckpt).parent / "ms"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ms_scores (뉴런별)는 별도 파일로 저장 (크기가 크므로)
        scores_only = result.pop("ms_scores")
        scores_path = out_dir / "ms_scores.json"
        with open(scores_path, "w") as f:
            json.dump({"ms_scores": scores_only}, f)
        print(f"Saved per-neuron MS scores: {scores_path}")

        # 요약 지표 저장
        summary_path = out_dir / "ms_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved MS summary: {summary_path}")

        result["ms_scores"] = scores_only  # 반환용
        results.append(result)

    # 복수 checkpoint 비교 요약
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("MS Score Comparison")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Valid':<8} {'Mean MS':<10} {'Median MS':<12} {'MS>0.5':<10} {'MS>0.7'}")
        for r in results:
            model_name = r.get("model_type", Path(r["checkpoint"]).parent.name)
            print(f"{model_name:<15} {r['valid_neurons']:<8} "
                  f"{r['mean_ms']:<10.4f} {r['median_ms']:<12.4f} "
                  f"{r['frac_ms_gt05']*100:<10.1f}% {r['frac_ms_gt07']*100:.1f}%")


if __name__ == "__main__":
    main()
