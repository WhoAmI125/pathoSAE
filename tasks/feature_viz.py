"""
pathoSAEv3 Feature Visualization
- Top-N features별 최고 활성 패치 + 공간 활성화 맵 렌더링
- Z-score 통계 표시 (pathoSAEv2 스타일)
- 단의미성(monosemanticity) 프록시 점수 계산

Usage:
    python tasks/feature_viz.py --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt
    python tasks/feature_viz.py --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \\
        --top_n 20 --top_k 5 --max_files 5 --gpu auto
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Allow running from arbitrary cwd (tmux/conda-run) without manual PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base_sae import BaseSAE
from src.config import SAEConfig
from src.extract.activation_store import ActivationStore


# ── GPU 유틸 ────────────────────────────────────────────────────

def find_free_gpu(min_free_mb: int = 20000) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            idx_s, free_s = line.split(",")
            candidates.append((int(idx_s.strip()), int(free_s.strip())))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for idx, free in candidates:
            if free >= min_free_mb:
                return idx
        return candidates[0][0] if candidates else 0
    except Exception:
        return 0


def resolve_device(gpu_arg: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    idx = find_free_gpu() if gpu_arg == "auto" else int(gpu_arg)
    torch.cuda.set_device(idx)
    return f"cuda:{idx}"


# ── 공간 활성화 마스크 ──────────────────────────────────────────

def apply_spatial_mask(orig_img: Image.Image, patch_acts: torch.Tensor, base_opacity: float = 0.15) -> np.ndarray:
    """
    patch_acts: [196] - 14×14 그리드의 활성화 값
    → 정규화 후 원본 이미지에 밝기 마스크 적용
    """
    w, h = orig_img.size
    mask = patch_acts.reshape(14, 14).float()

    lo, hi = mask.min(), mask.max()
    if (hi - lo) > 1e-8:
        mask = (mask - lo) / (hi - lo)
    else:
        mask = torch.zeros_like(mask)

    mask_up = F.interpolate(
        mask.view(1, 1, 14, 14), size=(h, w), mode="bilinear", align_corners=False
    )[0, 0]  # [H,W] float32 tensor

    # Use torch ops to avoid broken numpy.core arithmetic
    img_t = torch.tensor(
        np.array(orig_img.convert("RGB")), dtype=torch.float32
    )  # [H,W,3]
    mask_t = mask_up.unsqueeze(-1)  # [H,W,1]
    darkened = img_t * base_opacity
    overlay = img_t * mask_t + darkened * (1.0 - mask_t)
    return overlay.clamp(0, 255).byte().numpy()


# ── Z-score 유의성 ──────────────────────────────────────────────

def zscore_label(z: float) -> tuple[str, str]:
    if z >= 3:
        return "***", "red"
    elif z >= 2:
        return "**", "orange"
    elif z >= 1:
        return "*", "steelblue"
    else:
        return "ns", "gray"


# ── Vinje-Gallant Sparseness ────────────────────────────────────

def vinje_gallant_sparseness(mean_acts: np.ndarray, eps: float = 1e-8) -> float:
    """
    mean_acts: [hidden_dim] - 전체 데이터에서 각 feature의 평균 활성화
    반환값 [0, 1]: 1에 가까울수록 선택적(sparse) = monosemantic 경향
    torch 연산 사용 (numpy.core 버전 충돌 회피)
    """
    a = torch.tensor(mean_acts, dtype=torch.float32).clamp(min=0)
    n = float(len(a))
    num = (a.sum() / n) ** 2
    den = (a.pow(2).sum() / n) + eps
    return float((1 - num / den) / (1 - 1 / n + eps))


# ── Pass 1: feature별 통계 계산 ─────────────────────────────────

def compute_feature_stats(
    model: BaseSAE,
    data_files: list[Path],
    device: str,
    batch_size: int = 4096,
    max_images_per_file: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        feature_mean [H], feature_std [H], activation_freq [H]
    max_images_per_file: 파일당 최대 이미지 수 (None=전체, quick test용)
    """
    hidden_dim = model.hidden_dim
    f_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    f_sq  = torch.zeros(hidden_dim, dtype=torch.float64)
    f_nz  = torch.zeros(hidden_dim, dtype=torch.float64)
    total = 0

    model.eval()
    with torch.no_grad():
        for fpath in tqdm(data_files, desc="Pass1 stats"):
            data = torch.load(str(fpath), map_location="cpu")
            X = data["vectors"].float()          # [N, 196, 768]
            N, P, D = X.shape

            if max_images_per_file is not None and N > max_images_per_file:
                idx = torch.randperm(N)[:max_images_per_file]
                X = X[idx]

            flat = X.reshape(-1, D)

            for i in range(0, flat.size(0), batch_size):
                batch = flat[i:i+batch_size].to(device)
                _, acts, _ = model(batch, step=0)
                acts = acts.double().cpu()
                f_sum += acts.sum(0)
                f_sq  += (acts ** 2).sum(0)
                f_nz  += (acts != 0).double().sum(0)
                total += batch.size(0)

            del data, X, flat
            torch.cuda.empty_cache()

    mean = (f_sum / total).float().numpy()
    var  = ((f_sq / total) - (f_sum / total) ** 2).float()
    std  = torch.clamp(var, min=1e-8).sqrt().numpy()
    freq = (f_nz / total).float().numpy()
    return mean, std, freq


# ── Pass 2: top-K 이미지 탐색 ───────────────────────────────────

def find_top_activations(
    model: BaseSAE,
    data_files: list[Path],
    top_feature_indices: list[int],
    top_k: int,
    device: str,
    batch_size: int = 4096,
    max_images_per_file: int | None = None,
) -> dict[int, list[tuple]]:
    """
    Returns:
        {feat_idx: [(max_val, img_path, patch_acts_196), ...]} top_k개
    배치 단위로 처리하여 속도 개선.
    """
    results: dict[int, list] = {idx: [] for idx in top_feature_indices}
    feat_indices_t = torch.tensor(top_feature_indices, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for fpath in tqdm(data_files, desc="Pass2 top-k"):
            data  = torch.load(str(fpath), map_location="cpu")
            X     = data["vectors"].float()   # [N, 196, 768]
            paths = data["patch_paths"]       # list[N]
            N     = X.shape[0]

            if max_images_per_file is not None and N > max_images_per_file:
                perm  = torch.randperm(N)[:max_images_per_file]
                X     = X[perm]
                paths = [paths[i] for i in perm.tolist()]
                N     = len(paths)

            # 이미지별로 196 패치를 배치로 묶어 처리
            for img_i in range(0, N, batch_size // 196 + 1):
                end_i = min(img_i + batch_size // 196 + 1, N)
                chunk = X[img_i:end_i]          # [B, 196, 768]
                B = chunk.shape[0]
                flat  = chunk.reshape(-1, 768).to(device)  # [B*196, 768]

                _, acts, _ = model(flat, step=0)           # [B*196, H]
                acts_cpu = acts.cpu().reshape(B, 196, -1)  # [B, 196, H]

                for b in range(B):
                    img_acts = acts_cpu[b]                  # [196, H]
                    img_path = paths[img_i + b]

                    for feat_idx in top_feature_indices:
                        col     = img_acts[:, feat_idx]    # [196]
                        max_val = col.max().item()
                        entry   = (max_val, img_path, col.clone())
                        heap    = results[feat_idx]
                        heap.append(entry)
                        heap.sort(key=lambda x: x[0], reverse=True)
                        results[feat_idx] = heap[:top_k]

            del data, X, acts_cpu
            torch.cuda.empty_cache()

    return results


# ── 시각화 ──────────────────────────────────────────────────────

def render_grid(
    top_feature_indices: list[int],
    top_results: dict[int, list],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_freq: np.ndarray,
    top_k: int,
    model_type: str,
    output_path: Path,
) -> None:
    n_features = len(top_feature_indices)
    n_cols = top_k * 2   # 원본 + 마스크
    fig, axes = plt.subplots(n_features, n_cols, figsize=(n_cols * 2.2, n_features * 2.4))
    if n_features == 1:
        axes = axes[None, :]

    for row, feat_idx in enumerate(top_feature_indices):
        mu    = feature_mean[feat_idx]
        sigma = feature_std[feat_idx]
        freq  = feature_freq[feat_idx]
        samples = top_results[feat_idx]

        for col in range(top_k):
            ax_orig = axes[row, col]
            ax_mask = axes[row, col + top_k]

            if col < len(samples):
                max_val, img_path, patch_acts = samples[col]
                z = (max_val - mu) / (sigma + 1e-8)
                sig, color = zscore_label(z)

                if Path(img_path).exists():
                    img = Image.open(img_path).convert("RGB")
                    ax_orig.imshow(img)
                    ax_orig.set_title(
                        f"act={max_val:.2f}\nZ={z:.1f}{sig}",
                        fontsize=7, color=color, pad=2,
                    )
                    masked = apply_spatial_mask(img, patch_acts)
                    ax_mask.imshow(masked)
                    ax_mask.set_title(f"map", fontsize=7, pad=2)
                else:
                    ax_orig.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
                    ax_mask.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)

            ax_orig.axis("off")
            ax_mask.axis("off")

        # 행 레이블
        axes[row, 0].text(
            -0.55, 0.5,
            f"F{feat_idx}\nμ={mu:.3f}\nσ={sigma:.3f}\nfreq={freq:.3f}",
            transform=axes[row, 0].transAxes,
            va="center", ha="right", fontsize=8, fontweight="bold",
        )

    # 범례
    legend = [
        mpatches.Patch(color="red",      label="*** Z≥3 (p<0.001)"),
        mpatches.Patch(color="orange",   label="**  Z≥2 (p<0.05)"),
        mpatches.Patch(color="steelblue",label="*   Z≥1 (p<0.16)"),
        mpatches.Patch(color="gray",     label="ns  Z<1"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, framealpha=0.8)
    plt.suptitle(
        f"{model_type.upper()} SAE — Top {n_features} Features (Z-score Statistics)",
        fontsize=13, fontweight="bold",
    )
    plt.subplots_adjust(left=0.13, wspace=0.08, hspace=0.55, bottom=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장됨: {output_path}")


# ── 단의미성 요약 ────────────────────────────────────────────────

def print_monosemanticity_summary(
    model_type: str,
    feature_mean: np.ndarray,
    feature_freq: np.ndarray,
    top_feature_indices: list[int],
) -> dict:
    sparseness = vinje_gallant_sparseness(feature_mean)
    mean_t = torch.tensor(feature_mean, dtype=torch.float32)
    freq_t = torch.tensor(feature_freq, dtype=torch.float32)
    n_active = int((mean_t > 0).sum().item())
    n_total  = len(feature_mean)

    # 선택적 feature 비율 (freq < 0.01 = 1% 미만에서만 발화)
    selective_ratio = float((freq_t < 0.01).float().mean().item())

    print(f"\n{'='*50}")
    print(f"  [{model_type}] 단의미성 프록시 지표")
    print(f"{'='*50}")
    print(f"  Vinje-Gallant Sparseness : {sparseness:.4f}  (↑ monosemantic)")
    print(f"  Active features          : {n_active}/{n_total} ({n_active/n_total*100:.1f}%)")
    print(f"  Selective features       : {selective_ratio*100:.1f}%  (freq<1%)")
    print(f"  Top feature indices      : {top_feature_indices[:10]}")
    print(f"{'='*50}\n")

    return {
        "model_type": model_type,
        "vg_sparseness": sparseness,
        "active_ratio": n_active / n_total,
        "selective_ratio": selective_ratio,
    }


# ── CLI ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pathoSAEv3 Feature Visualization")
    p.add_argument("--checkpoint", required=True, help="체크포인트 경로 (.pt)")
    p.add_argument("--data_dir",   default=None,  help="activation 디렉토리")
    p.add_argument("--top_n",      type=int, default=10,   help="시각화할 feature 수")
    p.add_argument("--top_k",      type=int, default=5,    help="feature당 상위 이미지 수")
    p.add_argument("--max_files",  type=int, default=None, help="사용할 최대 파일 수 (None=전체)")
    p.add_argument("--max_images", type=int, default=None, help="파일당 최대 이미지 수 (None=전체, quick test용)")
    p.add_argument("--gpu",        default="auto")
    p.add_argument("--results_dir",default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.gpu)
    print(f"Device: {device}")

    # 모델 로드
    model = BaseSAE.load(args.checkpoint, device=device)
    cfg   = model.cfg
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.results_dir:
        cfg.results_dir = args.results_dir

    run_name   = cfg.run_name or Path(args.checkpoint).parent.name
    output_dir = Path(cfg.results_dir) / run_name / "feature_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 파일 목록
    store = ActivationStore(cfg)
    all_files = store.train_files + store.val_files
    if args.max_files:
        all_files = all_files[:args.max_files]
    print(f"사용 파일: {len(all_files)}개")

    if args.max_images:
        print(f"파일당 최대 이미지: {args.max_images}개 (quick test 모드)")

    # Pass 1: 통계
    feature_mean, feature_std, feature_freq = compute_feature_stats(
        model, all_files, device, max_images_per_file=args.max_images
    )

    # 단의미성 요약 출력
    top_idx = torch.topk(torch.tensor(feature_mean), args.top_n).indices.tolist()
    summary = print_monosemanticity_summary(
        cfg.model_type, feature_mean, feature_freq, top_idx
    )

    # Pass 2: top-K 이미지 탐색
    top_results = find_top_activations(
        model, all_files, top_idx, args.top_k, device,
        max_images_per_file=args.max_images
    )

    # 시각화
    output_path = output_dir / f"top{args.top_n}_features_z.png"
    render_grid(
        top_feature_indices=top_idx,
        top_results=top_results,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_freq=feature_freq,
        top_k=args.top_k,
        model_type=cfg.model_type,
        output_path=output_path,
    )

    print(f"\n완료: {output_path}")


if __name__ == "__main__":
    main()
