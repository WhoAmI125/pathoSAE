from __future__ import annotations

import argparse
import subprocess

import torch

from src.evaluation.group_analysis import analyze_groups_for_checkpoint


def find_free_gpu(min_free_mb: int = 20000) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates: list[tuple[int, int]] = []
        for line in result.stdout.strip().splitlines():
            idx_str, free_mem_str = line.split(",")
            candidates.append((int(idx_str.strip()), int(free_mem_str.strip())))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for idx, free_mem in candidates:
            if free_mem >= min_free_mb:
                return idx

        return candidates[0][0] if candidates else 0
    except Exception:
        return 0


def resolve_device(gpu_arg: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"

    gpu_idx = find_free_gpu() if gpu_arg == "auto" else int(gpu_arg)
    torch.cuda.set_device(gpu_idx)
    return f"cuda:{gpu_idx}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-wise feature analysis for PathoSAEv3 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--meta_csv", type=str, required=True, help="Metadata CSV with columns: path,class,patient")
    parser.add_argument("--data_dir", type=str, default=None, help="Activation directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Results root directory")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU id or 'auto'")
    parser.add_argument("--split", type=str, choices=["val", "train", "full"], default="val")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of activation files")
    parser.add_argument("--top_n", type=int, default=50, help="Number of top features to aggregate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for feature aggregation")
    parser.add_argument("--min_group_samples", type=int, default=20, help="Minimum samples required for a group")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.gpu)

    result = analyze_groups_for_checkpoint(
        checkpoint_path=args.checkpoint,
        meta_csv=args.meta_csv,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=device,
        split=args.split,
        max_files=args.max_files,
        top_n=args.top_n,
        batch_size=args.batch_size,
        min_group_samples=args.min_group_samples,
    )

    print(f"Saved class matrix: {result['class_csv']}")
    print(f"Saved patient matrix: {result['patient_csv']}")
    if result["class_heatmap"] is not None:
        print(f"Saved class heatmap: {result['class_heatmap']}")
    else:
        print("Class heatmap skipped: no valid groups after filtering")

    if result["patient_heatmap"] is not None:
        print(f"Saved patient heatmap: {result['patient_heatmap']}")
    else:
        print("Patient heatmap skipped: no valid groups after filtering")


if __name__ == "__main__":
    main()
