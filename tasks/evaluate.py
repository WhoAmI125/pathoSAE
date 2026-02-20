from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from src.evaluation.metrics import evaluate_checkpoint
from src.evaluation.visualize import plot_activation_histogram, plot_sparsity_distribution


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
    parser = argparse.ArgumentParser(description="Evaluate PathoSAEv3 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default=None, help="Activation directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Results root directory")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU id or 'auto'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.gpu)

    artifacts = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=device,
    )

    output_dir = Path(artifacts.output_dir)

    plot_activation_histogram(
        artifacts.activation_samples,
        output_dir / "activation_hist.png",
    )
    plot_sparsity_distribution(
        artifacts.sparsity_samples,
        output_dir / "sparsity_dist.png",
    )

    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved plot: {output_dir / 'activation_hist.png'}")
    print(f"Saved plot: {output_dir / 'sparsity_dist.png'}")


if __name__ == "__main__":
    main()
