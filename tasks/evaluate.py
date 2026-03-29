from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from src.evaluation.feature_analysis import analyze_checkpoint_features
from src.evaluation.feature_figures import (
    build_feature_dataframe,
    plot_feature_entropy_hist,
    plot_feature_scatter,
    plot_top_feature_reference_grid,
    save_top_feature_reference_images_individual,
)
from src.evaluation.group_analysis import load_meta_csv
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
    parser.add_argument("--feature_figures", action="store_true", help="Generate feature-level evaluation figures")
    parser.add_argument("--split", type=str, choices=["val", "train", "full"], default="val")
    parser.add_argument("--meta_csv", type=str, default=None, help="Optional metadata CSV with path,class,patient")
    parser.add_argument("--top_n", type=int, default=10, help="Top-N features for reference grid")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K patches per feature")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of activation files")
    parser.add_argument(
        "--save_individual_top_features",
        action="store_true",
        help="Save top-feature reference images as per-feature individual PNG files",
    )
    parser.add_argument(
        "--individual_top_features_dir",
        type=str,
        default="top_feature_reference_individual",
        help="Output subdirectory for per-feature individual PNG files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.gpu)

    artifacts = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=device,
        split=args.split,
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

    if args.feature_figures:
        meta_lookup = None
        if args.meta_csv:
            meta_lookup = load_meta_csv(args.meta_csv)

        feature_analysis = analyze_checkpoint_features(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_path=output_dir / "feature_analysis.json",
            device=device,
            top_k=args.top_k,
            batch_size=1024,
            split=args.split,
            max_files=args.max_files,
        )

        df = build_feature_dataframe(feature_analysis, meta_lookup=meta_lookup)
        summary_path = output_dir / "feature_summary.csv"
        df.to_csv(summary_path, index=False)

        scatter_path = output_dir / "feature_scatter.png"
        plot_feature_scatter(df, scatter_path)

        entropy_path = output_dir / "feature_entropy_hist.png"
        has_entropy = plot_feature_entropy_hist(df, entropy_path)

        grid_path = output_dir / "top_feature_reference_grid.png"
        plot_top_feature_reference_grid(
            feature_analysis,
            output_path=grid_path,
            top_n=args.top_n,
            top_k=args.top_k,
        )

        print(f"Saved feature analysis: {output_dir / 'feature_analysis.json'}")
        print(f"Saved feature summary: {summary_path}")
        print(f"Saved feature scatter: {scatter_path}")
        if has_entropy:
            print(f"Saved entropy histogram: {entropy_path}")
        else:
            print("Skipped entropy histogram: no finite entropy values (meta_csv missing or unmatched).")
        print(f"Saved top feature grid: {grid_path}")

        if args.save_individual_top_features:
            individual_dir = output_dir / args.individual_top_features_dir
            saved_count = save_top_feature_reference_images_individual(
                feature_analysis,
                output_dir=individual_dir,
                top_n=args.top_n,
                top_k=args.top_k,
            )
            print(f"Saved individual top-feature images: {individual_dir} (count={saved_count})")


if __name__ == "__main__":
    main()
