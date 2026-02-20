from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import evaluate_checkpoint


DEFAULT_COLUMNS = [
    "run_name",
    "model_type",
    "mse",
    "fvu",
    "l0",
    "sparsity",
    "cosine_sim",
    "active_neurons",
    "dead_neurons",
    "total_neurons",
    "checkpoint",
]


def _pareto_front_mask(mse: pd.Series, l0: pd.Series) -> pd.Series:
    n = len(mse)
    is_pareto = [True] * n

    for i in range(n):
        dominated = ((mse <= mse.iloc[i]) & (l0 <= l0.iloc[i])) & (
            (mse < mse.iloc[i]) | (l0 < l0.iloc[i])
        )
        if dominated.any():
            is_pareto[i] = False

    return pd.Series(is_pareto, index=mse.index)


def _load_metrics_if_exists(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_metrics(
    checkpoints: Sequence[str],
    data_dir: str | None = None,
    results_dir: str = "results",
    device: str = "cpu",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        run_name = ckpt_path.parent.name
        metrics_path = Path(results_dir) / run_name / "metrics.json"

        metrics = _load_metrics_if_exists(metrics_path)
        if metrics is None:
            artifacts = evaluate_checkpoint(
                checkpoint_path=str(ckpt_path),
                data_dir=data_dir,
                results_dir=results_dir,
                device=device,
            )
            metrics = artifacts.metrics

        row = {
            "run_name": run_name,
            "checkpoint": str(ckpt_path),
            **metrics,
        }
        rows.append(row)

    return rows


def save_comparison_table(rows: list[dict[str, Any]], output_csv: str | Path) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[DEFAULT_COLUMNS].sort_values(by=["mse", "l0"], ascending=[True, True])
    df.to_csv(output_csv, index=False)
    return df


def plot_mse_vs_l0(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["l0"], df["mse"], s=70, c="#1f77b4", alpha=0.9)

    for _, row in df.iterrows():
        ax.annotate(row["run_name"], (row["l0"], row["mse"]), fontsize=8, xytext=(5, 3), textcoords="offset points")

    pareto_mask = _pareto_front_mask(df["mse"], df["l0"])
    pareto_df = df[pareto_mask].sort_values("l0")
    if len(pareto_df) >= 2:
        ax.plot(
            pareto_df["l0"],
            pareto_df["mse"],
            color="#d62728",
            linewidth=1.6,
            linestyle="--",
            label="Pareto Front",
        )
        ax.legend()

    ax.set_title("MSE vs L0")
    ax.set_xlabel("L0 (Active Features per Sample)")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compare_checkpoints(
    checkpoints: Sequence[str],
    data_dir: str | None = None,
    results_dir: str = "results",
    device: str = "cpu",
) -> tuple[pd.DataFrame, Path, Path]:
    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoints are required for comparison.")

    rows = collect_metrics(
        checkpoints=checkpoints,
        data_dir=data_dir,
        results_dir=results_dir,
        device=device,
    )

    comparison_dir = Path(results_dir) / "comparison"
    csv_path = comparison_dir / "comparison_table.csv"
    plot_path = comparison_dir / "mse_vs_l0.png"

    df = save_comparison_table(rows, csv_path)
    plot_mse_vs_l0(df, plot_path)
    return df, csv_path, plot_path
