from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_activation_histogram(
    activation_values: np.ndarray,
    output_path: str | Path,
    bins: int = 120,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(activation_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.9, edgecolor="none")
    ax.set_title("Activation Histogram")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sparsity_distribution(
    sparsity_values: np.ndarray,
    output_path: str | Path,
    bins: int = 80,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(sparsity_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#d62728", alpha=0.9, edgecolor="none")
    ax.set_title("Sparsity Distribution")
    ax.set_xlabel("Per-sample Sparsity (Fraction of Zeros)")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
