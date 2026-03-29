from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _safe_log10(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.clip(values.astype(np.float64), eps, None))


def _normalize_path(path: str) -> str:
    return os.path.normpath(str(path))


def _resolve_meta_entry(path: str, meta_lookup: Mapping[str, dict[str, str]] | None) -> dict[str, str] | None:
    if not meta_lookup:
        return None

    norm = _normalize_path(path)
    if norm in meta_lookup:
        return meta_lookup[norm]

    basename = Path(norm).name
    if basename in meta_lookup:
        return meta_lookup[basename]

    try:
        resolved = _normalize_path(str(Path(path).resolve()))
        if resolved in meta_lookup:
            return meta_lookup[resolved]
    except Exception:
        pass

    return None


def _weighted_entropy(labels: list[str], weights: list[float], eps: float = 1e-12) -> float:
    if not labels or not weights:
        return float("nan")

    total = float(sum(weights))
    if total <= 0:
        return float("nan")

    by_label: dict[str, float] = {}
    for label, weight in zip(labels, weights):
        by_label[label] = by_label.get(label, 0.0) + float(weight)

    probs = np.array([w / total for w in by_label.values()], dtype=np.float64)
    return float(-(probs * np.log(probs + eps)).sum())


def build_feature_dataframe(
    feature_analysis_result: dict[str, Any],
    meta_lookup: Mapping[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feat in feature_analysis_result.get("features", []):
        top_patches = feat.get("top_patches", [])

        labels: list[str] = []
        weights: list[float] = []
        for patch in top_patches:
            path = str(patch.get("path", ""))
            entry = _resolve_meta_entry(path, meta_lookup)
            if not entry:
                continue
            cls = str(entry.get("class", "")).strip()
            if not cls:
                continue
            labels.append(cls)
            weights.append(float(patch.get("activation", 0.0)))

        entropy = _weighted_entropy(labels, weights)
        class_coverage = len(set(labels)) if labels else 0
        top1 = float(top_patches[0]["activation"]) if top_patches else float("nan")

        rows.append(
            {
                "feature_idx": int(feat.get("feature_idx", -1)),
                "mean_activation": float(feat.get("mean_activation", 0.0)),
                "std_activation": float(feat.get("std_activation", 0.0)),
                "max_activation": float(feat.get("max_activation", 0.0)),
                "activation_frequency": float(feat.get("activation_frequency", 0.0)),
                "top1_activation": top1,
                "num_top_patches": int(len(top_patches)),
                "entropy": entropy,
                "class_coverage": int(class_coverage),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_idx",
                "mean_activation",
                "std_activation",
                "max_activation",
                "activation_frequency",
                "top1_activation",
                "num_top_patches",
                "entropy",
                "class_coverage",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values("mean_activation", ascending=False).reset_index(drop=True)


def plot_feature_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Feature Scatter (no data)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    x = _safe_log10(df["activation_frequency"].to_numpy(dtype=np.float64))
    y = _safe_log10(df["mean_activation"].to_numpy(dtype=np.float64))

    entropy = df["entropy"].to_numpy(dtype=np.float64)
    has_entropy = np.isfinite(entropy).any()
    color = entropy if has_entropy else df["max_activation"].to_numpy(dtype=np.float64)
    color_label = "Entropy" if has_entropy else "Max Activation"

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, c=color, cmap="viridis", s=18, alpha=0.85, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_label)

    ax.set_title("Feature Scatter: log10(freq) vs log10(mean activation)")
    ax.set_xlabel("log10(activation_frequency)")
    ax.set_ylabel("log10(mean_activation)")
    ax.grid(alpha=0.2)

    for _, row in df.head(5).iterrows():
        xi = math.log10(max(float(row["activation_frequency"]), 1e-12))
        yi = math.log10(max(float(row["mean_activation"]), 1e-12))
        ax.annotate(f"F{int(row['feature_idx'])}", (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_entropy_hist(df: pd.DataFrame, output_path: str | Path, bins: int = 40) -> bool:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or "entropy" not in df.columns:
        return False

    entropy = df["entropy"].to_numpy(dtype=np.float64)
    entropy = entropy[np.isfinite(entropy)]
    if entropy.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy, bins=bins, color="#2ca02c", alpha=0.9, edgecolor="none")
    ax.set_title("Feature Entropy Histogram")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def plot_top_feature_reference_grid(
    feature_analysis_result: dict[str, Any],
    output_path: str | Path,
    top_n: int = 10,
    top_k: int = 5,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = feature_analysis_result.get("features", [])
    if not features:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Top Feature Reference Grid (no data)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    features = sorted(features, key=lambda x: float(x.get("mean_activation", 0.0)), reverse=True)[:top_n]

    rows = len(features)
    cols = max(1, top_k)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.8))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for row, feature in enumerate(features):
        feat_idx = int(feature.get("feature_idx", -1))
        mean_act = float(feature.get("mean_activation", 0.0))
        freq = float(feature.get("activation_frequency", 0.0))
        top_patches = feature.get("top_patches", [])

        for col in range(cols):
            ax = axes[row, col]
            if col < len(top_patches):
                patch = top_patches[col]
                path = str(patch.get("path", ""))
                act = float(patch.get("activation", 0.0))
                if path and Path(path).exists():
                    try:
                        img = Image.open(path).convert("RGB")
                        ax.imshow(img)
                        ax.set_title(f"#{col + 1} act={act:.2f}", fontsize=8)
                    except Exception:
                        ax.text(0.5, 0.5, "load error", ha="center", va="center", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
            ax.axis("off")

        axes[row, 0].text(
            -0.45,
            0.5,
            f"F{feat_idx}\nmu={mean_act:.3f}\nfreq={freq:.4f}",
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=8,
            fontweight="bold",
        )

    fig.suptitle("Top Feature Reference Images", fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.16, wspace=0.08, hspace=0.45)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_top_feature_reference_images_individual(
    feature_analysis_result: dict[str, Any],
    output_dir: str | Path,
    top_n: int = 10,
    top_k: int = 5,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = feature_analysis_result.get("features", [])
    if not features:
        return 0

    features = sorted(features, key=lambda x: float(x.get("mean_activation", 0.0)), reverse=True)[:top_n]
    saved_count = 0

    for feature in features:
        feat_idx = int(feature.get("feature_idx", -1))
        mean_act = float(feature.get("mean_activation", 0.0))
        freq = float(feature.get("activation_frequency", 0.0))
        top_patches = feature.get("top_patches", [])[:top_k]

        feat_dir = output_dir / f"feature_{feat_idx:05d}"
        feat_dir.mkdir(parents=True, exist_ok=True)

        meta_path = feat_dir / "meta.txt"
        meta_path.write_text(
            f"feature_idx: {feat_idx}\n"
            f"mean_activation: {mean_act:.8f}\n"
            f"activation_frequency: {freq:.8f}\n"
            f"num_top_patches: {len(top_patches)}\n",
            encoding="utf-8",
        )

        for rank, patch in enumerate(top_patches, start=1):
            path = str(patch.get("path", ""))
            act = float(patch.get("activation", 0.0))
            out_name = f"rank_{rank:02d}_act_{act:.6f}.png"
            out_path = feat_dir / out_name

            if path and Path(path).exists():
                try:
                    img = Image.open(path).convert("RGB")
                    img.save(out_path)
                    saved_count += 1
                    continue
                except Exception:
                    pass

            # fallback placeholder when source is missing/unreadable
            placeholder = Image.new("RGB", (224, 224), color=(245, 245, 245))
            draw = ImageDraw.Draw(placeholder)
            draw.text((10, 10), f"missing\nrank={rank}\nact={act:.4f}", fill=(30, 30, 30))
            placeholder.save(out_path)
            saved_count += 1

    return saved_count
