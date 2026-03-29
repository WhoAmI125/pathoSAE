from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from src.evaluation.feature_analysis import iter_activation_batches, resolve_activation_files_for_split
from src.models.base_sae import BaseSAE


def normalize_path(path: str) -> str:
    return os.path.normpath(str(path))


def load_meta_csv(
    meta_csv: str | Path,
    path_col: str = "path",
    class_col: str = "class",
    patient_col: str = "patient",
) -> dict[str, dict[str, str]]:
    meta_path = Path(meta_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")

    df = pd.read_csv(meta_path)
    required = [path_col, class_col, patient_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Meta CSV is missing required columns: {missing}. "
            f"Expected columns include: {required}"
        )

    lookup: dict[str, dict[str, str]] = {}
    basename_rows: dict[str, list[dict[str, str]]] = {}

    for _, row in df.iterrows():
        path_raw = str(row[path_col]).strip()
        if not path_raw:
            continue

        cls = str(row[class_col]).strip()
        patient = str(row[patient_col]).strip()
        record = {"class": cls, "patient": patient}

        norm = normalize_path(path_raw)
        lookup[norm] = record

        try:
            resolved = normalize_path(str(Path(path_raw).resolve()))
            lookup[resolved] = record
        except Exception:
            pass

        basename = Path(norm).name
        basename_rows.setdefault(basename, []).append(record)

    # Allow basename fallback only when it is unique.
    for basename, rows in basename_rows.items():
        unique_pairs = {(r["class"], r["patient"]) for r in rows}
        if len(unique_pairs) == 1:
            lookup[basename] = rows[0]

    if not lookup:
        raise ValueError(f"No usable rows found in meta CSV: {meta_path}")

    return lookup


def match_meta_record(path: str, meta_lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    norm = normalize_path(path)
    if norm in meta_lookup:
        return meta_lookup[norm]

    basename = Path(norm).name
    if basename in meta_lookup:
        return meta_lookup[basename]

    try:
        resolved = normalize_path(str(Path(path).resolve()))
        if resolved in meta_lookup:
            return meta_lookup[resolved]
    except Exception:
        pass

    return None


@torch.no_grad()
def _compute_top_feature_indices(
    model: BaseSAE,
    activation_files: list[Path],
    meta_lookup: dict[str, dict[str, str]],
    device: str,
    top_n: int,
    batch_size: int,
) -> list[int]:
    model.eval()
    device_obj = torch.device(device)

    sum_act = torch.zeros(model.hidden_dim, dtype=torch.float64)
    n_samples = 0

    iterator = iter_activation_batches(activation_files, model.input_dim, batch_size)
    iterator = tqdm(iterator, desc="Group Pass1 (top features)", unit="batch")

    for batch, batch_paths, _ in iterator:
        matched_idx = [i for i, p in enumerate(batch_paths) if match_meta_record(p, meta_lookup)]
        if not matched_idx:
            continue

        acts = model(batch.to(device_obj, dtype=torch.float32))[1].detach().cpu()
        selected = acts[matched_idx]
        sum_act += selected.sum(dim=0, dtype=torch.float64)
        n_samples += int(selected.shape[0])

    if n_samples == 0:
        raise RuntimeError("No activation rows matched metadata for group analysis.")

    mean_act = (sum_act / float(n_samples)).to(torch.float32)
    top_n = min(top_n, mean_act.numel())
    return torch.topk(mean_act, k=top_n).indices.tolist()


def _to_group_df(
    sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    top_feature_indices: list[int],
    min_group_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group, count in counts.items():
        if count < min_group_samples:
            continue

        mean_vec = (sums[group] / float(count)).cpu().numpy()
        row: dict[str, Any] = {"group": group, "sample_count": int(count)}
        for i, feat_idx in enumerate(top_feature_indices):
            row[f"f_{feat_idx}"] = float(mean_vec[i])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["group", "sample_count"] + [f"f_{i}" for i in top_feature_indices])

    df = pd.DataFrame(rows)
    return df.sort_values("sample_count", ascending=False).reset_index(drop=True)


@torch.no_grad()
def compute_group_feature_matrices(
    model: BaseSAE,
    activation_files: list[Path],
    meta_lookup: dict[str, dict[str, str]],
    device: str,
    top_feature_indices: list[int],
    batch_size: int = 1024,
    min_group_samples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    device_obj = torch.device(device)

    class_sums: dict[str, torch.Tensor] = {}
    class_counts: dict[str, int] = {}
    patient_sums: dict[str, torch.Tensor] = {}
    patient_counts: dict[str, int] = {}

    iterator = iter_activation_batches(activation_files, model.input_dim, batch_size)
    iterator = tqdm(iterator, desc="Group Pass2 (aggregate)", unit="batch")

    feat_idx_t = torch.tensor(top_feature_indices, dtype=torch.long)
    dim = len(top_feature_indices)

    for batch, batch_paths, _ in iterator:
        acts = model(batch.to(device_obj, dtype=torch.float32))[1].detach().cpu()
        selected = acts.index_select(dim=1, index=feat_idx_t)

        for row_idx, path in enumerate(batch_paths):
            record = match_meta_record(path, meta_lookup)
            if not record:
                continue

            vec = selected[row_idx].to(torch.float64)

            cls = str(record.get("class", "")).strip()
            if cls:
                if cls not in class_sums:
                    class_sums[cls] = torch.zeros(dim, dtype=torch.float64)
                    class_counts[cls] = 0
                class_sums[cls] += vec
                class_counts[cls] += 1

            patient = str(record.get("patient", "")).strip()
            if patient:
                if patient not in patient_sums:
                    patient_sums[patient] = torch.zeros(dim, dtype=torch.float64)
                    patient_counts[patient] = 0
                patient_sums[patient] += vec
                patient_counts[patient] += 1

    class_df = _to_group_df(class_sums, class_counts, top_feature_indices, min_group_samples)
    patient_df = _to_group_df(patient_sums, patient_counts, top_feature_indices, min_group_samples)
    return class_df, patient_df


def plot_group_heatmap(
    group_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    max_groups: int = 60,
) -> bool:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group_df.empty:
        return False

    feature_cols = [c for c in group_df.columns if c.startswith("f_")]
    if not feature_cols:
        return False

    plot_df = group_df.sort_values("sample_count", ascending=False).head(max_groups).copy()
    plot_df = plot_df.set_index("group")

    fig_h = max(5.0, min(20.0, 0.35 * len(plot_df) + 2.0))
    fig_w = max(8.0, min(22.0, 0.25 * len(feature_cols) + 6.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(plot_df[feature_cols], cmap="viridis", ax=ax, cbar_kws={"label": "Mean activation"})
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def analyze_groups_for_checkpoint(
    checkpoint_path: str,
    meta_csv: str,
    data_dir: str | None = None,
    results_dir: str | None = None,
    device: str = "cpu",
    split: str = "val",
    max_files: int | None = None,
    top_n: int = 50,
    batch_size: int = 1024,
    min_group_samples: int = 20,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = BaseSAE.load(str(checkpoint), device=device)
    cfg = model.cfg

    if data_dir is not None:
        cfg.data_dir = data_dir
    if results_dir is not None:
        cfg.results_dir = results_dir

    files = resolve_activation_files_for_split(cfg, split=split, max_files=max_files)
    meta_lookup = load_meta_csv(meta_csv)

    top_feature_indices = _compute_top_feature_indices(
        model=model,
        activation_files=files,
        meta_lookup=meta_lookup,
        device=device,
        top_n=top_n,
        batch_size=batch_size,
    )

    class_df, patient_df = compute_group_feature_matrices(
        model=model,
        activation_files=files,
        meta_lookup=meta_lookup,
        device=device,
        top_feature_indices=top_feature_indices,
        batch_size=batch_size,
        min_group_samples=min_group_samples,
    )

    run_name = cfg.run_name if cfg.run_name else checkpoint.parent.name
    output_dir = Path(cfg.results_dir) / run_name / "group"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_csv = output_dir / "class_feature_matrix.csv"
    patient_csv = output_dir / "patient_feature_matrix.csv"
    class_df.to_csv(class_csv, index=False)
    patient_df.to_csv(patient_csv, index=False)

    class_heatmap = output_dir / "class_feature_heatmap.png"
    patient_heatmap = output_dir / "patient_feature_heatmap.png"
    class_ok = plot_group_heatmap(class_df, class_heatmap, "Class-wise Feature Heatmap")
    patient_ok = plot_group_heatmap(patient_df, patient_heatmap, "Patient-wise Feature Heatmap")

    return {
        "output_dir": output_dir,
        "class_csv": class_csv,
        "patient_csv": patient_csv,
        "class_heatmap": class_heatmap if class_ok else None,
        "patient_heatmap": patient_heatmap if patient_ok else None,
        "top_feature_indices": top_feature_indices,
        "num_files": len(files),
        "split": split,
    }
