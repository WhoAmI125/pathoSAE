from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm import tqdm

from src.extract.activation_store import ActivationStore
from src.models.base_sae import BaseSAE


def resolve_activation_files_for_split(
    cfg,
    split: str = "full",
    max_files: int | None = None,
) -> list[Path]:
    split_norm = split.lower()
    if split_norm not in {"train", "val", "full"}:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: train, val, full.")

    store = ActivationStore(cfg)
    if split_norm == "train":
        files = list(store.train_files)
    elif split_norm == "val":
        files = list(store.val_files)
    else:
        files = list(store.train_files) + list(store.val_files)

    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be >= 1 when provided.")
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(
            f"No activation files selected for split='{split_norm}' under '{cfg.data_dir}'."
        )

    return files


def _get_path_list(payload: dict[str, Any]) -> list[str] | None:
    if "paths" in payload:
        return [str(p) for p in payload["paths"]]
    if "patch_paths" in payload:
        return [str(p) for p in payload["patch_paths"]]
    return None


def iter_activation_batches(
    files: Sequence[Path],
    input_dim: int,
    batch_size: int,
) -> Iterable[tuple[torch.Tensor, list[str], torch.Tensor]]:
    for file_path in files:
        payload = torch.load(file_path, map_location="cpu")
        if "vectors" not in payload:
            raise KeyError(f"Missing 'vectors' in activation file: {file_path}")

        vectors = payload["vectors"]
        if vectors.ndim != 3 or vectors.shape[-1] != input_dim:
            raise ValueError(
                f"Invalid vectors shape in {file_path}: {tuple(vectors.shape)}; expected [N, 196, {input_dim}]"
            )

        n_images, n_patches, _ = vectors.shape
        flat = vectors.reshape(-1, input_dim).to(torch.float32)
        image_paths = _get_path_list(payload)

        for start in range(0, flat.shape[0], batch_size):
            end = min(start + batch_size, flat.shape[0])
            rows = torch.arange(start, end, dtype=torch.long)

            image_indices = torch.div(rows, n_patches, rounding_mode="floor")
            patch_indices = torch.remainder(rows, n_patches)

            if image_paths is None:
                source_paths = [
                    f"{file_path.name}::image_{int(idx)}" for idx in image_indices.tolist()
                ]
            else:
                source_paths = [image_paths[int(idx)] for idx in image_indices.tolist()]

            yield flat[start:end], source_paths, patch_indices


def _build_path_index(batch_paths: list[str], cache: dict[str, int], reverse: list[str]) -> torch.Tensor:
    source_ids: list[int] = []
    for path in batch_paths:
        if path not in cache:
            cache[path] = len(reverse)
            reverse.append(path)
        source_ids.append(cache[path])
    return torch.tensor(source_ids, dtype=torch.long)


@torch.no_grad()
def analyze_features(
    model: BaseSAE,
    activation_files: Sequence[Path],
    device: str,
    top_k: int = 5,
    batch_size: int = 1024,
    show_progress: bool = True,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    model.eval()
    device_obj = torch.device(device)

    sum_act: torch.Tensor | None = None
    sum_sq_act: torch.Tensor | None = None
    max_act: torch.Tensor | None = None
    nonzero_count: torch.Tensor | None = None

    top_values: torch.Tensor | None = None
    top_source_ids: torch.Tensor | None = None
    top_patch_indices: torch.Tensor | None = None

    source_to_id: dict[str, int] = {}
    id_to_source: list[str] = []

    n_samples = 0

    iterator = iter_activation_batches(
        files=activation_files,
        input_dim=model.input_dim,
        batch_size=batch_size,
    )
    if show_progress:
        iterator = tqdm(iterator, desc="Feature analysis", unit="batch")

    for batch, batch_paths, batch_patch_indices in iterator:
        acts = model(batch.to(device_obj, dtype=torch.float32))[1].detach().cpu()

        if sum_act is None:
            hidden_dim = acts.shape[1]
            sum_act = torch.zeros(hidden_dim, dtype=torch.float64)
            sum_sq_act = torch.zeros(hidden_dim, dtype=torch.float64)
            max_act = torch.full((hidden_dim,), float("-inf"), dtype=torch.float32)
            nonzero_count = torch.zeros(hidden_dim, dtype=torch.int64)

            top_values = torch.full((hidden_dim, top_k), float("-inf"), dtype=torch.float32)
            top_source_ids = torch.full((hidden_dim, top_k), -1, dtype=torch.long)
            top_patch_indices = torch.full((hidden_dim, top_k), -1, dtype=torch.long)

        sum_act += acts.sum(dim=0, dtype=torch.float64)
        sum_sq_act += acts.pow(2).sum(dim=0, dtype=torch.float64)
        max_act = torch.maximum(max_act, acts.max(dim=0).values)
        nonzero_count += (acts != 0).sum(dim=0)

        n_samples += int(acts.shape[0])

        source_ids = _build_path_index(batch_paths, source_to_id, id_to_source)

        k_batch = min(top_k, acts.shape[0])
        batch_vals, batch_rows = torch.topk(acts, k=k_batch, dim=0)

        batch_source_ids = source_ids[batch_rows]
        batch_patch = batch_patch_indices.to(torch.long)[batch_rows]

        candidate_values = torch.cat([top_values.t(), batch_vals], dim=0)
        candidate_source_ids = torch.cat([top_source_ids.t(), batch_source_ids], dim=0)
        candidate_patch = torch.cat([top_patch_indices.t(), batch_patch], dim=0)

        best_values, best_idx = torch.topk(candidate_values, k=top_k, dim=0)
        best_source_ids = torch.gather(candidate_source_ids, 0, best_idx)
        best_patch = torch.gather(candidate_patch, 0, best_idx)

        top_values = best_values.t().contiguous()
        top_source_ids = best_source_ids.t().contiguous()
        top_patch_indices = best_patch.t().contiguous()

    if n_samples == 0 or sum_act is None:
        raise RuntimeError("No activation samples available for feature analysis.")

    assert sum_sq_act is not None
    assert max_act is not None
    assert nonzero_count is not None
    assert top_values is not None
    assert top_source_ids is not None
    assert top_patch_indices is not None

    mean_act = (sum_act / float(n_samples)).to(torch.float32)
    var_act = (sum_sq_act / float(n_samples)) - (sum_act / float(n_samples)).pow(2)
    std_act = var_act.clamp_min(0).sqrt().to(torch.float32)
    activation_freq = (nonzero_count.to(torch.float64) / float(n_samples)).to(torch.float32)

    features: list[dict[str, Any]] = []
    hidden_dim = mean_act.shape[0]

    for feat_idx in range(hidden_dim):
        top_patches: list[dict[str, Any]] = []
        for rank in range(top_k):
            value = float(top_values[feat_idx, rank].item())
            if not torch.isfinite(top_values[feat_idx, rank]):
                continue

            source_id = int(top_source_ids[feat_idx, rank].item())
            if source_id < 0:
                continue

            top_patches.append(
                {
                    "rank": rank + 1,
                    "activation": value,
                    "path": id_to_source[source_id],
                    "patch_idx": int(top_patch_indices[feat_idx, rank].item()),
                }
            )

        features.append(
            {
                "feature_idx": feat_idx,
                "mean_activation": float(mean_act[feat_idx].item()),
                "std_activation": float(std_act[feat_idx].item()),
                "max_activation": float(max_act[feat_idx].item()),
                "activation_frequency": float(activation_freq[feat_idx].item()),
                "top_patches": top_patches,
            }
        )

    return {
        "model_type": model.cfg.model_type,
        "num_samples": n_samples,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "features": features,
    }


def analyze_checkpoint_features(
    checkpoint_path: str,
    data_dir: str | None = None,
    output_path: str | Path | None = None,
    device: str = "cpu",
    top_k: int = 5,
    batch_size: int = 1024,
    split: str = "full",
    max_files: int | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = BaseSAE.load(str(checkpoint), device=device)
    cfg = model.cfg

    if data_dir is not None:
        cfg.data_dir = data_dir

    files = resolve_activation_files_for_split(cfg, split=split, max_files=max_files)

    result = analyze_features(
        model=model,
        activation_files=files,
        device=device,
        top_k=top_k,
        batch_size=batch_size,
        show_progress=True,
    )
    result["split"] = split
    result["num_files"] = len(files)

    if output_path is None:
        run_name = cfg.run_name if cfg.run_name else checkpoint.parent.name
        output_path = Path(cfg.results_dir) / run_name / "feature_analysis.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
