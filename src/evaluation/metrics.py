from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.extract.activation_store import ActivationStore
from src.models.base_sae import BaseSAE


@dataclass
class EvaluationArtifacts:
    metrics: dict[str, Any]
    activation_samples: np.ndarray
    sparsity_samples: np.ndarray
    output_dir: Path


def _sample_tensor_values(
    tensor: torch.Tensor,
    buffer: list[np.ndarray],
    current_size: int,
    max_size: int,
    generator: torch.Generator,
) -> int:
    if current_size >= max_size:
        return current_size

    flat = tensor.reshape(-1).detach().cpu()
    if flat.numel() == 0:
        return current_size

    remaining = max_size - current_size
    if flat.numel() <= remaining:
        sampled = flat
    else:
        indices = torch.randperm(flat.numel(), generator=generator)[:remaining]
        sampled = flat[indices]

    buffer.append(sampled.numpy())
    return current_size + int(sampled.numel())


@torch.no_grad()
def compute_reconstruction_metrics(
    model: BaseSAE,
    data_loader,
    device: str,
    max_activation_samples: int = 500_000,
    max_sparsity_samples: int = 250_000,
    sample_seed: int = 42,
) -> EvaluationArtifacts:
    model.eval()
    device_obj = torch.device(device)

    sum_sq_error = 0.0
    sum_sq_x = 0.0
    sum_x = torch.zeros(model.input_dim, dtype=torch.float64)
    sample_count = 0

    l0_sum = 0.0
    zero_count = 0
    total_activation_count = 0
    cosine_sum = 0.0

    active_mask = torch.zeros(model.hidden_dim, dtype=torch.bool, device=device_obj)
    fire_count = torch.zeros(model.hidden_dim, dtype=torch.long, device=device_obj)

    activation_buffers: list[np.ndarray] = []
    sparsity_buffers: list[np.ndarray] = []
    activation_size = 0
    sparsity_size = 0

    g = torch.Generator()
    g.manual_seed(sample_seed)

    for batch in data_loader:
        x = batch.to(device_obj, dtype=torch.float32, non_blocking=True)
        sae_out, feature_acts, _ = model(x, step=0)

        diff = sae_out - x
        sum_sq_error += float(diff.pow(2).sum().item())
        sum_sq_x += float(x.pow(2).sum().item())
        sum_x += x.sum(dim=0, dtype=torch.float64).cpu()

        batch_size = int(x.shape[0])
        sample_count += batch_size

        nonzero_mask = feature_acts != 0
        l0_per_sample = nonzero_mask.float().sum(dim=-1)
        l0_sum += float(l0_per_sample.sum().item())

        zero_count += int((~nonzero_mask).sum().item())
        total_activation_count += int(nonzero_mask.numel())

        cosine_sum += float(
            F.cosine_similarity(x, sae_out, dim=-1, eps=1e-8).sum().item()
        )

        active_mask |= nonzero_mask.any(dim=0)
        fire_count += nonzero_mask.long().sum(dim=0)

        sample_sparsity = (1.0 - l0_per_sample / feature_acts.shape[1]).detach().cpu()
        sparsity_size = _sample_tensor_values(
            sample_sparsity,
            sparsity_buffers,
            sparsity_size,
            max_sparsity_samples,
            g,
        )
        activation_size = _sample_tensor_values(
            feature_acts,
            activation_buffers,
            activation_size,
            max_activation_samples,
            g,
        )

    if sample_count == 0:
        raise RuntimeError("No evaluation samples were produced by data_loader.")

    mse = sum_sq_error / float(sample_count * model.input_dim)

    mean_x = sum_x / float(sample_count)
    total_variance = sum_sq_x - float(sample_count * (mean_x.pow(2).sum().item()))
    fvu = float("nan") if total_variance <= 0 else (sum_sq_error / total_variance)

    l0 = l0_sum / float(sample_count)
    sparsity = zero_count / float(total_activation_count)
    cosine_sim = cosine_sum / float(sample_count)

    active_neurons = int(active_mask.sum().item())
    total_neurons = int(model.hidden_dim)
    dead_neurons = total_neurons - active_neurons

    # Frequency-based dead neuron counts (more meaningful than "never fired")
    freq = fire_count.float() / float(sample_count)
    dead_freq_1e4 = int((freq < 1e-4).sum().item())   # CytoSAE-style threshold
    dead_freq_1e3 = int((freq < 1e-3).sum().item())
    active_freq_1pct = int((freq >= 0.01).sum().item())

    metrics = {
        "model_type": model.cfg.model_type,
        "mse": float(mse),
        "fvu": float(fvu),
        "l0": float(l0),
        "sparsity": float(sparsity),
        "cosine_sim": float(cosine_sim),
        "active_neurons": active_neurons,
        "dead_neurons": dead_neurons,
        "total_neurons": total_neurons,
        "dead_freq_1e4": dead_freq_1e4,
        "dead_freq_1e3": dead_freq_1e3,
        "active_freq_1pct": active_freq_1pct,
    }

    activation_samples = (
        np.concatenate(activation_buffers) if activation_buffers else np.zeros(1, dtype=np.float32)
    )
    sparsity_samples = (
        np.concatenate(sparsity_buffers) if sparsity_buffers else np.zeros(1, dtype=np.float32)
    )

    return EvaluationArtifacts(
        metrics=metrics,
        activation_samples=activation_samples,
        sparsity_samples=sparsity_samples,
        output_dir=Path("."),
    )


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str | None = None,
    results_dir: str | None = None,
    device: str = "cpu",
    split: str = "val",
    max_activation_samples: int = 500_000,
    max_sparsity_samples: int = 250_000,
) -> EvaluationArtifacts:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = BaseSAE.load(str(checkpoint), device=device)
    cfg = model.cfg

    if data_dir is not None:
        cfg.data_dir = data_dir
    if results_dir is not None:
        cfg.results_dir = results_dir

    run_name = cfg.run_name if cfg.run_name else checkpoint.parent.name
    output_dir = Path(cfg.results_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    store = ActivationStore(cfg)
    eval_loader = store.get_loader(split=split)

    artifacts = compute_reconstruction_metrics(
        model=model,
        data_loader=eval_loader,
        device=device,
        max_activation_samples=max_activation_samples,
        max_sparsity_samples=max_sparsity_samples,
        sample_seed=cfg.seed,
    )
    artifacts.output_dir = output_dir
    artifacts.metrics["split"] = split

    save_metrics(artifacts.metrics, output_dir / "metrics.json")
    return artifacts
