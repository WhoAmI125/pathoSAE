from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.config import SAEConfig


SUPPORTED_DECAY = {"cosine", "linear"}


def _lr_scale(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_start_step: int,
    min_ratio: float,
    decay_type: str,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if step < decay_start_step:
        return 1.0

    decay_steps = max(1, total_steps - decay_start_step)
    progress = min(1.0, max(0.0, float(step - decay_start_step) / float(decay_steps)))

    if decay_type == "linear":
        return min_ratio + (1.0 - progress) * (1.0 - min_ratio)

    # cosine
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + cosine * (1.0 - min_ratio)


def build_scheduler(
    optimizer: Optimizer,
    cfg: SAEConfig,
    steps_per_epoch: int,
    decay_type: str = "cosine",
    last_epoch: int = -1,
) -> tuple[LambdaLR, int]:
    """Warmup + cosine/linear decay scheduler."""
    if decay_type not in SUPPORTED_DECAY:
        raise ValueError(f"Unsupported decay_type '{decay_type}'. Use one of {SUPPORTED_DECAY}.")

    total_steps = max(1, cfg.epochs * max(1, steps_per_epoch))
    warmup_steps = min(max(0, cfg.lr_warmup_steps), total_steps)

    decay_start_step = int(total_steps * cfg.lr_decay_start)
    decay_start_step = max(warmup_steps, min(decay_start_step, total_steps))

    min_ratio = cfg.lr_min / max(cfg.lr, 1e-12)

    def lr_lambda(step: int) -> float:
        return _lr_scale(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_start_step=decay_start_step,
            min_ratio=min_ratio,
            decay_type=decay_type,
        )

    if last_epoch != -1:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    return scheduler, total_steps
