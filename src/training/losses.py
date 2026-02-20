from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(reconstruction, target)


def l1_loss(feature_acts: Tensor) -> Tensor:
    return feature_acts.abs().sum(dim=-1).mean()


def l0_loss(feature_acts: Tensor) -> Tensor:
    return (feature_acts != 0).float().sum(dim=-1).mean()


def sparsity(feature_acts: Tensor) -> Tensor:
    return (feature_acts == 0).float().mean()


def count_active_neurons(feature_acts: Tensor) -> tuple[int, int]:
    fired = (feature_acts != 0).any(dim=0)
    active = int(fired.sum().item())
    total = fired.numel()
    return active, total - active
