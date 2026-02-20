from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import SAEConfig
from src.models.base_sae import BaseSAE


class VanillaSAE(BaseSAE):
    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)

    def activate(self, pre_act: Tensor) -> Tensor:
        return torch.relu(pre_act)

    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        mse_loss = F.mse_loss(sae_out, x)
        l1 = feature_acts.abs().sum(dim=-1).mean()
        sparsity_loss = self.cfg.l1_coeff * self.get_warmup_scale(step) * l1
        total_loss = mse_loss + sparsity_loss
        l0 = (feature_acts > 0).float().sum(dim=-1).mean()

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "sparsity_loss": sparsity_loss,
            "l0": l0,
        }
