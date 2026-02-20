from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import SAEConfig
from src.models.base_sae import BaseSAE


class TopKSAE(BaseSAE):
    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.k = int(cfg.topk_k)
        self.nesting_list = list(cfg.topk_nesting_list)

    def activate(self, pre_act: Tensor) -> Tensor:
        if self.k <= 0:
            return torch.zeros_like(pre_act)

        k = min(self.k, pre_act.size(-1))
        values, indices = torch.topk(pre_act, k=k, dim=-1)
        out = torch.zeros_like(pre_act)
        out.scatter_(-1, indices, torch.relu(values))
        return out

    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        mse_loss = F.mse_loss(sae_out, x)
        sparsity_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        l0 = (feature_acts != 0).float().sum(dim=-1).mean()

        return {
            "loss": mse_loss,
            "mse_loss": mse_loss,
            "sparsity_loss": sparsity_loss,
            "l0": l0,
        }
