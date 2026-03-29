from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import SAEConfig
from src.models.base_sae import BaseSAE


class GatedSAE(BaseSAE):
    """
    Gated SAE (Rajamanoharan et al., 2024 https://arxiv.org/abs/2404.16014)

    Gate path  : pi_gate = pre_act + b_gate  →  f_gate = ReLU(pi_gate)
    Mag path   : f_mag   = ReLU(pre_act * exp(r_mag) + b_mag)
    Output     : feature_acts = (pi_gate > 0) * f_mag

    Loss = L_recon + λ * L_sparse + L_aux
      L_recon  : MSE(sae_out, x)
      L_sparse : ||f_gate||_1        ← L1 on gate, not on feature_acts
      L_aux    : MSE(f_gate @ W_dec.detach(), x)  ← gate-only recon (decoder frozen)
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_dim))
        self.r_mag  = nn.Parameter(torch.zeros(self.hidden_dim))  # log-scale magnitude
        self.b_mag  = nn.Parameter(torch.zeros(self.hidden_dim))

    def activate(self, pre_act: Tensor) -> Tensor:
        pi_gate = pre_act + self.b_gate
        gate    = (pi_gate > 0).float()
        f_mag   = torch.relu(pre_act * torch.exp(self.r_mag) + self.b_mag)
        return gate * f_mag

    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        mse_loss = F.mse_loss(sae_out, x)

        # f_gate: gate pre-activation (for L1 and L_aux)
        pre_act = self._cached_pre_act          # set by BaseSAE.forward()
        f_gate  = torch.relu(pre_act + self.b_gate)

        # L_sparse: L1 on f_gate so b_gate receives gradient
        l1            = f_gate.sum(dim=-1).mean()
        sparsity_loss = self.cfg.l1_coeff * self.get_warmup_scale(step) * l1

        # L_aux: gate-path reconstruction with frozen decoder
        x_hat_gate = f_gate @ self.W_dec.detach() + self.b_dec.detach()
        aux_loss   = F.mse_loss(x_hat_gate, x)

        total_loss = mse_loss + sparsity_loss + aux_loss
        l0 = (feature_acts > 0).float().sum(dim=-1).mean()

        return {
            "loss":          total_loss,
            "mse_loss":      mse_loss,
            "sparsity_loss": sparsity_loss,
            "aux_loss":      aux_loss,
            "l0":            l0,
        }
