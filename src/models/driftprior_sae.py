from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import SAEConfig
from src.models.base_sae import BaseSAE


class SpikeSlab:
    """Spike-and-Slab Prior: p(z_i) = pi*delta(0) + (1-pi)*Exp(lambda)."""

    def __init__(self, sparsity: float = 0.95, scale: float = 1.0):
        self.sparsity = sparsity
        self.scale = scale

    def sample(self, shape: tuple, device: torch.device) -> Tensor:
        n, d = shape
        mask = torch.bernoulli(
            torch.full((n, d), 1.0 - self.sparsity, device=device)
        )
        values = torch.empty(n, d, device=device).exponential_(1.0 / self.scale)
        return mask * values


class DriftPriorSAE(BaseSAE):
    """
    DriftPrior-SAE: distribution-matching drift loss for latent sparsity.

    Activation is ReLU (same as Vanilla).
    Sparsity is enforced by matching encoder output distribution
    to a Spike-and-Slab prior via a drifting field.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)

        # Frozen random projection: latent_dim -> phi_dim
        gen = torch.Generator().manual_seed(cfg.seed)
        W_phi = torch.randn(self.hidden_dim, cfg.drift_phi_dim, generator=gen)
        W_phi = F.normalize(W_phi, dim=0) * math.sqrt(
            cfg.drift_phi_dim / self.hidden_dim
        )
        self.register_buffer("W_phi", W_phi)

        self.prior = SpikeSlab(cfg.drift_prior_sparsity, cfg.drift_prior_scale)

    def activate(self, pre_act: Tensor) -> Tensor:
        return torch.relu(pre_act)

    def _get_beta(self, step: int) -> float:
        """Beta warmup: beta_start -> beta_end over sparsity_warmup fraction."""
        if self.total_training_steps is None or self.total_training_steps <= 0:
            return self.cfg.drift_beta_end
        warmup_steps = max(
            1, int(self.total_training_steps * self.cfg.sparsity_warmup)
        )
        if step >= warmup_steps:
            return self.cfg.drift_beta_end
        ratio = step / warmup_steps
        return (
            self.cfg.drift_beta_start
            + (self.cfg.drift_beta_end - self.cfg.drift_beta_start) * ratio
        )

    def _phi(self, z: Tensor) -> Tensor:
        """Project latent z to low-dim phi space."""
        return z @ self.W_phi

    @torch.no_grad()
    def _compute_drift_field(
        self, phi_gen: Tensor, phi_pos: Tensor, tau: float
    ) -> Tensor:
        """Drift field V = V+ (attraction) - V- (repulsion) in phi space."""
        # V+: prior samples attract
        dist_pos = torch.cdist(phi_gen, phi_pos)
        k_pos = (-dist_pos / tau).exp()
        Z_pos = k_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w_pos = k_pos / Z_pos
        V_plus = w_pos @ phi_pos - phi_gen

        # V-: generated samples repel (prevent mode collapse)
        dist_neg = torch.cdist(phi_gen, phi_gen)
        dist_neg.fill_diagonal_(1e6)
        k_neg = (-dist_neg / tau).exp()
        Z_neg = k_neg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w_neg = k_neg / Z_neg
        V_minus = w_neg @ phi_gen - phi_gen

        return V_plus - V_minus

    def _compute_drift_loss(self, feature_acts: Tensor) -> Tensor:
        """Drift loss with sub-batching for memory efficiency."""
        B, D = feature_acts.shape
        device = feature_acts.device
        cfg = self.cfg

        sub_size = min(cfg.drift_sub_batch, B)
        sub_idx = torch.randperm(B, device=device)[:sub_size]
        z_sub = feature_acts[sub_idx]

        z_prior = self.prior.sample((cfg.drift_n_pos, D), device=device)

        with torch.no_grad():
            phi_gen_ng = self._phi(z_sub.detach())
            phi_pos_ng = self._phi(z_prior)

        V = self._compute_drift_field(phi_gen_ng, phi_pos_ng, tau=cfg.drift_tau)

        phi_z = self._phi(z_sub)  # gradient flows through here
        target = (phi_z.detach() + V).detach()
        return F.mse_loss(phi_z, target)

    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        mse_loss = F.mse_loss(sae_out, x)
        drift_loss = self._compute_drift_loss(feature_acts)
        beta = self._get_beta(step)
        sparsity_loss = beta * drift_loss
        total_loss = mse_loss + sparsity_loss
        l0 = (feature_acts > 0).float().sum(dim=-1).mean()
        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "sparsity_loss": sparsity_loss,
            "l0": l0,
        }
