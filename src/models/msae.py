"""
MSAE — Matryoshka (Multi-Scale) Sparse Autoencoder

참고:
  - /home/kimhj/projects/MSAE/sae.py (MatryoshkaAutoencoder)
  - /home/kimhj/projects/pathoSAEv2/scripts/sae/msae_model.py (PathoMSAE)

아키텍처:
  - 공유 Encoder/Decoder (W_enc, W_dec, b_enc, b_dec)
  - 동일한 pre-activation에 k값이 다른 TopK를 각각 독립 적용
  - 각 level별 reconstruction MSE의 가중 합으로 loss 계산 (L1 없음, TopK가 sparsity 강제)

  nesting_list = [64, 128, 256, 512]:
    level 0: TopK(k=64)  → decoder → MSE₀  (가장 sparse, 핵심 feature)
    level 1: TopK(k=128) → decoder → MSE₁
    level 2: TopK(k=256) → decoder → MSE₂
    level 3: TopK(k=512) → decoder → MSE₃  (가장 dense, 세밀한 feature)

  loss = Σᵢ (wᵢ × MSEᵢ) / Σ wᵢ   [UW: wᵢ=1.0 / RW: wᵢ=n-i]

  forward() returns:
    sae_out      = 마지막 level (k=512) reconstruction
    feature_acts = 마지막 level (k=512) activations
    loss_dict    = {loss, mse_loss, mse_k64, mse_k128, ..., sparsity_loss, l0}
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import SAEConfig
from src.models.base_sae import BaseSAE


class MSAESAE(BaseSAE):
    """Matryoshka SAE: multi-scale TopK 구조."""

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.nesting_list: list[int] = list(cfg.msae_nesting_list)
        assert len(self.nesting_list) >= 1, "msae_nesting_list must have at least 1 element"
        assert all(k > 0 for k in self.nesting_list), "all k values must be positive"
        assert self.nesting_list == sorted(self.nesting_list), "msae_nesting_list must be sorted ascending"

        # 가중치: UW(uniform) 또는 RW(reverse)
        if cfg.msae_importance == "reverse":
            n = len(self.nesting_list)
            self.importance_weights = [float(n - i) for i in range(n)]
        else:  # "uniform" (default)
            self.importance_weights = [1.0] * len(self.nesting_list)

    # ── abstract method stubs (실제 로직은 forward 오버라이드에서 처리) ───
    def activate(self, pre_act: Tensor) -> Tensor:
        """사용 안 함 — forward()가 직접 multi-scale 처리."""
        raise NotImplementedError("MSAESAE.activate() is not called directly; use forward()")

    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        """사용 안 함 — forward()에 통합됨."""
        raise NotImplementedError("MSAESAE.compute_loss() is not called directly; use forward()")

    # ── 핵심: multi-scale forward ────────────────────────────────────────
    def forward(self, x: Tensor, step: int = 0) -> tuple[Tensor, Tensor, dict]:
        """
        Returns:
            sae_out      : [batch, d_in]  — 마지막 level(최대 k) reconstruction
            feature_acts : [batch, d_sae] — 마지막 level activations
            loss_dict    : {loss, mse_loss, mse_k<k>, ..., sparsity_loss, l0}
        """
        # ── Encode ───────────────────────────────────────────────────────
        pre_act = self.encode(x)
        hidden_dim = pre_act.size(-1)

        # ── Multi-scale TopK activation ───────────────────────────────────
        all_acts: list[Tensor] = []
        for k in self.nesting_list:
            k_clamped = min(k, hidden_dim)
            values, indices = torch.topk(pre_act, k=k_clamped, dim=-1)
            acts = torch.zeros_like(pre_act)
            acts.scatter_(-1, indices, torch.relu(values))
            all_acts.append(acts)

        # ── Decode each level & compute per-level MSE ─────────────────────
        level_mses: list[Tensor] = []
        for acts in all_acts:
            recon = self.decode(acts)
            level_mses.append(F.mse_loss(recon, x))

        # ── Weighted reconstruction loss ──────────────────────────────────
        w_sum = sum(self.importance_weights)
        weighted_mse = sum(
            w * mse for w, mse in zip(self.importance_weights, level_mses)
        ) / w_sum

        # ── Loss dict ─────────────────────────────────────────────────────
        loss_dict: dict[str, Tensor] = {}
        for k, mse in zip(self.nesting_list, level_mses):
            loss_dict[f"mse_k{k}"] = mse
        loss_dict["mse_loss"] = weighted_mse
        loss_dict["sparsity_loss"] = torch.zeros((), device=x.device, dtype=x.dtype)
        loss_dict["loss"] = weighted_mse

        # L0: 마지막 level (가장 dense) 기준
        last_acts = all_acts[-1]
        loss_dict["l0"] = (last_acts != 0).float().sum(dim=-1).mean()

        # ── Return 마지막 level as primary output ─────────────────────────
        sae_out = self.decode(last_acts)
        return sae_out, last_acts, loss_dict
