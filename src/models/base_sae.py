from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

from src.config import SAEConfig


class BaseSAE(nn.Module, ABC):
    """모든 SAE variant의 공통 인터페이스."""

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim

        # 공통 파라미터
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.input_dim, self.hidden_dim))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_dim))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.hidden_dim, self.input_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))

        with torch.no_grad():
            self.set_decoder_norm_to_unit_norm()

        self.total_training_steps: int | None = None
        self._cached_pre_act: Tensor | None = None

    def encode(self, x: Tensor) -> Tensor:
        """[batch, d_in] → [batch, d_sae] pre-activation"""
        sae_in = x - self.b_dec
        return sae_in @ self.W_enc + self.b_enc

    @abstractmethod
    def activate(self, pre_act: Tensor) -> Tensor:
        """[batch, d_sae] → [batch, d_sae] sparse activations"""

    def decode(self, feature_acts: Tensor) -> Tensor:
        """[batch, d_sae] → [batch, d_in] reconstruction"""
        return feature_acts @ self.W_dec + self.b_dec

    @abstractmethod
    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        """Returns: {"loss", "mse_loss", "sparsity_loss", "l0"}"""

    def forward(self, x: Tensor, step: int = 0) -> tuple[Tensor, Tensor, dict]:
        """Returns: (sae_out, feature_acts, loss_dict)"""
        pre_act = self.encode(x)
        self._cached_pre_act = pre_act
        feature_acts = self.activate(pre_act)
        sae_out = self.decode(feature_acts)
        loss_dict = self.compute_loss(x, sae_out, feature_acts, step)
        return sae_out, feature_acts, loss_dict

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(eps)
        self.W_dec.data /= norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        if self.W_dec.grad is None:
            return

        parallel_component = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel_component * self.W_dec.data

    def get_warmup_scale(self, step: int) -> float:
        if self.total_training_steps is None or self.total_training_steps <= 0:
            return 1.0
        warmup_steps = max(1, int(self.total_training_steps * self.cfg.sparsity_warmup))
        return min(1.0, float(step) / float(warmup_steps))

    def save(self, path: str):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "config": asdict(self.cfg),
            "model_type": self.cfg.model_type,
        }
        torch.save(payload, save_path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseSAE":
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        if isinstance(checkpoint, dict) and "config" in checkpoint:
            cfg_data = checkpoint["config"]
            cfg = cfg_data if isinstance(cfg_data, SAEConfig) else SAEConfig(**cfg_data)
        else:
            raise ValueError("Checkpoint must include 'config' to load SAE model.")

        if cls is BaseSAE:
            # Base class load must dispatch to specific variant.
            from src.models import create_sae

            model = create_sae(cfg)
        else:
            model = cls(cfg)

        model.load_state_dict(state_dict)
        model.to(device)
        return model
