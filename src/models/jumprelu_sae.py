from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import SAEConfig
from src.models.base_sae import BaseSAE


def rectangle_kernel(x: Tensor) -> Tensor:
    return (x.abs() <= 0.5).float()


def gaussian_kernel(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def triangular_kernel(x: Tensor) -> Tensor:
    return F.relu(1 - x.abs())


KERNEL_FUNCTIONS = {
    "rectangle": rectangle_kernel,
    "gaussian": gaussian_kernel,
    "triangular": triangular_kernel,
}


class JumpReLU(torch.autograd.Function):
    """Custom autograd for JumpReLU activation."""

    @staticmethod
    def forward(ctx, z: Tensor, threshold: Tensor, bandwidth: float, kernel_fn):
        mask = (z > threshold).float()
        activated = z * mask

        ctx.save_for_backward(z, threshold, mask)
        ctx.bandwidth = bandwidth
        ctx.kernel_fn = kernel_fn
        return activated

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        z, threshold, mask = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        kernel_fn = ctx.kernel_fn

        grad_z = grad_output * mask

        delta = (z - threshold) / bandwidth
        kernel_values = kernel_fn(delta)
        grad_threshold = -(z / bandwidth) * kernel_values * grad_output
        grad_threshold = grad_threshold.sum(dim=0)

        return grad_z, grad_threshold, None, None


class StepFunction(torch.autograd.Function):
    """Custom autograd for Heaviside step in L0 loss."""

    @staticmethod
    def forward(ctx, z: Tensor, threshold: Tensor, bandwidth: float, kernel_fn):
        mask = (z > threshold).float()
        ctx.save_for_backward(z, threshold)
        ctx.bandwidth = bandwidth
        ctx.kernel_fn = kernel_fn
        return mask

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        z, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        kernel_fn = ctx.kernel_fn

        delta = (z - threshold) / bandwidth
        kernel_values = kernel_fn(delta)

        grad_z = (kernel_values / bandwidth) * grad_output
        grad_threshold = -(kernel_values / bandwidth) * grad_output
        grad_threshold = grad_threshold.sum(dim=0)

        return grad_z, grad_threshold, None, None


class JumpReLUSAE(BaseSAE):
    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)

        if cfg.jumprelu_init_threshold <= 0:
            raise ValueError("jumprelu_init_threshold must be positive")

        init_log_threshold = math.log(cfg.jumprelu_init_threshold)
        self.log_threshold = nn.Parameter(
            torch.full((self.hidden_dim,), init_log_threshold)
        )

        self.bandwidth = cfg.jumprelu_bandwidth
        self.kernel_fn = KERNEL_FUNCTIONS["rectangle"]

    @property
    def threshold(self) -> Tensor:
        return torch.exp(self.log_threshold)

    def activate(self, pre_act: Tensor) -> Tensor:
        return JumpReLU.apply(pre_act, self.threshold, self.bandwidth, self.kernel_fn)

    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        mse_loss = F.mse_loss(sae_out, x)

        if self._cached_pre_act is None:
            raise RuntimeError("Pre-activation cache is missing for JumpReLU loss.")

        l0_proxy = StepFunction.apply(
            self._cached_pre_act,
            self.threshold,
            self.bandwidth,
            self.kernel_fn,
        )
        l0 = l0_proxy.sum(dim=-1).mean()

        sparsity_loss = self.cfg.l0_coeff * self.get_warmup_scale(step) * l0
        total_loss = mse_loss + sparsity_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "sparsity_loss": sparsity_loss,
            "l0": l0,
        }
