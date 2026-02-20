from __future__ import annotations

from src.config import SAEConfig
from src.models.base_sae import BaseSAE
from src.models.gated_sae import GatedSAE
from src.models.jumprelu_sae import JumpReLUSAE
from src.models.topk_sae import TopKSAE
from src.models.vanilla_sae import VanillaSAE

MODEL_REGISTRY = {
    "vanilla": VanillaSAE,
    "gated": GatedSAE,
    "topk": TopKSAE,
    "jumprelu": JumpReLUSAE,
}


def create_sae(cfg: SAEConfig) -> BaseSAE:
    if cfg.model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{cfg.model_type}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[cfg.model_type](cfg)


__all__ = [
    "BaseSAE",
    "VanillaSAE",
    "GatedSAE",
    "TopKSAE",
    "JumpReLUSAE",
    "MODEL_REGISTRY",
    "create_sae",
]
