from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SAEConfig:
    """논문용 통합 설정. 모든 variant가 이 Config 하나를 공유한다."""

    # === Model ===
    model_type: str = "vanilla"  # {vanilla, gated, topk, jumprelu, msae}
    input_dim: int = 768  # EXAONEPath ViT-B output dim
    expansion_factor: int = 32  # hidden_dim = input_dim * expansion_factor
    # derived: hidden_dim computed in __post_init__

    # === Training (통일 — 논문 비교용) ===
    batch_size: int = 4096
    epochs: int = 10
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_warmup_steps: int = 500
    lr_decay_start: float = 0.7  # epoch 비율
    weight_decay: float = 0.0
    seed: int = 42
    grad_clip: float = 1.0

    # === Sparsity ===
    l1_coeff: float = 1e-4  # vanilla, gated
    l0_coeff: float = 1e-4  # jumprelu
    topk_k: int = 64  # topk
    sparsity_warmup: float = 0.3  # warmup 비율

    # === JumpReLU ===
    jumprelu_init_threshold: float = 0.05
    jumprelu_bandwidth: float = 0.001

    # === TopK ===
    topk_nesting_list: list = field(default_factory=lambda: [64])

    # === MSAE (Matryoshka Multi-Scale SAE) ===
    msae_nesting_list: list = field(default_factory=lambda: [64, 128, 256, 512])
    msae_importance: str = "uniform"  # "uniform" | "reverse"

    # === Data ===
    data_dir: str = "data/activations"
    num_workers: int = 4
    val_ratio: float = 0.1

    # === Checkpoint ===
    checkpoint_dir: str = "models/checkpoints"
    save_every_epoch: int = 2
    run_name: str = ""

    # === Logging ===
    log_to_wandb: bool = True
    wandb_project: str = "pathoSAEv3"
    wandb_log_freq: int = 100

    # === Backbone ===
    backbone_path: str = "models/backbone/EXAONEPath.ckpt"

    # === Evaluation ===
    results_dir: str = "results"

    def __post_init__(self):
        self.hidden_dim = self.input_dim * self.expansion_factor
        if not self.run_name:
            self.run_name = f"{self.model_type}_e{self.expansion_factor}_ep{self.epochs}"

    def to_dict(self) -> dict:
        return asdict(self) | {"hidden_dim": self.hidden_dim}

    @property
    def run_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.run_name
