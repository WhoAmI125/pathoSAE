from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SAEConfig:
    """논문용 통합 설정. 모든 variant가 이 Config 하나를 공유한다."""

    # === Model ===
    model_type: str = "vanilla"  # {vanilla, gated, topk, jumprelu, msae}
    encoder_name: str = "exaonepath"  # {exaonepath, uni}
    input_dim: int = 768  # auto-set by encoder_name in __post_init__
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

    # === DriftPrior-SAE ===
    drift_phi_dim: int = 256  # frozen projection 출력 차원
    drift_sub_batch: int = 512  # drift 계산용 sub-batch 크기
    drift_n_pos: int = 512  # prior에서 샘플링할 positive 수
    drift_tau: float = 0.1  # kernel temperature
    drift_beta_start: float = 0.01  # β warmup 시작값
    drift_beta_end: float = 0.1  # β warmup 종료값
    drift_prior_sparsity: float = 0.95  # spike-and-slab sparsity (95% inactive)
    drift_prior_scale: float = 1.0  # active 뉴런의 exponential scale

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
    backbone_path: str = "models/backbone/EXAONEPath.ckpt"  # auto-set by encoder_name

    # === Evaluation ===
    results_dir: str = "results"

    # encoder presets: (input_dim, default_backbone_path)
    ENCODER_PRESETS: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        self.ENCODER_PRESETS = {
            "exaonepath": (768, "models/backbone/EXAONEPath.ckpt"),
            "uni": (1024, "src/models/uni/pytorch_model.bin"),
        }
        if self.encoder_name in self.ENCODER_PRESETS:
            preset_dim, preset_path = self.ENCODER_PRESETS[self.encoder_name]
            # input_dim이 기본값(768)이면 encoder preset으로 덮어쓴다
            if self.input_dim == 768 and preset_dim != 768:
                self.input_dim = preset_dim
            # backbone_path가 기본값이면 encoder preset으로 덮어쓴다
            if self.backbone_path == "models/backbone/EXAONEPath.ckpt" and self.encoder_name != "exaonepath":
                self.backbone_path = preset_path

        self.hidden_dim = self.input_dim * self.expansion_factor
        if not self.run_name:
            encoder_suffix = f"_{self.encoder_name}" if self.encoder_name != "exaonepath" else ""
            self.run_name = f"{self.model_type}_e{self.expansion_factor}_ep{self.epochs}{encoder_suffix}"

    def to_dict(self) -> dict:
        return asdict(self) | {"hidden_dim": self.hidden_dim}

    @property
    def run_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.run_name
