# contracts.md
> Opus 작성 · Codex **엄격 준수** · `v1.0.0` · 2026-02-20

---

## 규칙
1. 아래 I/O 스펙은 **계약**이다. Codex는 반드시 준수한다.
2. 스펙 변경이 필요하면 구현하지 말고 `codex_notes.md`에 변경 요청을 남긴다.
3. 타입은 Python typing 표기법을 따른다.

---

## 1. Config — `src/config.py`

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class SAEConfig:
    """논문용 통합 설정. 모든 variant가 이 Config 하나를 공유한다."""

    # === Model ===
    model_type: str = "vanilla"          # {vanilla, gated, topk, jumprelu}
    input_dim: int = 768                 # EXAONEPath ViT-B output dim
    expansion_factor: int = 32           # hidden_dim = input_dim * expansion_factor
    # derived: hidden_dim computed in __post_init__

    # === Training (통일 — 논문 비교용) ===
    batch_size: int = 4096
    epochs: int = 10
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_warmup_steps: int = 500
    lr_decay_start: float = 0.7          # epoch 비율
    weight_decay: float = 0.0
    seed: int = 42
    grad_clip: float = 1.0

    # === Sparsity ===
    l1_coeff: float = 1e-4               # vanilla, gated
    l0_coeff: float = 1e-4               # jumprelu
    topk_k: int = 64                     # topk
    sparsity_warmup: float = 0.3         # warmup 비율

    # === JumpReLU ===
    jumprelu_init_threshold: float = 0.05
    jumprelu_bandwidth: float = 0.001

    # === TopK (MSAE multi-level) ===
    topk_nesting_list: list = field(default_factory=lambda: [64])

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
```

---

## 2. BaseSAE Interface — `src/models/base_sae.py`

```python
class BaseSAE(nn.Module, ABC):
    """모든 SAE variant의 공통 인터페이스."""

    def __init__(self, cfg: SAEConfig): ...

    # 공통 파라미터: W_enc, b_enc, W_dec, b_dec

    def encode(self, x: Tensor) -> Tensor:
        """[batch, d_in] → [batch, d_sae] pre-activation"""

    @abstractmethod
    def activate(self, pre_act: Tensor) -> Tensor:
        """[batch, d_sae] → [batch, d_sae] sparse activations"""

    def decode(self, feature_acts: Tensor) -> Tensor:
        """[batch, d_sae] → [batch, d_in] reconstruction"""

    @abstractmethod
    def compute_loss(self, x, sae_out, feature_acts, step) -> dict:
        """Returns: {"loss", "mse_loss", "sparsity_loss", "l0"}"""

    def forward(self, x: Tensor, step: int = 0) -> tuple[Tensor, Tensor, dict]:
        """Returns: (sae_out, feature_acts, loss_dict)"""

    def set_decoder_norm_to_unit_norm(self): ...
    def remove_gradient_parallel_to_decoder_directions(self): ...
    def save(self, path: str): ...

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseSAE": ...
```

---

## 3. Variant 구현 계약

### VanillaSAE
- `activate`: `ReLU(pre_act)`
- `compute_loss`: `MSE + l1_coeff * warmup * L1(acts)`

### GatedSAE
- 추가 파라미터: `b_gate: [d_sae]`
- `activate`: `ReLU(pre_act) * (pre_act + b_gate > 0).float()`
- `compute_loss`: `MSE + l1_coeff * warmup * L1(acts)`

### TopKSAE
- `activate`: `TopK(pre_act, k=topk_k)` — top-k만 남기고 나머지 0
- `compute_loss`: `MSE` (L1 불필요, k가 sparsity 직접 제어)

### JumpReLUSAE
- 추가 파라미터: `log_threshold: [d_sae]`
- `activate`: `pre_act * H(pre_act - exp(log_threshold))` (Heaviside)
- `compute_loss`: `MSE + l0_coeff * warmup * L0(acts)`
- **Custom autograd**: threshold gradient에 kernel-based pseudo-gradient 적용
- pathoSAEv2 `scripts/sae/jumprelu_model.py` 참조하여 구현

---

## 4. Model Factory — `src/models/__init__.py`

```python
MODEL_REGISTRY = {
    "vanilla": VanillaSAE,
    "gated": GatedSAE,
    "topk": TopKSAE,
    "jumprelu": JumpReLUSAE,
}

def create_sae(cfg: SAEConfig) -> BaseSAE:
    return MODEL_REGISTRY[cfg.model_type](cfg)
```

---

## 5. SAETrainer — `src/training/trainer.py`

```python
class SAETrainer:
    def __init__(self, sae: BaseSAE, cfg: SAEConfig, device: str): ...

    def train(self) -> None:
        """
        전체 학습 루프.

        Saves to: models/checkpoints/{run_name}/
            best.pt, final.pt, epoch_{n}.pt, config.json

        Logs per step: loss, mse_loss, sparsity_loss, l0, lr
        Logs per epoch: val_mse, val_l0, val_sparsity, dead_neurons, active_neurons
        """

    def _train_step(self, batch: Tensor) -> dict: ...
    def _validate(self) -> dict: ...
```

---

## 6. ActivationStore — `src/extract/activation_store.py`

```python
class ActivationStore:
    """
    data/activations/spatial_folder_*.pt 로딩.

    Interface:
        store = ActivationStore(cfg)
        train_loader, val_loader = store.get_dataloaders()
        # yields: [batch_size, 768] tensors

    파일별 lazy loading → flatten [N, 196, 768] → [N*196, 768]
    """
```

---

## 7. Tasks CLI

### `tasks/train.py`
```
python tasks/train.py --model {vanilla|gated|topk|jumprelu} [옵션]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | required | SAE variant |
| `--expansion_factor` | int | 32 | hidden = input * expansion |
| `--batch_size` | int | 4096 | 배치 크기 |
| `--epochs` | int | 10 | 에폭 수 |
| `--lr` | float | 3e-4 | 학습률 |
| `--seed` | int | 42 | 랜덤 시드 |
| `--l1_coeff` | float | 1e-4 | L1 계수 |
| `--l0_coeff` | float | 1e-4 | L0 계수 |
| `--topk_k` | int | 64 | TopK k값 |
| `--data_dir` | str | data/activations | 데이터 경로 |
| `--run_name` | str | auto | 실험 이름 |
| `--no_wandb` | flag | False | W&B 비활성화 |
| `--resume` | str | None | 체크포인트 재개 |
| `--gpu` | int | auto | GPU ID |

### `tasks/evaluate.py`
```
python tasks/evaluate.py --checkpoint <path> [--data_dir <path>]
```

### `tasks/compare.py`
```
python tasks/compare.py --checkpoints <path1> <path2> ...
```

### `tasks/extract.py`
```
python tasks/extract.py --image_dir <path> --output_dir data/activations [--backbone_path <path>]
```

---

## 8. Checkpoint Format

```python
{
    "state_dict": OrderedDict,     # model.state_dict()
    "config": SAEConfig,           # dataclass
    "epoch": int,
    "step": int,
    "best_val_mse": float,
    "optimizer_state": dict,       # resume용
}
```

---

## 9. Metrics Output — `results/{run_name}/metrics.json`

```json
{
    "mse": 0.065,
    "fvu": 0.032,
    "l0": 174.0,
    "sparsity": 0.993,
    "cosine_sim": 0.987,
    "active_neurons": 1200,
    "dead_neurons": 23376,
    "total_neurons": 24576
}
```

---

> NOTE: **CODEX**: 타입과 시그니처를 정확히 따라. forward()의 리턴 형식 (sae_out, feature_acts, loss_dict) 변경 금지.
