from __future__ import annotations

import argparse
import random
import subprocess

import numpy as np
import torch

from src.config import SAEConfig
from src.extract.activation_store import ActivationStore
from src.models import create_sae
from src.training.trainer import SAETrainer


def find_free_gpu(min_free_mb: int = 20000) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates: list[tuple[int, int]] = []
        for line in result.stdout.strip().splitlines():
            idx_str, free_mem_str = line.split(",")
            candidates.append((int(idx_str.strip()), int(free_mem_str.strip())))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for idx, free_mem in candidates:
            if free_mem >= min_free_mb:
                return idx
        return candidates[0][0] if candidates else 0
    except Exception:
        return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PathoSAEv3 sparse autoencoder")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vanilla", "gated", "topk", "jumprelu"],
        help="SAE variant",
    )
    parser.add_argument("--expansion_factor", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l1_coeff", type=float, default=1e-4)
    parser.add_argument("--l0_coeff", type=float, default=1e-4)
    parser.add_argument("--topk_k", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data/activations")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU id or 'auto'",
    )

    return parser.parse_args()


def resolve_device(gpu_arg: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"

    if gpu_arg == "auto":
        gpu_idx = find_free_gpu()
    else:
        gpu_idx = int(gpu_arg)

    torch.cuda.set_device(gpu_idx)
    return f"cuda:{gpu_idx}"


def main() -> None:
    args = parse_args()

    cfg = SAEConfig(
        model_type=args.model,
        expansion_factor=args.expansion_factor,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        l1_coeff=args.l1_coeff,
        l0_coeff=args.l0_coeff,
        topk_k=args.topk_k,
        data_dir=args.data_dir,
        run_name=args.run_name,
        log_to_wandb=not args.no_wandb,
    )

    set_seed(cfg.seed)
    device = resolve_device(args.gpu)

    activation_store = ActivationStore(cfg)
    cfg._activation_store = activation_store  # type: ignore[attr-defined]

    sae = create_sae(cfg)

    # QD-003: decoder bias는 train 데이터 mean으로 초기화.
    with torch.no_grad():
        sae.b_dec.copy_(activation_store.compute_train_mean())

    trainer = SAETrainer(sae=sae, cfg=cfg, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
