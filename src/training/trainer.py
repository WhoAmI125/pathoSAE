from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.config import SAEConfig
from src.extract.activation_store import ActivationStore
from src.models.base_sae import BaseSAE
from src.training.scheduler import build_scheduler

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


class SAETrainer:
    def __init__(self, sae: BaseSAE, cfg: SAEConfig, device: str):
        self.sae = sae
        self.cfg = cfg
        self.device = torch.device(device)

        self.sae.to(self.device, dtype=torch.float32)

        self.activation_store = getattr(cfg, "_activation_store", None)
        if self.activation_store is None:
            self.activation_store = ActivationStore(cfg)

        self.train_loader, self.val_loader = self.activation_store.get_dataloaders()

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        self.scheduler, total_steps = build_scheduler(
            optimizer=self.optimizer,
            cfg=self.cfg,
            steps_per_epoch=max(1, len(self.train_loader)),
            decay_type="cosine",
            last_epoch=-1,
        )
        self.sae.total_training_steps = total_steps

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_mse = float("inf")

        self.run_dir = Path(cfg.checkpoint_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_config_json()

        self._wandb_run = None
        if self.cfg.log_to_wandb and wandb is not None:
            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                config=self.cfg.to_dict(),
                reinit=True,
            )

    def _save_config_json(self) -> None:
        cfg_path = self.run_dir / "config.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        checkpoint = {
            "state_dict": self.sae.state_dict(),
            "config": self.cfg,
            "epoch": epoch,
            "step": self.global_step,
            "best_val_mse": self.best_val_mse,
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.sae.load_state_dict(checkpoint["state_dict"])

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("step", 0))
        self.best_val_mse = float(checkpoint.get("best_val_mse", float("inf")))

        self.scheduler, total_steps = build_scheduler(
            optimizer=self.optimizer,
            cfg=self.cfg,
            steps_per_epoch=max(1, len(self.train_loader)),
            decay_type="cosine",
            last_epoch=self.global_step - 1,
        )
        self.sae.total_training_steps = total_steps

    def train(self) -> None:
        """
        전체 학습 루프.

        Saves to: models/checkpoints/{run_name}/
            best.pt, final.pt, epoch_{n}.pt, config.json

        Logs per step: loss, mse_loss, sparsity_loss, l0, lr
        Logs per epoch: val_mse, val_l0, val_sparsity, dead_neurons, active_neurons
        """
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.activation_store.set_epoch(epoch)
            self.sae.train()

            running_loss = 0.0
            running_mse = 0.0
            running_sparsity_loss = 0.0
            running_l0 = 0.0
            batch_count = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs}")
            for batch in pbar:
                step_log = self._train_step(batch)

                running_loss += step_log["loss"]
                running_mse += step_log["mse_loss"]
                running_sparsity_loss += step_log["sparsity_loss"]
                running_l0 += step_log["l0"]
                batch_count += 1

                pbar.set_postfix(
                    loss=f"{step_log['loss']:.4f}",
                    mse=f"{step_log['mse_loss']:.4f}",
                    l0=f"{step_log['l0']:.1f}",
                    lr=f"{step_log['lr']:.2e}",
                )

                if self.cfg.log_to_wandb and self._wandb_run is not None:
                    if self.global_step % self.cfg.wandb_log_freq == 0:
                        wandb.log(step_log, step=self.global_step)

                self.global_step += 1

            if batch_count == 0:
                raise RuntimeError("No training batches were produced by train_loader.")

            val_metrics = self._validate()

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": running_loss / batch_count,
                "train_mse": running_mse / batch_count,
                "train_sparsity_loss": running_sparsity_loss / batch_count,
                "train_l0": running_l0 / batch_count,
                **val_metrics,
            }

            if self.cfg.log_to_wandb and self._wandb_run is not None:
                wandb.log(epoch_log, step=self.global_step)

            if val_metrics["val_mse"] < self.best_val_mse:
                self.best_val_mse = val_metrics["val_mse"]
                self._save_checkpoint(self.run_dir / "best.pt", epoch + 1)

            if (epoch + 1) % self.cfg.save_every_epoch == 0:
                self._save_checkpoint(self.run_dir / f"epoch_{epoch + 1}.pt", epoch + 1)

        self._save_checkpoint(self.run_dir / "final.pt", self.cfg.epochs)

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def _train_step(self, batch: Tensor) -> dict:
        self.optimizer.zero_grad(set_to_none=True)

        batch = batch.to(self.device, dtype=torch.float32, non_blocking=True)

        self.sae.set_decoder_norm_to_unit_norm()
        sae_out, feature_acts, loss_dict = self.sae(batch, step=self.global_step)

        loss = loss_dict["loss"]
        loss.backward()

        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            clip_grad_norm_(self.sae.parameters(), self.cfg.grad_clip)

        self.sae.remove_gradient_parallel_to_decoder_directions()
        self.optimizer.step()
        self.scheduler.step()
        self.sae.set_decoder_norm_to_unit_norm()

        return {
            "loss": float(loss.detach().item()),
            "mse_loss": float(loss_dict["mse_loss"].detach().item()),
            "sparsity_loss": float(loss_dict["sparsity_loss"].detach().item()),
            "l0": float(loss_dict["l0"].detach().item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    @torch.no_grad()
    def _validate(self) -> dict:
        self.sae.eval()

        total_mse = 0.0
        total_l0 = 0.0
        total_sparsity = 0.0
        n_batches = 0

        active_mask = torch.zeros(self.sae.hidden_dim, dtype=torch.bool, device=self.device)

        for batch in self.val_loader:
            batch = batch.to(self.device, dtype=torch.float32, non_blocking=True)
            sae_out, feature_acts, _ = self.sae(batch, step=self.global_step)

            mse = F.mse_loss(sae_out, batch)
            l0 = (feature_acts != 0).float().sum(dim=-1).mean()
            sparsity = (feature_acts == 0).float().mean()

            total_mse += float(mse.item())
            total_l0 += float(l0.item())
            total_sparsity += float(sparsity.item())
            n_batches += 1

            active_mask |= (feature_acts != 0).any(dim=0)

        if n_batches == 0:
            raise RuntimeError("No validation batches were produced by val_loader.")

        active_neurons = int(active_mask.sum().item())
        dead_neurons = int((~active_mask).sum().item())

        return {
            "val_mse": total_mse / n_batches,
            "val_l0": total_l0 / n_batches,
            "val_sparsity": total_sparsity / n_batches,
            "dead_neurons": dead_neurons,
            "active_neurons": active_neurons,
        }
