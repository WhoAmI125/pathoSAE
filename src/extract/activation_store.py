from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.config import SAEConfig


class _ActivationBatchDataset(IterableDataset):
    def __init__(
        self,
        files: list[Path],
        file_patch_counts: dict[Path, int],
        input_dim: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.files = files
        self.file_patch_counts = file_patch_counts
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(
            math.ceil(self.file_patch_counts[path] / self.batch_size)
            for path in self.files
        )

    def _iter_assigned_files(self) -> list[Path]:
        worker_info = get_worker_info()
        files = self.files

        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        if self.shuffle and len(files) > 1:
            worker_seed = self.seed + self.epoch
            if worker_info is not None:
                worker_seed += worker_info.id
            g = torch.Generator()
            g.manual_seed(worker_seed)
            perm = torch.randperm(len(files), generator=g).tolist()
            files = [files[idx] for idx in perm]

        return files

    def __iter__(self) -> Iterable[torch.Tensor]:
        files = self._iter_assigned_files()

        worker_info = get_worker_info()
        local_seed = self.seed + self.epoch
        if worker_info is not None:
            local_seed += worker_info.id
        g = torch.Generator()
        g.manual_seed(local_seed)

        for path in files:
            data = torch.load(path, map_location="cpu")
            if "vectors" not in data:
                raise KeyError(f"Missing 'vectors' key in activation file: {path}")

            vectors = data["vectors"]
            if vectors.ndim != 3 or vectors.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Invalid vectors shape in {path}: {tuple(vectors.shape)}; expected [N, 196, {self.input_dim}]"
                )

            flat = vectors.reshape(-1, self.input_dim).to(torch.float32)

            if self.shuffle and flat.size(0) > 1:
                perm = torch.randperm(flat.size(0), generator=g)
                flat = flat[perm]

            for start in range(0, flat.size(0), self.batch_size):
                yield flat[start : start + self.batch_size]


class ActivationStore:
    """
    data/activations/spatial_folder_*.pt 로딩.

    Interface:
        store = ActivationStore(cfg)
        train_loader, val_loader = store.get_dataloaders()
        # yields: [batch_size, 768] tensors

    파일별 lazy loading → flatten [N, 196, 768] → [N*196, 768]
    """

    def __init__(self, cfg: SAEConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)

        pattern = str(self.data_dir / "spatial_folder_*.pt")
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(
                f"No activation files found under '{self.data_dir}'. Expected pattern: spatial_folder_*.pt"
            )

        self.file_patch_counts = self._scan_patch_counts(self.files)
        self.train_files, self.val_files = self._split_train_val_files(self.files)

        self._train_dataset: _ActivationBatchDataset | None = None
        self._val_dataset: _ActivationBatchDataset | None = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._train_mean: torch.Tensor | None = None

    def _scan_patch_counts(self, files: list[Path]) -> dict[Path, int]:
        counts: dict[Path, int] = {}
        for path in files:
            try:
                data = torch.load(str(path), map_location="cpu", mmap=True)
            except (TypeError, ValueError):
                data = torch.load(path, map_location="cpu")

            if "vectors" not in data:
                raise KeyError(f"Missing 'vectors' key in activation file: {path}")

            vectors = data["vectors"]
            if vectors.ndim != 3 or vectors.shape[-1] != self.cfg.input_dim:
                raise ValueError(
                    f"Invalid vectors shape in {path}: {tuple(vectors.shape)}; expected [N, 196, {self.cfg.input_dim}]"
                )
            counts[path] = int(vectors.shape[0] * vectors.shape[1])
        return counts

    def _split_train_val_files(self, files: list[Path]) -> tuple[list[Path], list[Path]]:
        n_files = len(files)

        if n_files == 1:
            # 최소 데이터셋에서는 train/val을 동일 파일로 사용.
            return files, files

        val_count = max(1, int(n_files * self.cfg.val_ratio))
        val_count = min(val_count, n_files - 1)

        train_files = files[:-val_count]
        val_files = files[-val_count:]
        return train_files, val_files

    def set_epoch(self, epoch: int) -> None:
        if self._train_dataset is not None:
            self._train_dataset.set_epoch(epoch)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        if self._train_loader is not None and self._val_loader is not None:
            return self._train_loader, self._val_loader

        self._train_dataset = _ActivationBatchDataset(
            files=self.train_files,
            file_patch_counts=self.file_patch_counts,
            input_dim=self.cfg.input_dim,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            seed=self.cfg.seed,
        )
        self._val_dataset = _ActivationBatchDataset(
            files=self.val_files,
            file_patch_counts=self.file_patch_counts,
            input_dim=self.cfg.input_dim,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            seed=self.cfg.seed,
        )

        common_loader_kwargs = {
            "batch_size": None,
            "num_workers": self.cfg.num_workers,
            "pin_memory": True,
            "persistent_workers": self.cfg.num_workers > 0,
        }

        self._train_loader = DataLoader(self._train_dataset, **common_loader_kwargs)
        self._val_loader = DataLoader(self._val_dataset, **common_loader_kwargs)
        return self._train_loader, self._val_loader

    def compute_train_mean(self) -> torch.Tensor:
        if self._train_mean is not None:
            return self._train_mean

        # 캐시 파일이 있으면 즉시 로드 (재계산 생략)
        cache_path = self.data_dir / "train_mean.pt"
        if cache_path.exists():
            self._train_mean = torch.load(str(cache_path), map_location="cpu")
            return self._train_mean

        import gc

        running_sum = torch.zeros(self.cfg.input_dim, dtype=torch.float64)
        running_count = 0

        for i, path in enumerate(self.train_files):
            data = torch.load(path, map_location="cpu")
            vectors = data["vectors"]
            flat = vectors.reshape(-1, self.cfg.input_dim).to(torch.float32)
            running_sum += flat.sum(dim=0, dtype=torch.float64)
            running_count += flat.shape[0]
            # 명시적 메모리 해제
            del flat, vectors, data
            gc.collect()
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(self.train_files)}] files processed", flush=True)

        if running_count == 0:
            raise RuntimeError("No training activations available to compute mean.")

        self._train_mean = (running_sum / running_count).to(torch.float32)
        torch.save(self._train_mean, str(cache_path))
        return self._train_mean

    @property
    def n_train_files(self) -> int:
        return len(self.train_files)

    @property
    def n_val_files(self) -> int:
        return len(self.val_files)
