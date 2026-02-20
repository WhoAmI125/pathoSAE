from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.extract.macenko import MacenkoNormalizer
from src.extract.vit_encoder import EXAONEPathViTEncoder


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def resolve_device(gpu_arg: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"

    gpu_idx = find_free_gpu() if gpu_arg == "auto" else int(gpu_arg)
    torch.cuda.set_device(gpu_idx)
    return f"cuda:{gpu_idx}"


class PathologyTileDataset(Dataset):
    def __init__(self, image_paths: list[Path], normalizer: MacenkoNormalizer):
        self.image_paths = image_paths
        self.normalizer = normalizer
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]

        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        try:
            normalized = self.normalizer(image)
            if not isinstance(normalized, torch.Tensor):
                normalized = transforms.functional.to_tensor(normalized)
        except Exception:
            normalized = transforms.functional.to_tensor(image)

        out = self.resize(normalized)
        out = self.normalize(out.float())
        return out, str(image_path)


def _collect_pngs(folder: Path) -> list[Path]:
    pngs = list(folder.glob("*.png")) + list(folder.glob("*.PNG"))
    return sorted(set(pngs))


def collect_folders(image_dir: Path) -> list[tuple[Path, list[Path]]]:
    subdirs = sorted([p for p in image_dir.iterdir() if p.is_dir()])

    folders: list[tuple[Path, list[Path]]] = []
    if subdirs:
        for subdir in subdirs:
            images = _collect_pngs(subdir)
            if images:
                folders.append((subdir, images))
    else:
        images = _collect_pngs(image_dir)
        if images:
            folders.append((image_dir, images))

    return folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EXAONEPath spatial activations")
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="data/activations", help="Output directory")
    parser.add_argument(
        "--backbone_path",
        type=str,
        default="models/backbone/EXAONEPath.ckpt",
        help="Backbone checkpoint path",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument(
        "--macenko_target",
        type=str,
        default=None,
        help="Optional target image for Macenko fitting",
    )
    parser.add_argument("--gpu", type=str, default="auto", help="GPU id or 'auto'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir does not exist or is not a directory: {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.gpu)
    normalizer = MacenkoNormalizer(target_path=args.macenko_target)
    encoder = EXAONEPathViTEncoder(backbone_path=args.backbone_path, device=device)

    folder_items = collect_folders(image_dir)
    if not folder_items:
        raise RuntimeError(f"No PNG images found under: {image_dir}")

    for folder_idx, (folder_path, image_paths) in enumerate(folder_items):
        dataset = PathologyTileDataset(image_paths=image_paths, normalizer=normalizer)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
            persistent_workers=args.num_workers > 0,
        )

        vectors_per_folder: list[torch.Tensor] = []
        saved_paths: list[str] = []

        for images, paths in tqdm(loader, desc=f"Extract {folder_path.name}", unit="batch"):
            spatial_tokens = encoder.extract_spatial_tokens(images)
            vectors_per_folder.append(spatial_tokens.detach().cpu().to(torch.float16))
            saved_paths.extend(paths)

        if not vectors_per_folder:
            continue

        merged = torch.cat(vectors_per_folder, dim=0)
        save_path = output_dir / f"spatial_folder_{folder_idx:02d}.pt"

        torch.save(
            {
                "vectors": merged,
                "paths": saved_paths,
            },
            save_path,
        )

        print(
            f"Saved {save_path} | folder={folder_path.name} | "
            f"shape={tuple(merged.shape)}"
        )

        del merged
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
