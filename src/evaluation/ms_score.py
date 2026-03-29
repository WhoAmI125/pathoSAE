"""
MonoSemanticity (MS) Score — Rajamanoharan et al. 스타일

각 SAE 뉴런이 "얼마나 하나의 개념만 포착하는가"를 정량화.
독립 encoder(DINOv2)로 이미지 유사도를 측정하여 activation 가중 평균 cosine similarity 계산.

MS_j = Σ_{n<m} ã_n·ã_m · cos(E(x_n), E(x_m))
       ──────────────────────────────────────────
                    Σ_{n<m} ã_n·ã_m

  ã_n = min-max normalized activation of neuron j for sample n
  E(x_n) = DINOv2 CLS embedding of the patch image x_n
  MS → 1.0: monosemantic  |  MS → 0.0: polysemantic
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from src.models.base_sae import BaseSAE


# ──────────────────────────────────────────────────────────
# DINOv2 독립 encoder
# ──────────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _pil_to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """PIL RGB → [3, 224, 224] float32, ImageNet normalized."""
    img = img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0          # [H, W, 3]
    t = torch.from_numpy(arr).permute(2, 0, 1)             # [3, H, W]
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD            # normalized


def load_dinov2(variant: str = "dinov2_vitb14", device: str = "cuda") -> torch.nn.Module:
    """DINOv2 모델 로드 (torch.hub 캐시 사용)."""
    model = torch.hub.load("facebookresearch/dinov2", variant, pretrained=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def encode_images_dinov2(
    model: torch.nn.Module,
    image_paths: list[str],
    device: str,
    batch_size: int = 64,
) -> Tensor:
    """이미지 리스트 → DINOv2 CLS 임베딩 [N, D]."""
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(_pil_to_tensor_normalized(img))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model(batch)        # [B, D]  CLS token
        embeddings.append(feats.cpu())
    return torch.cat(embeddings, dim=0)   # [N, D]


# ──────────────────────────────────────────────────────────
# 샘플링 + SAE 활성화 추출 (파일당 1회 로드, 메모리 효율)
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_samples(
    sae_model: BaseSAE,
    activation_files: list[Path],
    sample_size: int,
    device: str,
    sae_batch_size: int = 512,
    seed: int = 42,
) -> tuple[Tensor, list[str]]:
    """
    activation .pt 파일들에서 파일당 1회 로드로 샘플 수집.

    각 파일에서 균등하게 n_per_file = ceil(sample_size / n_files) 샘플 수집.
    파일을 한번만 로드 → 샘플링 + SAE 활성화 추출 동시 수행.

    Returns:
        A:           [N, H] SAE 활성화 (float32)
        patch_paths: [N] 원본 이미지 경로
    """
    rng = random.Random(seed)
    n_files = len(activation_files)
    n_per_file = max(1, (sample_size + n_files - 1) // n_files)

    sae_model.eval()
    device_obj = torch.device(device)

    all_acts: list[Tensor] = []
    all_paths: list[str] = []

    for fi, fpath in enumerate(activation_files):
        if len(all_paths) >= sample_size:
            break

        remaining = sample_size - len(all_paths)
        take = min(n_per_file, remaining)

        data = torch.load(fpath, map_location="cpu")
        n_patches = data["vectors"].shape[0]     # [N_patches, 196, 768]
        paths = data["patch_paths"]

        # 균등 간격 샘플링 (patch 단위, token은 랜덤)
        step = max(1, n_patches // take)
        patch_indices = list(range(0, n_patches, step))[:take]
        if len(patch_indices) < take:
            extra = rng.sample(range(n_patches), take - len(patch_indices))
            patch_indices.extend(extra)
        rng.shuffle(patch_indices)
        patch_indices = patch_indices[:take]

        token_indices = [rng.randint(0, 195) for _ in patch_indices]

        # SAE 활성화 추출 (배치 단위)
        for i in range(0, len(patch_indices), sae_batch_size):
            pi_batch = patch_indices[i : i + sae_batch_size]
            ti_batch = token_indices[i : i + sae_batch_size]

            vecs = [data["vectors"][pi, ti].float() for pi, ti in zip(pi_batch, ti_batch)]
            x = torch.stack(vecs).to(device_obj)

            pre_act = sae_model.encode(x)
            acts = sae_model.activate(pre_act)
            all_acts.append(acts.cpu())

            for pi in pi_batch:
                all_paths.append(paths[pi])

        print(f"  File {fi+1}/{n_files}: loaded {len(all_paths)} samples so far")
        del data  # 메모리 해제

    A = torch.cat(all_acts, dim=0)[:sample_size].float()
    all_paths = all_paths[:sample_size]
    return A, all_paths


# ──────────────────────────────────────────────────────────
# MS Score 계산 (핵심)
# ──────────────────────────────────────────────────────────

def compute_ms_score_from_tensors(
    A: Tensor,    # [N, H] SAE activations (raw)
    E: Tensor,    # [N, D] independent encoder embeddings
    min_activation_samples: int = 10,
) -> Tensor:
    """
    A, E가 주어졌을 때 MS Score [H] 계산.

    효율적 벡터화:
      S = cosine_sim(E, E)           [N, N]
      ã = min-max normalized A       [N, H]
      SA = S @ ã                     [N, H]
      numerator[j]   = (ã[:,j] * SA[:,j]).sum() - (ã[:,j]**2).sum()
      denominator[j] = ã[:,j].sum()**2 - (ã[:,j]**2).sum()
      MS[j] = numerator[j] / denominator[j]
    """
    N, H = A.shape

    # cosine similarity matrix S [N, N]
    E_norm = F.normalize(E.float(), dim=-1)         # [N, D]
    S = E_norm @ E_norm.T                           # [N, N]

    # min-max normalize each neuron's activation over N samples
    a_min = A.min(dim=0, keepdim=True).values       # [1, H]
    a_max = A.max(dim=0, keepdim=True).values       # [1, H]
    denom_norm = (a_max - a_min).clamp(min=1e-8)
    A_norm = (A - a_min) / denom_norm               # [N, H]  in [0, 1]

    # 활성 샘플 수 필터: 너무 적게 켜진 뉴런은 NaN 처리
    nonzero_counts = (A > 0).sum(dim=0).float()    # [H]
    valid_mask = nonzero_counts >= min_activation_samples

    # SA = S @ A_norm  [N, H]
    SA = S @ A_norm                                  # [N, H]

    # numerator = Σ_{n,m} ã_n ã_m S_nm  - Σ_n ã_n²
    numerator = (A_norm * SA).sum(dim=0) - (A_norm ** 2).sum(dim=0)  # [H]

    # denominator = (Σ_n ã_n)² - Σ_n ã_n²
    sum_a = A_norm.sum(dim=0)                       # [H]
    sum_a2 = (A_norm ** 2).sum(dim=0)               # [H]
    denominator = sum_a ** 2 - sum_a2               # [H]

    ms = torch.where(
        valid_mask & (denominator > 1e-8),
        numerator / denominator.clamp(min=1e-8),
        torch.full_like(numerator, float("nan")),
    )
    return ms   # [H], nan = insufficient activation


# ──────────────────────────────────────────────────────────
# 메인 진입점
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ms_score(
    sae_model: BaseSAE,
    activation_files: list[Path],
    dino_model: torch.nn.Module,
    device: str = "cuda",
    sample_size: int = 5000,
    dino_batch_size: int = 64,
    sae_batch_size: int = 512,
    seed: int = 42,
    min_activation_samples: int = 10,
) -> dict[str, Any]:
    """
    MS Score 전체 파이프라인.

    Returns dict:
        ms_scores: list[float | None]  — 뉴런별 MS Score (None = insufficient activation)
        valid_neurons: int             — 유효한 뉴런 수 (nan 제외)
        mean_ms: float                 — valid 뉴런의 평균 MS Score
        median_ms: float               — valid 뉴런의 중앙값 MS Score
        frac_ms_gt05: float            — MS > 0.5인 뉴런 비율
        frac_ms_gt07: float            — MS > 0.7인 뉴런 비율
        sample_size: int               — 실제 사용된 샘플 수
    """
    print(f"[MS Score] Sampling & extracting SAE activations "
          f"({sample_size} samples from {len(activation_files)} files)...")
    A, patch_paths = collect_samples(
        sae_model=sae_model,
        activation_files=activation_files,
        sample_size=sample_size,
        device=device,
        sae_batch_size=sae_batch_size,
        seed=seed,
    )
    actual_n = len(patch_paths)
    print(f"[MS Score] SAE activation matrix: {A.shape}  "
          f"(unique patches: {len(set(patch_paths))})")

    # DINOv2 이미지 임베딩
    print("[MS Score] Extracting DINOv2 embeddings...")
    E = encode_images_dinov2(
        dino_model, patch_paths, device=device, batch_size=dino_batch_size
    )   # [N, D]
    print(f"[MS Score] DINOv2 embedding matrix: {E.shape}")

    # MS Score 계산 (GPU 사용 — 대형 행렬 연산)
    print("[MS Score] Computing MS Score for all neurons...")
    device_obj = torch.device(device)
    ms = compute_ms_score_from_tensors(
        A.to(device_obj), E.to(device_obj),
        min_activation_samples=min_activation_samples,
    ).cpu()   # [H]

    # torch만 사용 (numpy 호환성 이슈 회피)
    valid_mask_t = torch.isfinite(ms)
    valid_ms_t   = ms[valid_mask_t]
    n_valid      = int(valid_mask_t.sum().item())
    total        = int(ms.numel())

    mean_ms   = float(valid_ms_t.mean().item())   if n_valid > 0 else float("nan")
    median_ms = float(valid_ms_t.median().item()) if n_valid > 0 else float("nan")
    gt05      = float((valid_ms_t > 0.5).float().mean().item()) if n_valid > 0 else float("nan")
    gt07      = float((valid_ms_t > 0.7).float().mean().item()) if n_valid > 0 else float("nan")

    # 뉴런별 MS Score 리스트 (None = insufficient activation)
    ms_scores_list = []
    for v in ms.tolist():
        import math
        ms_scores_list.append(None if not math.isfinite(v) else float(v))

    result = {
        "ms_scores":      ms_scores_list,
        "valid_neurons":  n_valid,
        "total_neurons":  total,
        "mean_ms":        mean_ms,
        "median_ms":      median_ms,
        "frac_ms_gt05":   gt05,
        "frac_ms_gt07":   gt07,
        "sample_size":    actual_n,
        "encoder":        "dinov2_vitb14",
    }

    print(f"[MS Score] Done.")
    print(f"  Valid neurons : {n_valid} / {total}")
    print(f"  Mean MS Score : {mean_ms:.4f}")
    print(f"  Median MS     : {median_ms:.4f}")
    print(f"  MS > 0.5      : {gt05*100:.1f}%")
    print(f"  MS > 0.7      : {gt07*100:.1f}%")

    return result
