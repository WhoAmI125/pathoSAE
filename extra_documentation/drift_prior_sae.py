"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              DriftPrior-SAE: 완전한 구현 + 이론 설명서                        ║
║                                                                              ║
║  Drifting Field 기반 분포 매칭형 Sparse Autoencoder                          ║
║  PathoSAEv3의 5번째 variant로 설계                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

이 파일은 코드 + 이론 설명서를 겸합니다.
Drifting 논문을 읽지 않은 사람도 이해할 수 있도록 작성했습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 목차
  0. 한줄 요약
  1. 배경: SAE(Sparse Autoencoder)란?
  2. 문제: 기존 SAE의 한계
  3. 해법: Drifting이란 무엇인가?
  4. 핵심 아이디어: DriftPrior-SAE
  5. Toy Demo → Production 변환 포인트
  6. 사용법
  7. VRAM 예상치

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. 한줄 요약
===========
"SAE의 latent z가 우리가 원하는 sparse 분포를 따르도록,
 Drifting이라는 새로운 분포 매칭 기법으로 직접 밀어넣는다."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 배경: SAE(Sparse Autoencoder)란?
===================================

딥러닝 모델(예: ViT) 내부의 뉴런 활성화를 "해석"하고 싶다.
하지만 ViT의 768차원 벡터는 여러 개념이 뒤섞여(superposition) 있어서
사람이 직접 해석하기 어렵다.

SAE는 이 뒤섞인 표현을 "풀어헤치는" 도구다:

    h (768D, 뒤섞인 표현)
        ↓ Encoder
    z (24,576D, 풀어헤친 표현) ← 대부분 0, 소수만 활성
        ↓ Decoder
    ĥ (768D, 복원된 표현) ≈ h

핵심 아이디어:
- z의 차원을 입력보다 훨씬 크게 (32배) 만들어서 "자리"를 많이 확보
- z가 sparse하게 (대부분 0) 되도록 강제
- 각 z_i가 하나의 "개념"에 대응하길 기대 (monosemanticity)

예: z_1234가 활성 → "암세포 패턴", z_5678이 활성 → "정상 조직 패턴"

Loss:
    L = ||h - ĥ||² + λ·R(z)
        ─────────   ─────────
        재구성 품질    희소성 정규화
        (h를 잘 복원)  (z를 sparse하게)

R(z)를 어떻게 설계하느냐가 SAE variant의 핵심 차이점이다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. 문제: 기존 SAE 4종의 한계
============================

기존 4가지 방식 모두 "개별 뉴런의 활성값"에 페널티를 주는 간접적 방법이다:

  ┌────────────┬──────────────────────────┬───────────────────────┐
  │  Variant   │  R(z) 방식               │  한계                  │
  ├────────────┼──────────────────────────┼───────────────────────┤
  │  L1-SAE    │  λ·Σ|z_i|               │  λ 튜닝 어려움,        │
  │            │  각 뉴런에 L1 페널티      │  dead neuron 많음      │
  ├────────────┼──────────────────────────┼───────────────────────┤
  │  TopK-SAE  │  상위 K개만 남기고 0      │  K가 고정이라 유연성↓   │
  │            │  나머지 강제 제거          │  재구성 성능은 강력     │
  ├────────────┼──────────────────────────┼───────────────────────┤
  │  JumpReLU  │  학습 가능한 threshold    │  뉴런마다 다른 기준    │
  │            │  θ 이하면 0으로 점프      │  전역 분포 제어 불가    │
  ├────────────┼──────────────────────────┼───────────────────────┤
  │  L0-SAE    │  ||z||₀를 미분가능하게    │  relaxation 근사의     │
  │            │  근사해서 최적화           │  불안정성               │
  └────────────┴──────────────────────────┴───────────────────────┘

공통 문제:
  → "얼마나 sparse한가"만 제어 (값의 크기/개수)
  → "어떤 형태로 sparse한가"는 제어 불가 (분포의 모양)
  → 결과: dead neuron 발생, 활성화 불균형, polysemantic 뉴런 잔존

비유하면:
  기존 = "살 좀 빼세요" (얼마나 빼는지만 말함)
  DriftPrior = "이 체형(분포)을 따라가세요" (목표 형태를 명시)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. 해법: Drifting이란 무엇인가?
==============================

원 논문: Deng et al. "Generative Modeling via Drifting" (2026, MIT/Harvard)

Drifting은 원래 이미지 생성 모델을 위해 개발되었다.
핵심 아이디어를 직관적으로 설명하면:

[3.1] 기존 생성 모델 vs Drifting

  ● Diffusion/Flow: "inference 시 여러 번 반복" (50~1000 NFE)
    노이즈 → (반복 정제) → 이미지

  ● Drifting: "training 시 반복, inference는 1번" (1 NFE)
    노이즈 → (한 번에) → 이미지
    대신 학습 중에 "어디로 가야 하는지" 방향(drift)을 계산해서 밀어줌

[3.2] Drift Field V — 핵심 메커니즘

  현재 내가 생성한 분포(q)와 목표 분포(p)가 다를 때,
  각 샘플을 "올바른 방향"으로 밀어주는 벡터장(vector field)이 V이다.

  V = V⁺ (attraction) - V⁻ (repulsion)

  ● V⁺(x): "목표 데이터(p)가 x를 끌어당기는 힘"
    → x 근처의 목표 샘플들이 "이쪽으로 와!" 하고 당김
    → 수식: V⁺(x) = (1/Z_p) · Σ_{y⁺~p} k(x,y⁺)·(y⁺ - x)

  ● V⁻(x): "다른 생성 샘플(q)이 x를 밀어내는 힘"
    → x 근처의 다른 생성 샘플들이 "나랑 겹치지 마!" 하고 밈
    → 수식: V⁻(x) = (1/Z_q) · Σ_{y⁻~q} k(x,y⁻)·(y⁻ - x)

  여기서 k(x,y) = exp(-||x-y||/τ) 는 "가까울수록 강한 힘" 커널.
  τ (temperature)가 작으면 가까운 것만, 크면 멀리까지 영향.

  [핵심 성질] p = q이면 V = 0
  → 생성 분포가 목표에 도달하면 더 이상 밀어줄 필요 없음 → "평형 상태"
  → 이것이 수렴 보장의 이론적 근거

[3.3] Training Loss

  L = E[||x - stopgrad(x + V(x))||²]

  해석: "현재 위치 x"와 "밀려난 위치 x+V(x)" 사이의 거리를 줄여라.
  stopgrad: V는 방향 지시만 하고, gradient는 모델(encoder)만 업데이트.

  이 loss가 0이 되면 → V=0 → p=q → 목표 분포 도달!

[3.4] Feature Space Drifting (고차원 해결)

  문제: 고차원(예: 24,576D)에서는 모든 샘플 간 거리가 비슷해짐
        → k(x,y) 값이 전부 uniform → V가 의미 없어짐 ("curse of dimensionality")

  해결: 저차원 feature space φ로 투영한 뒤 drift 계산
        φ(z) ∈ R^256 에서 거리 계산 → 유의미한 kernel weight 확보

  논문에서도 feature encoder 없이 ImageNet에서 실패했다고 보고.
  → φ 설계가 성공의 핵심!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. 핵심 아이디어: DriftPrior-SAE
================================

"Drifting을 이미지 생성이 아니라, SAE의 latent 분포 제어에 쓰자!"

원래 Drifting:
    noise → Generator → image  (생성 분포 q → 데이터 분포 p로 매칭)

DriftPrior-SAE:
    h → Encoder → z  (encoder 출력 q(z) → sparse prior p(z)로 매칭)

즉, "encoder가 만드는 z의 분포"를
     "우리가 원하는 sparse 분포(prior)"로 직접 밀어넣는다!

[4.1] 전체 아키텍처

    h (768D, ViT 출력)
      │
      ▼ ─── pre_bias 빼기 (centering)
      │
    Encoder: z = ReLU(W_enc · h + b)    ← 24,576D latent
      │
      ├──▶ Decoder: ĥ = W_dec · z + pre_bias  → L_recon = ||h - ĥ||²
      │
      └──▶ [Training Only] Drift Loss 계산:
            │
            φ(z) ──▶ drift field V 계산 ──▶ L_drift
            │
            φ(z_prior) ◀── Spike-and-Slab에서 샘플링

    Total Loss = L_recon + β · L_drift
                 ────────   ──────────
                 "잘 복원"  "z 분포를 prior로 매칭"

[4.2] Spike-and-Slab Prior — "이런 분포를 따라라"

    ReLU 기반 SAE에서 z ≥ 0 이므로, prior도 z ≥ 0 위에 정의:

    p(z_i) = π · δ(0) + (1-π) · Exp(λ)

    ● π = 0.95: 각 뉴런이 95% 확률로 0 (inactive)
    ● 1-π = 0.05: 5% 확률로 양수값 (Exponential 분포, active)

    시각화:
    빈도
     │████                           (95%가 정확히 0)
     │
     │  ██
     │    ██
     │      ████
     │          ████████████         (5%가 양수, 지수적 감소)
     └──0──────────────────── z값

    이것의 의미:
    - 24,576개 뉴런 중 평균 ~1,229개만 활성 (L0 ≈ 1,229)
    - 활성화 값은 Exp(λ) 분포 → 적당히 다양한 크기
    - Dead neuron 방지: prior가 "5%는 반드시 활성"을 요구

[4.3] Frozen Random Projection φ — "어디서 거리를 재는가"

    문제: z ∈ R^24,576 에서 직접 cdist → 모든 거리 비슷 → kernel 무의미
    해결: φ(z) = z · W_frozen ∈ R^256 으로 투영 후 거리 계산

    W_frozen은:
    - 학습하지 않음 (frozen) → drift와 자가증명(collapsing) 방지
    - 랜덤 초기화 + L2 normalize → 등방성(isotropic) 보장
    - Training-time only → inference에서는 존재하지 않음 (추가 비용 0)

    왜 학습하면 안 되나?
    → φ가 학습되면 "drift loss를 줄이기 쉬운 방향"으로 변형됨
    → encoder가 실제로 sparse해지지 않아도 loss가 줄어드는 치팅 발생

[4.4] β Warmup — "언제 밀어넣기 시작하는가"

    β (drift loss 가중치)를 처음부터 크게 하면:
    → encoder가 prior만 따라가고 reconstruction 무시
    → h를 전혀 복원 못하는 모델이 됨

    전략: β를 0.01에서 시작, 5000 step에 걸쳐 0.1로 증가

    Step:    0 ─────── 2500 ─────── 5000 ──── ...
    β:     0.01        0.055        0.1
    의미:  recon 위주   점진 투입     drift 본격화

[4.5] 기존 SAE들과의 비교

    ┌─────────────┬──────────────────┬──────────────────┐
    │             │  기존 4종 SAE     │  DriftPrior-SAE  │
    ├─────────────┼──────────────────┼──────────────────┤
    │ 희소성 제어  │ 개별 뉴런에 페널티 │ 분포 전체를 매칭  │
    │ 제어 수준    │ "얼마나 sparse"  │ "어떤 형태로"    │
    │ Dead neuron │ 구조적 해결 없음  │ prior가 활성 요구 │
    │ 분포 인식    │ 없음             │ 있음 (prior 명시) │
    │ 구현 난이도  │ 한 줄 (L1 등)    │ 중간 (drift 계산) │
    │ Inference   │ 기본              │ 동일 (drift 불필요)│
    └─────────────┴──────────────────┴──────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. Toy Demo → Production 변환 포인트
=====================================

원본 노트북(drifting_model_demo.ipynb)은 2D 시각화용 toy code.
실제 병리학 데이터에 적용하려면 아래 6가지를 반드시 수정해야 함:

  ┌──┬─────────────────┬──────────────────────┬───────────────────────┐
  │# │ Toy Demo         │ 문제                  │ 이 코드의 해결          │
  ├──┼─────────────────┼──────────────────────┼───────────────────────┤
  │1 │ Generator 구조    │ noise→MLP→output     │ h→Encoder→z→Decoder→ĥ │
  │  │ (생성 모델)       │ SAE는 autoencoder    │ L_recon + L_drift     │
  ├──┼─────────────────┼──────────────────────┼───────────────────────┤
  │2 │ Raw dim cdist    │ 24,576D에서 cdist    │ φ: 24576→256 frozen   │
  │  │ (2D라서 문제없음) │ = 수백 GB OOM        │ projection 추가       │
  ├──┼─────────────────┼──────────────────────┼───────────────────────┤
  │3 │ Full batch drift │ 4096² pairwise       │ Sub-batch 512개만     │
  │  │ (2048개 OK)      │ = VRAM 폭발          │ 랜덤 추출해서 계산    │
  ├──┼─────────────────┼──────────────────────┼───────────────────────┤
  │4 │ 실제 데이터를     │ SAE에서 target은      │ Spike-and-Slab prior  │
  │  │ positive로 사용   │ "원하는 분포"여야 함  │ 에서 샘플링           │
  ├──┼─────────────────┼──────────────────────┼───────────────────────┤
  │5 │ V⁺/V⁻ 결합 근사  │ 고차원에서 drift      │ V⁺/V⁻ 독립 정규화    │
  │  │ sqrt(Σrow·Σcol)  │ 방향 왜곡 가능        │ (논문 원문 방식)      │
  ├──┼─────────────────┼──────────────────────┼───────────────────────┤
  │6 │ β 고정 (없음)    │ 초기에 recon 붕괴     │ β: 0.01→0.1 warmup   │
  │  │                  │                      │ (5000 steps)          │
  └──┴─────────────────┴──────────────────────┴───────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. 사용법
=========

  # 합성 데이터로 빠른 테스트 (수 분)
  python drift_prior_sae.py --config debug

  # PathoSAEv3 activation으로 실제 학습
  python drift_prior_sae.py --config pathosae \\
      --data_path /path/to/exaonepath_tokens.pt \\
      --output_dir ./outputs/driftprior_v1

  # 하이퍼파라미터 개별 오버라이드
  python drift_prior_sae.py --config pathosae \\
      --tau 0.05 --beta_end 0.2 --phi_dim 128

  # τ sweep (temperature 비교)
  for tau in 0.05 0.1 0.3; do
      python drift_prior_sae.py --config pathosae \\
          --data_path act.pt --tau $tau \\
          --output_dir ./outputs/tau_${tau}
  done

  # 체크포인트에서 재개
  python drift_prior_sae.py --config pathosae \\
      --data_path act.pt --resume ./outputs/driftprior_v1/ckpt_step_10000.pt

Requirements: pip install torch tqdm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7. VRAM 예상치
=============

  ┌───────────────────────────┬─────────┬──────────┐
  │ 구성 요소                  │ pathosae │ small    │
  ├───────────────────────────┼─────────┼──────────┤
  │ SAE 파라미터 (enc+dec)     │ ~144 MB │ ~24 MB   │
  │ Batch activations         │ ~384 MB │ ~48 MB   │
  │ φ projection matrix       │ ~24 MB  │ ~2 MB    │
  │ Drift cdist peak          │ ~2 MB   │ ~0.5 MB  │
  │ Backward + optimizer      │ ~3 GB   │ ~500 MB  │
  │ PyTorch overhead          │ ~2 GB   │ ~1 GB    │
  ├───────────────────────────┼─────────┼──────────┤
  │ 총 예상                    │ ~6-8 GB │ ~2-3 GB  │
  │ 권장 GPU                   │ RTX 4090│ RTX 3060 │
  └───────────────────────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

원본 논문: Deng et al. "Generative Modeling via Drifting" (arXiv 2602.04770v2, 2026)
적용 대상: PathoSAEv3 — EXAONEPath 768D spatial tokens
구현: 김하종 (성균관대 인공지능융합전공, SKKAI)
"""

import os
import math
import json
import logging
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# Section 1: Configuration
# ============================================================
# 모든 하이퍼파라미터를 한 곳에서 관리.
# 커맨드라인에서 --tau 0.05 같은 식으로 개별 오버라이드 가능.
# ============================================================

@dataclass
class DriftPriorConfig:
    """
    DriftPrior-SAE 전체 설정.

    SAE 관련:
        input_dim: ViT 출력 차원 (EXAONEPath = 768)
        latent_dim: Dictionary 크기 (보통 input_dim의 32배)
        tied_weights: True면 decoder weight = encoder weight의 전치
        normalize_decoder: True면 매 step decoder 열벡터를 unit norm으로 정규화

    Drift 관련:
        phi_dim: frozen projection 출력 차원 (높을수록 정밀, 낮을수록 빠름)
        drift_sub_batch: drift 계산에 사용할 샘플 수 (전체 batch의 부분집합)
        n_pos: prior에서 샘플링할 positive 수
        tau: kernel temperature (작으면 가까운 것만, 크면 넓게 영향)
        prior_sparsity: 뉴런이 inactive일 확률 (0.95 = 95% 꺼짐)
        prior_scale: active 뉴런의 값 스케일

    β 스케줄:
        beta_start → beta_end 까지 beta_warmup_steps 동안 증가
        recon 안정 후에 drift를 점진 투입하는 전략
    """

    # --- SAE Architecture ---
    input_dim: int = 768
    latent_dim: int = 24_576
    tied_weights: bool = False
    normalize_decoder: bool = True

    # --- Drift Prior ---
    phi_dim: int = 256
    drift_sub_batch: int = 512
    n_pos: int = 512
    n_neg: int = 512
    tau: float = 0.1
    prior_type: str = "spike_slab"
    prior_sparsity: float = 0.95
    prior_scale: float = 1.0

    # --- Loss Weights ---
    beta_start: float = 0.01
    beta_end: float = 0.1
    beta_warmup_steps: int = 5000
    beta_schedule: str = "linear"

    # --- Training ---
    batch_size: int = 4096
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 50_000
    grad_clip: float = 1.0
    seed: int = 42

    # --- Logging ---
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    output_dir: str = "./outputs/driftprior_sae"
    device: str = "cuda"

    def __post_init__(self):
        self.n_neg = self.drift_sub_batch


CONFIGS = {
    "pathosae": DriftPriorConfig(
        input_dim=768, latent_dim=24_576, phi_dim=256,
        drift_sub_batch=512, n_pos=512, tau=0.1,
        batch_size=4096, max_steps=50_000,
    ),
    "small": DriftPriorConfig(
        input_dim=768, latent_dim=4096, phi_dim=128,
        drift_sub_batch=256, n_pos=256, tau=0.1,
        batch_size=2048, max_steps=20_000,
    ),
    "debug": DriftPriorConfig(
        input_dim=64, latent_dim=256, phi_dim=32,
        drift_sub_batch=64, n_pos=64, tau=0.1,
        batch_size=128, max_steps=500,
        log_every=10, eval_every=50, save_every=100,
    ),
}
CONFIGS["default"] = CONFIGS["pathosae"]


# ============================================================
# Section 2: Prior Distributions
# ============================================================
# "z가 따라야 할 목표 분포"를 정의한다.
#
# 왜 prior가 필요한가?
#   기존 SAE: "z를 sparse하게 만들어라" (간접적 — 값을 줄여라)
#   DriftPrior: "z가 이 분포를 따라라" (직접적 — 목표 형태 지정)
#
# Spike-and-Slab이 적합한 이유:
#   ReLU SAE에서 z ≥ 0이므로, prior도 z ≥ 0 위에 정의해야 함.
#   - Spike (z=0): inactive neuron → π=0.95 (95%)
#   - Slab (z>0): active neuron → Exp(λ) 분포 (5%)
#   이것이 "대부분 꺼져있고, 소수만 양수로 활성"이라는
#   이상적인 sparse 패턴을 수학적으로 정확히 표현한다.
# ============================================================

class SpikeSlab:
    """
    Spike-and-Slab Prior: p(z_i) = π·δ(0) + (1-π)·Exp(λ)

    시각적 이해:
        z=0에서 거대한 spike (확률 질량 π=0.95)
        z>0에서 완만한 exponential tail (확률 밀도 (1-π)·λ·exp(-λz))

    SAE에서의 의미:
        24,576개 뉴런 중 평균 0.05 × 24,576 ≈ 1,229개만 활성화
        → L0 ≈ 1,229 (prior가 기대하는 sparsity)
    """

    def __init__(self, sparsity: float = 0.95, scale: float = 1.0):
        self.sparsity = sparsity
        self.scale = scale

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """prior에서 n×d 크기의 샘플을 뽑는다."""
        n, d = shape
        # step 1: 각 뉴런이 active(1)인지 inactive(0)인지
        mask = torch.bernoulli(
            torch.full((n, d), 1.0 - self.sparsity, device=device)
        )
        # step 2: active인 뉴런에 Exponential 값 할당
        values = torch.empty(n, d, device=device).exponential_(1.0 / self.scale)
        return mask * values

    def expected_l0(self, dim: int) -> float:
        return dim * (1.0 - self.sparsity)


class LaplacePrior:
    """Half-Laplace Prior: |Laplace(0,b)| (ReLU 호환 대안)"""

    def __init__(self, scale: float = 0.1):
        self.scale = scale

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        return torch.distributions.Laplace(0.0, self.scale).sample(shape).to(device).abs()


def get_prior(config: DriftPriorConfig):
    if config.prior_type == "spike_slab":
        return SpikeSlab(config.prior_sparsity, config.prior_scale)
    elif config.prior_type == "laplace":
        return LaplacePrior(config.prior_scale)
    raise ValueError(f"Unknown prior: {config.prior_type}")


# ============================================================
# Section 3: Frozen Random Projection φ
# ============================================================
# Drifting의 핵심 제약: 고차원에서 kernel이 무의미해진다.
#
# 문제 상세:
#   z ∈ R^24,576 에서 ||z_i - z_j|| 를 계산하면,
#   고차원 현상으로 인해 모든 쌍의 거리가 거의 동일해짐.
#   → k(x,y) = exp(-||x-y||/τ) 가 전부 비슷한 값
#   → 가중치가 uniform → drift 방향이 의미 없어짐
#
# 해결:
#   z를 φ(z) = z · W ∈ R^256 으로 투영한 뒤 거리 계산.
#   256차원에서는 거리의 분산이 충분히 유지되어 kernel이 유의미.
#
# 왜 frozen(학습 불가)인가?
#   φ가 학습되면 "drift loss를 줄이기 쉬운 공간"으로 변형됨.
#   → encoder가 실제로 sparse해지지 않아도 φ가 loss를 줄여버림.
#   → frozen이면 φ는 고정된 "측정 도구" 역할만 함.
#
# Inference에서는?
#   φ는 training 시 drift loss 계산에만 사용.
#   학습 완료 후 inference에서는 φ 불필요 → 추가 비용 0!
# ============================================================

class FrozenProjection(nn.Module):
    """
    φ: R^D → R^d (예: 24,576 → 256)

    register_buffer → gradient 없음, state_dict에는 포함.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(in_dim, out_dim, generator=gen)
        W = F.normalize(W, dim=0) * math.sqrt(out_dim / in_dim)
        self.register_buffer("W", W)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W


# ============================================================
# Section 4: Drift Field Computation
# ============================================================
# 이 부분이 Drifting의 핵심 알고리즘이다.
#
# 직관적 이해 (비유):
#   - 당신(x)이 운동장에 서 있다
#   - "목표 위치들"(prior 샘플 y⁺)이 "이쪽으로 와!" 하고 당긴다 (V⁺)
#   - "다른 사람들"(encoder 출력 y⁻)이 "나랑 겹치지 마!" 하고 민다 (V⁻)
#   - 두 힘의 합 V = V⁺ - V⁻ 가 당신이 이동할 방향
#
# 수학적 정의 (논문 Eq. 1-2):
#   V⁺(x) = Σ [k(x,y⁺)/Z_p] · (y⁺ - x)   ← attraction
#   V⁻(x) = Σ [k(x,y⁻)/Z_q] · (y⁻ - x)   ← repulsion
#   V(x)  = V⁺(x) - V⁻(x)
#
#   k(x,y) = exp(-||x-y|| / τ)
#   Z_p = Σ k(x,y⁺),  Z_q = Σ k(x,y⁻)
#
# ⚠️ Toy Demo와의 핵심 차이:
#   Toy: V⁺/V⁻를 결합해서 sqrt(Σrow · Σcol)로 한번에 정규화
#        → 이것은 compact form (Eq.3)의 Monte Carlo 근사
#   여기: V⁺와 V⁻를 각각 독립적으로 Z_p, Z_q로 정규화
#        → 논문 원문(Eq.1-2)의 직접 구현
#        → 고차원에서 결합 근사는 drift 방향을 왜곡할 수 있음
# ============================================================

@torch.no_grad()
def compute_drift_projected(
    phi_gen: torch.Tensor,    # [G, d] — encoder 출력의 φ 투영
    phi_pos: torch.Tensor,    # [P, d] — prior 샘플의 φ 투영
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Mean-shift drifting field V = V⁺ - V⁻

    Args:
        phi_gen: encoder가 생성한 z를 φ로 투영한 것 [G, d]
                 → 이 점들 각각에 대해 V를 계산
        phi_pos: prior에서 샘플링한 z를 φ로 투영한 것 [P, d]
                 → 이 점들이 phi_gen을 "끌어당김"
        tau: kernel temperature
             0.05: 매우 가까운 것만 → 세밀한 매칭
             0.1:  적절한 범위 (기본값)
             0.3:  넓은 범위 → 부드러운 매칭

    Returns:
        V: drift vectors [G, d] (각 점이 이동해야 할 방향·크기)
    """

    # ── V⁺: Attraction (prior가 끌어당김) ──
    # dist[i,j] = ||phi_gen[i] - phi_pos[j]||
    # k[i,j] = exp(-dist/τ): 가까우면 강하게 당김
    # w[i,j] = k[i,j] / Σ_j k[i,j]: 정규화 (합=1)
    # V⁺[i] = Σ_j w[i,j]·phi_pos[j] - phi_gen[i]
    dist_pos = torch.cdist(phi_gen, phi_pos)
    k_pos = (-dist_pos / tau).exp()
    Z_pos = k_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w_pos = k_pos / Z_pos
    V_plus = w_pos @ phi_pos - phi_gen  # Σw=1이므로 성립

    # ── V⁻: Repulsion (encoder 출력끼리 밀어냄) ──
    # mode collapse 방지: 모든 z가 같은 곳에 몰리면 V⁻가 강하게 밀어냄
    # self-mask: 자기가 자기를 밀면 안 됨 → 거리 무한대 설정
    dist_neg = torch.cdist(phi_gen, phi_gen)
    dist_neg.fill_diagonal_(1e6)
    k_neg = (-dist_neg / tau).exp()
    Z_neg = k_neg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w_neg = k_neg / Z_neg
    V_minus = w_neg @ phi_gen - phi_gen

    return V_plus - V_minus


def compute_drift_loss(
    z: torch.Tensor,
    prior: object,
    phi: FrozenProjection,
    config: DriftPriorConfig,
) -> torch.Tensor:
    """
    Drift loss (메모리 효율적인 sub-batching 포함).

    L_drift = E[||φ(z) - sg(φ(z) + V(φ(z)))||²] = E[||V||²]

    Sub-batching이 필요한 이유:
      cdist([4096, 256], [4608, 256]) → 18M entries → 72MB (이것만은 OK)
      하지만 backward 시 추가 메모리 → sub-batch 512로 줄여서 안전하게

    흐름:
      1. z에서 512개 랜덤 추출 (sub-batch)
      2. Prior에서 512개 샘플링 (positive)
      3. 둘 다 φ(256d)로 투영
      4. φ 공간에서 drift V 계산 (no grad)
      5. Loss = ||φ(z_sub) - sg(φ(z_sub) + V)||²
         → gradient는 encoder로 흐름
    """
    B, D = z.shape
    device = z.device

    sub_idx = torch.randperm(B, device=device)[:config.drift_sub_batch]
    z_sub = z[sub_idx]

    z_prior = prior.sample((config.n_pos, D), device=device)

    with torch.no_grad():
        phi_gen_ng = phi(z_sub.detach())
        phi_pos_ng = phi(z_prior)

    V = compute_drift_projected(phi_gen_ng, phi_pos_ng, tau=config.tau)

    phi_z = phi(z_sub)  # gradient 살아있음!
    target = (phi_z.detach() + V).detach()
    return F.mse_loss(phi_z, target)


# ============================================================
# Section 5: DriftPrior-SAE Model
# ============================================================
# Inference 시에는 일반 SAE와 완전히 동일:
#   h → encode → z → decode → ĥ
# φ와 drift는 training에서만 사용, inference에서는 무시.
# ============================================================

class DriftPriorSAE(nn.Module):
    """
    DriftPrior-SAE: Sparse Autoencoder + Distribution-Matching Drift Loss

    학습: L_total = L_recon + β(step)·L_drift
    추론: 일반 SAE와 동일 (추가 비용 없음)
    """

    def __init__(self, config: DriftPriorConfig):
        super().__init__()
        self.config = config

        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))
        self.encoder = nn.Linear(config.input_dim, config.latent_dim)

        if config.tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(config.input_dim))
        else:
            self.decoder = nn.Linear(config.latent_dim, config.input_dim, bias=True)

        self.phi = FrozenProjection(config.latent_dim, config.phi_dim, seed=config.seed)
        self.prior = get_prior(config)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        if not self.config.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.decoder.bias)
        if self.config.normalize_decoder:
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Decoder 열벡터 unit norm 강제 (개념 방향의 크기 통일)"""
        if self.config.tied_weights:
            return
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(h - self.pre_bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.tied_weights:
            return z @ self.encoder.weight + self.decoder_bias + self.pre_bias
        return self.decoder(z) + self.pre_bias

    def forward(self, h: torch.Tensor):
        z = self.encode(h)
        h_hat = self.decode(z)
        info = {
            "l0": (z > 0).float().sum(dim=-1).mean().item(),
            "z_mean": z.mean().item(),
            "z_max": z.max().item(),
            "dead_frac": (z.sum(dim=0) == 0).float().mean().item(),
        }
        return h_hat, z, info


# ============================================================
# Section 6: β Warmup Schedule
# ============================================================

def get_beta(step: int, config: DriftPriorConfig) -> float:
    """β(step): 0.01에서 시작해 5000 step에 걸쳐 0.1로 증가"""
    if step >= config.beta_warmup_steps:
        return config.beta_end
    ratio = step / config.beta_warmup_steps
    if config.beta_schedule == "cosine":
        ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return config.beta_start + (config.beta_end - config.beta_start) * ratio


# ============================================================
# Section 7: Trainer
# ============================================================

class DriftPriorTrainer:
    def __init__(self, config: DriftPriorConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = DriftPriorSAE(config).to(self.device)

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"DriftPrior-SAE: {config.input_dim}→{config.latent_dim} | "
            f"φ: {config.latent_dim}→{config.phi_dim} | "
            f"trainable params: {n_train:,}"
        )
        if hasattr(self.model.prior, 'expected_l0'):
            logger.info(f"Prior expected L0: {self.model.prior.expected_l0(config.latent_dim):.0f}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.step = 0
        self.loss_history = []
        os.makedirs(config.output_dir, exist_ok=True)

    def train_step(self, h: torch.Tensor) -> dict:
        self.model.train()
        h = h.to(self.device)

        h_hat, z, info = self.model(h)
        loss_recon = F.mse_loss(h, h_hat)
        loss_drift = compute_drift_loss(z, self.model.prior, self.model.phi, self.config)
        beta = get_beta(self.step, self.config)
        loss = loss_recon + beta * loss_drift

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        if self.config.normalize_decoder:
            self.model._normalize_decoder()

        self.step += 1

        with torch.no_grad():
            fvu = loss_recon.item() / (h.var().item() + 1e-12)

        metrics = {
            "step": self.step, "loss": loss.item(),
            "loss_recon": loss_recon.item(), "loss_drift": loss_drift.item(),
            "beta": beta, "fvu": fvu, **info,
        }
        self.loss_history.append(metrics)
        return metrics

    def train(self, data_loader: DataLoader):
        logger.info(f"Training for {self.config.max_steps} steps...")
        with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        data_iter = iter(data_loader)
        pbar = tqdm(range(1, self.config.max_steps + 1), desc="DriftPrior-SAE")
        ema = None

        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            h = batch[0] if isinstance(batch, (list, tuple)) else batch
            m = self.train_step(h)

            ema = m["loss"] if ema is None else 0.95 * ema + 0.05 * m["loss"]
            pbar.set_postfix(
                loss=f"{ema:.4f}", R=f"{m['loss_recon']:.4f}",
                D=f"{m['loss_drift']:.4f}", b=f"{m['beta']:.3f}",
                L0=f"{m['l0']:.0f}", dead=f"{m['dead_frac']:.1%}",
            )

            if step % self.config.log_every == 0:
                logger.info(
                    f"[{step}] loss={m['loss']:.5f} recon={m['loss_recon']:.5f} "
                    f"drift={m['loss_drift']:.5f} β={m['beta']:.4f} "
                    f"FVU={m['fvu']:.4f} L0={m['l0']:.1f} dead={m['dead_frac']:.1%}"
                )
            if step % self.config.eval_every == 0:
                ev = self.evaluate(data_loader)
                logger.info(f"[Eval {step}] {ev}")
            if step % self.config.save_every == 0:
                self.save_checkpoint(f"step_{step}")

        self.save_checkpoint("final")
        logger.info("Training complete!")

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, max_batches: int = 10) -> dict:
        self.model.eval()
        mse_sum, var_sum, l0_sum, n = 0.0, 0.0, 0.0, 0
        z_activity = torch.zeros(self.config.latent_dim, device=self.device)

        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            h = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            h_hat, z, _ = self.model(h)
            mse_sum += F.mse_loss(h, h_hat, reduction="sum").item()
            var_sum += h.var(dim=0).sum().item() * h.shape[0]
            l0_sum += (z > 0).float().sum(dim=-1).sum().item()
            z_activity += (z > 0).float().sum(dim=0)
            n += h.shape[0]

        dead = (z_activity == 0).float().mean().item()
        return {
            "mse": mse_sum / max(n, 1),
            "fvu": mse_sum / max(var_sum, 1e-12),
            "l0": l0_sum / max(n, 1),
            "dead_frac": dead, "n": n,
        }

    def save_checkpoint(self, tag: str):
        path = os.path.join(self.config.output_dir, f"ckpt_{tag}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "step": self.step, "config": asdict(self.config),
        }, path)
        logger.info(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.step = ckpt["step"]
        logger.info(f"Resumed from step {self.step}: {path}")


# ============================================================
# Section 8: Data Loading
# ============================================================

def load_activations(path: str, key: str = "activations") -> torch.Tensor:
    """사전 추출 activation 로드 (.pt/.npy/.safetensors/.h5)"""
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        data = torch.load(path, weights_only=True)
        return (data[key] if isinstance(data, dict) else data).float()
    elif path.suffix == ".npy":
        import numpy as np
        return torch.from_numpy(np.load(path)).float()
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))[key].float()
    elif path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            return torch.from_numpy(f[key][:]).float()
    raise ValueError(f"Unsupported: {path.suffix}")


def create_synthetic_data(n: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Debug용 합성 데이터 (ViT 출력 모방)"""
    gen = torch.Generator().manual_seed(seed)
    data = torch.randn(n, dim, generator=gen) * 0.5
    centers = torch.randn(10, dim, generator=gen) * 0.3
    idx = torch.randint(0, 10, (n,), generator=gen)
    return data + centers[idx]


# ============================================================
# Section 9: Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DriftPrior-SAE")
    parser.add_argument("--config", default="default", choices=list(CONFIGS.keys()))
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_key", default="activations")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None)

    for k, t in [("tau", float), ("beta_end", float), ("beta_start", float),
                 ("phi_dim", int), ("drift_sub_batch", int), ("lr", float),
                 ("max_steps", int), ("batch_size", int), ("seed", int),
                 ("latent_dim", int), ("prior_sparsity", float)]:
        parser.add_argument(f"--{k}", type=t, default=None)

    args = parser.parse_args()
    config = CONFIGS[args.config]
    if args.output_dir:
        config.output_dir = args.output_dir

    for k in ["tau", "beta_end", "beta_start", "phi_dim", "drift_sub_batch",
              "lr", "max_steps", "batch_size", "seed", "latent_dim", "prior_sparsity"]:
        v = getattr(args, k, None)
        if v is not None:
            setattr(config, k, v)
    config.__post_init__()

    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(config.output_dir, "train.log"), mode="a")],
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if args.data_path:
        logger.info(f"Loading: {args.data_path}")
        data = load_activations(args.data_path, key=args.data_key)
        if data.shape[-1] != config.input_dim:
            logger.warning(f"input_dim: {config.input_dim} → {data.shape[-1]}")
            config.input_dim = data.shape[-1]
    else:
        logger.info("Using synthetic data (no --data_path)")
        data = create_synthetic_data(
            max(10000, config.batch_size * 20), config.input_dim, config.seed
        )

    logger.info(f"Data: {data.shape} ({data.numel() * 4 / 1e6:.0f} MB)")
    loader = DataLoader(
        TensorDataset(data), batch_size=config.batch_size,
        shuffle=True, drop_last=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
    )

    trainer = DriftPriorTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train(loader)

    ev = trainer.evaluate(loader, max_batches=50)
    logger.info(f"Final: {json.dumps(ev, indent=2)}")
    with open(os.path.join(config.output_dir, "final_eval.json"), "w") as f:
        json.dump(ev, f, indent=2)


if __name__ == "__main__":
    main()
