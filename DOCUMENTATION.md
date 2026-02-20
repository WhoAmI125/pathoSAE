# PathoSAEv3 — Sparse Autoencoder Variants for Pathology Foundation Model Interpretability

---

## 1. Why This Project Exists

병리 AI 모델은 "이 조직은 암이다"라고 판단하지만, **왜 그렇게 판단했는지** 설명하지 못한다.
EXAONEPath 같은 병리 특화 Vision Transformer는 768차원 임베딩을 생성하지만, 이 벡터는 사람이 해석할 수 없다.

**Sparse Autoencoder(SAE)**는 이 블랙박스를 해부한다.
768차원의 밀집 표현을 ~24,576개의 희소한 잠재 특징으로 분해하면,
각 뉴런이 특정 형태학적 패턴(선 구조, 림프구 군집, 괴사 영역 등)에 대응하게 된다.
임상의가 "이 뉴런이 활성화됐으니 선 구조가 보인다"고 이해할 수 있는 수준으로.

**그런데 SAE에도 여러 종류가 있다.**
기존 연구(CytoSAE, PLUTO-SAE)는 Vanilla SAE만 사용했고,
"어떤 SAE 아키텍처가 임상 해석성에 최적인가?"는 아직 답이 없는 질문이다.

PathoSAEv3는 이 질문에 답하기 위해 **4가지 SAE variant를 동일 조건에서 비교**한다.

---

## 2. Research Goal

### 핵심 가설

> "SAE의 아키텍처 선택이 병리 파운데이션 모델의 임상 해석성에 유의미한 영향을 미친다."

### 구체적 목표

1. **공정 비교 (Fair Comparison)**
   - Vanilla, Gated, TopK, JumpReLU를 동일 하이퍼파라미터로 학습
   - 동일 데이터, 동일 epoch, 동일 lr, 동일 expansion factor
   - 차이점은 activation 함수와 sparsity 메커니즘뿐

2. **형태학적 특징 추출 (Morphological Feature Extraction)**
   - 24,576개 잠재 뉴런 중 어떤 것이 임상적으로 의미 있는 패턴을 포착하는지 식별
   - Top-k activating patch 분석으로 각 뉴런의 "의미" 확인

3. **단의미성 검증 (Monosemanticity Validation)**
   - 개별 feature가 하나의 해석 가능한 개념에 대응하는지 병리학자 리뷰로 검증
   - SAE variant별 단의미성 점수 비교

4. **임상 유용성 평가 (Clinical Utility)**
   - 재구성 품질(MSE) vs 해석성(sparsity, L0) 트레이드오프 분석
   - 논문용 Pareto front 제시

---

## 3. Background

### 3.1 EXAONEPath — 병리 특화 Vision Transformer

| 속성 | 값 |
|------|-----|
| 아키텍처 | ViT-Base/16 |
| 입력 | 224 x 224 타일 이미지 |
| 패치 크기 | 16 x 16 → 14 x 14 = **196 spatial tokens** |
| 임베딩 차원 | **768** |
| 깊이 | 12 transformer blocks |
| 개발 | LG AI Research |
| 도메인 | 병리 이미지 특화 (H&E staining) |

**왜 EXAONEPath인가?**
- PLUTO(384-dim) 대비 2배 표현력 (768-dim)
- DinoBloom-B(일반 의료) 대비 병리 도메인 특화
- 196개 **spatial token** 사용 → CLS 단일 벡터 대비 공간적 해석 가능

### 3.2 SAE 4가지 Variant

```
입력 x ∈ ℝ^768
    │
    ├── [Vanilla]    ReLU(x·W + b)                    → L1 penalty
    ├── [Gated]      ReLU(x·W + b) × Gate(x·W + b_g) → L1 penalty
    ├── [TopK]       TopK(x·W + b, k=64)              → k가 sparsity 직접 제어
    └── [JumpReLU]   x·W·H(x·W - θ)                   → L0 penalty (θ 학습)
    │
    ▼
잠재 표현 z ∈ ℝ^24,576 (sparse)
    │
    ▼
재구성 x̂ = z·W_dec + b_dec ∈ ℝ^768
```

| Variant | Activation | Sparsity Control | 학습 파라미터 | 핵심 특성 |
|---------|-----------|-----------------|-------------|----------|
| **Vanilla** | ReLU | L1 penalty (간접) | W_enc, W_dec, b_enc, b_dec | 안정적 기준선 |
| **Gated** | Gate × ReLU | L1 on gated acts | + b_gate | 이진 게이트로 선택적 활성화 |
| **TopK** | TopK selection | k 직접 제어 | (추가 없음) | 정확한 sparsity, L1 불필요 |
| **JumpReLU** | Heaviside step | L0 penalty (뉴런 수) | + log_threshold | 뉴런별 적응적 임계값 |

### 3.3 선행 연구와의 차별점

| | CytoSAE (MICCAI'25) | PLUTO-SAE (NeurIPS'25 WS) | **PathoSAEv3 (Ours)** |
|---|---|---|---|
| 도메인 | 혈액학 (단일 세포) | 병리 (조직) | **병리 (조직)** |
| 백본 | DinoBloom-B (768d) | PLUTO (384d) | **EXAONEPath (768d)** |
| SAE 종류 | Vanilla 1종 | Vanilla 1종 | **4종 비교 (핵심 기여)** |
| 확장 배율 | 64x | 8x | **32x** |
| 토큰 전략 | CLS only | CLS only | **196 spatial (14x14)** |
| 추출 방식 | Online | - | **Offline (메모리 효율)** |

---

## 4. PathoSAEv2 결과 요약 (v3 설계 근거)

pathoSAEv2에서 각 variant를 **서로 다른 조건**으로 학습한 예비 결과:

| Metric | Vanilla | Gated (Spatial) | MSAE (TopK) | JumpReLU |
|--------|---------|-----------------|-------------|----------|
| Expansion | 32x | 16x | 32x | 32x |
| Epochs | 10 | 100 | 5-10 | 10 |
| Batch | 16,384 | 16,384 | 8,192 | 16,384 |
| Val MSE | 0.065 | ~0.066 | **0.063** | 0.065 |
| Sparsity | ~97% | ~95% | TopK 제어 | **99.3%** |
| Active Neurons | ~700 | - | 64-512 | **~174** |

**문제점**: variant별 하이퍼파라미터가 달라서 공정 비교 불가능.
**해결**: PathoSAEv3에서 **모든 조건을 통일** (expansion=32, epochs=10, batch=4096, lr=3e-4).

---

## 5. Code Architecture

### 5.1 설계 원칙

1. **Configuration-Driven**: 모든 하이퍼파라미터가 `SAEConfig` dataclass에 집중
2. **BaseSAE + Factory**: 추상 클래스 + `create_sae(cfg)` 팩토리로 variant 무관한 코드
3. **통합 CLI**: 하나의 `tasks/train.py --model {variant}`로 모든 학습
4. **Offline 추출**: 데이터 한번 추출 후 재사용 (symlink)
5. **논문 우선**: 모든 구조가 공정 비교 + 결과 재현을 위해 설계됨

### 5.2 디렉토리 구조

```
pathoSAEv3/
│
├── src/                              # 핵심 라이브러리
│   ├── config.py                     # SAEConfig — 통합 설정 dataclass
│   │
│   ├── models/                       # SAE 모델 정의
│   │   ├── base_sae.py               #   BaseSAE (ABC) — encode/activate/decode/forward
│   │   ├── vanilla_sae.py            #   VanillaSAE — ReLU + L1
│   │   ├── gated_sae.py              #   GatedSAE — b_gate + gated ReLU + L1
│   │   ├── topk_sae.py               #   TopKSAE — TopK activation, L1 불필요
│   │   ├── jumprelu_sae.py           #   JumpReLUSAE — learnable threshold + L0
│   │   └── __init__.py               #   MODEL_REGISTRY + create_sae() factory
│   │
│   ├── training/                     # 학습 파이프라인
│   │   ├── trainer.py                #   SAETrainer — 통합 학습 루프
│   │   ├── losses.py                 #   MSE, L1, L0 유틸 함수
│   │   └── scheduler.py              #   warmup + cosine decay LR 스케줄러
│   │
│   ├── evaluation/                   # 평가 & 시각화
│   │   ├── metrics.py                #   MSE, FVU, L0, sparsity, cosine_sim
│   │   ├── visualize.py              #   activation histogram, sparsity plot
│   │   ├── compare.py                #   variant 비교 CSV + Pareto plot
│   │   └── feature_analysis.py       #   feature별 통계, top-k patches
│   │
│   ├── extract/                      # ViT activation 추출
│   │   ├── activation_store.py       #   학습용 데이터 로딩 (lazy, file-based split)
│   │   ├── vit_encoder.py            #   EXAONEPath ViT-B/16 로딩 + layer 12 추출
│   │   └── macenko.py                #   Macenko stain normalization
│   │
│   └── utils.py                      # GPU 탐색, seed 설정 등
│
├── tasks/                            # CLI 엔트리포인트 (실행 스크립트)
│   ├── train.py                      #   python tasks/train.py --model vanilla
│   ├── evaluate.py                   #   python tasks/evaluate.py --checkpoint <path>
│   ├── compare.py                    #   python tasks/compare.py --checkpoints <p1> <p2>
│   └── extract.py                    #   python tasks/extract.py --image_dir <path>
│
├── scripts/                          # Shell 래퍼
│   ├── train_all.sh                  #   4 variant 순차 학습
│   └── evaluate_all.sh               #   4 variant 평가 + 비교
│
├── data/
│   └── activations/                  #   → symlink to pathoSAEv2 (749GB, 96 files)
│
├── models/
│   ├── backbone/EXAONEPath.ckpt      #   → symlink to pretrained ViT
│   └── checkpoints/                  #   학습된 SAE 체크포인트
│       ├── vanilla_e32_ep10/
│       ├── gated_e32_ep10/
│       ├── topk_e32_ep10/
│       └── jumprelu_e32_ep10/
│
├── results/                          # 평가 결과
│   ├── {run_name}/metrics.json       #   개별 variant 지표
│   └── comparison/                   #   비교 표 + 그래프
│       ├── comparison_table.csv
│       └── mse_vs_l0.png
│
├── environment.yaml                  # conda 환경 (pathosae3)
├── requirements.txt                  # pip 의존성
└── CLAUDE.md                         # Opus-Codex 파이프라인 설정
```

### 5.3 핵심 클래스 관계

```
                    SAEConfig
                        │
                        ▼
    ┌───────────────────────────────────────┐
    │              BaseSAE (ABC)            │
    │  encode() → activate() → decode()    │
    │  forward(x, step) → (out, acts, loss)│
    │  save() / load()                     │
    └───────┬───────┬───────┬───────┬──────┘
            │       │       │       │
      VanillaSAE GatedSAE TopKSAE JumpReLUSAE
            │       │       │       │
            └───────┴───┬───┴───────┘
                        │
                   create_sae(cfg)   ← MODEL_REGISTRY
                        │
                        ▼
                   SAETrainer
                   ├── ActivationStore (data loading)
                   ├── Adam optimizer
                   ├── LambdaLR scheduler
                   └── W&B logging
```

### 5.4 데이터 흐름

```
[WSI Tiles 224x224]
      │
      ▼ Macenko stain normalization
      │
      ▼ EXAONEPath ViT-B/16 (layer 12)
      │
[spatial_folder_XX.pt]  ←── {vectors: [N, 196, 768], paths: [...]}
      │                       96 files, ~749GB total
      ▼ flatten
[N × 196, 768]  ←── 각 패치를 독립 샘플로 취급 (~18.8M patches)
      │
      ▼ SAETrainer
      │   ├── encode: (x - b_dec) @ W_enc + b_enc → [batch, 24576]
      │   ├── activate: variant-specific → sparse [batch, 24576]
      │   └── decode: acts @ W_dec + b_dec → [batch, 768]
      │
[models/checkpoints/{run_name}/best.pt]
      │
      ▼ Evaluate
      │
[results/{run_name}/metrics.json]  +  activation_hist.png  +  sparsity_dist.png
      │
      ▼ Compare (all 4 variants)
      │
[results/comparison/comparison_table.csv]  +  mse_vs_l0.png (Pareto front)
```

---

## 6. Unified Training Protocol (논문 비교 기준)

모든 variant에 적용되는 **동일 학습 조건**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input dim | 768 | EXAONEPath ViT-B output |
| Expansion | 32x → 24,576 latents | pathoSAEv2 기준, 충분한 표현력 |
| Batch size | 4,096 | GPU 메모리 vs 학습 안정성 균형 |
| Epochs | 10 | pathoSAEv2 Vanilla/JumpReLU 기준 충분 |
| Learning rate | 3e-4 | Adam 표준 |
| LR schedule | 500 step warmup + cosine decay (70% 이후) | 안정적 수렴 |
| Sparsity warmup | 30% of training | 초기 재구성 학습 우선 |
| Seed | 42 | 재현성 |
| Precision | float32 | AMP 호환성 차이 방지 |
| L1 coeff | 1e-4 | Vanilla, Gated용 |
| L0 coeff | 1e-4 | JumpReLU용 |
| TopK k | 64 | 고정 sparsity |
| Validation | 파일 단위 split (마지막 ~10% 파일) | 깔끔한 분리 |

### Variant별 유일한 차이점

| | Activation | Sparsity Loss | Extra Params |
|---|---|---|---|
| Vanilla | `ReLU(pre_act)` | `l1_coeff * L1` | 없음 |
| Gated | `ReLU(pre_act) * Gate` | `l1_coeff * L1` | `b_gate` |
| TopK | `TopK(pre_act, k=64)` | 없음 (k가 제어) | 없음 |
| JumpReLU | `pre_act * H(pre_act - θ)` | `l0_coeff * L0` | `log_threshold` |

---

## 7. Evaluation Metrics

논문에 보고할 지표:

| Metric | 정의 | 의미 |
|--------|------|------|
| **MSE** | `mean((x - x̂)²)` | 재구성 품질 (낮을수록 좋음) |
| **FVU** | `Σ(x-x̂)² / Σ(x-μ)²` | 설명되지 않은 분산 비율 |
| **L0** | `mean(count(z > 0))` | 샘플당 평균 활성 뉴런 수 |
| **Sparsity** | `mean(z == 0)` | 전체 활성화 중 0인 비율 |
| **Cosine Sim** | `mean(cos(x, x̂))` | 방향 보존 정도 |
| **Active Neurons** | `count(any(z > 0))` | 한번이라도 활성화된 뉴런 수 |
| **Dead Neurons** | `total - active` | 완전히 비활성인 뉴런 수 |

### 논문용 출력물

1. **Comparison Table** (`comparison_table.csv`) — 4 variant 지표 비교
2. **Pareto Plot** (`mse_vs_l0.png`) — MSE vs L0 트레이드오프, Pareto front 표시
3. **Activation Histogram** — variant별 activation 분포
4. **Sparsity Distribution** — per-sample sparsity 분포

---

## 8. Usage

### 환경 설정

```bash
# conda 환경 활성화
conda activate pathosae3

# 프로젝트 루트에서 실행 (PYTHONPATH 필요)
cd /home/kimhj/projects/pathoSAEv3
```

### 학습

```bash
# 개별 variant 학습
PYTHONPATH=./ python tasks/train.py --model vanilla
PYTHONPATH=./ python tasks/train.py --model gated
PYTHONPATH=./ python tasks/train.py --model topk --topk_k 64
PYTHONPATH=./ python tasks/train.py --model jumprelu

# 또는 전체 순차 학습
PYTHONPATH=./ bash scripts/train_all.sh

# 하이퍼파라미터 override
PYTHONPATH=./ python tasks/train.py --model vanilla --epochs 20 --lr 1e-4 --no_wandb
```

### 평가

```bash
# 개별 체크포인트 평가
PYTHONPATH=./ python tasks/evaluate.py --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt

# 4 variant 비교 (논문 표 + Pareto plot 생성)
PYTHONPATH=./ python tasks/compare.py \
  --checkpoints models/checkpoints/vanilla_e32_ep10/best.pt \
               models/checkpoints/gated_e32_ep10/best.pt \
               models/checkpoints/topk_e32_ep10/best.pt \
               models/checkpoints/jumprelu_e32_ep10/best.pt

# 또는 전체 순차 평가
PYTHONPATH=./ bash scripts/evaluate_all.sh
```

### Activation 추출 (새 데이터)

```bash
PYTHONPATH=./ python tasks/extract.py \
  --image_dir /path/to/wsi/tiles \
  --output_dir data/activations \
  --backbone_path models/backbone/EXAONEPath.ckpt
```

---

## 9. Key Design Decisions

| ID | 결정 | 근거 |
|----|------|------|
| ADR-001 | 4 variant 동일 하이퍼파라미터 | 논문 공정 비교. 차이는 activation만 |
| ADR-002 | BaseSAE ABC + Factory | 코드 중복 제거, Trainer가 variant 무관 |
| ADR-003 | 단일 CLI (`--model` flag) | v2의 7개 스크립트 → 1개로 통합 |
| ADR-004 | 기존 activation symlink | 749GB 재추출 불필요 |
| ADR-005 | TopK 단일 k=64 | multi-level은 별도 실험 |
| ADR-006 | JumpReLU custom autograd | ICLR 2025 논문 구현 그대로 |
| ADR-007 | conda 분리 (pathosae3) | 재현성, 의존성 격리 |

---

## 10. Expected Outcomes

논문에서 기대하는 결과 패턴:

1. **JumpReLU**: 가장 높은 sparsity (99%+), 가장 적은 활성 뉴런 → **최고 해석성**
2. **TopK**: 정확한 sparsity 제어 (k=64 고정) → **가장 예측 가능한 동작**
3. **Vanilla**: 안정적 기준선, 중간 수준의 sparsity → **비교 기준**
4. **Gated**: gate 메커니즘의 효과 검증 → **선택적 활성화의 가치**
5. **Pareto front**: MSE vs L0에서 각 variant의 위치 → **트레이드오프 시각화**

### 궁극적 질문에 대한 답

> "임상의가 병리 AI의 판단 근거를 이해하기 위해, 어떤 SAE를 써야 하는가?"

이 프로젝트가 그 답을 제공한다.

---

## 11. Technical Stack

| Component | Choice | Version |
|-----------|--------|---------|
| Language | Python | 3.10 |
| DL Framework | PyTorch | 2.2.0 |
| CUDA | NVIDIA CUDA | 11.8 |
| ViT Backbone | EXAONEPath | ViT-B/16 |
| Stain Norm | torchstain (Macenko) | >= 1.3.0 |
| Experiment Tracking | Weights & Biases | latest |
| Environment | conda (pathosae3) | - |
| CLI | argparse | stdlib |

---

## 12. References

- **CytoSAE** — SAE for hematology cell interpretability (MICCAI 2025)
- **PLUTO-SAE** — SAE for pathology tissue (NeurIPS 2025 Workshop)
- **JumpReLU SAE** — Rajamanoharan et al., "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs" (ICLR 2025)
- **Matryoshka SAE** — Hierarchical TopK activation for multi-granularity
- **EXAONEPath** — LG AI Research, pathology-specialized ViT
- **Anthropic SAE** — Decoder norm constraint, gradient projection techniques
