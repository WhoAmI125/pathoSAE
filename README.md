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

PathoSAEv3는 이 질문에 답하기 위해 **5가지 SAE variant를 동일 조건에서 비교**한다.

---

## 2. Research Goal

### 핵심 가설

> "SAE의 아키텍처 선택이 병리 파운데이션 모델의 임상 해석성에 유의미한 영향을 미친다."

### 구체적 목표

1. **공정 비교 (Fair Comparison)**
   - Vanilla, Gated, TopK, JumpReLU, MSAE를 동일 하이퍼파라미터로 학습
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

### 3.2 SAE 5가지 Variant

```
입력 x ∈ ℝ^768
    │
    ├── [Vanilla]    ReLU(x·W + b)                          → L1 penalty
    ├── [Gated]      ReLU(x·W + b) × Gate(x·W + b_g)       → L1 penalty
    ├── [TopK]       TopK(x·W + b, k=64)                    → k가 sparsity 직접 제어
    ├── [JumpReLU]   x·W·H(x·W - θ)                         → L0 penalty (θ 학습)
    └── [MSAE]       TopK(x·W+b, k=64/128/256/512) × 4level → 멀티스케일 weighted MSE
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
| **MSAE** | Multi-scale TopK | nesting_list 직접 제어 | (추가 없음) | 계층적 feature 학습, Matryoshka |

### 3.3 선행 연구와의 차별점

| | CytoSAE (MICCAI'25) | PLUTO-SAE (NeurIPS'25 WS) | **PathoSAEv3 (Ours)** |
|---|---|---|---|
| 도메인 | 혈액학 (단일 세포) | 병리 (조직) | **병리 (조직)** |
| 백본 | DinoBloom-B (768d) | PLUTO (384d) | **EXAONEPath (768d)** |
| SAE 종류 | Vanilla 1종 | Vanilla 1종 | **5종 비교 (핵심 기여)** |
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
│   │   ├── msae.py                   #   MSAESAE — Matryoshka multi-scale TopK
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
│   │   ├── feature_analysis.py       #   feature별 통계, top-k patches
│   │   ├── feature_figures.py        #   scatter/entropy/top-reference grid figure
│   │   └── group_analysis.py         #   class/patient group matrix + heatmap
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
│   ├── group_analysis.py             #   python tasks/group_analysis.py --checkpoint <path> --meta_csv <csv>
│   ├── build_group_meta.py           #   activation patch_paths -> path,class,patient CSV 자동 생성
│   ├── generate_experiment_report.py #   full 실험 결과 요약 리포트 생성
│   ├── feature_viz.py                #   top feature spatial masking visualization
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
[results/{run_name}/metrics.json] + activation_hist.png + sparsity_dist.png
      + feature_analysis.json + feature_summary.csv + feature_scatter.png
      + feature_entropy_hist.png + top_feature_reference_grid.png
      + group/class_feature_matrix.csv + group/patient_feature_matrix.csv
      + group/class_feature_heatmap.png + group/patient_feature_heatmap.png
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
| MSAE | `TopK(pre_act, k) × 4 levels` | 없음 (각 k가 제어) | 없음 |

### MSAE (Matryoshka SAE) 상세

MSAE는 단일 encoder/decoder를 공유하되, **동일 pre-activation에 k값이 다른 TopK를 4번 적용**한다.
각 level이 다른 granularity의 특징을 담당 — coarse(k=64)~fine(k=512).

```
pre_act = W_enc @ (x - b_dec) + b_enc        # 공유 encoder, shape: [B, 24576]
    │
    ├── TopK(k=64)  → recon₀ → MSE₀   (가장 sparse, 핵심 형태학적 패턴)
    ├── TopK(k=128) → recon₁ → MSE₁
    ├── TopK(k=256) → recon₂ → MSE₂
    └── TopK(k=512) → recon₃ → MSE₃   (가장 dense, 세밀한 텍스처)

loss = (w₀·MSE₀ + w₁·MSE₁ + w₂·MSE₂ + w₃·MSE₃) / Σwᵢ
     UW (uniform):  w = [1, 1, 1, 1]
     RW (reverse):  w = [4, 3, 2, 1]  (coarse feature 우선)
```

**장점**: 단일 모델이 다중 해석 granularity 제공. k=64로 보면 핵심 구조, k=512로 보면 세밀한 패턴.
**pathoSAEv2 대비**: multi-level nesting 동일 구조이나, expansion=32로 통일 (v2는 16x 사용).

MSAE 전용 CLI 옵션:
```bash
--msae_nesting 64,128,256,512   # nesting k 값 (쉼표 구분)
--msae_importance uniform        # uniform | reverse
```

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

1. **Comparison Table** (`comparison_table.csv`) — 5 variant 지표 비교
2. **Pareto Plot** (`mse_vs_l0.png`) — MSE vs L0 트레이드오프, Pareto front 표시
3. **Activation Histogram** — variant별 activation 분포
4. **Sparsity Distribution** — per-sample sparsity 분포
5. **Feature Scatter** (`feature_scatter.png`) — `log10(freq)` vs `log10(mean_activation)` 분포
6. **Feature Entropy Histogram** (`feature_entropy_hist.png`) — metadata 기반 feature selectivity 분포
7. **Top Feature Reference Grid** (`top_feature_reference_grid.png`) — 상위 feature의 top-k reference 이미지
8. **Group Heatmaps** (`group/*_heatmap.png`) — class/patient 단위 평균 feature 활성 시각화

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

# feature figure 확장 (기본 split=val)
PYTHONPATH=./ python tasks/evaluate.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --feature_figures --split val --top_n 10 --top_k 5

# metadata 사용 시 entropy figure도 생성
PYTHONPATH=./ python tasks/evaluate.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --feature_figures --split val --meta_csv data/meta/group_meta.csv

# group metadata 자동 생성 (class=vital_status_<code>, patient=path parent)
PYTHONPATH=./ python tasks/build_group_meta.py \
  --split full \
  --clinical_csv "/home/kimhj/projects/past_try/images/data/tcga_brca_survival/clinical_data(labels).csv" \
  --patient_col bcr_patient_barcode \
  --class_col vital_status \
  --output_csv data/meta/group_meta_auto_vstatus_full.csv

# 4 variant 비교 (논문 표 + Pareto plot 생성)
PYTHONPATH=./ python tasks/compare.py \
  --checkpoints models/checkpoints/vanilla_e32_ep10/best.pt \
               models/checkpoints/gated_e32_ep10/best.pt \
               models/checkpoints/topk_e32_ep10/best.pt \
               models/checkpoints/jumprelu_e32_ep10/best.pt

# class/patient 그룹 분석
PYTHONPATH=./ python tasks/group_analysis.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --meta_csv data/meta/group_meta.csv \
  --split val --top_n 50 --min_group_samples 20

# full(96 files) 예시: feature_figures + group 분석
PYTHONPATH=./ python tasks/evaluate.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --feature_figures --split full --max_files 96 --top_n 10 --top_k 5 \
  --meta_csv data/meta/group_meta_auto_vstatus_full.csv

PYTHONPATH=./ python tasks/group_analysis.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --meta_csv data/meta/group_meta_auto_vstatus_full.csv \
  --split full --max_files 96 --top_n 50 --min_group_samples 20

# 또는 전체 순차 평가
PYTHONPATH=./ bash scripts/evaluate_all.sh

# 실험 요약 리포트 생성
PYTHONPATH=./ python tasks/generate_experiment_report.py \
  --output results/summary/experiment_design_full_report.md
```

### Group Metadata 형식

`tasks/group_analysis.py`와 entropy figure 생성을 위해 아래 컬럼을 포함한 CSV가 필요하다.

- `path`: 타일 이미지 경로 (`patch_paths`와 매칭)
- `class`: 클래스/진단 라벨
- `patient`: 환자 ID

자동 생성 스크립트 기본 규칙:
- `class`: clinical CSV의 `vital_status`를 `vital_status_<값>` 형태로 변환
- `patient`: patch path의 상위 디렉토리명 (예: `TCGA-3C-AALI`)

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

---

## 13. ACCV 2026 — 논문 Accept를 위한 실험 체크리스트 & 가이드라인

> **목표**: "어떤 SAE 아키텍처가 병리 FM의 임상 해석성에 최적인가?"에 답하는 논문.
> **타겟**: ACCV 2026 (Asian Conference on Computer Vision)
> **현재 상태**: 인프라 완료, 기본 재구성 지표 완료, **핵심 해석성 지표 미구현**

---

### 13.1 논문 스토리라인 → 실험 매핑

논문이 순서대로 답해야 할 질문과, 이를 뒷받침하는 실험:

```
Q1. "SAE가 원본 임베딩을 잘 보존하는가?"
    → Table 1: 5종 재구성 지표 비교                          [Tier 1, #1-2]

Q2. "어떤 SAE가 가장 해석 가능한가?"
    → Table 2: MS Score, Label Entropy, Concept Count       [Tier 1, #3,5 / Tier 2, #6]
    → Fig 3-4: MS Score CDF + Label Entropy scatter          [Tier 1, #3,5]

Q3. "SAE가 원본 FM보다 정말 monosemantic한가?"
    → Fig 3: SAE vs No-SAE baseline                          [Tier 2, #7]

Q4. "학습된 concept이 실제 형태학적 패턴과 대응하는가?"
    → Fig 5: Concept Card + 14×14 Spatial Heatmap            [Tier 1, #4]
    → Fig 6: Variant별 동일 tile heatmap 비교                 [Tier 2, #8]

Q5. "SAE feature가 임상적으로 유용한가?"
    → Table 3: Feature Discrimination Analysis               [Tier 2, #9]
    → Fig 7: Class-discriminative feature 시각화               [Tier 2, #9]
```

---

### 13.2 데이터 현황 & 제약

| 항목 | 값 |
|------|-----|
| 데이터셋 | TCGA BRCA (유방암 단일 암종) |
| 환자 수 | **96명** (activation 추출 완료) |
| 총 패치 수 | 2,666,816 (196 spatial tokens × tiles) |
| 사용 가능 라벨 | vital_status (2 class), pathologic_stage (4), T-stage (4), N-stage (4) |
| 패치 단위 조직 라벨 | **없음** (tumor/stroma/necrosis 등 annotation 미보유) |

**핵심 제약**: 96명 환자 + 단일 암종으로 CytoSAE식 multi-disease barcode classification 불가.
→ 대체 실험으로 **Feature Discrimination Analysis** 설계 (13.8절 참조).

---

### 13.3 Tier 1: 필수 실험 (없으면 Reject)

---

#### 실험 #1. 5종 전체 비교 테이블

| 항목 | 내용 |
|------|------|
| **상태** | 🟠 코드 완료, 2/5 variant만 실행 |
| **목적** | 논문 Table 1 — 모든 variant의 재구성/희소성 지표를 동일 조건에서 비교 |
| **산출물** | `results/comparison/comparison_table.csv`, 논문 **Table 1** |
| **소요** | ~2시간 (전체 variant 평가 실행) |

**구현 상태:**
- [x] `src/evaluation/metrics.py` — MSE, FVU, L0, Sparsity, Cosine Sim, Active/Dead Neurons
- [x] `tasks/evaluate.py` — 개별 체크포인트 평가
- [x] `tasks/compare.py` — 다중 체크포인트 비교 테이블 + Pareto plot
- [ ] **5종 전체 실행**: vanilla, gated, topk, jumprelu, msae 모두 evaluate + compare

**실행 방법:**
```bash
# 1) 전체 variant 평가
for v in vanilla gated topk jumprelu msae; do
    PYTHONPATH=./ python tasks/evaluate.py \
        --checkpoint models/checkpoints/${v}_e32_ep10/best.pt
done

# 2) 5종 비교 테이블 생성
PYTHONPATH=./ python tasks/compare.py \
    --checkpoints models/checkpoints/vanilla_e32_ep10/best.pt \
                  models/checkpoints/gated_e32_ep10/best.pt \
                  models/checkpoints/topk_e32_ep10/best.pt \
                  models/checkpoints/jumprelu_e32_ep10/best.pt \
                  models/checkpoints/msae_e32_ep10/best.pt
```

**논문 Table 1 형식:**

| Variant | MSE↓ | FVU↓ | R²↑ | L0 | Sparsity↑ | Active Neurons | Dead Neurons |
|---------|------|------|-----|-----|-----------|----------------|-------------|
| Vanilla | | | | | | | |
| Gated | | | | | | | |
| TopK | | | | | | | |
| JumpReLU | | | | | | | |
| MSAE | | | | | | | |

---

#### 실험 #2. Pareto Front (MSE vs L0)

| 항목 | 내용 |
|------|------|
| **상태** | 🟠 코드 완료, 2 variant만 포함 |
| **목적** | 논문 **Fig 2** — 재구성 품질 vs 희소성 트레이드오프 시각화 |
| **산출물** | `results/comparison/mse_vs_l0.png`, 논문 **Fig 2** |
| **소요** | 실험 #1과 동시 (compare.py가 자동 생성) |

**구현 상태:**
- [x] `src/evaluation/compare.py` — Pareto plot 자동 생성
- [ ] 5종 전체 포함된 Pareto plot
- [ ] R² vs L0 보조 Pareto plot 추가 (선택)

**필요 개선:**
- compare.py에서 MS Score를 color dimension으로 추가 (실험 #3 완료 후)
- Error bar 추가 (seed 반복 시)

---

#### 실험 #3. MonoSemanticity Score (MS Score) ⭐ 최우선

| 항목 | 내용 |
|------|------|
| **상태** | 🔴 미구현 |
| **목적** | label 없이 각 neuron의 해석성을 정량화 — variant 비교의 핵심 축 |
| **산출물** | `results/{variant}/ms_score.csv`, `results/comparison/ms_cdf.png`, 논문 **Table 2 + Fig 3** |
| **소요** | 구현 3일 + 실행 4~8시간 (GPU) |

**MS Score란:**
```
뉴런 j가 강하게 활성화시키는 이미지들이 서로 "비슷한가"를 정량화.
독립 encoder E로 이미지 embedding을 추출하고,
activation 가중 평균 cosine similarity로 계산.

MS_j = Σ_{n<m} (ã_n · ã_m) · cos(E(x_n), E(x_m))
       ─────────────────────────────────────────────
                    Σ_{n<m} (ã_n · ã_m)

MS → 1.0: 완벽한 monosemantic (한 개념만 활성)
MS → 0.0: polysemantic (여러 개념 혼합)
```

**구현 계획:**

```
신규 파일: src/evaluation/ms_score.py
```

```python
# 핵심 구조 (의사코드)
def compute_ms_score(
    sae_model,           # 학습된 SAE
    data_files,          # activation .pt 파일들
    independent_encoder, # CONCH, UNI, 또는 DINOv2 (EXAONEPath 아닌 것)
    tile_image_dir,      # 원본 tile 이미지 경로
    sample_size=50000,   # O(N²) 제한을 위한 서브샘플
    device="cuda",
):
    # 1) 서브샘플링: 전체 패치에서 sample_size개 랜덤 선택
    # 2) SAE activation 추출: [sample_size, 24576]
    # 3) 독립 encoder로 tile image embedding 추출: [sample_size, D_indep]
    # 4) Pairwise cosine similarity matrix: [N, N] (mini-batch 누적)
    # 5) 각 neuron j에 대해:
    #    - activation vector 추출 → min-max normalize → ã
    #    - relevance matrix R = outer(ã, ã)
    #    - MS_j = sum(R * S) / sum(R)  (upper triangle만)
    # 6) 결과: [24576] MS scores → CSV 저장
    pass
```

**독립 encoder 선택 (중요 — EXAONEPath 사용 불가, circular evaluation):**

| 옵션 | 장점 | 단점 | 권장 |
|------|------|------|------|
| **UNI** (병리 특화 ViT) | 조직학적 유사도에 근접 | weight 다운로드 필요 | ✅ 최우선 |
| **CONCH** (병리 VLM) | 병리 특화 + text 연동 | 셋업 복잡할 수 있음 | ✅ 차선 |
| **DINOv2** (범용 ViT) | 바로 사용 가능 | 병리 특화 아님 | ⚠️ 대안 |

**CLI 예시 (구현 후):**
```bash
PYTHONPATH=./ python tasks/eval_ms_score.py \
    --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
    --encoder uni \
    --sample_size 50000 \
    --output_dir results/vanilla_e32_ep10/ms/
```

**메모리 최적화:**
- Pairwise similarity는 O(N²)이므로 mini-batch 누적 또는 N≤50k로 제한
- Neuron별 MS는 독립이므로 병렬 가능

**논문 Fig 3 형식:**
```
┌─────────────────────────────────────┐
│  MS Score CDF (Cumulative Fraction) │
│                                      │
│  Y축: Fraction of neurons ≤ MS       │
│  X축: MS Score (0 ~ 1)               │
│                                      │
│  ─── No SAE (EXAONEPath 768-dim)    │
│  ─── Vanilla (blue)                  │
│  ─── Gated (orange)                  │
│  ─── TopK (green)                    │
│  ─── JumpReLU (red)                  │
│  ─── MSAE (purple)                   │
│                                      │
│  오른쪽으로 갈수록 monosemantic       │
└─────────────────────────────────────┘
```

---

#### 실험 #4. Concept Card + 14×14 Spatial Heatmap

| 항목 | 내용 |
|------|------|
| **상태** | 🟠 부분 구현 (`tasks/feature_viz.py` 존재) |
| **목적** | 논문 **Fig 5** — 정성적 해석성 증거 + spatial token의 장점 시각화 |
| **산출물** | `results/{variant}/feature_viz/`, 논문 **Fig 5** |
| **소요** | 코드 보완 1~2일 + 실행 variant당 30분 |

**구현 상태:**
- [x] `tasks/feature_viz.py` — top-N feature × top-K activating patches + spatial heatmap
- [x] Vinje-Gallant Sparseness (monosemanticity proxy) 계산
- [x] Z-score 기반 통계 표시
- [ ] **5종 전체 실행**: 현재 일부 variant만 feature_viz 생성됨
- [ ] **논문용 비교 Figure 생성**: variant별 같은 feature concept을 나란히 비교

**실행 방법:**
```bash
for v in vanilla gated topk jumprelu msae; do
    PYTHONPATH=./ python tasks/feature_viz.py \
        --checkpoint models/checkpoints/${v}_e32_ep10/best.pt \
        --top_n 20 --top_k 8 --max_files 10
done
```

**논문 Fig 5 구성 (concept card + spatial heatmap 3종 세트):**
```
┌──────────────────────────────────────────────────┐
│  Feature #1234 (MS=0.87, Entropy=0.3)            │
│                                                   │
│  ┌─────────┬─────────┬─────────┬─────────┐       │
│  │ Patch 1  │ Patch 2  │ Patch 3  │ Patch 4 │      │
│  │ (top act)│          │          │         │      │
│  ├─────────┼─────────┼─────────┼─────────┤       │
│  │ Patch 5  │ Patch 6  │ Patch 7  │ Patch 8 │      │
│  └─────────┴─────────┴─────────┴─────────┘       │
│                                                   │
│  Spatial Heatmap (14×14 → 224×224 overlay):       │
│  ┌──────────────┐                                 │
│  │ ▓▓▓░░░░░░░░░ │  ← 해당 feature가 활성화된      │
│  │ ▓▓▓▓░░░░░░░░ │     공간 영역 시각화              │
│  │ ░░▓▓▓░░░░░░░ │                                 │
│  └──────────────┘                                 │
└──────────────────────────────────────────────────┘
```

---

#### 실험 #5. Label Entropy (Class 기반 Feature Selectivity)

| 항목 | 내용 |
|------|------|
| **상태** | 🟠 부분 구현 (`src/evaluation/feature_figures.py`에 entropy 계산 존재) |
| **목적** | 논문 **Table 2 + Fig 4** — 각 feature가 특정 class에 특이적으로 활성화되는지 정량화 |
| **산출물** | `results/{variant}/label_entropy.csv`, `results/comparison/entropy_scatter_grid.png`, 논문 **Fig 4** |
| **소요** | 코드 보완 1~2일 + 실행 2~3시간 |

**Label Entropy란:**
```
각 feature j의 top-K activating patch를 수집 →
해당 patch들의 class label 분포에서 Shannon entropy 계산:

H(j) = -Σ p(class) × log₂(p(class))

H → 0: 한 class에만 활성 (selective/monosemantic)
H → log₂(C): 모든 class에 균등 활성 (non-selective)
```

**구현 상태:**
- [x] `src/evaluation/feature_figures.py` — entropy histogram 생성 로직 존재
- [x] `data/meta/group_meta_auto_vstatus_full.csv` — vital_status 기반 2-class label
- [ ] **다중 label 지원**: pathologic_stage, T-stage, N-stage로 확장
- [ ] **5종 variant 비교 scatter plot** (CytoSAE Fig.2A 스타일)

**추가 구현 필요:**

1) **Label 다양화** — vital_status(2 class)만으로는 entropy range가 좁음. 추가 label 생성:
```bash
# pathologic_stage 기반 group meta 생성
PYTHONPATH=./ python tasks/build_group_meta.py \
    --split full \
    --clinical_csv "/home/kimhj/projects/past_try/images/data/tcga_brca_survival/clinical_data(labels).csv" \
    --patient_col bcr_patient_barcode \
    --class_col pathologic_stage \
    --output_csv data/meta/group_meta_auto_stage_full.csv
```

2) **비교 scatter plot 생성** — `src/evaluation/feature_figures.py` 확장:
```python
# 각 variant에 대해:
# x축: log10(activation frequency)
# y축: log10(mean activation)
# 색상: label entropy
# → 5종을 2×3 grid로 배치 (+ No-SAE baseline)
```

**논문 Fig 4 형식 (CytoSAE Fig.2A 스타일):**
```
┌────────────────┬────────────────┬────────────────┐
│  Vanilla        │  Gated         │  TopK          │
│  (scatter)      │  (scatter)     │  (scatter)     │
│  색상=entropy   │  색상=entropy  │  색상=entropy  │
├────────────────┼────────────────┼────────────────┤
│  JumpReLU       │  MSAE          │  No-SAE (raw)  │
│  (scatter)      │  (scatter)     │  (scatter)     │
│  색상=entropy   │  색상=entropy  │  색상=entropy  │
└────────────────┴────────────────┴────────────────┘
```

---

### 13.4 Tier 2: 강력 권장 (Borderline → Accept)

---

#### 실험 #6. Concept Count (Threshold Sweep)

| 항목 | 내용 |
|------|------|
| **상태** | 🔴 미구현 |
| **목적** | "의미 있는 concept이 몇 개인가?"를 variant별로 정량화 |
| **산출물** | `results/comparison/concept_count_curve.png`, 논문 **Table 2** 보조 |
| **소요** | 구현 0.5일 |

**방법:**
```python
# 각 variant에서:
# 1) 전체 데이터에서 각 neuron의 mean activation 계산
# 2) log10(mean_activation) 기준 threshold sweep: -6, -5.5, ..., 0
# 3) 각 threshold에서 초과하는 neuron 수 = "concept count"
# 4) 4~5종 variant의 curve를 같은 plot에 overlay

thresholds = np.arange(-6, 0.5, 0.5)
for th in thresholds:
    count = (log10_mean_act > th).sum()
```

**시각화:**
```
Y축: Number of concepts above threshold
X축: Mean activation threshold (log10)
─── Vanilla, ─── Gated, ─── TopK, ─── JumpReLU, ─── MSAE
```

---

#### 실험 #7. SAE vs No-SAE 비교 ("왜 SAE를 써야 하는가?")

| 항목 | 내용 |
|------|------|
| **상태** | 🔴 미구현 |
| **목적** | EXAONEPath 원본 768-dim의 neuron도 MS Score/Entropy를 계산하여, SAE가 monosemanticity를 향상시키는지 증명 |
| **산출물** | Fig 3, Fig 4에 "No SAE" baseline 추가 |
| **소요** | 실험 #3, #5의 코드 재사용, 추가 0.5일 |

**방법:**
```
EXAONEPath 원본 768-dim의 각 차원을 "neuron"으로 취급:
- 각 차원의 activation을 추출 (이미 data/activations에 저장됨)
- 동일한 MS Score, Label Entropy 계산
- SAE variant와 같은 plot에 overlay

기대 결과: SAE neurons의 MS/Entropy가 원본 FM보다
           유의미하게 monosemantic해야 함 (논문의 핵심 근거)
```

---

#### 실험 #8. Spatial Attribution 비교 (Variant별 동일 Tile Heatmap)

| 항목 | 내용 |
|------|------|
| **상태** | 🟠 feature_viz.py에 spatial heatmap 존재, 비교 기능 미구현 |
| **목적** | 논문 **Fig 6** — 196 spatial token의 장점 + variant별 localization 차이 시각화 |
| **산출물** | `results/comparison/spatial_comparison_*.png`, 논문 **Fig 6** |
| **소요** | 구현 1~2일 |

**방법:**
```
동일 tile에 대해 5종 SAE의 특정 concept heatmap을 나란히 비교:

1) 대표 tile 5~10개 선정 (tumor, stroma, immune 등 다양한 영역)
2) 각 tile의 196 token에 대해 5종 SAE activation 추출
3) 특정 concept (high-MS neuron)의 14×14 activation map 생성
4) 원본 tile 위에 overlay → 5종 비교 Figure
```

**논문 Fig 6 형식:**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ 원본 Tile │ Vanilla  │ Gated    │ TopK     │ JumpReLU │ MSAE     │
│ (224×224) │ heatmap  │ heatmap  │ heatmap  │ heatmap  │ heatmap  │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Tile #2   │          │          │          │          │          │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

#### 실험 #9. Feature Discrimination Analysis ⭐ (Barcode Classification 대체)

| 항목 | 내용 |
|------|------|
| **상태** | 🔴 미구현 |
| **목적** | "SAE feature가 임상적으로 유용한가?"를 Barcode Classification 없이 증명 |
| **산출물** | `results/comparison/discrimination_table.csv`, 논문 **Table 3 + Fig 7** |
| **소요** | 구현 3~4일 |

**왜 Barcode Classification이 불가능한가:**
```
CytoSAE: 5개 질환(AML 4 subtype + healthy), 수백 명 환자 → LogReg로 F1=0.83
PathoSAEv3: 1개 암종(BRCA), 96명 환자, vital_status 2 class (91% vs 9%)
→ 환자 수 부족 + 극심한 class imbalance + 단일 암종으로 질환 분류 자체가 무의미
```

**대체 실험: Feature Discrimination Analysis**

Barcode classification 대신, SAE feature 자체가 임상 label과 얼마나 연관되는지를 **feature 단위**로 분석한다.
환자 수가 적어도 feature-level 분석은 패치 수(266만)가 충분하므로 통계적으로 유효하다.

**3가지 서브 분석:**

**(A) Feature-level AUROC (개별 feature의 class 분리력)**
```python
# 각 feature j에 대해:
# 1) 전체 패치의 activation a_j 추출
# 2) 각 패치의 class label (vital_status, stage 등) 확인
# 3) activation 값만으로 class를 분리하는 AUROC 계산
# 4) AUROC > 0.7인 feature 수 = "discriminative concept 수"

from sklearn.metrics import roc_auc_score
for j in range(n_latents):
    auroc_j = roc_auc_score(labels, activations[:, j])
    if auroc_j < 0.5:  # 방향 보정
        auroc_j = 1 - auroc_j
```

**산출물 (Table 3):**

| Variant | Discriminative Features (AUROC>0.7) | Mean Top-100 AUROC | Max AUROC |
|---------|------------------------------------|--------------------|-----------|
| Vanilla | | | |
| Gated | | | |
| TopK | | | |
| JumpReLU | | | |
| MSAE | | | |
| No-SAE (768-dim) | | | |

**(B) Class-Conditional Activation Analysis (클래스별 평균 활성 차이)**
```python
# 각 feature j에 대해:
# class 0의 평균 activation vs class 1의 평균 activation
# → effect size (Cohen's d) 계산
# → 상위 discriminative feature의 concept card 제시
#    "이 feature가 생존/사망 환자에서 다르게 활성화된다"

d_j = (mean_class1 - mean_class0) / pooled_std
```

**(C) Multi-label Discrimination Profile (다중 임상 라벨 활용)**
```python
# vital_status, pathologic_stage, T-stage, N-stage 4개 label에 대해
# 각각 AUROC 계산 → feature × label AUROC heatmap 생성
# → "이 feature는 stage에 특이적, 저 feature는 vital_status에 특이적"

# 논문 Fig 7: 상위 discriminative feature의 class-conditional activation 분포
```

**논문 Fig 7 형식:**
```
┌─────────────────────────────────────────────┐
│ (A) Discriminative Feature AUROC 분포        │
│     variant별 histogram overlay              │
│                                              │
│ (B) Top-3 Discriminative Feature의           │
│     class-conditional activation boxplot     │
│     + 해당 feature의 concept card             │
│                                              │
│ (C) Feature × Label AUROC heatmap            │
│     (상위 50 features × 4 labels)             │
└─────────────────────────────────────────────┘
```

---

### 13.5 Tier 3: 리뷰 대응용 (시간 있으면)

| # | 실험 | 상태 | 구현 난이도 | 언제 하는가 |
|---|------|------|---------|---------|
| 10 | Decoder Weight UMAP + Concept Clustering | 🔴 | 낮 (1일) | 논문 Figure 추가 시 |
| 11 | Feature Splitting Detection (Jaccard) | 🔴 | 낮 (0.5일) | Rebuttal 대응 |
| 12 | Universality (multi-seed matching) | 🔴 | 중 (학습 3회 추가) | Rebuttal 대응 |
| 13 | HIF Correlation (cell detection 기반) | 🔴 | 높 (HoVer-Net 필요) | Camera-ready |
| 14 | Stain/Scanner Robustness (AUROC≈0.5) | 🔴 | 중 (기관 metadata 필요) | Rebuttal 대응 |

---

### 13.6 논문 Figure/Table 최종 매핑

#### Tables

| Table # | 내용 | 대응 실험 | 상태 |
|---------|------|---------|------|
| **Table 1** | 5종 재구성/희소성 지표 비교 (MSE, FVU, R², L0, Sparsity, Active/Dead) | #1 | 🟠 |
| **Table 2** | 5종 해석성 지표 비교 (MS Score mean/median, Low-Entropy neuron %, Concept Count) | #3, #5, #6 | 🔴 |
| **Table 3** | 5종 Feature Discrimination (Discriminative feature 수, Mean AUROC, Max AUROC) | #9 | 🔴 |

#### Figures

| Figure # | 내용 | 대응 실험 | 상태 |
|----------|------|---------|------|
| **Fig 1** | 파이프라인 overview (EXAONEPath → 5 SAE → evaluation) | 도식 | 🔴 |
| **Fig 2** | Pareto front: MSE vs L0 (5종, MS Score color overlay) | #1, #2 | 🟠 |
| **Fig 3** | MS Score CDF (5종 + No-SAE baseline) | #3, #7 | 🔴 |
| **Fig 4** | Label Entropy scatter (2×3 grid: 5종 + No-SAE, CytoSAE 스타일) | #5, #7 | 🟠 |
| **Fig 5** | Concept Card + Spatial Heatmap (variant별 대표 3~4 features) | #4 | 🟠 |
| **Fig 6** | Variant별 동일 tile heatmap 비교 (spatial attribution) | #8 | 🔴 |
| **Fig 7** | Feature Discrimination Analysis (AUROC 분포 + class-conditional + heatmap) | #9 | 🔴 |

---

### 13.7 구현 타임라인 (권장)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Week 1: 기반 완성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Day 1-2:  [#1,#2] 5종 전체 evaluate + compare → Table 1, Fig 2
 Day 3-4:  [#4] 5종 feature_viz 실행 → Fig 5 (concept card + heatmap)
 Day 5:    [#5] Label Entropy 다중 label 확장 → Fig 4 초안

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Week 2: 핵심 해석성 지표
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Day 1-3:  [#3] MS Score 구현 (ms_score.py) + UNI/DINOv2 셋업
 Day 4:    [#3] MS Score 5종 실행 → Fig 3, Table 2
 Day 5:    [#7] No-SAE baseline 계산 → Fig 3,4에 추가

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Week 3: 임상 유용성 + 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Day 1-2:  [#9] Feature Discrimination Analysis 구현 → Table 3, Fig 7
 Day 3:    [#6] Concept Count threshold sweep → Table 2 보조
 Day 4-5:  [#8] Spatial Attribution 비교 Figure → Fig 6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Week 4: 논문 작성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Fig 1 도식 작성, 전체 Figure/Table 정리, 본문 작성
```

---

### 13.8 신규 구현 모듈 목록

```
src/evaluation/
├── metrics.py                  # ✅ 기존: MSE, FVU, L0, Sparsity, Cosine Sim
├── compare.py                  # ✅ 기존: variant 비교 CSV + Pareto plot
├── feature_analysis.py         # ✅ 기존: feature 통계, top-k patches
├── feature_figures.py          # ✅ 기존: scatter, entropy histogram
├── group_analysis.py           # ✅ 기존: class/patient group heatmap
├── ms_score.py                 # 🔴 신규: MonoSemanticity Score 계산
├── concept_count.py            # 🔴 신규: threshold sweep concept counting
├── discrimination.py           # 🔴 신규: Feature Discrimination Analysis
└── spatial_comparison.py       # 🔴 신규: variant간 동일 tile heatmap 비교

tasks/
├── eval_ms_score.py            # 🔴 신규: MS Score CLI
├── eval_discrimination.py      # 🔴 신규: Discrimination Analysis CLI
└── generate_paper_figures.py   # 🔴 신규: 논문 Figure 일괄 생성
```

---

### 13.9 실험별 상태 요약 체크리스트

```
Tier 1 (필수 — 없으면 Reject):
  [🟠] #1  5종 전체 비교 테이블          → Table 1
  [🟠] #2  Pareto Front (MSE vs L0)     → Fig 2
  [🔴] #3  MS Score (독립 encoder)       → Table 2, Fig 3
  [🟠] #4  Concept Card + Heatmap       → Fig 5
  [🟠] #5  Label Entropy                → Table 2, Fig 4

Tier 2 (강력 권장 — Accept 확률 상승):
  [🔴] #6  Concept Count Curve          → Table 2 보조
  [🔴] #7  SAE vs No-SAE 비교           → Fig 3,4 baseline
  [🟠] #8  Spatial Attribution 비교      → Fig 6
  [🔴] #9  Feature Discrimination       → Table 3, Fig 7

Tier 3 (리뷰 대응용):
  [🔴] #10 UMAP + Concept Clustering
  [🔴] #11 Feature Splitting Detection
  [🔴] #12 Universality (multi-seed)
  [🔴] #13 HIF Correlation
  [🔴] #14 Stain/Scanner Robustness

상태 범례: ✅ 완료 | 🟠 부분 구현 | 🔴 미구현
```

---

## 14. 현재 진행 현황 (2026-03-03 기준)

### 14.1 버그 수정 이력

| 파일 | 버그 | 수정 내용 | 날짜 |
|------|------|---------|------|
| `src/models/gated_sae.py` | b_gate에 gradient 미흐름 (Heaviside function 사용), L_aux 없음, L1 target 잘못됨 (feature_acts에 적용) | `r_mag`/`b_mag` 파라미터 추가, gate×magnitude 분리 구현, f_gate에 L1 + decoder detach L_aux 추가 (논문 수식 그대로) | 2026-03-02 |
| `src/evaluation/metrics.py` | Dead neuron을 "한번도 미활성" 기준으로만 측정 → Vanilla에서 dead=0% 오류 | frequency 기반 dead neuron 추가: `dead_freq_1e4` (<1e-4), `dead_freq_1e3` (<1e-3), `active_freq_1pct` (≥1%) | 2026-03-02 |
| `tasks/evaluate.py`, `src/evaluation/metrics.py`, `src/extract/activation_store.py` | `evaluate --split`가 feature 분석에만 반영되고 reconstruction metric은 항상 val split으로 계산됨 | `split={train,val,full}`를 reconstruction metric loader에도 반영 (`ActivationStore.get_loader`) | 2026-03-29 |

---

### 14.2 모델 학습 현황

| SAE | 상태 | GPU | 시작 | 완료/예상 | 비고 |
|-----|------|-----|------|---------|------|
| **Vanilla** | ✅ 학습 완료 | 0 | Feb 21 | Feb 23 (~35h) | |
| **Gated (v1)** | ❌ 결과 무효 | — | Feb 25 | Feb 26 (~40h) | 버그 있는 버전, Vanilla와 동일 결과 |
| **Gated (v2, 수정)** | 🔄 학습 중 | 5 | Mar 02 20:00 | Mar 04 ~12:00 (~40h) | gated_sae.py 버그 수정 후 재학습 |
| **TopK** | ✅ 학습 완료 | — | Feb 25 | Feb 26 (~35h) | |
| **JumpReLU** | ✅ 학습 완료 | — | Feb 21 | Feb 23 (~44h) | |
| **MSAE** | ✅ 학습 완료 | — | Feb 25 | Feb 28 (~79h) | nesting=[64,128,256,512], importance=uniform |
| **DriftPrior** | ✅ 학습 완료 | auto | Mar 01 03:00 | Mar 03 01:50 | `best.pt/final.pt` 저장 완료 |

---

### 14.3 평가 결과 현황

현재 확보된 지표 (val split 기준):

| SAE | MSE | FVU | L0 | Sparsity | Cosine Sim | Active | Dead (never) | dead_freq_1e4 | dead_freq_1e3 | active_1pct |
|-----|-----|-----|-----|---------|-----------|--------|-------------|-------------|-------------|------------|
| **Vanilla** | 2.75e-4 | 6.83e-4 | 953 | 96.1% | 0.9998 | 24,576 | 0 (0%) | 21,555 (87.7%) | 22,403 (91.2%) | 2,016 (8.2%) |
| **Gated (v1, 무효)** | 2.75e-4 | 6.83e-4 | 953 | 96.1% | 0.9998 | 24,576 | 0 | — | — | — |
| **Gated (v2)** | 🔄 | | | | | | | | | |
| **TopK** | 0.1139 | 0.2827 | 64 | 99.7% | 0.8831 | 2,860 | 21,716 (88.4%) | — | — | — |
| **JumpReLU** | 7.73e-3 | 1.92e-2 | 650 | 97.4% | 0.9925 | 11,589 | 12,987 (52.8%) | — | — | — |
| **MSAE** | 9.44e-3 | 2.34e-2 | 512 | 97.9% | 0.9911 | 3,947 | 20,629 (83.9%) | — | — | — |
| **DriftPrior** | 1.88e-7 | 4.66e-7 | 2,511 | 89.8% | 1.0000 | 2,512 | 22,064 (89.8%) | 22,064 (89.8%) | 22,064 (89.8%) | 2,512 (10.2%) |

> **주의:** TopK/JumpReLU/MSAE의 `dead_freq_*` 필드는 아직 계산 전 (metrics.py 수정 이전 결과).
> **주의:** Vanilla `dead (never) = 0`은 버그가 아닌, 524M 샘플에서 모든 뉴런이 최소 1회 활성화된 것. 실제 의미 있는 dead 기준은 `dead_freq_1e4 = 21,555 (87.7%)`.

---

### 14.4 현재 실행 중인 프로세스

```
GPU 0: Vanilla full eval (feature_figures, 96 WSI) — ~98% 완료
GPU 1: JumpReLU full eval (feature_figures, 96 WSI) — ~98% 완료
GPU 5: Gated SAE (v2) 학습 — Epoch 1/10 진행 중
GPU auto: DriftPrior 학습 — Epoch 8/10 완료, 9-10 진행 중
```

> **2026-03-29 업데이트:** DriftPrior 학습 프로세스는 종료되었고 체크포인트/평가 결과가 생성됨.

---

### 14.5 생성된 결과물

| SAE | metrics.json | feature_analysis.json | feature_viz | top_feature_grid | group heatmap |
|-----|-------------|--------------------|------------|----------------|--------------|
| Vanilla | ✅ (freq 포함) | ✅ (full, top_k=5) | ✅ | ✅ | ✅ |
| Gated (v1) | ❌ 무효 | — | — | — | — |
| TopK | 🟠 (freq 없음) | — | — | — | — |
| JumpReLU | 🟠 (freq 없음) | 🔄 (full eval 진행 중) | — | — | — |
| MSAE | 🟠 (freq 없음) | — | — | — | — |
| DriftPrior | ✅ (freq 포함) | ✅ | — | ✅ | — |

---

### 14.6 L0 해석 노트

**현재 L0 수치 판단:**

병리 이미지는 LLM과 달리 L0=50~100 기준이 직접 적용 안 됨:
- CytoSAE, PathAI PLUTO 모두 L0 수치를 보고하지 않음
- 두 논문 모두 "latent이 해석 가능한가" (monosemanticity) 중심으로 argument

**핵심 판단 기준:**
- L0=953이어도 논문 가능: 개별 feature가 monosemantic이면 → "953개 concept의 조합으로 표현"
- L0=953이면 문제: feature가 polysemantic이면 → dictionary learning 자체가 실패

**판단 방법:** MS Score 계산 또는 concept card 육안 검토

**현재 Vanilla 상위 feature 시각 검토 결과:**
- Feature #12085 (freq=73.7%): 지방조직/acellular 영역 — 시각적으로 coherent
- Feature #12002 (freq=73.0%): 유방 도관/소엽 — 시각적으로 coherent
- 상위 2개는 monosemantic으로 보이나, 전체 판단은 MS Score 계산 필요

---

### 14.7 다음 단계 우선순위 (2026-03-03 기준)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Phase A: 학습 완료 대기 (재학습 불필요, 병렬 진행)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [1] DriftPrior exploratory 후속 (matched-L0 sweep)
     → `prior_sparsity` sweep(0.96/0.97/0.98) 후 canonical 대비 비교

 [2] Gated (v2) 학습 완료 대기 (~Mar 04 12:00)
     → 완료 즉시 metrics-only evaluate 실행

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Phase B: 지표 정비 (재학습 불필요, 빠름)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [3] TopK: freq-based dead neuron 재평가 (val split, GPU 여유 시)
 [4] JumpReLU: freq-based 재평가 (현재 full eval 완료 후)
 [5] MSAE: freq-based 재평가
 [6] 모든 지표 확보 후 compare.py 실행 → Table 1, Fig 2 생성

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Phase C: 핵심 해석성 실험 (신규 구현 필요)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [7] MS Score 구현 (src/evaluation/ms_score.py)
     → 독립 encoder: UNI 또는 DINOv2
     → Vanilla + Gated(v2) 우선 실행
     → L0=953 monosemanticity 판단 → L1 sweep 필요 여부 결정

 [8] Concept Card 강화 (top_k=32, 상위 20개 뉴런 대상)
     → feature_analysis.py top_k 파라미터 높여 재실행
     → 4×8 patch grid 시각화 스크립트

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Phase D: 임상 유용성 + 논문 Figure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [9]  Feature Discrimination Analysis (AUROC per feature)
 [10] Label Entropy 다중 label 확장 (stage, T/N-stage 추가)
 [11] Concept Count threshold sweep
 [12] 논문 Figure 1 (파이프라인 도식) 작성

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 미결 결정 사항 (사용자 판단 필요)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 [ ] MS Score 결과 보고 L1 sweep 여부 결정 (L0 100~200 목표 시)
 [ ] TopK K sweep 필요 여부 (K=128, 256 추가 학습)
 [ ] Ghost gradient 적용 범위 결정
 [ ] 3 seeds 여부 결정 (현재 seed=42만 존재)
```

상태 범례: ✅ 완료 | 🟠 부분 구현 | 🔴 미구현 | 🔄 진행 중 | ❌ 무효/필요

---

## 14.8 빠른 업데이트 (2026-03-30)

- UNI activation extraction 완료:
  - `data/activations_uni/spatial_folder_000~191.pt` (총 192개 폴더)
- DriftPrior는 main canonical 비교가 아니라 exploratory 트랙으로 운용:
  - `matched-L0 sweep` 스크립트 추가: `scripts/run_driftprior_matched_l0_sweep.sh`
  - sweep 기본값: `prior_sparsity={0.96, 0.97, 0.98}`
- 평가 split 정합성 수정:
  - `tasks/evaluate.py --split`이 feature 분석뿐 아니라 reconstruction metrics에도 동일하게 반영되도록 수정
