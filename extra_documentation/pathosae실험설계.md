# PathoSAEv3 실험 지표 종합 전략서

## CytoSAE 분석 + 추가 지표 제안 + 도입 전략 + 구현 로드맵

---

## 목차

1. [현황 진단: 현재 PathoSAEv3에 없는 것](#1-현황-진단)
2. [CytoSAE 논문 지표 분석 및 도입 전략](#2-cytosae-논문-지표-분석)
3. [추가 제안 지표 — SAE / XAI / Medical 분야](#3-추가-제안-지표)
4. [전체 지표 맵: 질문 → 지표 → 구현](#4-전체-지표-맵)
5. [논문 Figure/Table 구성안](#5-논문-구성안)
6. [구현 로드맵 및 코드 아키텍처](#6-구현-로드맵)

---

## 1. 현황 진단

PathoSAEv3는 현재 **재구성 품질 지표만** 보유하고 있다. 이것만으로는 ACCV 2026 리뷰어를 설득할 수 없다.

### 현재 보유 지표

| 지표 | 측정 대상 | 논문 설득력 |
|------|----------|-----------|
| MSE | 재구성 오차 | ⚠️ 필수이나 차별화 불가 |
| FVU | 미설명 분산 비율 | ⚠️ MSE의 변형 |
| L0 | 활성 뉴런 수 | ⚠️ sparsity만 측정 |
| Sparsity % | 0인 활성화 비율 | ⚠️ L0와 중복 |
| Cosine Sim | 방향 보존 | ⚠️ 재구성에만 초점 |
| Dead Neurons | 비활성 뉴런 수 | ⚠️ 단순 카운트 |

### 핵심 부재 영역

**1) 해석성 정량화** — "SAE가 해석 가능하다"는 주장에 숫자가 없음
**2) 생물학적 의미 검증** — feature가 실제 조직 형태학과 대응하는지 증거 없음
**3) 임상 유용성** — 해석 가능한 feature로 실제 분류가 가능한지 미검증
**4) 공간적 해석 검증** — 196 spatial token의 장점을 입증하는 지표 없음

PathoSAEv3의 핵심 기여는 "4종 SAE 공정 비교"이므로, **재구성 품질 이외의 축에서 variant간 차이를 보여야** 논문이 성립한다.

---

## 2. CytoSAE 논문 지표 분석

CytoSAE(MICCAI 2025)는 혈액학 도메인에서 SAE의 임상 유용성을 보여준 첫 논문이다. 도메인은 다르지만 PathoSAEv3에 직접 도입 가능한 지표가 다수 있다.

---

### 2.1 Label Entropy 기반 Latent 품질 분류

#### 왜 필요한가

24,576개 잠재 뉴런을 모두 수동 검토하는 것은 불가능하다. 어떤 뉴런이 의미 있는 concept을 학습했는지 **자동으로 선별하는 기준**이 필요하며, label entropy가 그 역할을 한다. 낮은 entropy는 특정 조직 타입에 집중된 활성화를 의미하고, 이는 곧 해석 가능한 concept이다.

#### CytoSAE에서의 구현

CytoSAE는 각 SAE latent를 가장 강하게 활성화시키는 이미지들을 수집한 뒤, 해당 이미지들의 ground-truth cell type label 분포에서 Shannon entropy를 계산했다.

- **시각화**: Fig.2A — scatter plot에서 x축이 log10(activation frequency), y축이 log10(mean activation), **색상이 label entropy**
- 두 개의 뚜렷한 클러스터가 관찰됨: 고빈도/저entropy(의미있는 concept) vs 저빈도/고entropy(비특이적 feature)

#### PathoSAEv3 도입 방법

**필요 데이터**: EXAONEPath 타일에 대한 조직 타입 annotation이 필요하다. TCGA 데이터를 사용한다면 tissue type label(tumor, stroma, immune, necrosis, normal, adipose 등)을 확보해야 한다. 자체 annotation이 어려우면 HoVer-Net이나 기존 tissue classifier로 pseudo-label을 생성할 수 있다.

**테스트 절차**:

1. Validation set의 전체 패치에 대해 4개 SAE variant 각각의 activation 추출
2. 각 latent j에 대해, activation이 가장 높은 top-100 패치 수집
3. 해당 패치들의 tissue label 분포에서 Shannon entropy 계산:
   `H(j) = -Σ p(label) × log₂(p(label))`
4. 4개 variant의 entropy 분포를 비교

**비교 포인트**: variant별로 low-entropy latent(H < 1.0)의 **수**와 **비율**을 비교한다. JumpReLU의 극단적 sparsity가 concept 품질을 높이는지, 아니면 정보 손실로 이어지는지가 핵심 발견이 될 수 있다.

**시각화 예시**: CytoSAE Fig.2A 스타일의 scatter plot을 4개 variant에 대해 2×2 grid로 배치. 각 점이 하나의 latent이며 색상이 entropy를 나타낸다.

```
┌────────────────────┬────────────────────┐
│  Vanilla SAE       │  Gated SAE         │
│  (scatter plot)    │  (scatter plot)     │
│  색상: entropy     │  색상: entropy      │
├────────────────────┼────────────────────┤
│  TopK SAE          │  JumpReLU SAE      │
│  (scatter plot)    │  (scatter plot)     │
│  색상: entropy     │  색상: entropy      │
└────────────────────┴────────────────────┘
```

**구현 스케치**:

```python
def compute_label_entropy(activations, labels, topk=100):
    """
    activations: [N_patches, D_sae] — SAE latent activations
    labels: [N_patches] — tissue type labels (int)
    Returns: [D_sae] — entropy per latent
    """
    n_latents = activations.shape[1]
    n_classes = labels.max() + 1
    entropies = torch.zeros(n_latents)
    
    for j in range(n_latents):
        top_idx = torch.topk(activations[:, j], topk).indices
        top_labels = labels[top_idx]
        counts = torch.bincount(top_labels, minlength=n_classes).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies[j] = -(probs * probs.log2()).sum()
    
    return entropies
```

---

### 2.2 활성화 임계값 기반 Concept 수 카운팅

#### 왜 필요한가

"SAE가 얼마나 많은 의미 있는 concept을 발견했는가?"를 정량화해야 한다. Dead neuron 수의 반대 개념으로, **실제로 의미 있게 활성화되는 concept의 수**가 SAE의 표현력을 나타낸다.

#### CytoSAE에서의 구현

- 각 latent의 mean activation을 log10 스케일로 계산
- 다양한 threshold(-6, -5, ..., 0)에서 해당 threshold 이상인 latent 수를 카운트
- Expansion factor 64 기준, threshold -3에서 **2936±114 concepts** 발견
- 핵심 발견: expansion factor를 늘려도 highly activated cluster의 크기는 비교적 안정적

#### PathoSAEv3 도입 방법

**테스트 절차**:

1. Validation set에서 각 latent의 mean activation 계산
2. log10 scale로 변환
3. Threshold를 -6부터 0까지 0.5 간격으로 sweep
4. 각 threshold에서 초과하는 latent 수를 카운트
5. 4개 variant의 concept count curve를 같은 그래프에 오버레이

**비교 포인트**: 
- TopK(k=64)는 정확히 64개만 활성화하므로, mean activation 분포가 다를 것
- JumpReLU는 adaptive threshold로 인해 "의미있는" latent만 선별적으로 활성화할 가능성
- 같은 expansion factor(32x)에서 variant별 meaningful concept 수의 차이

**시각화**:

```
Y축: Number of concepts above threshold
X축: Mean activation threshold (log10)

        ─── Vanilla (blue)
        --- Gated (orange)  
        ··· TopK (green)
        ─·─ JumpReLU (red)

각 variant의 curve가 다른 위치에서 plateau를 형성
→ JumpReLU가 적지만 고품질 concept만 남기는지 확인
```

---

### 2.3 계층적 Concept 집계 — Barcode

#### 왜 필요한가

병리 진단은 단일 패치가 아닌 **WSI 전체**에서 이루어진다. patch-level activation을 tile → WSI → patient → disease 수준으로 집계하면, SAE feature의 **임상적 활용 가능성**을 직접 보여줄 수 있다.

#### CytoSAE의 계층적 집계 (Eq. 2-3)

```
patch-level:   a_{i,j}[s] = I(h_{i,j}[s] > τ)           ← 이진화
image-level:   a_i[s]     = Σ_j a_{i,j}[s]               ← 패치별 합산
patient-level: a_p[s]     = (1/N_p) Σ_{i∈I_p} a_i[s]     ← 환자 내 평균
disease-level: a_d[s]     = (1/N_d) Σ_{p∈P_d} a_p[s]     ← 질환별 평균
```

CytoSAE는 이 barcode로 AML 4 subtype + healthy를 weighted F1 0.832로 분류했다.

#### PathoSAEv3 도입 방법

PathoSAEv3는 196 spatial token을 사용하므로, CytoSAE보다 **더 세밀한 공간적 barcode**가 가능하다.

**테스트 절차**:

1. 각 tile의 196 token × 24,576 latent activation에서, latent별로 활성화된 token 수를 카운트 → **tile-level barcode** [24,576]
2. WSI 내 모든 tile의 barcode를 평균 → **WSI-level barcode** [24,576]
3. 질환별(TCGA cancer type) WSI barcode 평균 → **disease barcode** [24,576]
4. Disease barcode 간 차이가 큰 latent → disease-specific concept

**비교 포인트**: 어떤 SAE variant의 barcode가 가장 **discriminative**한가? 동일 질환의 barcode 간 유사도가 높고(intra-class), 다른 질환 간 유사도가 낮은(inter-class) variant가 우수하다.

**시각화**: CytoSAE Fig.5A 스타일 — 두 질환의 disease barcode를 scatter plot으로 비교. 축이 각 질환의 mean concept count이고, 대각선에서 벗어난 점이 disease-specific concept.

---

### 2.4 Barcode 기반 Linear Classification

#### 왜 필요한가

SAE feature가 **실제 임상 판단에 유용한지**의 최종 검증이다. 해석 가능한 feature만으로 질환을 분류할 수 있다면, 그 feature는 임상적 가치가 있다. 이것이 "black-box 성능과 동등한 해석 가능 시스템" 논증의 근거가 된다.

#### CytoSAE에서의 구현

- Patient barcode를 input으로 Logistic Regression(L2 regularization) 학습
- AML 4 subtype + healthy: **weighted F1 = 0.832 ± 0.044**
- 핵심 발견: latent 수를 줄여도(threshold를 높여도) 성능이 유지됨 → meaningful concept만으로 충분
- Deep learning baseline(Hehr et al.)과 **비슷한 성능** 달성

#### PathoSAEv3 도입 방법

**테스트 절차**:

1. TCGA에서 2~3개 분류 task 선정 (예: BRCA subtype, LUAD vs LUSC, 또는 multi-cancer)
2. 각 WSI의 barcode를 생성 (variant별)
3. Logistic Regression(L2, sklearn)으로 5-fold CV
4. Metric: weighted F1, balanced accuracy
5. Baseline 비교:
   - EXAONEPath 원본 768-dim CLS embedding 평균
   - EXAONEPath 원본 768-dim spatial embedding 평균
   - 4개 SAE variant의 barcode

**추가 분석**: CytoSAE Fig.4A처럼 activation threshold를 변화시키며 사용하는 latent 수를 줄여도 성능이 유지되는지 확인. 이는 "소수의 meaningful concept만으로 충분하다"는 해석성 논증의 핵심.

**시각화**:

```
┌─────────────────────────────┐
│  Fig: Classification F1     │
│  vs Number of Latents       │
│                              │
│  Y축: Weighted F1            │
│  X축: Mean activation        │
│       threshold (log10)      │
│                              │
│  각 variant의 curve           │
│  + baseline horizontal line  │
└─────────────────────────────┘
```

---

### 2.5 전문가 검증 (Expert Annotation)

#### 왜 필요한가

자동 지표만으로는 **"의학적 의미"**를 확인할 수 없다. Medical AI 논문에서 리뷰어가 가장 중시하는 것 중 하나가 "실제 의료진이 이 결과에 동의하는가?"이다. CytoSAE는 15년 경력 세포형태학자의 검증을 받았다.

#### CytoSAE에서의 프로토콜

1. KMeans(k=10)로 high-activation latent 클러스터링
2. 클러스터당 5개 랜덤 선택 → 총 50 latent
3. 각 latent의 top-activating image grid를 전문가에게 제시
4. "이 latent가 의미 있는 형태학적 개념을 나타내는가?" → Yes/No + concept label
5. 결과: disease-specific concept 중 CBFB::MYH11 32/50, PML::RARA 10/50 확인

#### PathoSAEv3 도입 방법

**리뷰 프로토콜 설계**:

1. 4개 variant 각각에서 high-activation latent 중 50개 선별 (KMeans 또는 stratified sampling)
2. 각 latent의 top-10 activating tile을 8×8 grid로 시각화
3. 14×14 spatial heatmap도 함께 제시 (어떤 영역이 활성화되는지)
4. 병리학자에게 제시:
   - "이 feature가 하나의 일관된 형태학적 패턴을 나타내는가?" (1-5 Likert scale)
   - 패턴 label 부여: glandular, stromal, immune infiltrate, necrosis, tumor border 등
5. **Variant별 "의미있는 concept" 비율 비교**: 예를 들어 Vanilla 60%, JumpReLU 80%

**협력처**: 이주상 교수님 팀 또는 삼성의료원 병리팀

**우선순위**: P3 (논문 제출 직전, 다른 자동 지표 완성 후)

---

### 2.6 Patch-level Attribution — Sub-region 분석

#### 왜 필요한가

CytoSAE의 차별점 중 하나는 **patch-level에서 sub-cellular 구조를 식별**한 것이다. PathoSAEv3는 196 spatial token을 사용하므로, CytoSAE보다 훨씬 풍부한 공간적 해석이 가능하다. 이것은 CLS-only 접근법(PLUTO-SAE) 대비 **PathoSAEv3의 고유 장점**이다.

#### CytoSAE에서의 구현 (Fig.3)

1. 특정 patch의 token을 SAE에 통과시켜 activated latent 확인
2. 해당 latent를 강하게 활성화하는 다른 이미지의 reference patch 수집
3. Segmentation map 생성: 각 patch를 latent activation 값으로 곱함
4. 핵심 발견: 다른 데이터셋(bone marrow 포함)에서도 동일한 morphological concept(eosinophilic granules) 일관되게 검출

#### PathoSAEv3 도입 방법

**테스트 절차**:

1. 특정 latent s에 대해, tile i의 14×14 spatial activation map 생성:
   `M_s[row, col] = activation[token_{row×14+col}][latent_s]`
2. 이 heatmap을 원본 tile 위에 오버레이
3. Ground-truth tissue annotation(있는 경우)과 IoU/Dice 계산
4. 다른 tile에서도 동일 latent가 같은 조직 구조를 highlight하는지 일관성 확인

**비교 포인트**: variant별로 spatial attribution의 **sharpness**와 **consistency**를 비교. TopK의 정확한 sparsity 제어가 더 깔끔한 heatmap을 만드는지, JumpReLU의 adaptive threshold가 더 정확한 localization을 보이는지.

**시각화**: CytoSAE Fig.3 스타일

```
┌──────────┬──────────────────┬──────────────────┐
│ 원본 Tile │ Vanilla heatmap  │ TopK heatmap     │
│ (224×224) │ (14×14 → 224)    │ (14×14 → 224)    │
├──────────┼──────────────────┼──────────────────┤
│ Latent   │ Gated heatmap    │ JumpReLU heatmap │
│ #1234    │ (14×14 → 224)    │ (14×14 → 224)    │
└──────────┴──────────────────┴──────────────────┘
``` 

---

### 2.7 Ablation Study — CytoSAE에서 배울 점

CytoSAE는 체계적인 ablation을 수행했으며, PathoSAEv3도 이를 참고해야 한다.

#### CytoSAE의 Ablation 항목과 발견

| 항목 | 실험 범위 | 핵심 발견 |
|------|----------|----------|
| Expansion factor | 16, 32, 64*, 128 | 높을수록 concept 수↑, 단 highly activated cluster는 안정적 |
| L1 coefficient | 8e-4, 8e-5*, 8e-6 | 10x 증가 → concept 128개로 급감, 10x 감소 → L0 폭증 |
| Learning rate | 4e-3, 4e-4*, 4e-5 | 너무 높으면 불안정, 4e-5는 유사 결과 |
| Residual layer | 2, 5, 11*, 12 | Layer 7-11이 concept 수 균형에 최적, layer 12는 과다 |
| Ghost gradient | True*, False | Concept 수 변화 미미, dead feature 비율 0.92→0.57 감소 |
| b_dec init | mean, geometric median* | Geometric median 선택 |

#### PathoSAEv3 ablation 제안

PathoSAEv3는 variant 비교가 핵심이므로, **variant 간 공정 비교를 유지하면서** 추가 ablation 가능:

1. **Expansion factor sweep**: 16x / 32x / 64x — 1개 대표 variant(예: Vanilla)에서만 수행
2. **TopK의 k sweep**: 32, 64, 128, 256 — sparsity와 MS Score의 관계
3. **Layer selection**: Layer 6, 9, 12 — 어떤 layer의 representation이 가장 해석 가능한가
4. **L1/L0 coefficient sweep**: 각 variant에서 1e-3, 1e-4, 1e-5

---

## 3. 추가 제안 지표

CytoSAE에서 직접 가져오는 지표 외에, SAE/XAI/Medical 분야에서 PathoSAEv3의 결과를 더 강력하게 증빙할 수 있는 지표들이다.

---

### 3.1 MonoSemanticity Score (MS Score) ⭐ 최우선

#### 출처
Pach et al. (NeurIPS 2025) — VLM 기반 SAE 해석성 평가

#### 왜 필요한가

Label entropy는 사전 정의된 annotation에 의존한다. MS Score는 **label 없이** 각 neuron의 의미적 일관성을 측정하는 concept-agnostic 지표이다. "이 neuron을 강하게 활성화시키는 이미지들끼리 서로 얼마나 비슷한가?"를 측정하며, human judgment와 높은 alignment(차이 0.8-0.9에서 100% alignment)를 보인다.

#### 테스트 방법

1. 이미지 집합 {x₁, ..., x_N}에서 **독립 encoder E**(CONCH 또는 UNI)로 embedding 추출
   - ⚠️ EXAONEPath 자체를 E로 쓰면 circular evaluation이므로 반드시 독립 모델 사용
2. Pairwise cosine similarity matrix S 계산: `s_nm = cos(E(x_n), E(x_m))`
3. 각 neuron k의 activation vector a^k를 min-max normalize → ã^k
4. Relevance matrix: `r^k_nm = ã^k_n × ã^k_m`
5. MS Score: `MS^k = Σ_{n<m} r^k_nm × s_nm / Σ_{n<m} r^k_nm`

#### PathoSAEv3 도입

- 4개 variant 전체 neuron의 MS 분포를 violin plot 또는 CDF curve로 비교
- "No SAE" baseline: EXAONEPath 원본 768-dim의 각 neuron에 대해서도 MS Score 계산 → SAE의 monosemanticity 향상 효과 입증
- **Pareto 확장**: MSE vs L0 기존 Pareto에 MS Score를 color dimension으로 추가 → 3D trade-off 시각화

**시각화**:

```
┌────────────────────────────────────┐
│  MS Score Distribution (CDF)       │
│                                     │
│  Y축: Cumulative fraction           │
│  X축: MS Score (0 to 1)             │
│                                     │
│  ─── No SAE (EXAONEPath raw)       │
│  ─── Vanilla                        │
│  --- Gated                          │
│  ··· TopK                           │
│  ─·─ JumpReLU                       │
│                                     │
│  JumpReLU curve가 가장 오른쪽이면     │
│  → 가장 monosemantic                │
└────────────────────────────────────┘
```

**구현 스케치**:

```python
def compute_ms_score(independent_embeddings, sae_activations, neuron_idx, 
                     sample_size=2000):
    """
    independent_embeddings: [N, D_indep] from CONCH/UNI
    sae_activations: [N, D_sae] from SAE
    neuron_idx: int — target neuron
    sample_size: subsampling for O(N²) pairwise computation
    """
    # Subsample for computational efficiency
    idx = np.random.choice(len(independent_embeddings), 
                           min(sample_size, len(independent_embeddings)), 
                           replace=False)
    emb = independent_embeddings[idx]
    act = sae_activations[idx, neuron_idx]
    
    # Pairwise cosine similarity
    S = cosine_similarity(emb)  # [N', N']
    
    # Min-max normalize activations
    a_min, a_max = act.min(), act.max()
    a_norm = (act - a_min) / (a_max - a_min + 1e-8)
    
    # Relevance matrix (outer product)
    R = np.outer(a_norm, a_norm)  # [N', N']
    
    # Upper triangle mask (exclude diagonal)
    mask = np.triu(np.ones_like(S, dtype=bool), k=1)
    
    numerator = (R[mask] * S[mask]).sum()
    denominator = R[mask].sum()
    
    return numerator / (denominator + 1e-8)
```

---

### 3.2 HIF Correlation Analysis

#### 출처
PathAI SAE (NeurIPS 2025 Workshop) — 병리 조직 SAE

#### 왜 필요한가

SAE latent이 실제 생물학적 개념(cell count, nuclear area 등)과 **수치적으로** 얼마나 강하게 연결되는지 정량화한다. 시각적 검토를 넘어, Pearson correlation이라는 객관적 증거를 제시할 수 있다.

#### 테스트 방법

1. 별도 세포/조직 분석 도구(HoVer-Net, StarDist)로 ground-truth HIF 추출
2. 추출 가능한 HIF 예시:
   - Cell count (tumor cells, lymphocytes, fibroblasts)
   - Nuclear area, roundness, solidity
   - Stroma ratio
   - Necrosis area fraction
   - Glandular density
3. 각 SAE latent activation과 각 HIF 간 Pearson ρ 계산
4. 각 HIF에 대해 가장 높은 ρ를 보이는 latent → "이 latent은 lymphocyte count를 나타낸다"

#### PathoSAEv3 도입

- 4개 variant 각각에서, 각 HIF에 대한 **best-correlating latent의 ρ** 비교
- Table 형태: rows = HIF (cell count, nuclear area, ...), columns = variant, values = max |ρ|
- **기대**: 특정 variant이 더 높은 max correlation을 보인다면, 해당 variant이 더 "생물학적으로 의미있는" feature를 학습한 것

```python
from scipy.stats import pearsonr

def compute_hif_correlations(sae_acts, hif_values):
    """
    sae_acts: [N_tiles, D_sae]
    hif_values: [N_tiles, N_hifs]
    Returns: [D_sae, N_hifs] correlation matrix
    """
    n_latents = sae_acts.shape[1]
    n_hifs = hif_values.shape[1]
    corr = np.zeros((n_latents, n_hifs))
    for i in range(n_latents):
        for j in range(n_hifs):
            if sae_acts[:, i].std() > 1e-8:  # skip dead neurons
                r, _ = pearsonr(sae_acts[:, i], hif_values[:, j])
                corr[i, j] = r
    return corr
```

---

### 3.3 Monosemanticity Entropy (SAE vs FM 비교)

#### 출처
PathAI SAE 논문 — HIF 기반 entropy

#### 왜 필요한가

SAE의 핵심 가치는 **polysemantic → monosemantic 변환**이다. 이를 정량화하지 않으면 "왜 SAE를 써야 하는가?"에 대한 답이 없다.

#### 테스트 방법

1. 각 neuron과 N개 HIF 간 correlation ρ₁, ..., ρ_N 계산
2. 확률 분포 변환: `p_i = |ρ_i| / Σ|ρ_j|`
3. Shannon entropy: `H = -Σ p_i × log(p_i)`
4. 낮은 entropy → monosemantic (하나의 HIF에만 높은 correlation)
5. 높은 entropy → polysemantic (여러 HIF에 고르게 correlation)

#### PathoSAEv3 도입

- EXAONEPath 원본 768-dim의 각 neuron entropy vs 4개 SAE variant의 entropy 비교
- Mann-Whitney U test로 통계적 유의성 확인
- **5-way 비교**: No SAE / Vanilla / Gated / TopK / JumpReLU

---

### 3.4 Concept Purity Score

#### 출처
XAI 분야 — Concept Bottleneck Models 관련 문헌

#### 왜 필요한가

Label entropy는 ground-truth label에 의존하지만, concept purity는 **embedding 공간에서의 일관성**을 측정한다. Top-activating patch들이 embedding 상에서 하나의 클러스터를 형성하는지 확인한다.

#### 테스트 방법

1. 각 latent의 top-k activating patch를 수집
2. 이 패치들의 embedding을 별도 encoder(CONCH/UNI)로 추출
3. K-means clustering (n_clusters=5)
4. Purity = 가장 큰 클러스터의 비율
5. Purity ≈ 1.0이면 완벽한 monosemanticity

#### PathoSAEv3 도입

- Variant별 전체 latent의 평균 purity 비교
- MS Score와의 correlation 확인 (cross-validation of interpretability metrics)

```python
from sklearn.cluster import KMeans

def concept_purity(embeddings_of_top_patches, n_clusters=5):
    if len(embeddings_of_top_patches) < n_clusters:
        return 1.0
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_of_top_patches)
    counts = np.bincount(labels)
    return counts.max() / counts.sum()
```

---

### 3.5 Sparse Linear Probe (k-sparse)

#### 출처
PathAI SAE — 예측력 평가

#### 왜 필요한가

단순 correlation을 넘어 **예측력(predictive power)**을 측정한다. 소수의 neuron만으로 생물학적 특성을 예측할 수 있다면, 그 neuron은 진정한 concept이다.

#### 테스트 방법

1. 각 HIF에 대해 가장 높은 correlation을 보이는 k개 neuron 선택
2. 이 k개 neuron으로 linear regression 학습
3. Test set에서 R² 계산
4. k = 1, 3, 5, 10으로 변화시키며 관찰

#### PathoSAEv3 도입

- EXAONEPath 원본 768-dim vs 4 SAE variant의 24,576-dim
- k=3 기준 R² 비교 (Table 형태)
- **핵심 질문**: 어떤 activation mechanism이 더 predictive한 feature를 만드는가?

---

### 3.6 Feature Splitting / Absorption Detection

#### 출처
Anthropic "Scaling Monosemanticity" (2024), Pach et al.

#### 왜 필요한가

SAE의 알려진 실패 모드 중 하나. 하나의 개념이 여러 latent으로 분할되면(splitting), concept 수가 부풀려지고 해석이 어려워진다.

#### 테스트 방법

1. 모든 latent pair의 top-k activating patch 집합 간 Jaccard similarity 계산
2. J > threshold인 pair의 수 = splitting 정도
3. Pach et al.: J > 0인 pair가 전체의 0.03%, J > 0.5인 pair가 20개뿐 → 낮은 splitting

#### PathoSAEv3 도입

- Variant별 splitting pair 수 비교
- **기대**: TopK가 정확한 k개만 활성화하므로 splitting이 적을 것
- JumpReLU의 adaptive threshold도 splitting 방지에 유리할 수 있음

```python
def detect_feature_splitting(activations, topk=100, jaccard_threshold=0.3):
    n_latents = activations.shape[1]
    # Precompute top-k indices for each latent
    top_sets = {}
    for j in range(n_latents):
        vals = activations[:, j]
        if vals.max() > 0:  # skip dead
            top_sets[j] = set(torch.topk(vals, min(topk, len(vals))).indices.tolist())
    
    splitting_pairs = []
    active_latents = list(top_sets.keys())
    for idx_a in range(len(active_latents)):
        for idx_b in range(idx_a + 1, len(active_latents)):
            a, b = active_latents[idx_a], active_latents[idx_b]
            inter = len(top_sets[a] & top_sets[b])
            union = len(top_sets[a] | top_sets[b])
            if union > 0:
                jacc = inter / union
                if jacc > jaccard_threshold:
                    splitting_pairs.append((a, b, jacc))
    
    return splitting_pairs
```

---

### 3.7 Scanner/Stain Robustness

#### 출처
PathAI SAE — confound 분리

#### 왜 필요한가

병리 이미지는 스캐너, 염색 프로토콜에 따라 크게 달라진다. 좋은 SAE feature는 **생물학적 정보만 담고**, 이런 technical artifact를 무시해야 한다. 이 지표가 있으면 "SAE가 stain variation에 robust하다"는 주장에 증거를 제시할 수 있다.

#### 테스트 방법

1. 생물학적 concept과 높은 correlation을 보이는 SAE latent 선택
2. 이 latent으로 scanner type / institution을 예측하는 logistic regression 학습
3. AUROC ≈ 0.5이면 confound-free

#### PathoSAEv3 도입

- TCGA는 다양한 기관에서 스캔되어 institution metadata 활용 가능
- 또는 Macenko normalization 전/후 activation 변화량 측정
- variant별 confound 분리 능력 비교

---

### 3.8 Cross-Dataset Generalization

#### 출처
CytoSAE (4개 데이터셋), PathAI (CPTAC out-of-domain)

#### 왜 필요한가

SAE가 학습 데이터에 overfitting되지 않고, **다른 기관/스캐너의 데이터에서도 동일한 concept을 발견하는지** 확인해야 한다. 이것은 임상 적용의 전제조건이다. CytoSAE는 peripheral blood에서 학습하여 bone marrow에서도 동일 concept을 검출한 것이 핵심 기여 중 하나였다.

#### PathoSAEv3 도입

- 학습: TCGA 데이터
- 평가: CPTAC, CAMELYON16, 또는 다른 공개 병리 데이터셋
- 동일 SAE를 OOD 데이터에 적용하여:
  - MSE가 크게 증가하지 않는지
  - Label entropy가 유사한 분포를 보이는지
  - MS Score가 유지되는지
- Variant별 generalization gap 비교

---

### 3.9 Feature Universality (재현성)

#### 출처
PathAI SAE — 독립 학습 모델 간 일관성

#### 왜 필요한가

다른 random seed로 학습한 SAE에서도 동일한 feature가 발견된다면, 그 feature는 training artifact가 아닌 **실제 생물학적 개념**일 가능성이 높다.

#### 테스트 방법

1. 동일 variant를 **다른 random seed**로 2~3번 학습
2. 두 모델의 latent activation을 같은 데이터셋에서 추출
3. Hungarian matching으로 최적 1:1 매칭
4. 매칭된 pair의 Pearson ρ > 0.5이면 "universal"
5. Universal feature의 비율 보고

#### PathoSAEv3 도입

- 1개 대표 variant(Vanilla 또는 best performer)에서 seed 3개로 학습
- Universal feature 비율 보고
- 시간이 허락하면 4개 variant 모두에서 universality 비교

---

### 3.10 UMAP/t-SNE Feature Landscape

#### 출처
일반 representation learning, PathAI Fig.2

#### 왜 필요한가

SAE latent space의 **전체적인 구조**를 시각화하여, concept들이 잘 분리되어 있는지 직관적으로 확인한다. 리뷰어와 독자에게 "이 SAE가 무엇을 학습했는지"를 한눈에 보여줄 수 있다.

#### PathoSAEv3 도입

1. SAE decoder weight (W_dec의 각 행)를 추출 — 각 latent의 "방향"
2. UMAP으로 2D 투영
3. 색상: label entropy 또는 tissue type
4. HDBSCAN으로 자동 클러스터링 → 클러스터별 대표 image grid
5. 4 variant를 나란히 비교

---

### 3.11 Inter-rater Agreement (Cohen's Kappa)

#### 출처
의학 AI 평가 표준

#### 왜 필요한가

전문가 1명의 주관적 평가만으로는 부족하다. 최소 2명 이상의 평가자 간 일치도를 보여야 "이 concept은 객관적으로 의미 있다"고 주장할 수 있다.

#### 테스트 방법

1. 50~100 latent의 top-activating patch grid를 2+ 평가자에게 제시
2. 각자 독립적으로 평가
3. Cohen's Kappa 계산
4. κ > 0.6이면 substantial agreement, κ > 0.8이면 almost perfect

---

## 4. 전체 지표 맵

### 질문 → 지표 → 우선순위

| # | 질문 | 답변 도구 (지표) | 우선순위 | 노력도 |
|---|------|----------------|---------|--------|
| Q1 | 어떤 SAE가 가장 잘 재구성하는가? | MSE, FVU, R², Cosine Sim | 🟢 완료 | - |
| Q2 | 어떤 SAE가 가장 sparse한가? | L0, Sparsity %, Dead Neurons | 🟢 완료 | - |
| Q3 | 어떤 SAE가 가장 해석 가능한가? | **MS Score**, **Label Entropy**, **Concept Purity** | 🔴 P0 | 중 |
| Q4 | 의미있는 concept이 몇 개인가? | **Concept Count** (threshold sweep) | 🔴 P0 | 낮 |
| Q5 | SAE가 원본보다 정말 monosemantic한가? | **Monosemanticity Entropy** (SAE vs FM) | 🟠 P1 | 중 |
| Q6 | Feature가 생물학적으로 의미있는가? | **HIF Correlation**, **k-sparse Probe** | 🟠 P1 | 중~높 |
| Q7 | 해석 가능한 feature로 분류가 되는가? | **Barcode Linear Classification** | 🟠 P1 | 중 |
| Q8 | 공간적 해석이 올바른가? | **Patch Attribution** (heatmap + IoU) | 🟠 P1 | 중 |
| Q9 | Feature splitting이 있는가? | **Jaccard-based splitting detection** | 🟡 P2 | 낮 |
| Q10 | Feature가 재현 가능한가? | **Universality** (multi-seed) | 🟡 P2 | 중 |
| Q11 | Latent space 구조가 어떤가? | **UMAP visualization** | 🟡 P2 | 낮 |
| Q12 | 다른 데이터에서도 작동하는가? | **Cross-Dataset Generalization** | 🟡 P2 | 높 |
| Q13 | Stain/Scanner에 robust한가? | **Confound AUROC** | ⚪ P3 | 중 |
| Q14 | 병리학자가 동의하는가? | **Expert Annotation + Inter-rater κ** | ⚪ P3 | 높 |

### 우선순위 기준

- **P0 (필수)**: 논문의 핵심 주장("SAE variant 비교")을 지탱하는 지표. 이것 없이는 recon metrics만으로 reject 가능성 높음
- **P1 (강력 권장)**: 논문의 설득력을 크게 높이는 지표. top-tier venue에는 필요
- **P2 (권장)**: 논문의 completeness를 높이는 추가 분석
- **P3 (보너스)**: 있으면 좋지만 시간 제약 시 생략 가능

---

## 5. 논문 Figure/Table 구성안

### Tables

| Table | 내용 | 대응 지표 |
|-------|------|----------|
| **Table 1** | 4 variant 기본 지표 (MSE, FVU, R², L0, Cosine Sim, Dead Neurons, Active Neurons) | 기존 |
| **Table 2** | 4 variant 해석성 지표 (MS Score mean/median, Label Entropy mean, Concept Count @threshold, Concept Purity) | Q3, Q4 |
| **Table 3** | Linear Probe Classification — variant × classification task (BRCA, LUAD/LUSC) | Q7 |
| **Table 4** | HIF Correlation — rows=HIF, columns=variant, values=max |ρ| | Q6 |
| **Table 5** | k-sparse Probe R² — rows=HIF, columns=variant (k=3) | Q6 |

### Figures

| Figure | 내용 | 대응 지표 |
|--------|------|----------|
| **Fig 1** | Architecture overview: EXAONEPath → 4 SAE variants → multi-level evaluation | 전체 |
| **Fig 2** | MSE vs L0 Pareto front + MS Score color overlay | Q1+Q2+Q3 |
| **Fig 3** | Label Entropy scatter plots — 2×2 grid (4 variants), CytoSAE Fig.2A 스타일 | Q3 |
| **Fig 4** | MS Score distribution curves — 4 variants + No SAE baseline (CDF) | Q3, Q5 |
| **Fig 5** | Concept Count curve — threshold sweep, 4 variants 오버레이 | Q4 |
| **Fig 6** | Top-activating patch grids — 대표 monosemantic neurons (variant별 3~4개) | 정성적 |
| **Fig 7** | 14×14 spatial heatmap — 같은 latent concept의 variant별 localization | Q8 |
| **Fig 8** | Barcode classification — F1 vs latent count curve | Q7 |
| **Fig 9** | UMAP of SAE latents with concept clusters (4 variants 비교) | Q11 |
| **Fig 10** | HIF correlation heatmap (variant × HIF) | Q6 |

---

## 6. 구현 로드맵

### 코드 아키텍처 확장

```
evaluation/
├── metrics.py                 # 기존: MSE, FVU, L0, Cosine Sim, Dead Neurons
├── monosemanticity.py         # NEW: MS Score, Label Entropy, Concept Purity
├── concept_analysis.py        # NEW: Concept Count, Feature Splitting, UMAP
├── probing.py                 # NEW: Linear Probe, k-sparse Probe, Barcode
├── hif_analysis.py            # NEW: HIF Correlation, Monosemanticity Entropy
├── spatial_attribution.py     # NEW: 14×14 heatmap, IoU with annotation
├── robustness.py              # NEW: Scanner/Stain AUROC, Cross-Dataset
└── visualization.py           # 확장: scatter, CDF, heatmap, UMAP, grid
```

### Phase 1: 자동화 가능 정량 지표 (1~2주)

**목표**: 코드 실행만으로 얻을 수 있는 해석성 지표 완성

**Step 1: `monosemanticity.py`**
- MS Score 구현 (CONCH 또는 UNI를 독립 encoder로)
- Label Entropy 구현 (tissue type annotation 필요)
- Concept Purity 구현

핵심 의존성:
- 독립 encoder: `pip install conch` 또는 UNI weight 다운로드
- Tissue annotation: 기존 TCGA annotation 활용 또는 pseudo-label 생성

**Step 2: `concept_analysis.py`**
- Concept Count threshold sweep
- Feature Splitting (Jaccard)
- UMAP + HDBSCAN visualization

핵심 의존성:
- `pip install umap-learn hdbscan`

**Step 3: `probing.py`**
- Barcode generation pipeline (tile → WSI → disease)
- Linear Probe (sklearn LogisticRegression)
- k-sparse Probe (top-k correlation → LinearRegression)

핵심 의존성:
- TCGA WSI-level labels

### Phase 2: 생물학적 검증 (2~4주)

**Step 4: `hif_analysis.py`**
- HoVer-Net 또는 StarDist 셋업
- Cell detection → HIF 추출 파이프라인
- HIF-latent correlation matrix
- Monosemanticity Entropy (SAE vs FM)

핵심 의존성:
- `pip install hovernet` 또는 StarDist
- 상당한 GPU 시간 (cell detection은 compute-intensive)

**Step 5: `spatial_attribution.py`**
- 14×14 activation heatmap 생성
- 원본 tile 오버레이
- IoU/Dice with tissue annotation (있는 경우)

### Phase 3: 전문가 검증 + 추가 분석 (논문 제출 전)

**Step 6**: Expert annotation 프로토콜 실행
**Step 7**: Cross-dataset evaluation (CPTAC)
**Step 8**: Scanner robustness analysis

### 실행 스크립트 예시

```bash
# Phase 1: 해석성 지표 일괄 계산
PYTHONPATH=./ python tasks/eval_interpretability.py \
  --checkpoints models/checkpoints/vanilla_e32_ep10/best.pt \
               models/checkpoints/gated_e32_ep10/best.pt \
               models/checkpoints/topk_e32_ep10/best.pt \
               models/checkpoints/jumprelu_e32_ep10/best.pt \
  --independent_encoder conch \
  --tissue_labels data/tcga_tissue_labels.csv \
  --output_dir results/interpretability/

# Phase 2: HIF 분석
PYTHONPATH=./ python tasks/eval_hif.py \
  --checkpoint models/checkpoints/vanilla_e32_ep10/best.pt \
  --cell_detector hovernet \
  --output_dir results/hif/

# 전체 비교 리포트 생성
PYTHONPATH=./ python tasks/generate_paper_figures.py \
  --results_dir results/ \
  --output_dir paper/figures/
```

### 리소스 예상

| Phase | GPU 시간 | 추가 데이터 | 외부 모델 |
|-------|---------|----------|----------|
| Phase 1 | ~4h (MS Score 계산) | Tissue labels | CONCH or UNI |
| Phase 2 | ~12h (cell detection) | HIF ground truth | HoVer-Net |
| Phase 3 | 0 (human effort) | CPTAC dataset | — |

---

## 최종 요약: 논문의 스토리라인

PathoSAEv3 논문은 다음 스토리를 따라야 한다:

1. **문제 제기**: 병리 FM은 블랙박스 → SAE로 해석 가능 → 하지만 어떤 SAE가 최적인지 모름
2. **방법론**: 4종 SAE를 동일 조건에서 공정 비교 (Table 1: 재구성 지표)
3. **해석성 입증**: MS Score, Label Entropy, Concept Purity로 variant별 해석성 정량화 (Table 2, Fig 3-4)
4. **생물학적 의미**: HIF correlation으로 feature가 실제 조직 형태학에 대응함을 증명 (Table 4, Fig 10)
5. **임상 유용성**: Barcode classification으로 해석 가능한 feature만으로도 분류 가능함을 보임 (Table 3, Fig 8)
6. **공간적 장점**: 14×14 heatmap으로 CLS-only 대비 spatial token의 우위 시각화 (Fig 7)
7. **결론**: "임상의가 병리 AI의 판단 근거를 이해하기 위해, [best variant]을 써야 한다"

**핵심**: 재구성 품질(Table 1)은 baseline이고, **해석성 + 생물학적 의미 지표(Table 2-4)가 논문의 설득력을 결정**한다. ACCV 2026 리뷰어를 설득하려면 P0 + P1 지표가 반드시 있어야 한다.
