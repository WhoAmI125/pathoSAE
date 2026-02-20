# decisions.md
> Opus 작성 · Codex **반드시 읽고 준수** · 2026-02-20

---

## 목적

Codex는 자체 판단으로 아키텍처를 변형하는 경향이 있다.
이 문서는 각 설계 결정의 **"왜"**를 명시하여 임의 변경을 방지한다.

**Codex 규칙**: 동의하지 않더라도 먼저 설계대로 구현하고, `codex_notes.md`에 대안을 제안하라.

---

### ADR-001: 4가지 SAE variant를 하나의 학습 설정으로 통일

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | pathoSAEv2에서 각 variant(vanilla, gated, msae, jumprelu)가 서로 다른 expansion, batch_size, epochs, lr로 학습됨 → 논문에서 공정 비교 불가 |
| **결정** | 모든 variant를 동일한 하이퍼파라미터로 학습 (expansion=32, batch=4096, epochs=10, lr=3e-4) |
| **근거** | 논문 Table에서 variant 간 비교 시, 학습 조건이 동일해야 공정한 비교. variant별 차이는 activation 함수와 sparsity 메커니즘만 반영 |
| **거부한 대안** | variant별 최적 하이퍼파라미터 사용 — 비교 불공정, 리뷰어 지적 가능 |
| **결과** | 일부 variant가 최적 성능에 도달 못할 수 있으나, 공정 비교가 논문에서 더 중요 |
| **Codex 지시** | `SAEConfig` 기본값은 절대 variant별로 분기하지 마. CLI에서 override 가능하게만 하되, 기본값은 모두 동일 |

---

### ADR-002: BaseSAE 추상 클래스 + Factory 패턴

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | pathoSAEv2에서 각 SAE가 독립적으로 정의됨 (일부는 train 스크립트에 inline). 코드 중복, 인터페이스 불일치 |
| **결정** | `BaseSAE` ABC에서 `encode/activate/decode/compute_loss` 인터페이스 정의. variant는 `activate`와 `compute_loss`만 override |
| **근거** | cytoSAE의 `SparseAutoencoder` 패턴 참조. 동일 `forward()` 시그니처로 Trainer가 variant 무관하게 동작 |
| **거부한 대안** | 각 variant를 완전 독립 클래스로 — 코드 중복, Trainer 분기 필요 |
| **Codex 지시** | `forward(x, step) → (sae_out, feature_acts, loss_dict)` 시그니처 절대 변경 금지. loss_dict에는 반드시 "loss", "mse_loss", "sparsity_loss", "l0" 키 포함 |

---

### ADR-003: 통합 CLI (tasks/train.py --model 플래그)

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | pathoSAEv2에서 7개의 train 스크립트가 존재 (train_sae_vanilla.py, train_sae_spatial.py 등). 대부분 80% 코드 중복 |
| **결정** | 단일 `tasks/train.py`에 `--model {vanilla,gated,topk,jumprelu}` 플래그 |
| **근거** | 하나의 스크립트로 모든 variant 학습. shell 스크립트에서 `--model` 만 바꿔서 순차 실행 가능 |
| **거부한 대안** | Hydra config — 오버엔지니어링. argparse로 충분 |
| **Codex 지시** | argparse 사용. Config dataclass 필드에 대응하는 CLI 인자를 1:1 매핑. variant별 무관한 인자(예: topk_k for vanilla)는 무시 |

---

### ADR-004: 기존 activation 데이터 symlink 재사용

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | pathoSAEv2의 data/encode_spatial/ (96파일, ~749GB)을 다시 추출하면 시간 낭비 |
| **결정** | `data/activations/` → pathoSAEv2 data를 symlink. 데이터 파일은 이동/복사하지 않음 |
| **근거** | 동일한 EXAONEPath + 동일 이미지 = 동일 activation. 재추출 불필요 |
| **Codex 지시** | `ActivationStore`는 `data/activations/spatial_folder_*.pt` 경로를 glob으로 탐색. 절대경로 하드코딩 금지 |

---

### ADR-005: TopK에서 MSAE multi-level 제거 (논문 비교 기준)

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | pathoSAEv2 MSAE는 nesting_list=[64,128,256,512]로 multi-level 학습. 다른 variant와 비교 불공정 |
| **결정** | TopKSAE는 기본적으로 단일 k (topk_k=64)만 사용. nesting_list는 고급 실험용 옵션으로 남김 |
| **근거** | 논문 비교 기준: 각 variant의 핵심 메커니즘만 비교. multi-level은 TopK의 확장 실험으로 별도 기술 |
| **Codex 지시** | topk_nesting_list 기본값은 `[64]` (단일 k). len(nesting_list)==1이면 일반 TopK처럼 동작 |

---

### ADR-006: JumpReLU Custom Autograd 유지

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | JumpReLU의 Heaviside 함수는 gradient가 0이므로 STE + kernel-based pseudo-gradient 필요 |
| **결정** | pathoSAEv2의 `jumprelu_model.py`에서 `JumpReLU` autograd.Function을 그대로 가져옴 |
| **근거** | 이미 검증된 구현. ICLR 2025 논문 기반 |
| **Codex 지시** | `torch.autograd.Function` 사용. `jumprelu_model.py`의 `JumpReLU`, `StepFunction` 클래스 참조. forward/backward 로직 변경 금지 |

---

### ADR-007: conda 가상환경 분리 (pathosae3)

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2026-02-20 |
| **컨텍스트** | pathoSAEv2는 `exaonepath` 환경 공유. 의존성 충돌 방지, 재현성 확보 필요 |
| **결정** | `conda create -n pathosae3 python=3.10`. environment.yaml + requirements.txt로 관리 |
| **근거** | 프로젝트 독립성. 논문 재현 시 환경 재구축 용이 |
| **Codex 지시** | environment.yaml과 requirements.txt를 프로젝트 루트에 생성. PyTorch 2.2.0 + CUDA 11.8 기준 |

---

## Quick Decision Log

| ID | 결정 | 이유 | Codex 주의 |
|----|------|------|-----------|
| QD-001 | W_dec unit norm 제약 유지 | Anthropic SAE 논문 표준 | `set_decoder_norm_to_unit_norm()` 매 step 호출 |
| QD-002 | gradient projection 유지 | decoder 방향으로의 gradient 제거 | `remove_gradient_parallel_to_decoder_directions()` 매 step 호출 |
| QD-003 | b_dec 초기화 = mean | cytoSAE는 geometric_median, 여기선 mean (데이터 특성상 충분) | ActivationStore에서 전체 mean 계산 후 b_dec에 설정 |
| QD-004 | AMP 사용 안함 | 논문 비교 공정성 — variant별 AMP 호환성 차이 방지 | `torch.float32` 고정 |
| QD-005 | validation은 마지막 파일 1개 | 파일 단위 split이 깔끔 | val_ratio는 파일 수 기준 |
