# PathoSAEv3 이론 정렬 + 모듈 분담 스펙 (v1, KR)

## 1) 목표와 범위

- 목표: 다음 구현 단계에 바로 투입 가능한 단일 의사결정 완료(hand-off) 문서를 고정한다.
- 범위: PathoSAEv3 코드베이스와 베이스 논문 간 이론 정렬 + 모듈 기준 역할 분담.
- 범위 외: 이 단계에서 코드/CLI/체크포인트 포맷 변경은 하지 않는다.

가정:
- 현재 워크트리 상태를 기준으로 사용한다.
- 논문 해석은 아래 본문의 수치/정의를 우선한다.
  - `/home/kimhj/projects/cytoSAE/cytosae.pdf`
  - `/home/kimhj/projects/cytoSAE/plutosae.pdf`

---

## 2) 이론 정렬 (논문 -> PathoSAEv3)

### 2.1 공통 SAE 목적함수 축

두 논문에서 공통으로 쓰는 목적함수 축:

\[
\mathcal{L} = \mathcal{L}_{recon}(x, \hat{x}) + \lambda \cdot \mathcal{L}_{sparsity}(z)
\]

- 재구성 항: MSE.
- 희소성 항: variant별 L1/L0/TopK 계열 제약.

코드 앵커:
- Base 인터페이스/forward 계약:
  - [`src/models/base_sae.py:38`](/home/kimhj/projects/pathoSAEv3/src/models/base_sae.py:38)
  - [`src/models/base_sae.py:55`](/home/kimhj/projects/pathoSAEv3/src/models/base_sae.py:55)
- Trainer에서 `loss/mse_loss/sparsity_loss/l0` 추적:
  - [`src/training/trainer.py:112`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:112)

### 2.2 Variant별 이론-구현 매핑

| Variant | 이론 의도 | PathoSAEv3 구현 앵커 |
|---|---|---|
| Vanilla | `ReLU + L1` 기본 희소 사전 | [`src/models/vanilla_sae.py:15`](/home/kimhj/projects/pathoSAEv3/src/models/vanilla_sae.py:15), [`src/models/vanilla_sae.py:18`](/home/kimhj/projects/pathoSAEv3/src/models/vanilla_sae.py:18) |
| Gated | gate/magnitude 분리, gate 경로 희소화 | [`src/models/gated_sae.py:31`](/home/kimhj/projects/pathoSAEv3/src/models/gated_sae.py:31), [`src/models/gated_sae.py:37`](/home/kimhj/projects/pathoSAEv3/src/models/gated_sae.py:37) |
| TopK | L1 대신 hard-k 활성 | [`src/models/topk_sae.py:17`](/home/kimhj/projects/pathoSAEv3/src/models/topk_sae.py:17), [`src/models/topk_sae.py:27`](/home/kimhj/projects/pathoSAEv3/src/models/topk_sae.py:27) |
| JumpReLU | 학습 임계값 + L0 proxy | [`src/models/jumprelu_sae.py:107`](/home/kimhj/projects/pathoSAEv3/src/models/jumprelu_sae.py:107), [`src/models/jumprelu_sae.py:110`](/home/kimhj/projects/pathoSAEv3/src/models/jumprelu_sae.py:110) |
| MSAE | multi-scale TopK + 가중 multi-level MSE | [`src/models/msae.py:63`](/home/kimhj/projects/pathoSAEv3/src/models/msae.py:63), [`src/models/msae.py:95`](/home/kimhj/projects/pathoSAEv3/src/models/msae.py:95) |
| DriftPrior | latent 분포를 sparse prior로 정합 | [`src/models/driftprior_sae.py:29`](/home/kimhj/projects/pathoSAEv3/src/models/driftprior_sae.py:29), [`src/models/driftprior_sae.py:117`](/home/kimhj/projects/pathoSAEv3/src/models/driftprior_sae.py:117) |

Registry/CLI 커버리지:
- Registry는 6종 지원: [`src/models/__init__.py:12`](/home/kimhj/projects/pathoSAEv3/src/models/__init__.py:12)
- Train CLI도 6종 허용: [`tasks/train.py:57`](/home/kimhj/projects/pathoSAEv3/tasks/train.py:57)

### 2.3 해석성 축

유지해야 할 논문 개념:
- entropy/monosemanticity (`p_i = |rho_i| / sum |rho_j|`, entropy 낮을수록 단의미성 높음).
- sparse probe 유틸리티 (`k`-sparse linear probe, R2/F1 계열 검증).
- barcode 계층(패치 -> 이미지 -> 환자/군집).

PathoSAEv3 앵커:
- top patch + metadata 기반 entropy proxy:
  - [`src/evaluation/feature_figures.py:44`](/home/kimhj/projects/pathoSAEv3/src/evaluation/feature_figures.py:44)
  - [`src/evaluation/feature_figures.py:60`](/home/kimhj/projects/pathoSAEv3/src/evaluation/feature_figures.py:60)
- 그룹/환자 집계:
  - [`src/evaluation/group_analysis.py:157`](/home/kimhj/projects/pathoSAEv3/src/evaluation/group_analysis.py:157)
- monosemanticity 성격의 MS score 파이프라인:
  - [`src/evaluation/ms_score.py:163`](/home/kimhj/projects/pathoSAEv3/src/evaluation/ms_score.py:163)
  - [`tasks/eval_ms_score.py:37`](/home/kimhj/projects/pathoSAEv3/tasks/eval_ms_score.py:37)

---

## 3) 모듈 기준 분담 스펙

이 섹션은 바로 역할 분담 가능한 수준으로 고정한다. 한 명이 한 모듈을 독립적으로 맡을 수 있어야 한다.

### Module A: Extract

책임:
- `PNG tile -> 정규화 텐서 -> spatial token activation -> .pt 저장`.

엔트리포인트:
- [`tasks/extract.py:142`](/home/kimhj/projects/pathoSAEv3/tasks/extract.py:142)
- [`src/extract/vit_encoder.py:423`](/home/kimhj/projects/pathoSAEv3/src/extract/vit_encoder.py:423)

입력 계약:
- PNG 타일 폴더.
- 인코더 선택 `{exaonepath, uni}`.

출력 계약:
- 폴더 단위 activation 파일:
  - key `vectors`: `[N, 196, D]`
  - key `paths`: 길이 `N`
  - 저장 로직: [`tasks/extract.py:193`](/home/kimhj/projects/pathoSAEv3/tasks/extract.py:193)

의존성:
- `macenko`, 백본 체크포인트, GPU 선택.

완료 정의(DoD):
- 산출 파일이 `ActivationStore`/feature/group analyzer에서 읽혀야 한다.

### Module B: Train

책임:
- config/model/store 구성, decoder bias 초기화, 통합 trainer 수행, 체크포인트 저장.

엔트리포인트:
- [`tasks/train.py:126`](/home/kimhj/projects/pathoSAEv3/tasks/train.py:126)
- [`src/training/trainer.py:23`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:23)

입력 계약:
- `data/activations/spatial_folder_*.pt` activation 파일.
- CLI로 variant/하이퍼파라미터 지정.

출력 계약:
- 체크포인트 폴더(`best.pt`, `final.pt`, `config.json`).
- 체크포인트 payload는 `state_dict`, `config`, optimizer state 포함:
  - [`src/training/trainer.py:79`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:79)

의존성:
- `create_sae` registry, `ActivationStore`, scheduler, optional wandb.

DoD:
- 학습 완료된 모델이 `BaseSAE.load`로 로드되어야 한다.

### Module C: Eval/Analysis

책임:
- 재구성 지표, feature 통계, group 행렬, optional MS score 생성.

엔트리포인트:
- Reconstruction eval: [`tasks/evaluate.py:84`](/home/kimhj/projects/pathoSAEv3/tasks/evaluate.py:84)
- Compare: [`src/evaluation/compare.py:131`](/home/kimhj/projects/pathoSAEv3/src/evaluation/compare.py:131)
- Group analysis: [`src/evaluation/group_analysis.py:156`](/home/kimhj/projects/pathoSAEv3/src/evaluation/group_analysis.py:156)
- MS score: [`tasks/eval_ms_score.py:87`](/home/kimhj/projects/pathoSAEv3/tasks/eval_ms_score.py:87)

입력 계약:
- 학습 체크포인트.
- activation 디렉토리.
- optional metadata CSV(`path,class,patient`).

출력 계약:
- `results/{run_name}/metrics.json` 및 plot.
- `feature_analysis.json`, `feature_summary.csv`, group matrix/heatmap.
- `results/comparison/comparison_table.csv`, Pareto plot.

의존성:
- feature/group/ms-score 전 구간에서 activation path key 소비 일관성.

DoD:
- 모든 결과물이 입력 체크포인트와 추적 가능해야 한다.

### Module D: Report

책임:
- 실험 결과 아티팩트를 단일 최종 markdown 리포트로 집계.

엔트리포인트:
- [`tasks/generate_experiment_report.py:151`](/home/kimhj/projects/pathoSAEv3/tasks/generate_experiment_report.py:151)

입력 계약:
- 기존 `results/`, `logs/`, optional comparison csv.

출력 계약:
- `results/summary/experiment_design_full_report.md`.

의존성:
- Eval 모듈의 출력 파일명 안정성.

DoD:
- 수동 편집 없이 리포트 재생성 가능해야 한다.

---

## 4) 다음 구현 전 인터페이스 잠금안

아래 3개를 다음 구현의 기본값으로 고정한다.

### Lock-1: Activation key 호환(`paths` vs `patch_paths`)

결정:
- canonical write key는 `paths`.
- canonical read는 전 구간에서 `paths`/`patch_paths` 모두 허용.

근거:
- extract는 현재 `paths`로 저장:
  - [`tasks/extract.py:196`](/home/kimhj/projects/pathoSAEv3/tasks/extract.py:196)
- 일부 소비자는 `patch_paths`만 직접 참조:
  - [`tasks/feature_viz.py:202`](/home/kimhj/projects/pathoSAEv3/tasks/feature_viz.py:202)
  - [`src/evaluation/ms_score.py:123`](/home/kimhj/projects/pathoSAEv3/src/evaluation/ms_score.py:123)

구현 수용 기준:
- 단일 key 가정 소비 코드가 없어야 한다.

### Lock-2: `evaluate --split`의 reconstruction metric 반영 범위

결정:
- `--split`이 다음 둘 다 제어해야 한다.
  - feature 분석 split
  - reconstruction metric split
- 기본값은 `val` 유지.

근거:
- CLI에서 split은 파싱됨:
  - [`tasks/evaluate.py:65`](/home/kimhj/projects/pathoSAEv3/tasks/evaluate.py:65)
- reconstruction은 현재 val loader 고정:
  - [`src/evaluation/metrics.py:209`](/home/kimhj/projects/pathoSAEv3/src/evaluation/metrics.py:209)

구현 수용 기준:
- 동일 `--split` 값에서 eval 산출물의 데이터 분할이 일치해야 한다.

상태 (2026-03-29):
- 코드 반영 완료:
  - [`tasks/evaluate.py`](/home/kimhj/projects/pathoSAEv3/tasks/evaluate.py)
  - [`src/evaluation/metrics.py`](/home/kimhj/projects/pathoSAEv3/src/evaluation/metrics.py)
  - [`src/extract/activation_store.py`](/home/kimhj/projects/pathoSAEv3/src/extract/activation_store.py)

### Lock-3: Variant 집합 정책(4/5/6 분리)

결정:
- `Core set`: `vanilla,gated,topk,jumprelu,msae`
- `Experimental set`: `driftprior` (opt-in)

근거:
- registry/CLI는 6종:
  - [`src/models/__init__.py:12`](/home/kimhj/projects/pathoSAEv3/src/models/__init__.py:12)
  - [`tasks/train.py:57`](/home/kimhj/projects/pathoSAEv3/tasks/train.py:57)
- 운영 스크립트는 불일치:
  - run.sh는 5종: [`run.sh:32`](/home/kimhj/projects/pathoSAEv3/run.sh:32)
  - train/evaluate_all은 4종: [`scripts/train_all.sh:4`](/home/kimhj/projects/pathoSAEv3/scripts/train_all.sh:4), [`scripts/evaluate_all.sh:4`](/home/kimhj/projects/pathoSAEv3/scripts/evaluate_all.sh:4)

구현 수용 기준:
- 모델 집합 정의가 단일 출처(Source of Truth)로 통합되어 scripts/CLI/docs/report에서 재사용되어야 한다.

---

## 5) 추적성 및 시나리오 검증

### 5.1 이론-코드 추적성

최소 요구:
- 아래 핵심 주장마다 코드 앵커 최소 1개.

| 주장 | 앵커 |
|---|---|
| SAE는 encode-activate-decode 계약을 사용 | [`src/models/base_sae.py:38`](/home/kimhj/projects/pathoSAEv3/src/models/base_sae.py:38) |
| 학습에서 MSE+sparsity가 추적됨 | [`src/training/trainer.py:119`](/home/kimhj/projects/pathoSAEv3/src/training/trainer.py:119) |
| L1/L0/TopK variant 동작이 분리되어 존재 | Section 2.2의 model 파일 |
| entropy/feature 해석 산출물이 존재 | [`src/evaluation/feature_figures.py:60`](/home/kimhj/projects/pathoSAEv3/src/evaluation/feature_figures.py:60) |
| group/patient 집계가 존재 | [`src/evaluation/group_analysis.py:157`](/home/kimhj/projects/pathoSAEv3/src/evaluation/group_analysis.py:157) |

### 5.2 파이프라인 시나리오

Scenario-1: Extract 스키마 일관성
- extract 결과 key를 모든 downstream reader가 허용하는지 검증.
- 현재는 mixed key 기대가 있어 리스크 오픈 상태.

Scenario-2: Split 일관성
- metrics와 feature/group 분석이 동일 split을 사용해야 함.
- 현재는 `evaluate_checkpoint`가 val-only라 리스크 오픈.

Scenario-3: 병렬 분담 가능성
- Extract/Train/Eval/Report가 산출물 기준으로 분리 가능.
- 의존성이 선형이어서 독립 분담 가능.

---

## 6) 실행 가능한 작업 패키지 (모듈 분할)

WP-A Extract
- 담당 파일: `tasks/extract.py`, `src/extract/*`
- 산출물: 안정적인 activation schema + 재현 실행 노트
- 선행: 백본 체크포인트, 이미지 타일

WP-B Train
- 담당 파일: `tasks/train.py`, `src/models/*`, `src/training/*`, `src/config.py`
- 산출물: core 모델셋 재현 체크포인트
- 선행: WP-A 산출물

WP-C Eval
- 담당 파일: `tasks/evaluate.py`, `tasks/compare.py`, `tasks/group_analysis.py`, `tasks/eval_ms_score.py`, `src/evaluation/*`
- 산출물: metrics/features/group/ms 아티팩트 + comparison table
- 선행: WP-B 산출물 + optional metadata

WP-D Report
- 담당 파일: `tasks/generate_experiment_report.py`
- 산출물: 단일 최종 실험 리포트
- 선행: WP-C 산출물

---

## 7) 수용 기준

아래를 모두 만족해야 문서 수용:

1. 신규 기여자가 문서만 보고 모듈 1개를 바로 착수할 수 있어야 한다.
2. Section 4의 인터페이스 잠금 3개가 기본값으로 취급되어야 한다.
3. Section 2의 주요 이론 주장들이 코드 앵커로 추적 가능해야 한다.
4. 역할 경계와 I/O 계약 이해를 위해 추가 코드 수정이 필요 없어야 한다.
