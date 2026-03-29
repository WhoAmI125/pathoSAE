# PathoSAEv3 통합 심층분석 리포트 (2026-03-29)

## 1) 핵심 결론 (1페이지 요약)

### 최종 판정
- 전체 방향은 타당하다. 특히 `offline extraction`, `multi-variant SAE comparison`, `pathology token interpretability` 축은 유지 가능하다.
- 다만 논문화/재현성 관점에서 즉시 닫아야 하는 `Critical 4개`가 존재한다.
- 본 프로젝트의 안전한 novelty 문장은 다음으로 고정하는 것이 가장 합리적이다.
  - **"Pathology tissue foundation model에서 SAE variant를 체계적으로 비교한 연구"**
- 아래 문장은 현재 상태에서 금지하는 것이 안전하다.
  - "pathology에 SAE를 처음 적용했다"
  - "CytoSAE는 CLS-only"
  - "EXAONEPath가 기본적으로 spatial token을 반환한다"
  - "JumpReLU가 최고 해석성"
  - "pathologist validation 완료"

### 즉시 닫아야 할 Critical 4개
1. `paths` vs `patch_paths` key 불일치
- Extract는 `paths` 저장 (`tasks/extract.py:197-201`), 일부 분석 코드는 `patch_paths`만 가정 (`src/evaluation/ms_score.py:123`, `tasks/feature_viz.py:202`).

2. `MS score`와 `MSAE` 구현 경로 충돌
- MS score는 `sae_model.activate()`를 직접 호출 (`src/evaluation/ms_score.py:144-145`)하지만, MSAE는 `activate()`를 미구현으로 명시 (`src/models/msae.py:54-56`).

3. 공식 비교군 수(4/5/6) 불일치
- README/스크립트/레지스트리 간 모델 세트가 다름 (`README.md:265, 469`, `run.sh:32`, `scripts/train_all.sh:4`, `scripts/evaluate_all.sh:4`, `src/models/__init__.py:12-18`).

4. Clinical discrimination 설계의 patient leakage 위험
- 현재 제안은 patch-level AUROC 중심 (`README.md:955-966`), split은 파일 기준 (`src/extract/activation_store.py:151-156`), label은 patient label을 patch에 전파 (`tasks/build_group_meta.py:117-120`).

### 제출 버전 최소 스토리
- **문제**: pathology FM representation은 강력하지만 해석이 어렵다.
- **방법**: EXAONEPath internal spatial token(196, CLS 제거)에 SAE variant를 동일 조건으로 학습.
- **평가 4축**: reconstruction/fidelity + sparsity/utilization + MS score + 제한적 병리 검증(audit).
- **범위 통제**: DriftPrior는 exploratory로 분리, clinical claim은 patient-level 재설계 전까지 보수적으로 기술.

---

## 2) 근거 소스와 검증 방법

### 2.1 근거 소스
- 코드/문서 로컬 리포지토리
  - `/home/kimhj/projects/pathoSAEv3/README.md`
  - `/home/kimhj/projects/pathoSAEv3/extra_documentation/THEORY_ALIGNMENT_MODULE_SPEC.md`
  - `/home/kimhj/projects/pathoSAEv3/src/**`, `/home/kimhj/projects/pathoSAEv3/tasks/**`, `/home/kimhj/projects/pathoSAEv3/scripts/**`
- 논문 자료 (PDF)
  - `/home/kimhj/projects/cytoSAE/plutosae.pdf`
  - `/home/kimhj/projects/cytoSAE/cytosae.pdf`
  - `/home/kimhj/projects/pathoSAEv3/extra_documentation/Generative Modeling via Drifting part 1.pdf`
- 회의 피드백(사용자 제공 21개 항목)

### 2.2 검증 규칙
- 코드/문서 근거는 `파일:라인`으로 표기.
- PDF 근거는 `section/page`로 표기.
- 도구 제한으로 page/section anchor가 불완전한 경우 `추정`으로 명시하고 단정 문장 금지.

### 2.3 근거 신뢰도 레벨
- `A (강)` 직접 코드 라인으로 확인.
- `B (중)` README/스펙 문장 확인.
- `C (보조)` PDF 해석(section/page 기반 요약), 또는 회의 진술 기반.

---

## 3) 피드백 21개 1:1 판정 매트릭스

| 항목번호 | 판정(유지/수정/보류) | 근거 | 위험도 | 즉시 교정 문장 | 필요한 구현/실험 |
|---|---|---|---|---|---|
| 1. 중심 가설 타당성 | 유지 | README의 문제정의/목표는 variant 비교 중심 (`README.md:15-19, 31-35`) | 중간 | "본 연구는 pathology tissue FM에서 SAE variant의 해석성 차이를 체계 비교한다." | 공식 비교군 수 확정(4종 or 5종) |
| 2. EXAONEPath 토큰 추출 정확성 | 수정 | 기본 `forward()`는 CLS 반환 (`src/extract/vit_encoder.py:285-290`), 현재 구현은 internal token 후 CLS 제거 (`src/extract/vit_encoder.py:371-378`) | 높음 | **EN**: "We extract last-block token sequences via internal layer access, remove CLS, and train SAEs on 196 spatial tokens." | Methods에 추출 경로 명시 |
| 3. Offline 추출 후 SAE 학습 타당성 | 유지 | 추출 파이프라인과 저장 구조 존재 (`tasks/extract.py:146-203`) | 중간 | "백본 고정 해석 연구 목적에 맞게 offline activation extraction 후 SAE를 학습한다." | layer/CLS/split 규칙 문서화 |
| 4. Stain normalization 해석 | 수정 | Macenko 전처리 적용 코드 존재 (`tasks/extract.py:13, 82-89`), README stack에도 stain norm 명시 (`README.md:505`) | 중간 | "Stain normalization은 모델 내부 레이어가 아니라 입력 전처리/학습 설정으로 다룬다." | EXAONEPath vs UNI/CONCH 전처리 정책 표준화 |
| 5. CytoSAE/DinoBloom 설명 오류 | 수정 | README 비교표가 `CytoSAE token strategy=CLS only`로 표기 (`README.md:103`) | 높음 | "CytoSAE: hematology single-cell, patch/token-level analysis with patient aggregation." | README 비교표 수정 |
| 6. Monosemanticity 정의 충돌(MS vs NMI) | 수정 | README는 MS score를 핵심 축으로 기술 (`README.md:640-661`) + 동시에 상태를 미구현으로 표기 (`README.md:644, 666`) | 높음 | "Primary monosemanticity metric은 MS score로 고정, morphology-label NMI는 라벨 확보 시 보조 지표로 격하." | 논문 본문 지표 체계 정리 |
| 7. Gated 수식 충실도 | 수정 | 현재 코드는 gate/magnitude 분리+aux loss 구현 (`src/models/gated_sae.py:15-23, 44-50`), README 서술 일부 단순화/이력 혼재 (`README.md:75, 1142`) | 중간 | "현 구현은 canonical gated 구조를 따르며 L1은 gate path에 적용한다." | Methods 수식과 코드 정합성 재작성 |
| 8. TopK/JumpReLU 해석성 선판정 문제 | 수정 | README가 JumpReLU 최고 해석성 단정 (`README.md:483`) | 중간 | "JumpReLU는 강력한 후보이지만, 해석성 우위는 측정 결과로만 주장한다." | variant별 MS score/CDF 실측 |
| 9. Matryoshka/MSAE 네이밍 주의 | 수정 | README는 MSAE를 Matryoshka로 단정 (`README.md:93, 150`) | 중간 | "Matryoshka-inspired multi-level TopK SAE"로 완화 | 용어 정리(본문/README 동기화) |
| 10. DriftPrior 메인 스토리 위험 | 유지(분리 권고) | 레지스트리에는 포함 (`src/models/__init__.py:18`)이나 README 핵심 비교군과 혼재 (`README.md:19, 1158`) | 높음 | "DriftPrior는 exploratory appendix/follow-up으로 분리 보고한다." | 메인 표/abstract에서 분리 |
| 11. README vs Draft 비교군 불일치 | 수정 | README 내부에서도 5종 vs 4종 혼재 (`README.md:19, 265, 410, 469`) | 높음 | "제출 버전의 공식 비교군을 단일 집합으로 고정한다." | 스크립트/문서 source-of-truth 통일 |
| 12. 결과표 해석(중간 상태) | 유지(표현 보수화) | 진행상태/결과 표가 미완료 포함 (`README.md:1168-1174, 1200-1203`) | 중간 | "현재 결과는 중간 상태이며 핵심 비교는 최종 재계산 중" | dead_freq 재계산, Gated v2 반영 |
| 13. 평가 구성 4축 제안 | 유지 | 기존 metric 축 존재 (`README.md:335-354`) + MS score 축 정의 (`README.md:640-661`) | 낮음 | "Fidelity/Sparsity/MS/Pathology validation의 4축으로 고정" | 표/그림 매핑 재구성 |
| 14. Feature discrimination leakage 위험 | 유지(중요) | patch-level AUROC 제안 (`README.md:955-966`), patient label patch 전파 (`tasks/build_group_meta.py:117-120`) | 높음 | "Clinical discrimination은 patient-level aggregation + patient-level CV로 계산" | 파이프라인 재설계 |
| 15. Patch-only 충분성 질문 | 유지(클레임 조정) | 현재 파이프라인은 tissue token 중심 (`README.md:58, 67`; `tasks/extract.py`) | 중간 | "본 연구는 tissue-level morphological concept 분석에 초점을 둔다." | cell-level claim 삭제/완화 |
| 16. 구현 고정안(A/B/C/D) | 유지 | 스펙에 lock/모듈 분할 명시 (`THEORY_ALIGNMENT_MODULE_SPEC.md:173-230`) | 낮음 | "샘플 단위, split 규칙, 추출 경로, variant 정의를 인터페이스 lock으로 고정" | lock 문서를 실제 코드 정책으로 반영 |
| 17. Monosemanticity 계산 절차 | 유지 | MS score 구현 파일/CLI 존재 (`src/evaluation/ms_score.py`, `tasks/eval_ms_score.py:71-81`) | 중간 | "독립 인코더 기반 activation-weighted similarity로 MS score 계산" | 인코더 옵션/샘플링/baseline 정규화 |
| 18. Pathologist 검증 위치 | 수정 | README에는 리뷰 방향 있으나 formal protocol 부재 (`README.md:40-43`) | 중간 | "Pathologist review는 전수 정량지표가 아니라 audit protocol로 보고" | 평가자/샘플링/합의지표 정의 |
| 19. 최소 제출 스토리 제안 | 유지 | 현재 코드/데이터 상태와 가장 정합적 | 낮음 | "Comparative SAE + MS score + 제한적 audit + patient-level 보조분석" | abstract/method/results 문장 통일 |
| 20. 즉시 수정 문장 체크리스트 | 유지 | README 오표기/과장 문장 확인 (`README.md:103, 469, 483`) | 높음 | 본 문서 7절 체크리스트를 즉시 반영 | README/Methods 교정 커밋 |
| 21. 최종 판정(방향 타당 + 문서 불일치 수정 필요) | 유지 | 코드 정합성 이슈 4개 + 문헌 표기 리스크 병존 | 높음 | "방향은 타당하나, 선행연구 표기·지표정의·비교군 정의를 즉시 정렬해야 한다." | 문서+코드 동시 정렬 |

> 주: 항목 6, 11, 18은 Draft 본문 원문 파일이 저장소에 없어(추정) 회의 피드백과 README/스펙 증거를 결합해 판정했다.

---

## 4) 재현성/논문화 Critical 이슈

### C1. `paths` vs `patch_paths` key 불일치
- 근거 라인
  - 저장: `tasks/extract.py:197-201`
  - 소비(강제 `patch_paths`): `src/evaluation/ms_score.py:123`, `tasks/feature_viz.py:202`
  - 일부는 양쪽 허용: `src/evaluation/feature_analysis.py:44-49`, `tasks/build_group_meta.py:78-83`
- 영향
  - 새로 추출한 activation 파일에서 일부 평가/시각화가 즉시 실패하거나 결과 누락 위험.
- 수정안
  - 읽기 경로 전역에서 `paths` 우선 + `patch_paths` fallback으로 통일.

### C2. `MS score`–`MSAE` 경로 충돌
- 근거 라인
  - MS score: `src/evaluation/ms_score.py:144-145`
  - MSAE 미구현 stub: `src/models/msae.py:54-56`
- 영향
  - "5종 비교"를 주장하면서 MSAE의 MS score를 동일 경로로 계산 불가.
- 수정안
  - 옵션 A: MSAE용 activation 추출 함수를 MS score 경로에서 분기.
  - 옵션 B: MSAE에 `activate()` 계약 호환 레이어 제공.

### C3. 공식 비교군 정의 불일치(4/5/6)
- 근거 라인
  - README 5종 주장: `README.md:19, 32, 69`
  - README 4종 흔적: `README.md:265, 410, 469`
  - 레지스트리 6종: `src/models/__init__.py:12-18`
  - train CLI 6종: `tasks/train.py:57`
  - run.sh 5종: `run.sh:32`
  - train/evaluate_all 4종: `scripts/train_all.sh:4`, `scripts/evaluate_all.sh:4`
- 영향
  - 비교표/결론의 모집단이 달라져 재현성 및 리뷰 신뢰 저하.
- 수정안
  - 제출 버전 source-of-truth를 1개로 고정.
  - 권장: `Core=4종(Vanilla/Gated/TopK/JumpReLU) + Optional=MSAE + Exploratory=DriftPrior`.

### C4. Patient-level leakage 위험
- 근거 라인
  - patch-level AUROC 설계: `README.md:955-966`
  - patch -> patient label 부여: `tasks/build_group_meta.py:117-120`
  - 파일 기반 split: `src/extract/activation_store.py:151-156`
- 영향
  - 동일 환자의 다수 패치가 train/test에 동시에 반영되어 임상 유효성 과대평가 가능.
- 수정안
  - patient-level aggregation table 생성 후, patient-level CV/bootstrapping으로 AUROC/PR-AUC 계산.

---

## 5) 선행연구 정합성 섹션 (안전 claim vs 금지 claim)

### 5.1 PLUTO-SAE

안전한 claim
- "PLUTO-SAE는 pathology 도메인에서 CLS residual embedding 기반 SAE 해석을 수행했고, monosemanticity 개선과 mixed probe utility를 보고했다."
- 근거: `plutosae.pdf §2.1 p4`, `§2.2 p4`, `§4.5 p9`, `§6 p10`.

금지 claim
- "PLUTO-SAE가 완전한 monosemanticity를 증명했다."
- "PLUTO-SAE가 모든 downstream에서 일관 우위다."

### 5.2 CytoSAE / DinoBloom-B

안전한 claim
- "CytoSAE는 hematology single-cell setting에서 token-level 표현을 사용하고 patch→image→patient/disease aggregation을 수행했다."
- 근거: `cytosae.pdf Abstract p1`, `§2 p3-4`, `Eq.(2)-(3) p4`, `§3.3 p6-7`.

금지 claim
- "CytoSAE는 CLS-only 접근이다."
- "CytoSAE 결과를 tissue pathology와 직접 leaderboard 비교할 수 있다."

### 5.3 Generative Modeling via Drifting

안전한 claim
- "Drifting은 training-time distribution evolution 기반 one-step generative framework이며 canonical SAE variant는 아니다."
- "Pathology SAE 해석성 메인 comparative method보다는 exploratory regularization 아이디어로 분리하는 것이 안전하다."
- 근거: `Generative Modeling via Drifting part 1.pdf` (main text method/experiments; section/page anchor는 도구 한계로 `추정`).

금지 claim
- "Drifting은 표준 SAE variant다."
- "Drifting의 자연영상 결과가 pathology interpretability 향상을 직접 보장한다."

---

## 6) 최소 제출 스토리 (Submission-safe narrative)

### 6.1 Abstract용 안전 문장 가이드

국문
- "본 연구는 pathology tissue foundation model representation의 해석 가능성을 높이기 위해 SAE variant를 동일 조건에서 비교한다."
- "EXAONEPath의 공개 forward가 반환하는 CLS embedding 대신, 마지막 블록 내부 token sequence에서 CLS를 제거한 196 spatial token을 사용해 SAE를 학습한다."
- "평가는 reconstruction/fidelity, sparsity/utilization, independent encoder 기반 MS score, 제한적 전문가 audit를 포함한다."

영문
- "We perform a controlled comparison of SAE variants for interpretability of pathology tissue foundation-model representations."
- "Instead of the default public forward output (CLS only), we extract last-block token sequences, remove CLS, and train SAEs on 196 spatial tokens."
- "We evaluate fidelity, sparsity/utilization, MS score with an independent encoder, and limited expert audit."

### 6.2 Methods 고정 포인트
- 추출 경로 명시: `forward()` CLS vs internal token extraction 차이 (`src/extract/vit_encoder.py:285-290`, `371-378`).
- 데이터 단위 명시: tile 내부 token 또는 tile-level 중 무엇인지 본문에서 고정.
- split 규칙 명시: patient-level split 여부를 임상 분석 파트에서 명확화.
- 비교군 고정: 공식 비교군과 exploratory 군을 분리.

### 6.3 Results/Discussion 표현 가이드
- 가능한 표현
  - "일부 variant의 reconstruction metric은 확보되었고, 핵심 interpretability metric의 전수 비교는 진행 중이다."
  - "dead neuron은 never-fired뿐 아니라 frequency-threshold 지표로 재해석했다."
- 피해야 할 표현
  - "최고 해석성 확정"
  - "임상 검증 완료"
  - "모든 변형에서 일관된 우위"

---

## 7) 즉시 수정 체크리스트 (우선순위 + 완료 기준)

### P0 (즉시)
1. 비교군 정의 단일화
- 작업: README/스크립트/리포트에서 4/5/6 혼재 제거.
- 완료 기준: 모든 엔트리포인트가 동일한 core set을 참조.

2. activation key 호환 통일
- 작업: `paths`/`patch_paths` 전역 호환 처리.
- 완료 기준: extract 직후 파일로 `evaluate`, `feature_viz`, `ms_score`가 모두 실행.

3. MS score–MSAE 충돌 해소
- 작업: MSAE 경로 호환 구현 또는 MSAE를 MS score에서 명시 제외(근거 포함).
- 완료 기준: 비교표에서 MSAE 처리 정책이 명확하고 실행 결과가 재현 가능.

4. EXAONEPath 토큰 추출 문장 교정
- 작업: README/Methods에 internal extraction 문장 삽입.
- 완료 기준: 리뷰어 질문("CLS만 반환하는데 spatial token은? ")에 문장 하나로 대응 가능.

### P1 (차주)
5. Clinical discrimination 재설계
- 작업: patient-level aggregation + patient-level CV로 AUROC/PR-AUC 재계산.
- 완료 기준: patch-level exploratory와 patient-level primary를 명확히 분리 보고.

6. Monosemanticity 지표 체계 확정
- 작업: Primary=MS score, Secondary=NMI(if labels available)로 문서 고정.
- 완료 기준: README/Methods/표/그림 캡션이 동일 정의 사용.

7. 선행연구 비교표 교정
- 작업: CytoSAE token strategy와 domain scope 수정.
- 완료 기준: "CytoSAE=CLS only" 문구 삭제 및 안전 문장 반영.

### P2 (후속)
8. Pathologist audit 프로토콜 정식화
- 작업: 평가자 수, feature 선정 기준, agreement 지표, adjudication 절차 문서화.
- 완료 기준: 재현 가능한 audit 섹션 1개.

9. DriftPrior 위치 고정
- 작업: 본문 메인 비교군에서 분리해 appendix/follow-up로 이동.
- 완료 기준: abstract/conclusion에서 DriftPrior 과대주장 제거.

10. 의존성/재현성 정리
- 작업: `timm` 의존성 명시 (`src/extract/vit_encoder.py:385` 대비 `requirements.txt:1-13`, `environment.yaml:7-23`), 대용량 경로/외부 symlink 문서화 (`data/activations`, `models/backbone/EXAONEPath.ckpt`).
- 완료 기준: clean env에서 재현 단계 문서화 완료.

---

## 부록 A) 재현성 리스크 참고 메모

- 하드코딩 경로 다수 존재
  - `run.sh:24`, `scripts/launch_training.sh:18`, `README.md:367`.
- 외부 symlink 의존
  - `data/activations -> /home/kimhj/projects/pathoSAEv2/...`
  - `models/backbone/EXAONEPath.ckpt -> /home/kimhj/projects/pathoSAEv2/...`
- 저장소 용량/추적 리스크
  - `du -sh`: data `2.2T`, models `20G`, logs `1.9G`, `src/models/uni` `1.2G`.
  - `.gitignore`는 `data/activations`만 제외, `data/activations_uni`는 명시 제외 없음 (`.gitignore:9-11`).

---

## 부록 B) 본 문서의 전제(Assumptions)

- 공식 비교군은 최종적으로 `Vanilla/TopK/Gated/JumpReLU` 또는 `+MSAE` 중 하나로 고정한다.
- `DriftPrior`는 exploratory로 분리한다.
- monosemanticity 주지표는 MS score로 둔다.
- morphology-label 기반 NMI는 라벨 확보 시 secondary로만 사용한다.
- clinical discrimination은 patient-level 재설계 전까지 exploratory로만 기술한다.

