# context_snapshot.md
> Opus 갱신 · 매 세션 시작 시 전달 · 2026-02-20

---

## 프로젝트

| 항목 | 내용 |
|------|------|
| **이름** | PathoSAEv3 |
| **한줄 설명** | 병리 이미지 해석을 위한 Sparse Autoencoder 통합 학습·비교 프레임워크 |
| **기술** | Python 3.10, PyTorch 2.2.0, EXAONEPath ViT-B/16 |
| **단계** | 설계 완료 → 초기 구현 진입 |
| **목표** | 논문 작성을 위한 4가지 SAE variant 공정 비교 |

---

## 현재 상태 (3-5문장)

> pathoSAEv2의 리팩토링 프로젝트. cytoSAE의 깔끔한 구조를 참고하여 설계 완료.
> 4가지 SAE variant (Vanilla, Gated, TopK, JumpReLU)를 동일 하이퍼파라미터로 학습하여 논문용 비교 분석이 가능하도록 구조화.
> .pipeline/ 설계 문서 (architecture, contracts, decisions) 작성 완료.
> 다음 단계: Codex에게 구현 위임 (T-001 ~ T-011).

---

## 완료된 모듈

| 모듈 | 상태 | 핵심 파일 | 비고 |
|------|------|----------|------|
| 설계 문서 | OK | .pipeline/*.md | Opus 작성 |

---

## 미해결 이슈

| ID | 요약 | 심각도 | 담당 |
|----|------|--------|------|
| - | 없음 | - | - |

---

## 최근 변경 (최신 5개)

| 날짜 | 변경 | 담당 |
|------|------|------|
| 2026-02-20 | 프로젝트 초기화, git init | [USER] |
| 2026-02-20 | opus-codex pipeline 복사 | [USER] |
| 2026-02-20 | architecture.md 작성 | [OPUS] |
| 2026-02-20 | contracts.md 작성 | [OPUS] |
| 2026-02-20 | decisions.md + task_queue.md 작성 | [OPUS] |

---

## 다음 단계

1. Codex에게 T-001 (프로젝트 골격) 위임 → `/run-codex`
2. T-002 ~ T-006 순차 구현 (P0 태스크)
3. 구현 완료 후 Opus 리뷰

---

## 참조 프로젝트

| 프로젝트 | 경로 | 용도 |
|---------|------|------|
| pathoSAEv2 | `/home/kimhj/projects/pathoSAEv2` | 기존 구현 참조 (SAE 모델 코드, activation 데이터) |
| cytoSAE | `/home/kimhj/projects/cytoSAE` | 구조 참조 (클린 아키텍처) |

## 데이터 참조

| 데이터 | 원본 경로 | symlink 대상 |
|--------|----------|-------------|
| activation 파일 | `/home/kimhj/projects/pathoSAEv2/data/encode_spatial/` | `data/activations/` |
| EXAONEPath.ckpt | `/home/kimhj/projects/pathoSAEv2/models/EXAONEPath.ckpt` | `models/backbone/EXAONEPath.ckpt` |
| ViT config | `/home/kimhj/projects/pathoSAEv2/configs/config.json` | `configs/vit_config.json` (복사) |
