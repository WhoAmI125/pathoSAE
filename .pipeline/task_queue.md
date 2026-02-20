# task_queue.md
> User 관리 · Opus/Codex 공통 참조 · 2026-02-20

---

## Status Flow

```
DESIGN → READY → IMPLEMENT → REVIEW → (FEEDBACK →) DONE
                                │
                                └── BLOCKED
```

---

## In Progress

| ID | Task | Owner | Status | Priority | Depends |
|----|------|-------|--------|----------|---------|
| T-001 | 프로젝트 골격 생성 (디렉토리, __init__.py, environment.yaml, requirements.txt) | [CODEX] | READY | P0 | - |
| T-002 | src/config.py — SAEConfig dataclass 구현 | [CODEX] | READY | P0 | T-001 |
| T-003 | src/models/ — BaseSAE + 4 variant 구현 | [CODEX] | READY | P0 | T-002 |
| T-004 | src/extract/activation_store.py — 데이터 로딩 | [CODEX] | READY | P0 | T-002 |
| T-005 | src/training/trainer.py + losses.py + scheduler.py | [CODEX] | READY | P0 | T-003, T-004 |
| T-006 | tasks/train.py — 통합 CLI 엔트리포인트 | [CODEX] | READY | P0 | T-005 |
| T-007 | src/evaluation/ — metrics, visualize, compare | [CODEX] | READY | P1 | T-003 |
| T-008 | tasks/evaluate.py + tasks/compare.py | [CODEX] | READY | P1 | T-007 |
| T-009 | src/extract/vit_encoder.py + macenko.py | [CODEX] | READY | P1 | T-001 |
| T-010 | tasks/extract.py — activation 추출 CLI | [CODEX] | READY | P1 | T-009 |
| T-011 | scripts/train_all.sh + evaluate_all.sh | [CODEX] | READY | P2 | T-006, T-008 |

---

## Backlog

| ID | Task | Owner | Priority | Note |
|----|------|-------|----------|------|
| | | | | |

---

## Done

| ID | Task | Date | Note |
|----|------|------|------|
| | | | |
