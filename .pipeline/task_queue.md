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
| T-001 | 프로젝트 골격 생성 | 2026-02-20 | Codex 구현, Opus 승인 |
| T-002 | src/config.py — SAEConfig | 2026-02-20 | Codex 구현, Opus 승인 |
| T-003 | src/models/ — BaseSAE + 4 variant | 2026-02-20 | Codex 구현, TopK ReLU Opus 수정 |
| T-004 | src/extract/activation_store.py | 2026-02-20 | Codex 구현, Opus 승인 |
| T-005 | src/training/ — trainer + losses + scheduler | 2026-02-20 | Codex 구현, Opus 승인 |
| T-006 | tasks/train.py — 통합 CLI | 2026-02-20 | Codex 구현, Opus 승인 |
