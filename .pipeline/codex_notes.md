# codex_notes.md
> Codex 작성 · Opus 리뷰

---

## 규칙
- Codex는 구현 중 발견한 이슈, 설계 변경 제안, 질문을 여기에 기록한다.
- 각 항목에 날짜와 관련 태스크 ID를 명시한다.
- Opus가 리뷰 후 해결 여부를 표기한다.

---

## Notes

| 날짜 | Task | 내용 | 상태 |
|------|------|------|------|
| 2026-02-20 | T-001~T-006 | P0 구현 완료: `src/config.py`, `src/models/*`, `src/extract/activation_store.py`, `src/training/{losses,scheduler,trainer}.py`, `tasks/train.py` 작성. QD-001/002/003/004/005 반영(디코더 norm, grad projection, b_dec mean init, float32 only, 파일 단위 val split). | RESOLVED |
| 2026-02-20 | T-003 (TopK) | ADR-005 기준으로 TopK는 기본 단일 `topk_k` 경로로 구현. `topk_nesting_list`는 Config/모델 속성으로 유지했으며 multi-level 학습 로직은 P0 범위에서 활성화하지 않음. | RESOLVED |
| 2026-02-20 | 검증 | `python -m compileall src tasks/train.py` 통과. 런타임 smoke test는 현재 셸 환경에 `torch` 미설치로 실행 불가(`ModuleNotFoundError`). | OPEN |
