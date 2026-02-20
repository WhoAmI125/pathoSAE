# decisions.md
> Opus 작성 · Codex **반드시 읽고 준수** · YYYY-MM-DD

---

## 목적

Codex는 자체 판단으로 아키텍처를 변형하는 경향이 있다.
이 문서는 각 설계 결정의 **"왜"**를 명시하여 임의 변경을 방지한다.

**Codex 규칙**: 동의하지 않더라도 먼저 설계대로 구현하고, `codex_notes.md`에 대안을 제안하라.

---

## ADR Format

### ADR-001: {제목}

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` / `PROPOSED` / `DEPRECATED` / `SUPERSEDED by ADR-XXX` |
| **날짜** | YYYY-MM-DD |
| **컨텍스트** | 어떤 문제를 해결하려는지 |
| **결정** | 무엇을 선택했는지 |
| **근거** | 왜 이 선택인지 |
| **거부한 대안** | 검토했지만 선택하지 않은 옵션 |
| **결과** | 트레이드오프 |
| **Codex 지시** | 구현 시 반드시 지킬 사항 |

---

## 예시

### ADR-001: SAE 학습에 TopK 활성화 사용

| 항목 | 내용 |
|------|------|
| **상태** | `ACCEPTED` |
| **날짜** | 2025-02-18 |
| **컨텍스트** | PathoSAEv2에서 sparse autoencoder 활성화 함수 선택. ReLU, TopK, JumpReLU 검토. |
| **결정** | TopK 활성화를 메인으로 사용 |
| **근거** | sparsity level 직접 제어 가능 → 해석성 실험에 유리 |
| **거부한 대안** | ReLU (sparsity 간접 제어), JumpReLU (학습 초기 collapse) |
| **결과** | k값 튜닝이 추가 하이퍼파라미터. k=32,64,128 실험 필요 |
| **Codex 지시** | `activation_fn` 파라미터 설정 가능하게. 하드코딩 금지. 기본값 `topk` |

---

## Quick Decision Log (경량 ADR)

| ID | 결정 | 이유 | Codex 주의 |
|----|------|------|-----------|
| QD-001 | | | |
| QD-002 | | | |
