# Opus-Codex Pipeline

> Opus가 설계하고, Codex가 구현한다.
> Glue layer는 Claude Code `/run-codex` custom skill.

---

## 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  사용자                                                       │
│  태스크 정의 → Claude Code 세션 시작                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  [OPUS] Claude Code (Opus 4.6)                                  │
│                                                              │
│  1. 설계  →  .pipeline/*.md 생성·수정                          │
│  3. 리뷰  →  구현 결과 + codex_notes → review_feedback.md     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │  /run-codex  (Custom Skill · ~50 LOC)           │      │
│  │                                                    │      │
│  │  2. .pipeline/*.md 수집 → 프롬프트 조립              │      │
│  │     codex exec -m gpt-5.3-codex --full-auto        │      │
│  └──────────────────────┬─────────────────────────────┘      │
│                         │ headless spawn                      │
└─────────────────────────┼────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  [CODEX] Codex CLI (GPT-5.3-Codex · xhigh)                       │
│                                                              │
│  비동기 독립 실행 · fire-and-forget                            │
│  src/ 구현 · tests/ 작성 · codex_notes.md 이슈 기록           │
└──────────────────────────────────────────────────────────────┘
```

---

## 전제 조건

| 항목 | 요구사항 |
|------|---------|
| Claude Code | Opus 4.6 모델 사용 가능한 상태 |
| Codex CLI | `npm i -g @openai/codex` 설치 완료 |
| Codex 인증 | `codex login` 완료 (ChatGPT Pro 또는 API key) |
| Git | 프로젝트가 git repo인 상태 |

---

## 설치

### 1. 프로젝트에 파이프라인 복사

```bash
# 다운로드한 zip을 프로젝트 루트에 풀기
cp -r opus-codex-pipeline/.pipeline  your-project/.pipeline
cp -r opus-codex-pipeline/.claude    your-project/.claude
cp    opus-codex-pipeline/CLAUDE.md  your-project/CLAUDE.md
```

### 2. 디렉토리 구조 확인

```
your-project/
├── CLAUDE.md                         # Claude Code가 자동으로 읽는 프로젝트 지침
├── .claude/
│   └── skills/
│       └── run-codex/
│           └── SKILL.md              # /run-codex 커스텀 스킬
├── .pipeline/
│   ├── architecture.md               # 아키텍처 (ASCII 다이어그램)
│   ├── contracts.md                  # I/O 인터페이스 계약
│   ├── decisions.md                  # 설계 근거 (ADR)
│   ├── task_queue.md                 # 태스크 관리
│   ├── codex_notes.md                # Codex → Opus 이슈 리포트
│   ├── review_feedback.md            # Opus 코드 리뷰
│   └── context_snapshot.md           # 프로젝트 상태 요약
└── src/                              # 실제 코드 (Codex가 생성)
```

### 3. Codex CLI 설치 & 인증

```bash
npm i -g @openai/codex
codex login           # 브라우저에서 ChatGPT 인증
codex login status    # 인증 확인
```

---

## 사용법

### Phase 1: 설계 (Opus)

Claude Code 세션을 열고 설계를 요청한다:

```
나: 이미지 분류 파이프라인을 설계해줘.
    입력은 pathology 이미지, 출력은 cancer/normal classification.
    SAE feature를 중간 표현으로 사용.

    .pipeline/ 문서를 업데이트해줘. 코드는 작성하지 마.
```

Opus가 아래 문서를 생성/업데이트한다:
- `architecture.md` — 시스템 다이어그램
- `contracts.md` — 모듈 간 I/O 스펙
- `decisions.md` — 왜 이 설계인지
- `task_queue.md` — 태스크 등록 (READY 상태)
- `context_snapshot.md` — 상태 갱신

### Phase 2: 구현 (Codex via /run-codex)

설계가 완료되면 Codex에게 위임한다:

```
나: /run-codex
```

스킬이 자동으로:
1. task_queue.md에서 READY 태스크를 찾아 제안
2. 모델/reasoning level 선택 (기본: gpt-5.3-codex, xhigh)
3. 모드 선택 (implement/fix/test/refactor)
4. .pipeline/ 문서를 수집하여 프롬프트 조립
5. `codex exec` headless 실행
6. 결과 수거

### Phase 3: 리뷰 (Opus)

Codex 실행 완료 후 리뷰를 요청한다:

```
나: Codex가 구현 완료했어. 리뷰해줘.
```

Opus가:
1. 변경된 코드를 architecture/contracts와 대조
2. `codex_notes.md` 확인 (이슈 있는지)
3. `review_feedback.md` 작성
4. `task_queue.md` 상태 업데이트
5. 문제 없으면 DONE, 수정 필요하면 FEEDBACK

### Phase 4: 피드백 반영 (필요 시)

FEEDBACK 상태가 되면:

```
나: /run-codex    (모드: review-fix)
```

review_feedback.md의 "Codex에게 보내는 지시"를 기반으로 Codex가 수정.
→ 다시 Phase 3 리뷰 → DONE.

---

## 실전 워크플로우 예시

### 예시: PathoSAEv2 Feature Visualization 모듈 구현

```
── Session 1: 설계 ──────────────────────────────

나: PathoSAEv2의 feature visualization 모듈을 설계해줘.
    학습된 SAE의 각 feature가 어떤 pathology 패턴에
    반응하는지 시각화하는 모듈이야.
    .pipeline/ 문서 업데이트해줘. 코드는 쓰지 마.

[OPUS] Opus: (architecture.md, contracts.md, decisions.md 업데이트)
          ADR-003: matplotlib 대신 plotly 사용 — 인터랙티브 필요
          Task T-005: READY 상태로 등록

── Session 1: 구현 위임 ─────────────────────────

나: /run-codex

Skill:
   → 태스크: T-005 Feature Visualization (READY)
   → 모델: gpt-5.3-codex / xhigh
   → 모드: implement
   → .pipeline/ 수집 → 프롬프트 조립
   → codex exec 실행 (headless)

[CODEX] Codex: (src/visualization/ 생성, tests/ 작성)
          (codex_notes.md: "plotly 대신 matplotlib이
           서버 환경에서 더 안정적. 변경 제안.")

── Session 1: 리뷰 ──────────────────────────────

나: 리뷰해줘.

[OPUS] Opus: contracts.md 준수 OK
          architecture.md 정합 OK
          decisions.md 위반 NOTE: — ADR-003에서 plotly로 결정했는데
          Codex가 matplotlib으로 구현함.

          판단: Codex 의견 수용. 서버 환경 고려하면 matplotlib이 맞음.
          → ADR-003 SUPERSEDED by ADR-004
          → review_feedback.md: APPROVED (with ADR update)
          → T-005: DONE
```

---

## 토큰 절약 전략

| 상황 | Opus 사용 | Codex 사용 |
|------|-----------|------------|
| 아키텍처 설계 | OK | NG |
| 코드 구현 | NG | OK |
| 코드 리뷰 (핵심만) | OK | NG |
| 버그 디버깅 | NG | OK |
| 설계 변경 판단 | OK | NG |
| 테스트 작성 | NG | OK |
| 리팩토링 방향 | OK | NG |
| 리팩토링 실행 | NG | OK |

**원칙**: Opus = "무엇을, 왜" · Codex = "어떻게"

### 추가 절약 팁

1. **세션 시작 시** context_snapshot.md만 전달 (전체 .pipeline/ 불필요)
2. **Opus 압축 모드**: "불릿 포인트만. 코드 예시 불필요. 핵심만."
3. **모드별 선택적 전달**: implement는 3개 문서, fix는 contracts만

---

## /run-codex 모드별 전달 문서

| 모드 | 전달 문서 | 용도 |
|------|----------|------|
| `implement` | architecture + contracts + decisions | 새 기능 구현 |
| `fix` | contracts + 에러 로그 + 관련 코드 | 버그 수정 |
| `test` | contracts + 구현 코드 | 테스트 작성 |
| `refactor` | review_feedback + contracts + 코드 | 리팩토링 |
| `review-fix` | review_feedback + contracts + 코드 | 리뷰 반영 |

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| `codex exec` 실패 | `codex login status`로 인증 확인 |
| Codex가 설계를 무시함 | decisions.md의 "Codex 지시"를 더 구체적으로 작성 |
| 너무 큰 태스크 | task_queue.md에서 세분화하여 분할 |
| context 유실 | context_snapshot.md 갱신 주기를 줄이기 |
| Codex 네트워크 오류 | `-c 'sandbox_workspace_write.network_access=true'` |
| 결과물 품질 낮음 | reasoning level을 `xhigh`로, contracts를 더 상세하게 |

---

## 파일 역할 요약

```
CLAUDE.md                    ← Claude Code가 자동으로 읽는 프로젝트 지침
.claude/skills/run-codex/    ← /run-codex 커스텀 스킬
.pipeline/
  ├── architecture.md        ← [OPUS] Opus 작성, [CODEX] Codex 참조
  ├── contracts.md           ← [OPUS] Opus 작성, [CODEX] Codex 엄격 준수
  ├── decisions.md           ← [OPUS] Opus 작성, [CODEX] Codex 반드시 읽기
  ├── task_queue.md          ← [USER] User + [OPUS] Opus 관리
  ├── codex_notes.md         ← [CODEX] Codex 작성, [OPUS] Opus 리뷰
  ├── review_feedback.md     ← [OPUS] Opus 작성, [CODEX] Codex 반영
  └── context_snapshot.md    ← [OPUS] Opus 갱신, 모든 세션 시작 시 참조
```
