---
name: run-codex
description: Codex CLI를 headless worker로 실행하여 구현 태스크를 위임한다. .pipeline/ 문서를 자동으로 수집·조립하여 Codex에게 전달하고, 결과를 수거한다.
---

# /run-codex — Codex Headless Worker Skill

Claude Code(Opus)가 아키텍트로서 설계한 뒤, Codex CLI를 headless worker로 spawn하여 구현을 위임하는 skill이다.

---

## 실행 흐름

```
/run-codex 호출
  │
  ├─ Step 1: 태스크 확인 (AskUserQuestion)
  ├─ Step 2: 모델 & reasoning level 선택
  ├─ Step 3: 모드 선택 (implement / fix / test / refactor)
  ├─ Step 4: .pipeline/ 문서 수집 → 프롬프트 조립
  ├─ Step 5: codex exec 실행 (headless)
  └─ Step 6: 결과 수거 → 리뷰 트리거
```

---

## Step 1 — 태스크 확인

AskUserQuestion으로 어떤 태스크를 Codex에게 위임할지 확인한다.

task_queue.md에 `READY` 상태인 태스크가 있으면 자동으로 제안한다.
없으면 사용자에게 자유 입력을 받는다.

---

## Step 2 — 모델 & Reasoning Level 선택

AskUserQuestion으로 모델과 reasoning effort를 선택한다.

**모델 옵션:**
- `gpt-5.3-codex` — 최신, 최고 성능 (기본값)
- `gpt-5.2-codex` — 안정적, 코드 리뷰 특화

**Reasoning Level 옵션:**
- `low` — 단순 작업 (파일 이동, 이름 변경)
- `medium` — 일반 구현 (기본값)
- `high` — 복잡한 로직
- `xhigh` — 아키텍처급 복잡도

---

## Step 3 — 모드 선택

태스크 유형에 따라 Codex에게 전달할 문서 조합이 달라진다.

| 모드 | 설명 | 전달 문서 |
|------|------|----------|
| `implement` | 새 기능 구현 | architecture + contracts + decisions + task |
| `fix` | 버그 수정 | contracts + 관련 코드 + 에러 로그 |
| `test` | 테스트 작성 | contracts + 구현 코드 |
| `refactor` | 리팩토링 실행 | review_feedback + contracts + 대상 코드 |
| `review-fix` | 리뷰 피드백 반영 | review_feedback + contracts + 대상 코드 |

---

## Step 4 — 프롬프트 조립

선택된 모드에 따라 .pipeline/ 문서를 읽어서 하나의 프롬프트로 조립한다.

### implement 모드 프롬프트 템플릿

```
cat <<'PROMPT_END'
# 역할
너는 구현 엔지니어다. 아래 설계문서를 기반으로 정확히 구현해.

# 절대 규칙
- architecture.md의 구조를 임의로 변경하지 마
- contracts.md의 타입과 시그니처를 정확히 따라
- decisions.md의 결정을 위반하지 마
- 설계와 다르게 구현해야 할 이유가 있으면 .pipeline/codex_notes.md에 기록하되, 일단 설계대로 구현해
- 구현 완료 후 .pipeline/codex_notes.md에 구현 메모를 남겨

PROMPT_END

echo "## architecture.md"
cat .pipeline/architecture.md

echo "## contracts.md"
cat .pipeline/contracts.md

echo "## decisions.md"
cat .pipeline/decisions.md

echo "## 현재 태스크"
# task_queue.md에서 해당 태스크 정보 추출
grep -A5 "READY\|IMPLEMENT" .pipeline/task_queue.md
```

### fix 모드 프롬프트 템플릿

```
cat <<'PROMPT_END'
# 역할
아래 버그를 수정해. contracts.md의 I/O 스펙을 반드시 준수해.
수정 사항을 .pipeline/codex_notes.md에 기록해.

PROMPT_END

echo "## contracts.md"
cat .pipeline/contracts.md

echo "## 버그 설명"
echo "${BUG_DESCRIPTION}"

echo "## 관련 코드"
# 사용자가 지정한 파일을 cat
```

### test 모드 프롬프트 템플릿

```
cat <<'PROMPT_END'
# 역할
아래 모듈의 테스트를 작성해.
contracts.md의 I/O 스펙에 명시된 입출력과 제약조건을 기준으로 테스트 케이스를 설계해.
엣지 케이스와 에러 케이스를 반드시 포함해.

PROMPT_END

echo "## contracts.md"
cat .pipeline/contracts.md

echo "## 구현 코드"
# 테스트 대상 코드를 cat
```

### review-fix 모드 프롬프트 템플릿

```
cat <<'PROMPT_END'
# 역할
아키텍트의 리뷰 피드백을 반영하여 코드를 수정해.
"Codex에게 보내는 지시" 섹션의 항목을 하나씩 처리해.
수정 완료 후 .pipeline/codex_notes.md에 기록해.

PROMPT_END

echo "## review_feedback.md"
cat .pipeline/review_feedback.md

echo "## contracts.md (참고)"
cat .pipeline/contracts.md

echo "## 수정 대상 코드"
# 대상 파일을 cat
```

---

## Step 5 — codex exec 실행

조립된 프롬프트를 파일로 저장하고, codex exec를 headless로 실행한다.

```bash
# 프롬프트를 임시 파일에 저장
PROMPT_FILE=$(mktemp /tmp/codex-prompt-XXXXX.md)
# ... (Step 4에서 조립한 내용을 $PROMPT_FILE에 기록)

# Codex exec 실행
codex exec \
  -c model="${CODEX_MODEL}" \
  -c model_reasoning_effort="${REASONING_LEVEL}" \
  --full-auto \
  - < "$PROMPT_FILE"

# 임시 파일 정리
rm -f "$PROMPT_FILE"
```

### 옵션 설명

| 플래그 | 설명 |
|--------|------|
| `-c model=` | 사용할 Codex 모델 |
| `-c model_reasoning_effort=` | reasoning depth 설정 |
| `--full-auto` | 자동 승인 (구현 모드) |
| `--sandbox read-only` | 읽기 전용 (리뷰/분석 시) |
| `--output-last-message <file>` | 마지막 메시지를 파일로 저장 (선택) |

### sandbox 모드 선택

| 태스크 유형 | sandbox 설정 |
|------------|-------------|
| implement / fix / refactor | `--full-auto` (쓰기 필요) |
| test | `--full-auto` (테스트 파일 생성 필요) |
| 코드 분석 / 리뷰 | `--sandbox read-only` |

---

## Step 6 — 결과 수거 & 리뷰 트리거

Codex 실행 완료 후:

1. **변경된 파일 확인**: `git diff --name-only` 또는 `git status`로 Codex가 생성/수정한 파일 목록 수거
2. **codex_notes.md 확인**: Codex가 이슈를 기록했는지 확인
3. **자동 리뷰 트리거**: Claude Code(Opus)가 아래를 수행
   - 변경된 코드를 architecture.md, contracts.md와 대조
   - codex_notes.md에 새 이슈가 있으면 확인
   - review_feedback.md 작성
   - task_queue.md 상태 업데이트 (IMPLEMENT → REVIEW → DONE 또는 FEEDBACK)
   - context_snapshot.md 갱신

---

## 에러 처리

| 상황 | 대응 |
|------|------|
| codex exec 실패 (exit code ≠ 0) | 에러 로그 출력, 사용자에게 알림 |
| Codex가 설계를 변형함 | review_feedback.md에 기록, FEEDBACK 상태로 전환 |
| codex_notes.md에 설계 변경 요청 | Opus가 판단하여 수락/거부 |
| 네트워크 / 인증 오류 | `codex login` 재실행 안내 |

---

## 주의사항

- Codex는 **ChatGPT Pro ($200/mo) 또는 API key** 인증이 필요하다
- 첫 실행 전 `codex login`으로 인증을 완료해야 한다
- `--full-auto`는 Codex에게 파일 쓰기를 허용하므로, git commit 후 실행하는 것을 권장한다
- 장시간 작업(7hr+)이 가능하지만, 중간 체크포인트가 없으므로 큰 태스크는 분할 권장
