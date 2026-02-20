# CLAUDE.md

## 프로젝트 아키텍처

이 프로젝트는 **Opus-Codex 파이프라인**을 사용한다.

- **Claude Code (Opus 4.6)** = 아키텍트. "무엇을, 왜"에만 답한다.
- **Codex CLI (GPT-5.3)** = 구현 엔지니어. "어떻게"를 실행한다.
- **`/run-codex` skill** = Claude Code 안에서 Codex를 headless worker로 spawn한다.

## 설계 문서 위치

모든 설계 문서는 `.pipeline/` 폴더에 있다:

| 파일 | 역할 |
|------|------|
| `architecture.md` | 시스템 구조 (ASCII 다이어그램) |
| `contracts.md` | 모듈 간 I/O 인터페이스 계약 |
| `decisions.md` | 설계 근거 (ADR) |
| `task_queue.md` | 태스크 상태 관리 |
| `codex_notes.md` | Codex → Opus 역방향 커뮤니케이션 |
| `review_feedback.md` | Opus 코드 리뷰 결과 |
| `context_snapshot.md` | 프로젝트 상태 요약 (세션 시작 시 참조) |

## 워크플로우

1. 설계 시: `.pipeline/` 문서만 수정. 코드 작성 금지.
2. 구현 위임 시: `/run-codex` skill 사용.
3. 리뷰 시: Codex 결과물 + `codex_notes.md` → `review_feedback.md` 작성.
4. 매 세션 시작 시: `context_snapshot.md`를 읽어 현재 상태 파악.

## 코드 컨벤션

- 설계 문서 변경 시 반드시 `context_snapshot.md` 갱신
- `task_queue.md` 상태를 항상 최신으로 유지
- Codex 구현 전 반드시 `git commit`으로 체크포인트 생성
