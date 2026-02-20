# architecture.md
> Opus 작성 · Codex 참조 · `v0.0.0` · YYYY-MM-DD

---

## System Overview

```
┌─────────────────────────────────────────────────┐
│                  시스템 이름                       │
│                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Module A  │───▶│ Module B  │───▶│ Module C  │  │
│  │          │    │          │    │          │  │
│  │ 역할:    │    │ 역할:    │    │ 역할:    │  │
│  └──────────┘    └──────────┘    └──────────┘  │
│       │                               ▲        │
│       │         ┌──────────┐          │        │
│       └────────▶│ Module D  │─────────┘        │
│                 └──────────┘                   │
└─────────────────────────────────────────────────┘
```

### 다이어그램 규칙
```
───▶  데이터 흐름        ─ ─ ▶  비동기/선택적
◄──▶  양방향             ┌──┐   내부 모듈
╔══╗  외부 시스템         ═══    중요 경계
```

---

## Module Definitions

### Module A: {이름}
- **책임**: 
- **입력**: 
- **출력**: 
- **의존**: 
- **제약**: 

### Module B: {이름}
- **책임**: 
- **입력**: 
- **출력**: 
- **의존**: 
- **제약**: 

---

## Data Flow

```
[Input] ──▶ Module A ──▶ Module B ──▶ [Output]
                │                        ▲
                └──▶ Module D ───────────┘
```

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Lang | | |
| Framework | | |
| DB | | |
| Infra | | |

---

## Directory Structure

```
src/
├── module_a/
│   ├── __init__.py
│   └── core.py
├── module_b/
│   └── core.py
├── shared/
│   ├── types.py
│   └── config.py
└── main.py
```

---

## Error Strategy

```
정상:  A ───▶ B ───▶ C ───▶ Output
에러:  A ───▶ B ──╳──▶ ErrorHandler → retry(3) / fallback / log
```

---

> NOTE: **CODEX**: 이 구조를 임의 변경 금지. 변경 필요 시 `codex_notes.md`에 기록.
