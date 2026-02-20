# contracts.md
> Opus 작성 · Codex **엄격 준수** · `v0.0.0` · YYYY-MM-DD

---

## 규칙
1. 아래 I/O 스펙은 **계약**이다. Codex는 반드시 준수한다.
2. 스펙 변경이 필요하면 구현하지 말고 `codex_notes.md`에 변경 요청을 남긴다.
3. 타입은 Python typing 표기법을 따른다.

---

## Module A → Module B

```python
def process(data: InputTypeA) -> OutputTypeA:
    """
    Input:
        data: InputTypeA
            - field_1: str          # 설명
            - field_2: int          # 범위: 0-100
            - field_3: Optional[list[float]]

    Output:
        OutputTypeA
            - result: str
            - score: float          # 범위: 0.0-1.0
            - metadata: dict        # keys: "timestamp", "source"

    Raises:
        ValueError: field_2 범위 밖
        TypeError: field_1이 str 아님

    Constraints:
        - 응답 < 100ms
        - 메모리 < 512MB
    """
```

---

## Shared Types

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class InputTypeA:
    field_1: str
    field_2: int
    field_3: Optional[list[float]] = None

@dataclass
class OutputTypeA:
    result: str
    score: float
    metadata: dict
```

---

## Config Contract

```python
@dataclass
class AppConfig:
    debug: bool = False
    max_retries: int = 3
    timeout_sec: float = 30.0
```

---

## API Endpoints (해당 시)

| Method | Path | Request | Response | Codes |
|--------|------|---------|----------|-------|
| POST | /api/process | `InputTypeA` | `OutputTypeA` | 200, 400, 500 |

---

> NOTE: **CODEX**: 타입을 `src/shared/types.py`에 구현. 시그니처 변경 금지.
