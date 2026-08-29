from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Cheap deterministic fallback suitable for budgeting and tests.

    CJK characters are commonly close to one token; ASCII prose/code is roughly four
    characters per token. Over-counting is intentional because budget overflow is worse
    than compacting slightly early.
    """

    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    cjk = sum(1 for char in value if "\u3400" <= char <= "\u9fff")
    other = len(value) - cjk
    return max(1, cjk + (other + 3) // 4)


def total_tokens(values: Iterable[Any]) -> int:
    return sum(estimate_tokens(value) for value in values)
