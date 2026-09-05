from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def estimate_tokens(value: Any) -> int:
    """便宜、确定性的兜底估算，预算和测试用它就够。

    中文字符一般接近 1 token；英文正文和代码大约 4 个字符 1 token。宁可往高了估——
    预算撑爆的代价，比稍微提前一点压缩大得多。
    """

    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    cjk = sum(1 for char in value if "\u3400" <= char <= "\u9fff")
    other = len(value) - cjk
    return max(1, cjk + (other + 3) // 4)


def total_tokens(values: Iterable[Any]) -> int:
    return sum(estimate_tokens(value) for value in values)
