from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True)
class RecentUsage:
    sampled_messages: int
    usage_samples: int
    cache_samples: int
    input_tokens: int
    output_tokens: int
    cache_input_tokens: int
    cache_read_tokens: int

    @property
    def cache_hit_rate(self) -> float | None:
        if not self.cache_samples or not self.cache_input_tokens:
            return None
        return self.cache_read_tokens / self.cache_input_tokens

    @classmethod
    def from_metadata(cls, records: Sequence[Any]) -> RecentUsage:
        usage_samples = cache_samples = input_tokens = output_tokens = 0
        cache_input_tokens = cache_read_tokens = 0
        for metadata in records:
            if not isinstance(metadata, dict):
                continue
            inputs = _count(metadata.get("input_tokens"))
            outputs = _count(metadata.get("output_tokens"))
            if inputs is None or outputs is None:
                continue
            usage_samples += 1
            input_tokens += inputs
            output_tokens += outputs
            details = metadata.get("input_token_details")
            cached = _count(details.get("cache_read")) if isinstance(details, dict) else None
            if cached is None or cached > inputs:
                continue
            # LangChain 的 input_tokens 已经把缓存读取和缓存创建算在里面了。
            cache_samples += 1
            cache_input_tokens += inputs
            cache_read_tokens += cached
        return cls(
            len(records),
            usage_samples,
            cache_samples,
            input_tokens,
            output_tokens,
            cache_input_tokens,
            cache_read_tokens,
        )
