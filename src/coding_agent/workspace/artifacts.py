from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..context.tokens import estimate_tokens


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:100] or "unknown"


def truncate_to_tokens(text: str, limit: int) -> str:
    if estimate_tokens(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def offload_tool_result(
        self, *, thread_id: str, user_seq: int, tool_call_id: str, content: str, inline_limit: int
    ) -> tuple[str, Path | None]:
        if estimate_tokens(content) <= inline_limit:
            return content, None
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        destination = (
            self.root
            / "tool-results"
            / _safe_component(thread_id)
            / str(user_seq)
            / f"{_safe_component(tool_call_id)}-{digest}.txt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        prefix = truncate_to_tokens(content, inline_limit)
        notice = (
            f"\n\n[工具结果超过 {inline_limit} tokens，完整内容已保存到 "
            f"{destination}。需要细节时请使用 read_file 工具读取。]"
        )
        return prefix + notice, destination
