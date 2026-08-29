from __future__ import annotations

import json
from typing import Any

from rich.markdown import Markdown
from rich.theme import Theme

TUI_THEME = Theme(
    {
        # Rich defaults to "bold cyan on black", which clashes with light terminals.
        "markdown.code": "bold blue",
        "markdown.code_block": "none",
    }
)


class ContentRenderer:
    HIDDEN_BLOCK_TYPES = frozenset(
        {"thinking", "reasoning", "tool_use", "tool_call", "tool_result"}
    )

    def text(self, content: Any) -> str:
        return self._extract(content).strip()

    def summary(self, content: Any, *, limit: int = 500) -> str:
        text = self.text(content)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"

    def markdown(self, content: Any) -> Markdown:
        return Markdown(self.text(content), code_theme="ansi_light")

    def _extract(self, content: Any) -> str:
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith(("[", "{")):
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    return content
                return self._extract(decoded)
            return content
        if isinstance(content, list):
            parts = [self._extract(block) for block in content]
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            block_type = content.get("type")
            if block_type in self.HIDDEN_BLOCK_TYPES:
                return ""
            if "text" in content:
                return self._extract(content["text"])
            if "content" in content:
                return self._extract(content["content"])
            return json.dumps(content, ensure_ascii=False, indent=2, default=str)
        if content is None:
            return ""
        return str(content)
