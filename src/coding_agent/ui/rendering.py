from __future__ import annotations

import json
from typing import Any

from markdown_it import MarkdownIt
from rich.markdown import Markdown
from rich.theme import Theme

TUI_THEME = Theme(
    {
        # Rich defaults to "bold cyan on black", which clashes with light terminals.
        "markdown.code": "bold blue",
        "markdown.code_block": "none",
    }
)


class LiteralMarkdown(Markdown):
    """Markdown that keeps literal HTML-like text instead of silently dropping it.

    Rich's default renderer discards raw HTML (html_inline / html_block tokens),
    so agent prose such as ``<w></w>`` or ``</parameter>`` disappeared from the TUI
    while the full text was still stored in the database. Disabling markdown-it's
    HTML rules makes those sequences render verbatim as ordinary text (markdown
    headings, bold, code spans and fenced code keep working as before).
    """

    def __init__(
        self,
        markup: str,
        code_theme: str = "monokai",
        justify=None,
        style="none",
        hyperlinks: bool = True,
        inline_code_lexer: str | None = None,
        inline_code_theme: str | None = None,
    ) -> None:
        # Mirror rich.markdown.Markdown.__init__, but disable the HTML rules so that
        # tags are parsed as text rather than dropped during rendering.
        parser = (
            MarkdownIt()
            .disable("html_block")
            .disable("html_inline")
            .enable("strikethrough")
            .enable("table")
        )
        self.markup = markup
        self.parsed = parser.parse(markup)
        self.code_theme = code_theme
        self.justify = justify
        self.style = style
        self.hyperlinks = hyperlinks
        self.inline_code_lexer = inline_code_lexer
        self.inline_code_theme = inline_code_theme or code_theme


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
        return LiteralMarkdown(self.text(content), code_theme="ansi_light")

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
