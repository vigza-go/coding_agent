from __future__ import annotations

import json
from typing import Any

from markdown_it import MarkdownIt
from rich.markdown import Markdown
from rich.theme import Theme

TUI_THEME = Theme(
    {
        # Rich 默认用“黑底亮青”，在浅色终端里很刺眼。
        "markdown.code": "bold blue",
        "markdown.code_block": "none",
    }
)


class LiteralMarkdown(Markdown):
    """字面量 HTML 样文本照样看得见的 Markdown，不会被悄悄扔掉。

    Rich 默认的渲染器会把原始 HTML 直接丢掉（html_inline / html_block 两种 token），于是
    Agent 写的标签样正文（尖括号那类）在 TUI 里消失了，而库里明明存着完整文本。关掉
    markdown-it 的 HTML 规则之后，这些串按普通文本原样渲染（标题、加粗、行内代码、围栏
    代码块都照旧）。
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
        # 照搬 rich.markdown.Markdown.__init__，但关掉 HTML 规则：标签按文本解析，
        # 而不是在渲染时被丢掉。
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
