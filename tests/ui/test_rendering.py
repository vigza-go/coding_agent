from __future__ import annotations

import io
import json

from rich.console import Console

from coding_agent.ui.rendering import TUI_THEME, ContentRenderer
from coding_agent.ui.tui import TerminalUI


def test_renderer_extracts_visible_text_and_preserves_newlines():
    content = [
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "first\nsecond"},
        {"type": "tool_use", "id": "call-1"},
    ]

    assert ContentRenderer().text(content) == "first\nsecond"


def test_renderer_decodes_json_content_strings():
    content = json.dumps(
        [
            {"type": "text", "text": "hello"},
            {"type": "reasoning", "text": "hidden"},
        ]
    )

    assert ContentRenderer().text(content) == "hello"


def test_renderer_truncates_history_summaries():
    assert ContentRenderer().summary("abcdef", limit=4) == "abcd…"


def test_markdown_renderer_uses_light_terminal_code_styles():
    markdown = ContentRenderer().markdown("**strong** and `inline code`")

    assert markdown.code_theme == "ansi_light"
    assert TUI_THEME.styles["markdown.code"].bgcolor is None


def _render_plain(markdown: object, *, width: int = 120) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False, no_color=True)
    console.print(markdown)
    return buffer.getvalue()


def test_markdown_keeps_literal_html_like_text_visible():
    """Agent prose that looks like HTML must not vanish from the TUI.

    Rich's builtin markdown renderer drops raw HTML tokens, so tag-like replies such
    as "</parameter>" or "<w></w>" were stored in the database but invisible in the
    TUI. The renderer disables markdown-it's HTML rules so they render verbatim.
    """
    cases = {
        "</>": "</>",
        "<a123123/>": "<a123123/>",
        "<w></w>": "<w></w>",
        "<li></li>": "<li></li>",
        "</parameter>": "</parameter>",
        "这是正文</parameter>后面": "这是正文</parameter>后面",
    }
    for source, expected in cases.items():
        rendered = _render_plain(ContentRenderer().markdown(source))
        assert expected in rendered, f"{source!r} rendered as {rendered!r}"


def test_markdown_keeps_real_markdown_features():
    rendered = _render_plain(
        ContentRenderer().markdown("# title\n\n**bold** and `code`\n\n```\n<x>\n```")
    )
    assert "title" in rendered
    assert "bold" in rendered
    assert "code" in rendered
    assert "<x>" in rendered


def test_status_memory_timeline_preserves_block_order():
    assert TerminalUI._memory_timeline((2, 1, 0, 0)).plain == "L2 → L1 → L0 → L0  (4 块)"
    assert TerminalUI._memory_timeline(()).plain == "尚无记忆块"
