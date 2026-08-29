from __future__ import annotations

import json

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


def test_status_memory_timeline_preserves_block_order():
    assert TerminalUI._memory_timeline((2, 1, 0, 0)).plain == "L2 → L1 → L0 → L0  (4 块)"
    assert TerminalUI._memory_timeline(()).plain == "尚无记忆块"
