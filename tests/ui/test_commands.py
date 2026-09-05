from __future__ import annotations

import pytest

from coding_agent.ui.commands import CommandParseError, parse_command, positive_int


def test_parse_command_supports_aliases_and_quoted_thread_ids():
    history = parse_command("/list 5")
    thread = parse_command('/thread "feature work"')

    assert history is not None
    assert history.name == "history"
    assert history.argument == "5"
    assert thread is not None
    assert thread.name == "thread"
    assert thread.argument == "feature work"


def test_parse_command_rejects_unknown_or_invalid_commands():
    with pytest.raises(CommandParseError, match="未知命令"):
        parse_command("/remove-everything")
    with pytest.raises(CommandParseError, match="参数数量"):
        parse_command("/status extra")
    with pytest.raises(CommandParseError, match="大于等于 1"):
        positive_int("0", label="user_seq")


def test_plain_text_is_not_treated_as_a_command():
    assert parse_command("please inspect /src") is None


def test_usage_command_accepts_optional_recent_count():
    default = parse_command("/usage")
    explicit = parse_command("/usage 50")
    assert default is not None and default.argument is None
    assert explicit is not None and explicit.argument == "50"
    with pytest.raises(CommandParseError):
        parse_command("/usage 50 extra")


def test_clear_command_takes_no_argument():
    clear = parse_command("/clear")

    assert clear is not None
    assert clear.name == "clear"
    assert clear.argument is None
    with pytest.raises(CommandParseError, match="参数数量"):
        parse_command("/clear everything")
