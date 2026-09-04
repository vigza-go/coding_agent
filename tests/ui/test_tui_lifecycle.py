from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from coding_agent.application import TurnExecutionError
from coding_agent.config import Settings, TUISettings
from coding_agent.services.progress import TurnEvent, TurnEventKind
from coding_agent.services.usage import RecentUsage
from coding_agent.ui import tui
from coding_agent.ui.commands import CommandParseError, ParsedCommand


def make_ui(monkeypatch, settings=None):
    monkeypatch.setattr(tui, "PromptSession", lambda **kwargs: None)
    app = Mock()
    app.settings = settings or Settings()
    output = StringIO()
    console = Console(file=output, width=100)
    ui = tui.TerminalUI(app, console, thread_id="t1", debug=False)
    lifecycle = []

    @contextmanager
    def keep_awake():
        lifecycle.append("start")
        try:
            yield
        finally:
            lifecycle.append("stop")

    def notify(message):
        assert lifecycle[-1] == "stop"
        lifecycle.append(message)
        return True

    monkeypatch.setattr(
        ui, "desktop", SimpleNamespace(keep_awake=keep_awake, notify=Mock(side_effect=notify))
    )
    return ui, app, output, lifecycle


@pytest.mark.parametrize("answer", [AIMessage(content="done"), None])
def test_success_releases_sleep_before_notifying(monkeypatch, answer):
    ui, app, _, lifecycle = make_ui(monkeypatch)
    app.run_turn.return_value = answer
    ui._run_turn("task")
    assert lifecycle[:2] == ["start", "stop"]
    assert "本轮" in lifecycle[-1]
    cast(Mock, ui.desktop.notify).assert_called_once()


@pytest.mark.parametrize("interrupted", [False, True])
def test_failure_releases_sleep_and_notifies_before_undo_prompt(monkeypatch, interrupted):
    ui, app, _, lifecycle = make_ui(monkeypatch)
    app.run_turn.side_effect = TurnExecutionError(
        thread_id="t1",
        user_seq=1,
        cause=RuntimeError("failure"),
        interrupted=interrupted,
    )

    def confirm(*args, **kwargs):
        assert lifecycle[:2] == ["start", "stop"]
        assert ("已中断" if interrupted else "执行失败") in lifecycle[-1]
        return False

    monkeypatch.setattr(tui.Confirm, "ask", confirm)
    ui._run_turn("task")
    cast(Mock, ui.desktop.notify).assert_called_once()


def test_unwrapped_failure_still_releases_sleep_and_notifies(monkeypatch):
    ui, app, _, lifecycle = make_ui(monkeypatch)
    app.run_turn.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        ui._run_turn("task")
    assert lifecycle[:2] == ["start", "stop"]
    assert "执行失败" in lifecycle[-1]


def test_disabled_notifications_do_not_ring_terminal(monkeypatch):
    ui, app, _, _ = make_ui(monkeypatch, Settings(tui=TUISettings(notifications_enabled=False)))
    app.run_turn.return_value = AIMessage(content="done")
    bell = Mock()
    monkeypatch.setattr(ui.console, "bell", bell)
    ui._run_turn("task")
    cast(Mock, ui.desktop.notify).assert_not_called()
    bell.assert_not_called()


def test_notification_falls_back_to_terminal_bell(monkeypatch):
    ui, app, _, _ = make_ui(monkeypatch)
    app.run_turn.return_value = AIMessage(content="done")
    ui.desktop.notify = Mock(return_value=False)
    bell = Mock()
    monkeypatch.setattr(ui.console, "bell", bell)
    ui._run_turn("task")
    bell.assert_called_once()


def test_usage_command_renders_weighted_rate_without_running_a_turn(monkeypatch):
    ui, app, output, lifecycle = make_ui(
        monkeypatch, Settings(tui=TUISettings(usage_recent_messages=7))
    )
    app.recent_usage.return_value = RecentUsage(2, 2, 2, 1000, 20, 1000, 850)
    assert ui._handle_command(ParsedCommand("usage"))
    app.recent_usage.assert_called_once_with("t1", limit=7)
    assert "85.0%" in output.getvalue()
    assert "含撤销历史" in output.getvalue()
    assert not lifecycle
    app.run_turn.assert_not_called()
    ui._handle_command(ParsedCommand("usage", "40"))
    app.recent_usage.assert_called_with("t1", limit=40)
    with pytest.raises(CommandParseError):
        ui._handle_command(ParsedCommand("usage", "0"))


@pytest.mark.parametrize("samples", [0, 3])
def test_usage_empty_data_is_not_reported_as_zero_percent(monkeypatch, samples):
    ui, app, output, _ = make_ui(monkeypatch)
    app.recent_usage.return_value = RecentUsage(samples, 0, 0, 0, 0, 0, 0)
    ui._show_usage(20)
    assert "0.0%" not in output.getvalue()
    assert "暂无可计算数据" in output.getvalue() if samples else "还没有" in output.getvalue()


def test_mid_turn_prose_reaches_the_console(monkeypatch):
    ui, _app, output, _ = make_ui(monkeypatch)
    ui._render_event(Mock(), TurnEvent(TurnEventKind.MODEL_FINISHED, text="先解释一句再动手"))

    assert "先解释一句再动手" in output.getvalue()


def test_model_finished_without_prose_prints_nothing(monkeypatch):
    ui, _app, output, _ = make_ui(monkeypatch)
    ui._render_event(Mock(), TurnEvent(TurnEventKind.MODEL_FINISHED))

    assert output.getvalue() == ""


def test_thread_command_refuses_to_switch_into_occupied_thread(monkeypatch):
    from coding_agent.persistence.thread_lock import ThreadBusyError

    ui, app, output, _ = make_ui(monkeypatch)
    app.enter_thread.side_effect = ThreadBusyError("thread 'research' 正被另一个会话使用")

    assert ui._handle_command(ParsedCommand(name="thread", argument="research")) is True

    assert ui.thread_id == "t1"  # 没切过去
    assert "不切换" in output.getvalue()


def test_thread_command_switches_when_thread_is_free(monkeypatch):
    ui, app, output, _ = make_ui(monkeypatch)

    ui._handle_command(ParsedCommand(name="thread", argument="research"))

    assert ui.thread_id == "research"
    app.enter_thread.assert_called_once_with("research")
    assert "已切换" in output.getvalue()


def test_startup_refuses_to_run_inside_an_occupied_thread(monkeypatch):
    from coding_agent.persistence.thread_lock import ThreadBusyError

    ui, app, output, _ = make_ui(monkeypatch)
    app.enter_thread.side_effect = ThreadBusyError("thread 't1' 正被另一个会话使用")
    ui.session = SimpleNamespace(prompt=Mock(side_effect=AssertionError("不该开始问")))

    try:
        ui.run()
    except ThreadBusyError:
        pass  # 由 ui.run() 外面那层收成"启动失败"
    else:
        raise AssertionError("占用失败应该冒出去")
    assert "已切换" not in output.getvalue()


def test_busy_thread_is_reported_without_traceback(monkeypatch):
    from coding_agent.persistence.thread_lock import ThreadBusyError

    ui, app, output, _ = make_ui(monkeypatch)
    app.run_turn.side_effect = ThreadBusyError("thread 't1' 正被另一个会话使用")
    ui.session = SimpleNamespace(prompt=Mock(side_effect=["问一句", EOFError]))
    ui.run()

    printed = output.getvalue()
    assert "本轮未开始" in printed
    assert "Traceback" not in printed
