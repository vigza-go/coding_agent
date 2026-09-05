from __future__ import annotations

import os
import signal
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
from coding_agent.services.rollback import RollbackPreview, RollbackResult
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


def _watcher_probe(monkeypatch, *outcomes):
    """把 _watch_terminal 变成一段可观察的同步执行：``outcomes`` 是 ``_tty_lost()`` 的判定
    序列，用剩最后一个就一直重复它。返回观察到的动作。"""
    acts: list[object] = []
    seq = list(outcomes)

    def lost() -> bool:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(tui, "_tty_lost", lost)
    monkeypatch.setattr(tui, "_TTY_POLL_SECONDS", 0.01)
    monkeypatch.setattr(tui, "_TTY_EXIT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(tui, "sleep", lambda seconds: acts.append(("等", seconds)))
    monkeypatch.setattr(os, "kill", lambda pid, sig: acts.append(("信号", pid, sig)))

    def hard_exit(code: int) -> None:  # 真 os._exit 不返回，这里必须抛出来才收得住循环
        acts.append(("退出", code))
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", hard_exit)
    return acts


def test_closing_the_terminal_stops_a_turn_in_progress(monkeypatch):
    """关标签页是用户的显式意图（"我不想看了"），所以正在跑 turn 也得停：先给自己一发
    Ctrl-C 走既有中断收尾（杀本轮 bash 进程组、终止子代理、结清时间轴、放出座位），
    收尾不配合就硬退。"""
    ui, *_ = make_ui(monkeypatch)

    # 回归：曾把"打印一句提示"写在退出之前，往已撤销的终端写字抛 OSError(EIO) 把这个线程
    # 摔死在退出前面，座位照样没还（真机 22:27 / 22:30 两只鬼就是这么留下的）。这里连提示
    # 都不许有：console.print 一旦被动过就抛。
    def eio(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(ui.console, "print", eio)
    acts = _watcher_probe(monkeypatch, False, False, True, True)  # 先确认终端在，再连着没了

    with pytest.raises(SystemExit):
        ui._watch_terminal()

    assert ("信号", os.getpid(), signal.SIGINT) in acts
    assert acts[-1] == ("退出", 0)


def test_watcher_waits_while_the_terminal_is_alive(monkeypatch):
    """终端好好的，一个信号都不许发、一次都不许退。"""
    ui, *_ = make_ui(monkeypatch)
    acts = _watcher_probe(monkeypatch, False)

    def one_lap(seconds: float) -> None:  # 跑完一轮判定就收工
        raise SystemExit(0)

    monkeypatch.setattr(tui, "sleep", one_lap)
    with pytest.raises(SystemExit):
        ui._watch_terminal()

    assert acts == []


def test_watcher_stays_out_of_the_way_without_a_controlling_terminal(monkeypatch):
    """被脚本直接拉起（没有 shell 认领那块 pty）时，``tcgetpgrp`` 一上来就抛 ENOTTY，看着
    像"终端没了"。这时判据没资格说话：不许动信号、不许退出，自检自己收工。"""
    ui, *_ = make_ui(monkeypatch)
    acts = _watcher_probe(monkeypatch, True)

    ui._watch_terminal()  # 自己 return，走不到兜底的 os._exit

    assert not [a for a in acts if a[0] in ("信号", "退出")]


def test_terminal_probe_treats_revoked_terminal_as_gone(monkeypatch):
    """Terminal 关标签页只撤销 pty、不发挂断，所以只能靠自己问。两个问法各管一段坏状态。"""
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: 7)
    monkeypatch.setattr(os, "close", lambda fd: None)
    monkeypatch.setattr(os, "tcgetpgrp", lambda fd: os.getpgrp())
    assert tui._tty_lost() is False

    monkeypatch.setattr(os, "tcgetpgrp", lambda fd: 0)
    assert tui._tty_lost() is True  # 终端已销毁：没有前台作业组了

    def gone(fd):
        raise OSError(25, "Inappropriate ioctl for device")

    monkeypatch.setattr(os, "tcgetpgrp", gone)
    assert tui._tty_lost() is True

    def no_ctty(*args, **kwargs):
        raise OSError(6, "No such device or address")

    monkeypatch.setattr(os, "tcgetpgrp", lambda fd: os.getpgrp())
    monkeypatch.setattr(os, "open", no_ctty)
    assert tui._tty_lost() is True  # 中间态：master 关了、slave 还开着，tcgetpgrp 看不出来


def _preview(messages=2, work_states=1, file_mutations=1, files=1):
    # 用真 dataclass 而不是随手编的假对象：字段名写错时 Mock 会一路放过，真跑才炸。
    return RollbackPreview(
        messages=messages,
        file_mutations=file_mutations,
        files=files,
        work_states=work_states,
    )


def test_clear_command_reaches_the_app_only_after_confirmation(monkeypatch):
    ui, app, output, _ = make_ui(monkeypatch)
    app.rollback_preview.return_value = _preview()
    app.clear_context.return_value = RollbackResult(0, 2, (), None)
    command = ParsedCommand(name="clear")

    monkeypatch.setattr(tui.Confirm, "ask", classmethod(lambda cls, *a, **k: False))
    ui._handle_command(command)
    app.clear_context.assert_not_called()

    monkeypatch.setattr(tui.Confirm, "ask", classmethod(lambda cls, *a, **k: True))
    assert ui._handle_command(command) is True
    app.clear_context.assert_called_once_with("t1")
    app.rollback.assert_not_called()  # 清空不是撤销，别顺手把文件退了
    rendered = output.getvalue()
    assert "保持原样" in rendered
    assert "已停用 2 条消息" in rendered


def test_clear_command_on_an_empty_thread_does_nothing(monkeypatch):
    ui, app, output, _ = make_ui(monkeypatch)
    app.rollback_preview.return_value = _preview(messages=0, work_states=0)
    asked = []
    monkeypatch.setattr(
        tui.Confirm, "ask", classmethod(lambda cls, *a, **k: asked.append(1) or True)
    )

    ui._handle_command(ParsedCommand(name="clear"))

    assert asked == []
    assert app.clear_context.call_count == 0
    assert "没有可清空的上下文" in output.getvalue()
