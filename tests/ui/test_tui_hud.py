"""底栏 HUD：一眼看到 seq 与计划，而它绝不能变成"每敲一个键查一次库"。

TerminalUI 只在回合边界算这一行（`_refresh_hud`），`bottom_toolbar` 的回调只把算好的结果交出去
——`test_the_toolbar_never_queries_again` 就是盯这条线的。
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console

from coding_agent.application import ThreadStatus
from coding_agent.config import Settings
from coding_agent.services.progress import TurnEvent, TurnEventKind
from coding_agent.ui import tui


def make_status(todos=None) -> ThreadStatus:
    return ThreadStatus(
        thread_id="t1",
        active_head_seq=12,
        next_user_seq=13,
        memory_levels=(1, 0),
        memory_tokens=1200,
        memory_limit=250000,
        working_messages=8,
        working_tokens=900,
        working_trigger=500000,
        bash_enabled=True,
        search_enabled=False,
        work_state=None,
        todos=todos,
    )


def make_ui(monkeypatch, status: ThreadStatus | None = None, error: Exception | None = None):
    monkeypatch.setattr(tui, "PromptSession", lambda **kwargs: None)
    app = Mock()
    app.settings = Settings()
    if error is not None:
        app.thread_status.side_effect = error
    else:
        app.thread_status.return_value = status or make_status()
    ui = tui.TerminalUI(app, Console(file=StringIO(), width=100), thread_id="t1", debug=False)
    return ui, app


def bar_text(ui) -> str:
    return "".join(part for _, part in ui._toolbar())


PLAN = [
    {"id": "1", "title": "读 DESIGN.md", "status": "completed"},
    {"id": "2", "title": "写 todo.py", "status": "in_progress"},
    {"id": "3", "title": "补测试", "status": "pending"},
]


@pytest.mark.parametrize(
    "todos, expected",
    [
        (None, ""),
        ([], ""),
        (["不是字典"], ""),
        (PLAN, "计划 1/3 · 进行中 写 todo.py"),
        ([{"id": "1", "title": "甲", "status": "pending"}], "计划 0/1 · 下一步 甲"),
        (
            [
                {"id": "1", "title": "甲", "status": "completed"},
                {"id": "2", "title": "乙", "status": "cancelled"},
            ],
            "计划 2/2 · 全部交代",
        ),
        # 在做的那项优先：下一步是给"手头空了"看的，不该跟进行中的抢位置
        (
            [
                {"id": "1", "title": "甲", "status": "pending"},
                {"id": "2", "title": "乙", "status": "in_progress"},
            ],
            "计划 0/2 · 进行中 乙",
        ),
    ],
)
def test_the_plan_segment_covers_every_state(todos, expected):
    assert tui._hud_plan(todos) == expected


def test_a_rambling_title_is_cut_short():
    """标题上限 200 字，底栏只有一行——长了就截，别把 seq 挤没。"""

    segment = tui._hud_plan([{"id": "1", "title": "长" * 200, "status": "in_progress"}])

    assert segment == f"计划 0/1 · 进行中 {'长' * 24}…"


def test_the_bar_shows_seq_and_plan(monkeypatch):
    ui, _app = make_ui(monkeypatch, make_status(todos=PLAN))

    ui._refresh_hud()

    text = bar_text(ui)
    assert "t1" in text and "head 12" in text and "next 13" in text
    assert "计划 1/3 · 进行中 写 todo.py" in text


def test_no_plan_means_no_empty_segment(monkeypatch):
    ui, _app = make_ui(monkeypatch, make_status(todos=[]))

    ui._refresh_hud()

    assert "计划" not in bar_text(ui)


def test_the_toolbar_never_queries_again(monkeypatch):
    """回调每敲一个键调一次：算过就不能再算，否则键盘会把 SQL 敲出火星子。"""

    ui, app = make_ui(monkeypatch, make_status(todos=PLAN))

    ui._refresh_hud()
    for _ in range(5):
        bar_text(ui)

    assert app.thread_status.call_count == 1


def test_a_failing_status_read_is_visible_instead_of_swallowed(monkeypatch):
    """显示层不许把会话带走，但也不许装作没事——错因就印在底栏上。"""

    ui, _app = make_ui(monkeypatch, error=RuntimeError("database unavailable"))

    ui._refresh_hud()  # 不能抛

    assert "状态不可用" in bar_text(ui) and "RuntimeError" in bar_text(ui)


def test_a_status_payload_that_makes_no_sense_does_not_kill_the_session(monkeypatch):
    """读得到、但读到一坨根本没法看的东西：同样是底栏那一行的事，会话照跑。"""

    ui, _app = make_ui(monkeypatch, make_status(todos=42))

    ui._refresh_hud()  # 不能抛

    assert "状态不可用" in bar_text(ui)


def test_a_junk_plan_is_just_no_plan(monkeypatch):
    """能遍历但里面没有一条像样的（比如手写过的旧行）：当作没有计划，不报错也不显示。"""

    ui, _app = make_ui(monkeypatch, make_status(todos=["不是字典", None]))

    ui._refresh_hud()

    assert "状态不可用" not in bar_text(ui)
    assert "计划" not in bar_text(ui)


# ----------------------------------------------------------- 回合进行中那行进度（底栏那时不在）


def test_the_busy_line_carries_the_plan(monkeypatch):
    """回车之后底栏和输入框一起消失，整轮跑完才回来——所以进度行必须自己带上计划。"""

    ui, _app = make_ui(monkeypatch, make_status(todos=PLAN))

    ui._refresh_hud()

    assert ui._busy("模型思考中…") == (
        "[cyan]模型思考中…[/cyan]"
        " [bright_black]· 计划 1/3 · 进行中 写 todo.py[/bright_black]"
    )


def test_the_busy_line_is_just_the_action_without_a_plan(monkeypatch):
    ui, _app = make_ui(monkeypatch, make_status(todos=[]))

    ui._refresh_hud()

    assert ui._busy("模型思考中…") == "[cyan]模型思考中…[/cyan]"


def test_a_todo_write_shows_up_in_the_busy_line_right_away(monkeypatch):
    """计划一变进度行就换：等整轮跑完才露面已经晚了，模型那会儿早走过两步了。"""

    ui, app = make_ui(monkeypatch, make_status(todos=PLAN))
    ui._refresh_hud()
    app.thread_status.return_value = make_status(
        todos=[{"id": "9", "title": "写代码", "status": "in_progress"}]
    )
    status = Mock()

    ui._render_event(status, TurnEvent(kind=TurnEventKind.TOOL_FINISHED, name="todo"))

    line = status.update.call_args[0][0]
    assert "计划已更新" in line and "计划 0/1 · 进行中 写代码" in line


def test_other_tools_do_not_go_back_to_the_database(monkeypatch):
    """一轮里工具事件一个接一个，每个都去查库等于拿键盘打数据库；用缓存那份就好。"""

    ui, app = make_ui(monkeypatch, make_status(todos=PLAN))
    ui._refresh_hud()
    status = Mock()

    ui._render_event(status, TurnEvent(kind=TurnEventKind.TOOL_FINISHED, name="bash"))

    assert app.thread_status.call_count == 1
    assert "计划 1/3" in status.update.call_args[0][0], "缓存的那份照样带着计划"


def test_a_title_with_rich_markup_does_not_break_the_line(monkeypatch):
    """标题是模型写的，里头可能有 [red] 这种字样——不转义就会被 rich 当样式吃掉。"""

    ui, _app = make_ui(
        monkeypatch,
        make_status(todos=[{"id": "1", "title": "[red]危险", "status": "in_progress"}]),
    )

    ui._refresh_hud()

    assert r"\[red]危险" in ui._busy("跑着呢")
