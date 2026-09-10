"""计划工具的端到端测试。

与 `test_work_state_tool.py` 同样的理由：工厂忘了 ``return`` 会让 TUI 启动即炸，而别的测试
一律传假工具，所以这里覆盖"工厂返回真工具、参数能被解析、校验失败不写库、重复提交不落快照"。
"""

from __future__ import annotations

from langgraph.config import var_child_runnable_config
from sqlalchemy import select

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.langchain_agent import make_todo_tool
from coding_agent.persistence.models import TodoSnapshot

PLAN = [
    {"id": "1", "title": "读 DESIGN.md", "status": "completed"},
    {"id": "2", "title": "写 todo.py", "status": "in_progress"},
    {"id": "3", "title": "补测试", "status": "pending"},
]


def make_tool(database):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    return make_todo_tool(engine), engine


def invoke(tool, thread_id: str = "t1", user_seq: int = 1, **args) -> str:
    token = var_child_runnable_config.set(
        {"configurable": {"thread_id": thread_id, "user_seq": user_seq}}
    )
    try:
        return str(tool.invoke(args))
    finally:
        var_child_runnable_config.reset(token)


def todo_rows(database):
    with database.session() as session:
        return list(session.scalars(select(TodoSnapshot).order_by(TodoSnapshot.id)))


def test_factory_returns_a_real_tool_with_parsed_args(database):
    tool, _engine = make_tool(database)

    assert tool is not None, "工厂必须返回工具本体，否则 tools=[None] 会让 agent 起不来"
    assert tool.name == "todo"
    assert set(tool.args) == {"items"}


def test_a_write_materializes_one_snapshot_that_the_cache_follows(database):
    tool, engine = make_tool(database)

    invoke(tool, items=PLAN)

    rows = todo_rows(database)
    assert len(rows) == 1 and rows[0].user_seq == 1
    assert rows[0].items_json == PLAN, "整表替换：库里存的就是提交的那份完整列表"
    cached = engine.current_todos("t1")
    assert cached is not None and cached.items == PLAN


def test_the_receipt_carries_the_whole_table_so_the_plan_is_never_far(database):
    tool, _engine = make_tool(database)

    body = invoke(tool, items=PLAN)

    assert body.startswith("plan updated | 3 items: 1 pending, 1 in_progress, 1 completed")
    assert "changes:" in body
    assert "- [~] 2 写 todo.py" in body and "- [ ] 3 补测试" in body


def test_a_repeated_submission_does_not_materialize_a_new_snapshot(database):
    """内容没变就别落新快照：新快照 = 新 id = 下次剪裁时便签换新，白断一次前缀。"""

    tool, _engine = make_tool(database)
    invoke(tool, items=PLAN)

    body = invoke(tool, items=PLAN)

    assert body.startswith("plan unchanged")
    assert len(todo_rows(database)) == 1


def test_clearing_the_plan_writes_an_empty_snapshot_then_goes_quiet(database):
    tool, engine = make_tool(database)
    invoke(tool, items=PLAN)

    body = invoke(tool, items=[])

    assert "plan cleared" in body
    rows = todo_rows(database)
    assert len(rows) == 2 and rows[-1].items_json == []
    assert engine.current_todos("t1").items == []

    invoke(tool, items=[])
    assert len(todo_rows(database)) == 2, "已经是空的，再清一次不必落新行"


def test_invalid_items_return_error_text_and_write_nothing(database):
    tool, _engine = make_tool(database)

    assert invoke(tool, items=[{"id": "1", "title": "甲", "status": "doing"}]).startswith("Error:")
    assert invoke(tool, items=[{"id": "1", "title": "甲"}]).startswith("Error:")
    assert invoke(
        tool,
        items=[
            {"id": "1", "title": "甲", "status": "in_progress"},
            {"id": "2", "title": "乙", "status": "in_progress"},
        ],
    ).startswith("Error:")
    assert todo_rows(database) == [], "校验失败发生在写之前，不留半改状态"


def test_each_round_of_writes_can_be_revoked_together_by_user_seq(database):
    tool, _engine = make_tool(database)

    invoke(tool, user_seq=7, items=[{"id": "1", "title": "甲", "status": "in_progress"}])
    invoke(tool, user_seq=7, items=[{"id": "1", "title": "甲", "status": "completed"}])

    rows = todo_rows(database)
    assert [row.user_seq for row in rows] == [7, 7], "撤销按 user_seq 停用，天然覆盖整轮"
    assert len(rows) == 2
