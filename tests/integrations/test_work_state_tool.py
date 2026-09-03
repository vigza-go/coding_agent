"""工作状态工具的端到端测试。

存在的理由是一条真实事故：`make_work_state_tool` 一度忘了 ``return work_state``
而返回 ``None``，而 `tests/integrations/test_agent.py` 一律传 `tools=[]`，
于是"工厂返回 None"这个会让 TUI 启动即炸的 bug 在 150 条测试全绿的情况下活了下来。
这里覆盖三件此前没被测过的事：工厂返回真工具、docstring 能被解析成参数、
只读 op 绝不写库。
"""

from __future__ import annotations

from langgraph.config import var_child_runnable_config
from sqlalchemy import func, select

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.langchain_agent import make_work_state_tool
from coding_agent.persistence.models import WorkStateSnapshot


def make_tool(database):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    return make_work_state_tool(engine), engine


def invoke(tool, thread_id: str = "t1", user_seq: int = 1, **args) -> str:
    token = var_child_runnable_config.set(
        {"configurable": {"thread_id": thread_id, "user_seq": user_seq}}
    )
    try:
        return str(tool.invoke(args))
    finally:
        var_child_runnable_config.reset(token)


def snapshot_rows(database):
    with database.session() as session:
        return list(session.scalars(select(WorkStateSnapshot).order_by(WorkStateSnapshot.id)))


def test_factory_returns_a_real_tool_with_parsed_args(database):
    tool, _engine = make_tool(database)
    assert tool is not None, "工厂必须返回工具本体，否则 tools=[None] 会让 agent 起不来"
    assert tool.name == "work_state"
    assert set(tool.args) == {"op", "key", "value"}


def test_mutating_ops_persist_and_read_only_ops_do_not(database):
    tool, _engine = make_tool(database)

    assert "keys: goal" in invoke(tool, op="set", key="goal", value="目标正文")
    assert len(snapshot_rows(database)) == 1

    body = invoke(tool, op="get", key="goal")
    assert "目标正文" in body
    assert "keys:" in invoke(tool, op="list")
    assert len(snapshot_rows(database)) == 1, "只读 op 不得制造新快照"

    assert "appended to goal" in invoke(tool, op="append", key="goal", value="第二条")
    rows = snapshot_rows(database)
    assert len(rows) == 2
    assert rows[-1].state_json == {"goal": "目标正文\n- 第二条"}


def test_each_op_materializes_one_row_that_rollback_can_revoke_together(database):
    """同轮多次 key 级写入会各落一行；撤销按 user_seq 停用，天然覆盖这一整轮。"""

    tool, _engine = make_tool(database)
    invoke(tool, user_seq=7, op="set", key="a", value="1")
    invoke(tool, user_seq=7, op="set", key="b", value="2")
    invoke(tool, user_seq=7, op="delete", key="a")

    rows = snapshot_rows(database)
    assert [row.user_seq for row in rows] == [7, 7, 7]
    assert rows[-1].state_json == {"b": "2"}
    assert len(rows) == 3


def test_validation_failure_returns_error_text_and_writes_nothing(database):
    tool, _engine = make_tool(database)

    assert invoke(tool, op="get", key="nope").startswith("no such key")
    assert invoke(tool, op="rename", key="x").startswith("Error:")
    assert invoke(tool, op="set", key="multi\nline", value="y").startswith("Error:")
    assert invoke(tool, op="set", key="x").startswith("Error:")
    assert snapshot_rows(database) == []


def test_derived_cache_follows_the_canonical_row(database):
    tool, engine = make_tool(database)
    assert engine.current_work_state("t1") is None

    invoke(tool, op="set", key="goal", value="同步到缓存")
    with database.session() as session:
        count = session.scalar(select(func.count(WorkStateSnapshot.id)))
    assert count == 1
    cached = engine.current_work_state("t1")
    assert cached is not None
    assert cached.state_json == {"goal": "同步到缓存"}
