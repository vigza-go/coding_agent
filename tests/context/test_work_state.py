from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.context.work_state import (
    MAX_KEY_CHARS,
    WorkStateError,
    apply_op,
    flatten,
    ordered,
    render,
)
from coding_agent.persistence.models import MessageType
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.context_projection import ContextProjectionService


def make_engine(database, **overrides) -> ContextEngine:
    return ContextEngine(database, ContextSettings(**overrides), DeterministicSummarizer())


def save_state(database, thread_id: str, user_seq: int, state: dict) -> int:
    with database.session() as session:
        saved = AgentRepository(session).save_work_state(thread_id, user_seq, state)
    return int(saved.id)


def add_messages(database, thread_id: str, count: int, *, chars: int = 400, offset: int = 0):
    with database.session() as session:
        repo = AgentRepository(session)
        rows = [
            repo.add_message(
                thread_id=thread_id,
                user_seq=offset + index + 1,
                message_type=MessageType.USER,
                content_json={"type": "human", "data": {"content": str(index) + "x" * chars}},
                langchain_message_id=f"{thread_id}-m-{offset + index}",
            )
            for index in range(count)
        ]
    return rows


def pin_text(messages) -> str:
    return "\n".join(
        message.content
        for message in messages
        if getattr(message, "name", None) == "work_state"
    )


# 30 条 400 字的消息约 3000 token：触发线 1000 撑得到，压完尾部 ~600 又稳稳落回线下，
# 所以第二次 build 不会再触发——"便签冻住、投影逐字节不变"才测得出来。
COMPACT = {
    "total_tokens": 2000,
    "working_trigger_ratio": 0.5,
    "recent_tail_ratio": 0.2,
    "l0_block_count": 2,
    "summary_concurrency": 1,
}


# --------------------------------------------------------------------------- 字典操作


def test_set_then_get_touches_only_one_key():
    state, receipt = apply_op({"goal": "旧目标"}, "set", "goal", "新目标")
    assert receipt.startswith("set goal")
    assert state == {"goal": "新目标"}

    _, body = apply_op(state, "get", "goal")
    assert body == "## goal\n新目标"


def test_append_adds_a_bullet_without_rewriting_the_key():
    state, receipt = apply_op({"next": "- 甲"}, "append", "next", "乙")
    assert receipt.startswith("appended to next")
    assert state["next"] == "- 甲\n- 乙"


def test_append_creates_a_missing_key_instead_of_failing():
    state, receipt = apply_op({}, "append", "notes", "第一条")
    assert "created notes" in receipt
    assert state == {"notes": "第一条"}


def test_read_only_ops_return_the_state_unchanged():
    original = {"a": "1", "b": "2"}
    for op in ("list", "get"):
        state, receipt = apply_op(original, op, "a")
        assert state == original
        assert receipt
    assert apply_op(original, "list")[0] == original


def test_delete_removes_the_key_and_clear_empties_everything():
    state, _ = apply_op({"a": "1", "b": "2"}, "delete", "a")
    assert state == {"b": "2"}
    state, _ = apply_op({"b": "2"}, "delete", "absent")
    assert state == {"b": "2"}
    assert apply_op({"b": "2"}, "clear")[0] == {}


def test_unknown_op_and_bad_keys_are_rejected_without_a_write():
    with pytest.raises(WorkStateError, match="unknown op"):
        apply_op({}, "rename", "a")
    with pytest.raises(WorkStateError, match="non-empty key"):
        apply_op({}, "set", "  ", "value")
    with pytest.raises(WorkStateError, match="single line"):
        apply_op({}, "set", "two\nlines", "value")
    with pytest.raises(WorkStateError, match="longer than"):
        apply_op({}, "set", "k" * (MAX_KEY_CHARS + 1), "value")
    with pytest.raises(WorkStateError, match="value"):
        apply_op({}, "set", "key")


def test_nested_values_are_flattened_not_stored_as_structure():
    with pytest.raises(WorkStateError, match="markdown strings"):
        apply_op({}, "set", "key", {"nested": True})  # type: ignore[arg-type]
    assert flatten({"n": 1, "deep": {"k": ["v"]}}) == {
        "n": "1",
        "deep": '{"k": ["v"]}',
    }


def test_get_of_a_missing_key_reports_available_keys():
    _, receipt = apply_op({"goal": "x"}, "get", "nope")
    assert "no such key" in receipt and "goal" in receipt


def test_ordered_is_independent_of_insertion_order():
    """MySQL 的 JSON 列会规范化键序，所以渲染顺序必须由我们自己定死（按键名排序）。"""

    state = {"zz": "3", "aaa": "9", "mmm": "1"}
    assert [key for key, _ in ordered(state)] == ["aaa", "mmm", "zz"]
    assert render({"b": "2", "a": "1"}) == "## a\n1\n\n## b\n2"


# --------------------------------------------------------------------------- 投影 / 便签


def test_work_state_is_not_injected_every_turn(database):
    """没剪裁过的会话，工作状态不自动出现在上下文里（去掉了每轮无条件注入）。"""

    save_state(database, "t1", 1, {"goal": "不该每轮出现的正文"})
    engine = make_engine(database)
    messages = ContextProjectionService(engine).build("t1")
    assert not any(getattr(message, "name", None) == "work_state" for message in messages)


def test_pin_appears_only_after_trim_and_carries_the_latest_state(database):
    """撑过触发线、发生剪裁/压缩时，才把最新一版贴成便签，且用 HumanMessage 承载。"""

    add_messages(database, "t1", 30)
    save_state(database, "t1", 1, {"goal": "跨轮要记住的目标正文"})
    engine = make_engine(database, **COMPACT)
    projection = ContextProjectionService(engine)
    messages = projection.build("t1")

    pins = [message for message in messages if getattr(message, "name", None) == "work_state"]
    assert pins, "剪裁之后应当出现便签"
    assert "跨轮要记住的目标正文" in pins[0].content
    assert pins[0].content.startswith("<current_work_state>")
    assert isinstance(pins[0], HumanMessage), "便签必须是 HumanMessage，不能是 SystemMessage"

    # 位置：压缩块之后、原文之前。
    assert [piece.kind for piece in engine.rebuild("t1")] == ["memory", "work_state", "raw"]

    # 便签只在剪裁那一下换新：压完已经落回线下，下一轮必须逐字节不动。
    frozen = [message.content for message in messages]
    assert [message.content for message in projection.build("t1")] == frozen


def test_pin_is_a_view_and_never_reaches_storage(database):
    """便签是渲染时贴上的派生视图：不落库、不进 messages 表、不进 memory_blocks。"""

    add_messages(database, "t1", 30)
    save_state(database, "t1", 1, {"goal": "只该活在投影里的正文"})
    engine = make_engine(database, **COMPACT)
    messages = ContextProjectionService(engine).build("t1")
    assert any(getattr(message, "name", None) == "work_state" for message in messages), (
        "先确认便签确实出现了，否则这条断言是空的"
    )

    with database.session() as session:
        repo = AgentRepository(session)
        stored = " ".join(
            json.dumps(row.content_json, ensure_ascii=False) for row in repo.active_messages("t1")
        ) + " ".join(block.text for block in repo.memory_blocks("t1"))
    assert "只该活在投影里的正文" not in stored


def test_projection_is_byte_stable_when_nothing_trims(database):
    """两次 build 之间没有任何剪裁 → 投影逐字节不变（缓存前缀全员命中）。"""

    add_messages(database, "t1", 5)
    save_state(database, "t1", 1, {"goal": "目标"})
    engine = make_engine(database)  # 默认触发线 50 万，这几条消息撑不到
    projection = ContextProjectionService(engine)

    first = projection.build("t1")
    second = projection.build("t1")
    assert [message.content for message in first] == [message.content for message in second]
    assert not any(getattr(message, "name", None) == "work_state" for message in second)


def test_pin_recovers_on_cold_load_when_history_was_compressed(database):
    """重启/缓存失效后，只要历史上压过块且 work_state 有内容，便签要自己长回来。"""

    add_messages(database, "t1", 30)
    save_state(database, "t1", 1, {"goal": "冷启动也要有的正文"})
    engine = make_engine(database, **COMPACT)
    ContextProjectionService(engine).build("t1")  # 压出块、落下便签

    engine.invalidate("t1")  # 模拟重启：缓存清空，只能从库里重建
    messages = ContextProjectionService(engine).build("t1")
    assert "冷启动也要有的正文" in pin_text(messages)


def test_pin_refreshes_at_the_next_trim_not_on_every_write(database):
    """写完一版不算数：便签要等到下一次剪裁/压缩才换新（压缩那条路也算）。

    每次写就换，等于模型一调 work_state 工具就断一次前缀——便签在投影中段，改它等于把
    它后面整段原文的前缀也废掉。
    """

    add_messages(database, "t1", 30)
    engine = make_engine(database, **COMPACT)
    projection = ContextProjectionService(engine)
    engine.mutate_work_state("t1", 1, "set", "goal", "第一版")

    assert "第一版" in pin_text(projection.build("t1")), "越线剪裁那一下把便签贴上"

    engine.mutate_work_state("t1", 2, "set", "goal", "第二版")
    stale = pin_text(projection.build("t1"))
    assert "第一版" in stale and "第二版" not in stale, "没剪裁，便签必须冻着"

    engine.append_messages("t1", add_messages(database, "t1", 12, offset=30))  # 再顶过线
    refreshed = ContextProjectionService(engine).build("t1")
    assert "第二版" in pin_text(refreshed), "下一次剪裁/压缩时，便签换成最新一版"


def test_pin_is_not_refreshed_when_the_pass_changed_nothing(database):
    """越线了，但什么都没改动（没冷草稿、没旧工具结果、也切不出块）——便签不该换新。

    否则这条路上每次都要白断一次前缀，等于把便签降级成"写完就刷"。
    """

    add_messages(database, "t1", 30)
    save_state(database, "t1", 1, {"goal": "第一版"})
    ContextProjectionService(make_engine(database, **COMPACT)).build("t1")  # 压出块 + 便签

    # 换个"尾部比例接近 1"的引擎：同样越线，但切不出块、也没有可剪的东西。
    engine = make_engine(
        database,
        total_tokens=2000,
        working_trigger_ratio=0.5,
        recent_tail_ratio=0.99,
        summary_concurrency=1,
    )
    projection = ContextProjectionService(engine)
    assert "第一版" in pin_text(projection.build("t1")), "冷启动先把便签长回来"

    engine.mutate_work_state("t1", 2, "set", "goal", "第二版")
    assert "第一版" in pin_text(projection.build("t1")), "白过一遍不得动便签"


# --------------------------------------------------------------------- 便签：计划 + 工作状态


PLAN = [
    {"id": "1", "title": "读 DESIGN.md", "status": "completed"},
    {"id": "2", "title": "写 todo.py", "status": "in_progress"},
    {"id": "3", "title": "补测试", "status": "pending"},
]


def save_todos(database, thread_id: str, user_seq: int, items: list[dict]) -> int:
    with database.session() as session:
        saved = AgentRepository(session).save_todos(thread_id, user_seq, items)
    return int(saved.id)


def test_pin_puts_the_plan_before_the_work_state(database):
    add_messages(database, "t1", 30)
    save_state(database, "t1", 1, {"goal": "把计划机制做出来"})
    save_todos(database, "t1", 1, PLAN)

    text = pin_text(ContextProjectionService(make_engine(database, **COMPACT)).build("t1"))

    assert "- [~] 2 写 todo.py" in text and "- [ ] 3 补测试" in text, "勾选清单要看得懂"
    assert text.index("写 todo.py") < text.index("把计划机制做出来"), "计划在前，细节在后"


def test_a_thread_with_only_a_plan_still_gets_a_pin(database):
    """工具消息会被工具窗口换成占位符，计划必须由便签接住——哪怕没有 work_state。"""

    add_messages(database, "t1", 30)
    save_todos(database, "t1", 1, PLAN)

    text = pin_text(ContextProjectionService(make_engine(database, **COMPACT)).build("t1"))

    assert "补测试" in text
    assert "# 工作状态" not in text, "没有工作状态就别留一个空段"


def test_no_pin_at_all_when_there_is_neither_plan_nor_state(database):
    add_messages(database, "t1", 30)

    messages = ContextProjectionService(make_engine(database, **COMPACT)).build("t1")

    assert pin_text(messages) == "", "两样都空就不贴空标签"


def test_the_plan_reaches_the_pin_at_the_next_trim_and_survives_a_cold_start(database):
    """写完不算数（模型在尾部工具消息里看新的），剪裁那一下才换；重启后自己长回来。"""

    add_messages(database, "t1", 30)
    save_todos(database, "t1", 1, [{"id": "1", "title": "第一版", "status": "in_progress"}])
    engine = make_engine(database, **COMPACT)
    projection = ContextProjectionService(engine)
    assert "第一版" in pin_text(projection.build("t1"))

    engine.mutate_todos("t1", 2, [{"id": "1", "title": "第二版", "status": "in_progress"}])
    assert "第一版" in pin_text(projection.build("t1")), "没再剪裁就不换便签"

    engine.append_messages("t1", add_messages(database, "t1", 12, offset=30))
    assert "第二版" in pin_text(projection.build("t1")), "下次剪裁时换成最新一版计划"

    engine.invalidate("t1")  # 重启/缓存失效
    assert "第二版" in pin_text(projection.build("t1")), "冷启动按最新快照重新渲染便签"
