from __future__ import annotations

import pytest

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
    render_index,
)
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.context_projection import ContextProjectionService


def save_state(database, thread_id: str, user_seq: int, state: dict) -> int:
    """写一版工作状态并同步派生缓存，返回快照 id（与工具落库路径一致）。"""

    with database.session() as session:
        saved = AgentRepository(session).save_work_state(thread_id, user_seq, state)
    ContextEngine(database, ContextSettings(), DeterministicSummarizer()).update_work_state(
        thread_id, saved
    )
    return int(saved.id)


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


def test_ordered_pins_first_and_is_independent_of_insertion_order():
    """MySQL 的 JSON 列会规范化键序，所以渲染顺序必须由我们自己定死。"""

    state = {"zz": "3", "!vetoed": "0", "aaa": "9", "!pinned": "keep"}
    assert [key for key, _ in ordered(state)] == ["!pinned", "!vetoed", "aaa", "zz"]


def test_render_index_drops_bodies_but_keeps_pinned_ones():
    state = {"goal": "很长的正文内容", "!vetoed": "不得重提"}
    body = render(state)
    index = render_index(state)
    assert "很长的正文内容" in body and "很长的正文内容" not in index
    assert "不得重提" in index and "goal" in index


def test_projection_repeats_index_until_the_state_is_rewritten(database):
    """内容没变就只重发索引；重写一版之后必须重新给全文（钉住的键任何时候都给正文）。"""

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    projection = ContextProjectionService(engine)
    snapshot_id = save_state(
        database, "t1", 1, {"goal": "唯一一份目标正文", "!vetoed": "不得重提的否决"}
    )
    engine.update_work_state("t1", _reload(database, "t1", snapshot_id))

    first = projection.build("t1")[-1].content
    assert "唯一一份目标正文" in first

    second = projection.build("t1")[-1].content
    assert "唯一一份目标正文" not in second, "未变的内容不应重发全文"
    assert "不得重提的否决" in second, "钉住的键必须常驻"
    assert "goal" in second, "键名要留在索引里"

    assert not second.startswith("work state unchanged"), "同一快照的第三次仍是索引"
    third = projection.build("t1")[-1].content
    assert third == second

    new_id = save_state(database, "t1", 2, {"goal": "改写后的目标正文"})
    engine.update_work_state("t1", _reload(database, "t1", new_id))
    fourth = projection.build("t1")[-1].content
    assert "改写后的目标正文" in fourth


def _reload(database, thread_id: str, snapshot_id: int):
    from sqlalchemy import select

    from coding_agent.persistence.models import WorkStateSnapshot

    with database.session() as session:
        row = session.scalars(
            select(WorkStateSnapshot).where(WorkStateSnapshot.id == snapshot_id)
        ).one()
        session.expunge(row)
    return row


def test_rollback_to_an_older_snapshot_gets_the_full_text_again(database):
    """撤销后 latest 退回更早的 id：必须重新给全文，不能因为"发过"而只给索引。

    判据是"和上一轮发出去的 id 相同"，而不是"这个 id 曾经发过"，所以方向是安全的。
    """

    from sqlalchemy import update

    from coding_agent.persistence.models import WorkStateSnapshot

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    projection = ContextProjectionService(engine)

    first = save_state(database, "t1", 1, {"goal": "第一轮的目标正文"})
    engine.update_work_state("t1", _reload(database, "t1", first))
    assert "第一轮的目标正文" in projection.build("t1")[-1].content
    assert "第一轮的目标正文" not in projection.build("t1")[-1].content

    second = save_state(database, "t1", 2, {"goal": "第二轮的目标正文"})
    engine.update_work_state("t1", _reload(database, "t1", second))
    assert "第二轮的目标正文" in projection.build("t1")[-1].content

    with database.session() as session:
        session.execute(
            update(WorkStateSnapshot)
            .where(WorkStateSnapshot.id == second)
            .values(active=False)
        )
    engine.invalidate("t1")

    body = projection.build("t1")[-1].content
    assert "第一轮的目标正文" in body, "回滚到更早快照必须重新给全文"
