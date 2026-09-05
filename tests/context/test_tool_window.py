from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from coding_agent.config import ContextSettings
from coding_agent.context.compaction import message_tokens
from coding_agent.context.engine import ContextEngine
from coding_agent.context.records import MessageSnapshot
from coding_agent.persistence.message_codec import decode_message_data, encode_message, message_type
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.context_projection import ContextProjectionService
from coding_agent.services.rollback import RollbackService
from coding_agent.workspace.file_undo import FileMutationRecorder


class RecordingSummarizer:
    def __init__(self, *, fail: bool = False):
        self.sources: list[str] = []
        self.fail = fail

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str:
        del hard_limit, level, attempt
        self.sources.append(text)
        if self.fail:
            raise RuntimeError("summary unavailable")
        return "summary"


def snapshot(identifier: int, message: BaseMessage) -> MessageSnapshot:
    return MessageSnapshot(
        id=identifier,
        thread_id="t1",
        user_seq=1,
        type=message_type(message),
        content_json=encode_message(message),
        langchain_message_id=message.id,
    )


def tool_batch(count: int, *, text: bool = False, prefix: str = "call") -> list[BaseMessage]:
    calls = [
        {"name": "read_file", "id": f"{prefix}-{index}", "args": {"path": f"/{index}"}}
        for index in range(count)
    ]
    content: list[str | dict[str, Any]] = (
        [{"type": "text", "text": "ordinary assistant text"}] if text else []
    )
    content.extend(
        {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
        for call in calls
    )
    assistant = AIMessage(content=content, tool_calls=calls)
    return [
        assistant,
        *[
            ToolMessage(content=f"RESULT-{i}: " + "x" * 2000, tool_call_id=f"{prefix}-{i}")
            for i in range(count)
        ],
    ]


def persist(database, messages: list[BaseMessage], *, user_seq: int = 1):
    with database.session() as session:
        repo = AgentRepository(session)
        rows = [
            repo.add_message(
                thread_id="t1",
                user_seq=user_seq,
                message_type=message_type(message),
                content_json=encode_message(message),
            )
            for message in messages
        ]
    return rows


def assert_paired(messages: list[BaseMessage]) -> None:
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
        else:
            assert not pending
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    assert call["id"] is not None
                    pending.add(call["id"])
    assert not pending


def compaction_settings(messages: list[MessageSnapshot], *, keep: int = 2):
    trimmed = ContextEngine._trim_old_tool_results(messages, keep)
    raw_tokens = sum(message_tokens(message) for message in messages)
    trimmed_tokens = sum(message_tokens(message) for message in trimmed)
    assert raw_tokens > trimmed_tokens
    # Trimming alone falls below the trigger; the already-triggered summarization still runs.
    return ContextSettings(
        total_tokens=raw_tokens + trimmed_tokens,
        recent_tool_interactions=keep,
        recent_tail_ratio=0.8,
        summary_concurrency=1,
    )


def test_window_only_replaces_old_result_content():
    messages = [snapshot(i + 1, message) for i, message in enumerate(tool_batch(12, text=True))]
    original = deepcopy(messages)

    projected = ContextEngine._trim_old_tool_results(messages, keep=10)

    assert messages == original
    assert len(projected) == len(messages)
    assert [row.id for row in projected] == [row.id for row in messages]
    assert projected[0] == messages[0]  # Both normalized calls and tool_use blocks stay intact.
    assert projected[3:] == messages[3:]
    for row in projected[1:3]:
        assert "剪裁" in row.content_json["data"]["content"]
    assert "RESULT-0:" not in str(projected)
    assert_paired([decode_message_data(row.content_json) for row in projected])


def test_structured_result_keeps_all_fields_except_content():
    message = snapshot(
        1,
        ToolMessage(
            id="tool-id",
            name="read_file",
            tool_call_id="call-id",
            content=[{"type": "text", "text": "original error"}],
            status="error",
            artifact={"path": "/output.txt"},
            response_metadata={"duration": 3},
        ),
    )
    original = deepcopy(message)
    projected = ContextEngine._trim_old_tool_results([message], keep=0)[0]
    expected = deepcopy(original)
    expected.content_json["data"]["content"] = projected.content_json["data"]["content"]
    assert "剪裁" in projected.content_json["data"]["content"]
    assert projected == expected
    assert message == original


def test_window_uses_result_order_not_assistant_call_order():
    batch = tool_batch(3)
    messages = [snapshot(i + 1, message) for i, message in enumerate([batch[0], *batch[:0:-1]])]

    projected = ContextEngine._trim_old_tool_results(messages, keep=1)

    assert projected[0] == messages[0]
    assert projected[-1] == messages[-1]
    assert projected[-1].content_json["data"]["tool_call_id"] == "call-0"
    assert all("剪裁" in row.content_json["data"]["content"] for row in projected[1:-1])


@pytest.mark.parametrize("keep", [0, 1, 10])
def test_window_preserves_all_messages_and_handles_small_histories(keep):
    messages = [
        snapshot(i + 1, message)
        for i, message in enumerate(
            [HumanMessage(content="task"), *tool_batch(1), AIMessage(content="finished")]
        )
    ]
    projected = ContextEngine._trim_old_tool_results(messages, keep=keep)
    assert len(projected) == len(messages)
    assert projected[:2] == messages[:2]
    assert projected[-1] == messages[-1]
    if keep == 0:
        assert "剪裁" in projected[2].content_json["data"]["content"]
    else:
        assert projected == messages
    assert ContextEngine._trim_old_tool_results([], keep=keep) == []


def test_normal_projection_does_not_roll_the_tool_window(database):
    rows = persist(database, [HumanMessage(content="task"), *tool_batch(12)])
    engine = ContextEngine(database, ContextSettings(), RecordingSummarizer())
    projection = ContextProjectionService(engine)
    before = projection.build("t1")[1:]
    assert sum(isinstance(message, ToolMessage) for message in before) == 12
    assert all(
        str(message.content).startswith("RESULT-")
        for message in before
        if isinstance(message, ToolMessage)
    )

    appended = persist(database, [HumanMessage(content="next")], user_seq=2)
    engine.append_messages("t1", appended)
    after = projection.build("t1")[1:]
    assert after[:-1] == before
    assert len(after) == len(rows) + 1


def sequential_history() -> list[BaseMessage]:
    messages: list[BaseMessage] = [HumanMessage(content="task")]
    for i in range(12):
        messages.extend(
            [
                AIMessage(
                    content=f"step {i}",
                    tool_calls=[{"id": f"call-{i}", "name": "read_file", "args": {}}],
                ),
                ToolMessage(content=f"RESULT-{i}: " + "x" * 2000, tool_call_id=f"call-{i}"),
            ]
        )
    return messages


def test_compaction_keeps_trimmed_tail_in_cache_without_sliding_on_append(database, tmp_path):
    rows = persist(database, sequential_history())
    snapshots = [MessageSnapshot.from_model(row) for row in rows]
    original = deepcopy(snapshots)
    settings = compaction_settings(snapshots)
    summarizer = RecordingSummarizer()
    engine = ContextEngine(database, settings, summarizer)

    result = engine.compact_if_needed("t1")

    assert result.l0_blocks_created > 0
    assert summarizer.sources
    assert all("RESULT-0:" not in source for source in summarizer.sources)
    before = engine.rebuild("t1")
    cached_tail = [row for piece in before for row in piece.messages]
    assert any(
        row.type == "tool" and "剪裁" in row.content_json["data"]["content"] for row in cached_tail
    )
    reloaded = ContextEngine(database, settings, RecordingSummarizer())
    restored = reloaded.rebuild("t1")
    assert [piece for piece in restored if piece.kind == "memory"] == [
        piece for piece in before if piece.kind == "memory"
    ]
    restored_tail = [row for piece in restored for row in piece.messages]
    assert [row.id for row in restored_tail] == [row.id for row in cached_tail]
    assert sum(map(message_tokens, restored_tail)) > sum(map(message_tokens, cached_tail))
    assert engine.usage("t1").working_tokens == sum(map(message_tokens, cached_tail))
    assert engine.usage("t1").working_messages == len(cached_tail)
    assert_paired(ContextProjectionService.render_pieces(restored))
    with database.session() as session:
        repo = AgentRepository(session)
        assert [MessageSnapshot.from_model(row) for row in repo.active_messages("t1")] == original
        assert len(repo.memory_blocks("t1")) == result.l0_blocks_created

    before_messages = ContextProjectionService(engine).build("t1")[1:]
    assert_paired(before_messages)
    appended = persist(
        database,
        [
            AIMessage(content="", tool_calls=[{"id": "new", "name": "ls", "args": {}}]),
            ToolMessage(content="new result", tool_call_id="new"),
        ],
        user_seq=2,
    )
    engine.append_messages("t1", appended)
    after_messages = ContextProjectionService(engine).build("t1")[1:]
    assert after_messages[:-2] == before_messages
    assert_paired(after_messages)
    full_results = [
        message
        for message in after_messages
        if isinstance(message, ToolMessage) and "剪裁" not in str(message.content)
    ]
    assert len(full_results) == 3  # The previous two results plus the appended result.

    RollbackService(database, engine, FileMutationRecorder(database, tmp_path)).rollback("t1", 2)
    assert engine.rebuild("t1") == restored  # Any cache rebuild restores the original raw tail.


def test_memory_ranges_and_tool_pairs_survive_compaction_and_reload(database):
    messages: list[BaseMessage] = []
    for index in range(12):
        messages.extend(
            [
                AIMessage(content="", tool_calls=[{"id": f"c{index}", "name": "ls", "args": {}}]),
                ToolMessage(content="x" * 2000, tool_call_id=f"c{index}"),
                HumanMessage(content=f"note {index}"),
            ]
        )
    rows = persist(database, messages)
    settings = compaction_settings([MessageSnapshot.from_model(row) for row in rows])
    engine = ContextEngine(database, settings, RecordingSummarizer())

    assert engine.compact_if_needed("t1").l0_blocks_created > 0

    before = engine.rebuild("t1")
    assert before[0].kind == "memory"
    assert before[0].begin_message_id == rows[0].id
    restored = ContextEngine(database, settings, RecordingSummarizer()).rebuild("t1")
    assert [piece for piece in restored if piece.kind == "memory"] == [
        piece for piece in before if piece.kind == "memory"
    ]
    assert_paired(ContextProjectionService.render_pieces(before))
    assert_paired(ContextProjectionService.render_pieces(restored))


def test_rollback_restores_original_uncompressed_tool_content(database, tmp_path):
    rows = persist(database, tool_batch(12))
    rows.extend(persist(database, [HumanMessage(content="trigger")], user_seq=2))
    snapshots = [MessageSnapshot.from_model(row) for row in rows]
    settings = compaction_settings(snapshots)
    engine = ContextEngine(database, settings, RecordingSummarizer())
    engine.compact_if_needed("t1")
    trimmed = [
        row for piece in engine.rebuild("t1") for row in piece.messages if row.type == "tool"
    ]
    assert len(trimmed) == 12
    assert sum("剪裁" in row.content_json["data"]["content"] for row in trimmed) == 10

    RollbackService(database, engine, FileMutationRecorder(database, tmp_path)).rollback("t1", 2)

    restored = [row for piece in engine.rebuild("t1") for row in piece.messages]
    assert sum(row.type == "tool" for row in restored) == 12
    assert len(restored[0].content_json["data"]["tool_calls"]) == 12
    assert restored == snapshots[:-1]


def test_failed_summary_does_not_change_cached_messages(database):
    rows = persist(database, sequential_history())
    snapshots = [MessageSnapshot.from_model(row) for row in rows]
    engine = ContextEngine(database, compaction_settings(snapshots), RecordingSummarizer(fail=True))
    before = engine.rebuild("t1")

    with pytest.raises(RuntimeError, match="summary unavailable"):
        engine.compact_if_needed("t1")

    assert engine.rebuild("t1") == before
    with database.session() as session:
        repo = AgentRepository(session)
        assert repo.memory_blocks("t1") == []


def test_summary_failure_keeps_the_cold_cot_trim_in_the_cache(database):
    """摘要失败不写 memory block，但越线那一次的冷 CoT 剪裁**留在缓存里不回退**。

    上一条测试的 fixture 没有任何 reasoning 块，所以它盖不住这条路径；这里补上。
    不回退是刻意的：剪 CoT 不花钱也不丢记忆，回滚只会让下一轮在同一位置重剪一次。
    """

    with database.session() as session:
        repo = AgentRepository(session)
        for index in range(1, 9):
            repo.add_message(
                thread_id="t1",
                user_seq=1,
                message_type="assistant",
                content_json={
                    "type": "ai",
                    "data": {
                        "content": [
                            {"type": "thinking", "thinking": "思" * 300, "signature": "sig"},
                            {"type": "text", "text": f"turn-{index} " + "x" * 40},
                        ]
                    },
                },
                langchain_message_id=f"a{index}",
            )

    settings = ContextSettings(
        total_tokens=200,
        working_trigger_ratio=0.5,
        reasoning_retain_ratio=0.15,
        recent_tail_ratio=0.2,
        l0_block_count=2,
        summary_concurrency=1,
    )

    def visible_reasoning(engine: ContextEngine) -> int:
        return sum(
            1
            for piece in engine.rebuild("t1")
            for row in piece.messages or ()
            for block in (row.content_json.get("data") or {}).get("content") or []
            if isinstance(block, dict) and block.get("type") == "thinking"
        )

    engine = ContextEngine(database, settings, RecordingSummarizer(fail=True))
    assert visible_reasoning(engine) == 8

    with pytest.raises(RuntimeError, match="summary unavailable"):
        engine.compact_if_needed("t1")

    assert visible_reasoning(engine) == 1  # 最老的那些不回读，只保住最新一段
    with database.session() as session:
        assert AgentRepository(session).memory_blocks("t1") == []
    # 库里仍是原文：缓存失效重载后 8 段全部回来。
    assert visible_reasoning(ContextEngine(database, settings, RecordingSummarizer())) == 8


def test_next_compaction_trims_new_old_results(database):
    rows = persist(database, tool_batch(12))
    settings = compaction_settings([MessageSnapshot.from_model(row) for row in rows])
    engine = ContextEngine(database, settings, RecordingSummarizer())
    engine.compact_if_needed("t1")

    rows = persist(database, tool_batch(12, prefix="new"), user_seq=2)
    engine.append_messages("t1", rows)
    assert engine.usage("t1").working_tokens > settings.working_trigger
    engine.compact_if_needed("t1")
    messages = ContextProjectionService.render_pieces(engine.rebuild("t1"))
    results = [message for message in messages if isinstance(message, ToolMessage)]
    # The 80% tail retains both atomic batches; only the newest two bodies stay complete.
    assert len(results) == 24
    assert sum("剪裁" in str(message.content) for message in results) == 22
    assert [message.tool_call_id for message in results[-2:]] == ["new-10", "new-11"]
    assert_paired(messages)


@pytest.mark.parametrize("keep", [0, 2])
def test_no_splittable_prefix_only_updates_cached_content(database, keep):
    rows = persist(database, tool_batch(12))
    snapshots = [MessageSnapshot.from_model(row) for row in rows]
    settings = compaction_settings(snapshots, keep=keep)
    summarizer = RecordingSummarizer()
    engine = ContextEngine(database, settings, summarizer)

    result = engine.compact_if_needed("t1")

    assert result.l0_blocks_created == 0
    assert not summarizer.sources
    trimmed = [row for piece in engine.rebuild("t1") for row in piece.messages]
    assert trimmed == ContextEngine._trim_old_tool_results(snapshots, keep=keep)
    assert len(trimmed) == len(snapshots)
    restored = ContextEngine(database, settings, RecordingSummarizer()).rebuild("t1")
    assert [row for piece in restored for row in piece.messages] == snapshots
    assert_paired(ContextProjectionService.render_pieces(restored))


def test_negative_tool_window_is_rejected():
    with pytest.raises(ValueError, match="recent_tool_interactions"):
        ContextSettings(recent_tool_interactions=-1)
