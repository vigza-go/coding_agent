from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock, get_ident

from coding_agent.config import ContextSettings
from coding_agent.context.compaction import (
    atomic_message_units,
    split_atomic_units_token_balanced,
    split_token_balanced,
)
from coding_agent.context.cover import greedy_cover
from coding_agent.context.engine import ContextEngine
from coding_agent.context.records import MemoryBlockSnapshot, MessageSnapshot
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.persistence.models import Message, MessageType
from coding_agent.persistence.repository import AgentRepository


def add_messages(database, count: int, *, chars: int = 80) -> list[Message]:
    with database.session() as session:
        repo = AgentRepository(session)
        rows = []
        for index in range(count):
            rows.append(
                repo.add_message(
                    thread_id="t1",
                    user_seq=index // 2 + 1,
                    message_type=MessageType.USER,
                    content_json={
                        "type": "human",
                        "data": {"content": str(index) + "x" * chars},
                    },
                    langchain_message_id=f"m-{index}",
                )
            )
        return rows


class ConcurrentTrackingSummarizer:
    def __init__(self, workers: int) -> None:
        self.barrier = Barrier(workers)
        self.lock = Lock()
        self.thread_ids: set[int] = set()

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str:
        del text, hard_limit, attempt
        if level == 0:
            with self.lock:
                self.thread_ids.add(get_ident())
            self.barrier.wait(timeout=3)
        return "x"


def test_token_balanced_split_preserves_order(database):
    rows = add_messages(database, 11)
    chunks = split_token_balanced([MessageSnapshot.from_model(row) for row in rows], 4)
    assert len(chunks) == 4
    assert [row.id for chunk in chunks for row in chunk] == [row.id for row in rows]
    assert all(chunks)


def test_tool_call_and_results_form_an_atomic_compaction_unit(database):
    with database.session() as session:
        repo = AgentRepository(session)
        assistant = repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.ASSISTANT,
            content_json={
                "type": "ai",
                "data": {
                    "content": "",
                    "tool_calls": [
                        {"name": "read_file", "args": {"file_path": "/a"}, "id": "call-1"}
                    ],
                },
            },
            langchain_message_id="assistant",
        )
        tool_result = repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.TOOL,
            content_json={
                "type": "tool",
                "data": {"content": "result", "tool_call_id": "call-1"},
            },
            langchain_message_id="tool",
        )
        human = repo.add_message(
            thread_id="t1",
            user_seq=2,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "next"}},
            langchain_message_id="human",
        )
    units = atomic_message_units(
        [MessageSnapshot.from_model(row) for row in [assistant, tool_result, human]]
    )
    assert [[message.id for message in unit] for unit in units] == [
        [assistant.id, tool_result.id],
        [human.id],
    ]
    chunks = split_atomic_units_token_balanced(units, 2)
    assert [[message.id for message in chunk] for chunk in chunks] == [
        [assistant.id, tool_result.id],
        [human.id],
    ]


def test_compaction_creates_l0_and_merges_oldest_pairs(database):
    add_messages(database, 12, chars=100)
    settings = ContextSettings(
        total_tokens=300,
        compression_ratio=0.10,
        working_trigger_ratio=0.50,
        recent_tail_ratio=0.20,
        l0_block_count=4,
        summary_target_ratio=0.50,
        summary_max_attempts=2,
    )
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    result = engine.compact_if_needed("t1")

    assert result.l0_blocks_created == 4
    assert result.merges_completed >= 1
    with database.session() as session:
        repo = AgentRepository(session)
        all_blocks = repo.memory_blocks("t1")
        selected_ids = {piece.block_id for piece in engine.rebuild("t1") if piece.kind == "memory"}
        selected = [block for block in all_blocks if block.id in selected_ids]
        assert len(all_blocks) > len(selected)
        assert sum(block.token_count for block in selected) <= settings.compression_limit
        assert selected == sorted(selected, key=lambda block: block.begin_message_id)

    usage = engine.usage("t1")
    assert usage.memory_levels == tuple(block.level for block in selected)
    assert usage.memory_tokens == sum(block.token_count for block in selected)
    assert usage.working_messages > 0
    assert usage.working_tokens > 0


def test_l0_chunk_summaries_run_concurrently(database):
    add_messages(database, 12, chars=100)
    summarizer = ConcurrentTrackingSummarizer(workers=4)
    settings = ContextSettings(
        total_tokens=300,
        compression_ratio=0.25,
        working_trigger_ratio=0.50,
        recent_tail_ratio=0.20,
        l0_block_count=4,
        summary_concurrency=4,
        summary_target_ratio=0.50,
        summary_max_attempts=2,
    )

    result = ContextEngine(database, settings, summarizer).compact_if_needed("t1")

    assert result.l0_blocks_created == 4
    assert len(summarizer.thread_ids) == 4


def test_greedy_cover_prefers_longest_valid_block(database):
    rows = add_messages(database, 6)
    with database.session() as session:
        repo = AgentRepository(session)
        child1 = repo.add_memory_block(
            thread_id="t1",
            text="child-1",
            begin_message_id=rows[0].id,
            end_message_id=rows[1].id,
            level=0,
            token_count=2,
        )
        child2 = repo.add_memory_block(
            thread_id="t1",
            text="child-2",
            begin_message_id=rows[2].id,
            end_message_id=rows[3].id,
            level=0,
            token_count=2,
        )
        parent = repo.add_memory_block(
            thread_id="t1",
            text="parent",
            begin_message_id=rows[0].id,
            end_message_id=rows[3].id,
            level=1,
            token_count=2,
        )
        messages = repo.active_messages("t1")
        blocks = repo.memory_blocks("t1")

    pieces = greedy_cover(
        [MessageSnapshot.from_model(row) for row in messages],
        [MemoryBlockSnapshot.from_model(block) for block in blocks],
    )
    selected = {piece.block_id for piece in pieces if piece.kind == "memory"}
    assert pieces[0].block_id == parent.id
    assert selected == {parent.id}
    assert child1.id not in selected and child2.id not in selected
    assert pieces[1].kind == "raw"


def test_context_cache_is_incremental_until_explicitly_invalidated(database):
    first = add_messages(database, 1)[0]
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())

    initial = engine.rebuild("t1")
    assert [message.id for piece in initial for message in piece.messages] == [first.id]

    with database.session() as session:
        second = AgentRepository(session).add_message(
            thread_id="t1",
            user_seq=2,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "second"}},
            langchain_message_id="second",
        )

    cached = engine.rebuild("t1")
    assert [message.id for piece in cached for message in piece.messages] == [first.id]

    engine.append_messages("t1", [second])
    appended = engine.rebuild("t1")
    assert [message.id for piece in appended for message in piece.messages] == [
        first.id,
        second.id,
    ]

    engine.invalidate("t1")
    reloaded = engine.rebuild("t1")
    assert [message.id for piece in reloaded for message in piece.messages] == [
        first.id,
        second.id,
    ]


def test_parallel_message_appends_remain_ordered_and_idempotent(database):
    first = add_messages(database, 1)[0]
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    engine.rebuild("t1")
    with database.session() as session:
        repo = AgentRepository(session)
        second = repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.TOOL,
            content_json={"type": "tool", "data": {"content": "second"}},
            langchain_message_id="parallel-second",
        )
        third = repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.TOOL,
            content_json={"type": "tool", "data": {"content": "third"}},
            langchain_message_id="parallel-third",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(engine.append_messages, "t1", [third]),
            executor.submit(engine.append_messages, "t1", [second]),
        ]
        for future in futures:
            future.result()
    engine.append_messages("t1", [third])

    pieces = engine.rebuild("t1")
    assert [message.id for piece in pieces for message in piece.messages] == [
        first.id,
        second.id,
        third.id,
    ]


def test_work_state_cache_updates_only_after_canonical_commit(database):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    assert engine.current_work_state("t1") is None

    with database.session() as session:
        snapshot = AgentRepository(session).save_work_state("t1", 1, {"step": 1})

    assert engine.current_work_state("t1") is None
    engine.update_work_state("t1", snapshot)
    cached_state = engine.current_work_state("t1")
    assert cached_state is not None
    assert cached_state.state_json == {"step": 1}
