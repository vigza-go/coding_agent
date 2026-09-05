from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Barrier, Lock, get_ident

from coding_agent.config import ContextSettings
from coding_agent.context.compaction import (
    atomic_message_units,
    message_tokens,
    split_atomic_units_token_balanced,
    split_token_balanced,
    visible_text,
)
from coding_agent.context.cover import greedy_cover
from coding_agent.context.engine import ContextEngine
from coding_agent.context.records import MemoryBlockSnapshot, MessageSnapshot
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.context.tokens import estimate_tokens
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


# 当传声筒用：LangChain 的回调就是挂在这种"当前这条执行线"的变量上往下传的，
# 用它验压缩工作线程还看不看得见外层的东西，比直接测回调更直白。
PROBE: ContextVar[str] = ContextVar("probe", default="")


def add_tool_call_messages(database, count: int) -> list[Message]:
    """带大块工具调用参数的助手消息 + 对应工具结果，不掺 thinking。

    挑这种形状有两个用处：write_file 那类几千字的参数在 transcript 里会被压成一行占位
    文字，在 JSON 包里则混在键名之间 —— 三种量法（JSON 包壳 / 模型可见 / 摘要器副本）
    在这儿彼此差得最开，断言才有区分度。另外 thinking 一条都没有，冷思维链剪裁无事可做，
    工具结果也不超过 10 条不会被剪，回读库里的原消息就能精确复算引擎看到的量。
    """

    payload = "写入的文件正文内容" * 200
    with database.session() as session:
        repo = AgentRepository(session)
        rows = []
        for index in range(count):
            rows.append(
                repo.add_message(
                    thread_id="t1",
                    user_seq=index + 1,
                    message_type=MessageType.ASSISTANT,
                    content_json={
                        "type": "ai",
                        "data": {
                            "content": [
                                {"type": "text", "text": f"我来写第 {index} 个文件"},
                                {
                                    "type": "tool_use",
                                    "id": f"call-{index}",
                                    "name": "write_file",
                                    "input": {"file_path": f"/f{index}.py", "content": payload},
                                },
                            ],
                            "tool_calls": [
                                {
                                    "name": "write_file",
                                    "args": {"file_path": f"/f{index}.py", "content": payload},
                                    "id": f"call-{index}",
                                    "type": "tool_call",
                                }
                            ],
                        },
                    },
                    langchain_message_id=f"a-{index}",
                )
            )
            rows.append(
                repo.add_message(
                    thread_id="t1",
                    user_seq=index + 1,
                    message_type=MessageType.TOOL,
                    content_json={
                        "type": "tool",
                        "data": {"content": "ok", "tool_call_id": f"call-{index}"},
                    },
                    langchain_message_id=f"t-{index}",
                )
            )
        return rows


class RecordingSummarizer:
    """把摘要器每次真正收到的文本和配额记下来，用于核对上限是按哪份文本算的。"""

    def __init__(self) -> None:
        self.l0_calls: list[tuple[str, int]] = []

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str:
        del attempt
        if level == 0:
            self.l0_calls.append((text, hard_limit))
        return "要点：一次压缩产物"


class ConcurrentTrackingSummarizer:
    def __init__(self, workers: int) -> None:
        self.barrier = Barrier(workers)
        self.lock = Lock()
        self.thread_ids: set[int] = set()
        self.probe_values: list[str] = []

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str:
        del text, hard_limit, attempt
        if level == 0:
            with self.lock:
                self.thread_ids.add(get_ident())
                self.probe_values.append(PROBE.get())
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


def test_l0_summaries_still_see_the_callers_context(database):
    """压缩跑在线程池里，但外层这条执行线上的东西必须还能看见 —— LangChain 的回调
    （界面上"正在压缩记忆 L0"就是它送的）正挂在这儿，看不见就等于全程隐身。
    """

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
    token = PROBE.set("外层")
    try:
        result = ContextEngine(database, settings, summarizer).compact_if_needed("t1")
    finally:
        PROBE.reset(token)

    assert result.l0_blocks_created == 4
    assert summarizer.probe_values == ["外层"] * 4


def test_message_tokens_measure_what_the_model_can_read():
    """尺子量的应该是「这条消息里模型能读到的字」：思考、工具调用参数、工具结果都算，包壳不算。"""

    assistant = MessageSnapshot(
        id=1,
        thread_id="t1",
        user_seq=1,
        type="assistant",
        content_json={
            "type": "ai",
            "data": {
                "content": [
                    {"type": "thinking", "thinking": "先把配额算清楚"},
                    {"type": "text", "text": "我来写文件"},
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "write_file",
                        "input": {"content": "整段文件正文"},
                    },
                ],
                # 与 tool_use 块同一批调用（库里实测 2,635/2,635 id 重合），不能数两遍。
                "tool_calls": [{"id": "c1", "name": "write_file", "args": {"content": "整段文件正文"}}],
                "response_metadata": {"usage": {"input_tokens": 99999}},
            },
        },
        langchain_message_id="ai-1",
    )
    rendered = visible_text(assistant)
    assert message_tokens(assistant) == estimate_tokens(rendered)
    assert "先把配额算清楚" in rendered  # 思考消息模型看得见
    assert "整段文件正文" in rendered and rendered.count("整段文件正文") == 1  # 参数只算一遍
    assert "99999" not in rendered and "response_metadata" not in rendered  # 包壳不是内容
    assert message_tokens(assistant) < estimate_tokens(assistant.content_json)

    tool = MessageSnapshot(
        id=2,
        thread_id="t1",
        user_seq=1,
        type="tool",
        content_json={
            "type": "tool",
            "data": {
                "content": "文件已写入",
                "tool_call_id": "c1",
                "artifact": {"note": "LangChain 侧附带物，不进模型"},
            },
        },
        langchain_message_id="tool-1",
    )
    assert "文件已写入" in visible_text(tool)
    assert "不进模型" not in visible_text(tool)


def test_l0_summary_quota_uses_the_same_yardstick_as_the_trigger(database):
    """摘要配额的分母 = 这段历史在模型上下文里占的量，跟触发线同一把尺子。

    另外两种量法都不对，这里一并排除：按 JSON 包壳量会把上限抬到比原文还长（键名、
    usage_metadata 都算钱了，工具调用参数还被数两遍）；按喂给摘要器的那份 transcript
    量又只剩真实量的零头（思考块与工具调用参数都被它抹掉了）。
    """

    rows = add_tool_call_messages(database, 8)
    settings = ContextSettings(
        total_tokens=8000,
        compression_ratio=0.10,
        working_trigger_ratio=0.30,
        recent_tail_ratio=0.20,
        l0_block_count=2,
        summary_target_ratio=0.50,
        summary_max_attempts=1,
    )
    summarizer = RecordingSummarizer()

    result = ContextEngine(database, settings, summarizer).compact_if_needed("t1")

    assert result.l0_blocks_created == 2
    assert len(summarizer.l0_calls) == 2
    with database.session() as session:
        blocks = sorted(
            AgentRepository(session).memory_blocks("t1"),
            key=lambda block: block.begin_message_id,
        )
    by_id = {row.id: MessageSnapshot.from_model(row) for row in rows}
    for block, (transcript, hard_limit) in zip(blocks, summarizer.l0_calls, strict=True):
        chunk = [
            by_id[i]
            for i in range(block.begin_message_id, block.end_message_id + 1)
            if i in by_id
        ]
        visible = sum(message_tokens(message) for message in chunk)
        envelope = sum(estimate_tokens(message.content_json) for message in chunk)
        assert hard_limit == math.floor(visible * 0.50)  # 配额 = 模型可见量的一半
        assert hard_limit > math.floor(estimate_tokens(transcript) * 0.50)  # 不是摘要器副本
        assert hard_limit < math.floor(envelope * 0.50)  # 也不是 JSON 包壳


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
