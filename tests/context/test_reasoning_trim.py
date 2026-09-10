"""Cold chain-of-thought is shed at the wall, never re-read, and never rewritten to storage."""

from __future__ import annotations

import json

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.records import MessageSnapshot
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.persistence.models import MessageType
from coding_agent.persistence.repository import AgentRepository

TRAILER = "[较早的模型思维链已剪裁，不再回读。]"


def reasoning(index: int, chars: int = 40) -> MessageSnapshot:
    return MessageSnapshot(
        id=index,
        thread_id="t1",
        user_seq=index,
        type="assistant",
        content_json={
            "type": "ai",
            "data": {
                "content": [
                    {"type": "thinking", "thinking": "思" * chars, "signature": "sig"},
                    {"type": "text", "text": f"turn-{index} " + "x" * 20},
                ],
            },
        },
        langchain_message_id=f"a-{index}",
    )


def blocks(snapshot: MessageSnapshot) -> list[dict]:
    return (snapshot.content_json.get("data") or {}).get("content") or []


def kinds(snapshot: MessageSnapshot) -> list[str]:
    return [str(block.get("type")) for block in blocks(snapshot)]


def trimmed(messages) -> set[int]:
    return {
        message.id
        for message in messages
        if any(
            block.get("type") == "text" and block.get("text") == TRAILER for block in blocks(message)
        )
    }


def retain(messages, budget):
    return ContextEngine._retain_reasoning_within_budget(messages, budget)


def test_nothing_is_trimmed_while_reasoning_fits_the_budget() -> None:
    messages = [reasoning(index) for index in range(1, 4)]
    assert retain(messages, budget=10_000) == messages
    assert trimmed(messages) == set()


def test_coldest_reasoning_goes_first_and_only_the_newest_survives() -> None:
    messages = [reasoning(index) for index in range(1, 11)]
    result = retain(messages, budget=60)
    survivors = {message.id for message in result if "thinking" in kinds(message)}
    assert survivors, "至少最新一段思维链必须留下"
    assert max(survivors) == 10
    assert survivors == set(range(min(survivors), 11)), "保留的必须是最新的一段连续区间"
    assert len(survivors) < 10, "超预算的冷草稿必须被剪"
    assert trimmed(result) == set(range(1, min(survivors))), "剪的必须是 survivors 之外那一段"
    looser = retain(messages, budget=600)
    assert len(trimmed(looser)) < len(trimmed(result)), "预算越宽，剪得越少"


def plain(index: int) -> MessageSnapshot:
    return MessageSnapshot(
        id=index,
        thread_id="t1",
        user_seq=index,
        type="assistant",
        content_json={"type": "ai", "data": {"content": [{"type": "text", "text": f"turn-{index}"}]}},
        langchain_message_id=f"a-{index}",
    )


def test_trimming_never_drops_reorders_or_leaks_none() -> None:
    """预算是"少投影一段草稿"，不是"少一条消息"——工作区条数是不可变契约。"""

    mixed = [
        reasoning(index) if index % 2 else plain(index) for index in range(1, 13)
    ]
    for budget in (0, 60, 120, 600, 10_000):
        result = retain(mixed, budget)
        assert len(result) == len(mixed), "条数绝不能变"
        assert [message.id for message in result] == [message.id for message in mixed], "顺序绝不能变"
        assert all(isinstance(message, MessageSnapshot) for message in result), "None 只是内部占位"
        assert all(str(blocks(message)) for message in result), "没有一条被剪成空"
        assert trimmed(result) <= {index for index in range(1, 13) if index % 2}, "只能剪带草稿的那些"

    cut = trimmed(retain(mixed, budget=60))
    assert cut and max(cut) < 12, "剪的是前缀，最新一条不受影响"


def test_the_newest_segment_survives_even_when_it_alone_breaches_the_budget() -> None:
    messages = [reasoning(index, chars=600) for index in range(1, 5)]
    result = retain(messages, budget=10)
    assert "thinking" in kinds(result[-1]), "当前这一轮绝不能丢自己刚写完的草稿"
    assert trimmed(result) == {1, 2, 3}


def test_trimming_leaves_visible_text_and_tool_calls_untouched() -> None:
    tool_use = {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"command": "ls"}}
    paired = MessageSnapshot(
        id=1,
        thread_id="t1",
        user_seq=1,
        type="assistant",
        content_json={
            "type": "ai",
            "data": {
                "content": [*blocks(reasoning(1)), tool_use],
                "tool_calls": [{"id": "call_1"}],
            },
        },
        langchain_message_id="a-1",
    )
    result = retain([paired, reasoning(2)], budget=0)
    assert trimmed(result) == {1}
    assert tool_use in blocks(result[0]), "tool_use 块必须原样留下，否则与 tool_result 的配对会断"
    assert result[0].content_json["data"]["tool_calls"] == [{"id": "call_1"}]
    assert f"turn-1 {'x' * 20}" in str(blocks(result[0])), "可见发言不能跟着草稿一起丢"
    assert kinds(result[0]).count("text") == 2, "一条消息只补一个标记"


def test_trimming_is_idempotent_and_only_ever_moves_forward() -> None:
    messages = [reasoning(index) for index in range(1, 11)]
    once = retain(messages, budget=120)
    assert retain(once, budget=120) == once, "对已剪结果再剪一次必须完全不动"

    grown = retain([*messages, reasoning(11, chars=400), reasoning(12, chars=400)], budget=120)
    lost = trimmed(once)
    assert lost and lost <= trimmed(grown), "剪掉的不能又长回来，否则每轮都在重写中段前缀"
    for index, snapshot in enumerate(once):
        if snapshot.id in lost:
            assert blocks(snapshot) == blocks(grown[index]), "已剪内容的必须逐字节稳定"


def _settings() -> ContextSettings:
    return ContextSettings(
        total_tokens=1_000,
        compression_ratio=0.10,
        working_trigger_ratio=0.50,
        recent_tail_ratio=0.20,
        l0_block_count=2,
        summary_target_ratio=0.50,
        summary_max_attempts=2,
        reasoning_retain_ratio=0.10,
    )


def test_trim_runs_only_at_the_wall_and_spares_the_summarizer(database) -> None:
    """冷草稿占了大头：剪完明显落到余量线以下，这一次摘要就该省下来。

    最新一轮的草稿按规矩一定留着（一轮对话不能丢掉"这次调用是怎么想出来的"），所以让它
    很小，冷的那几段给足体积——剪裁才有得赚。
    """

    settings = _settings()
    thinking_chars = [70] * 7 + [2]
    with database.session() as session:
        repo = AgentRepository(session)
        for index in range(1, 9):
            repo.add_message(
                thread_id="t1",
                user_seq=index,
                message_type=MessageType.ASSISTANT,
                content_json=reasoning(index, chars=thinking_chars[index - 1]).content_json,
                langchain_message_id=f"a-{index}",
            )

    engine = ContextEngine(database, settings, DeterministicSummarizer())
    assert engine.usage("t1").working_tokens > settings.working_trigger

    result = engine.compact_if_needed("t1")
    assert result.l0_blocks_created == 0, "剪掉冷草稿就剪出了余量，不该动用摘要器"
    assert result.merges_completed == 0
    with database.session() as session:
        assert AgentRepository(session).memory_blocks("t1") == [], "没有摘要块=一次 API 都没花"

    usage = engine.usage("t1")
    assert usage.working_messages == 8, "消息一条都不能少，只是草稿不再投影"
    assert usage.working_tokens <= settings.trim_sufficient_line, "剪完必须落到余量线以下"

    projected = next(
        piece.messages for piece in engine.rebuild("t1") if piece.kind == "raw"
    )
    cut = trimmed(projected)
    assert cut and 8 not in cut, "最新一轮的草稿必须还在，其余冷的被剪"

    with database.session() as session:
        stored = AgentRepository(session).active_messages("t1")
    assert all("thinking" in json.dumps(row.content_json, ensure_ascii=False) for row in stored), (
        "存储里必须仍留着原文思维链：剪的是投影，不是账本"
    )

    frozen = engine.usage("t1").working_tokens
    again = engine.compact_if_needed("t1")
    assert (again.l0_blocks_created, again.merges_completed) == (0, 0)
    assert engine.usage("t1").working_tokens == frozen, "回到余量线以下之后，下一轮不得再动投影"


def heavy_text(index: int, chars: int = 400) -> MessageSnapshot:
    """正文很占地方、草稿几乎没有：剪裁从它身上刮不出余量。"""

    snapshot = reasoning(index, chars=2)
    snapshot.content_json["data"]["content"][1]["text"] = "x" * chars
    return snapshot


def test_trimming_that_only_reaches_the_wall_still_compacts(database) -> None:
    """剪完只压到"线下但没剪出余量"→ 不省这一次摘要：下一轮一个工具结果就能再顶过线。

    这条是防退化的：判据一旦变成"剪到线下就算数"，剪裁就变成滑动窗口，每轮都改写投影。
    """

    settings = _settings()
    with database.session() as session:
        repo = AgentRepository(session)
        for index in range(1, 9):
            repo.add_message(
                thread_id="t1",
                user_seq=index,
                message_type=MessageType.ASSISTANT,
                content_json=heavy_text(index).content_json,
                langchain_message_id=f"a-{index}",
            )

    engine = ContextEngine(database, settings, DeterministicSummarizer())
    assert engine.usage("t1").working_tokens > settings.working_trigger

    result = engine.compact_if_needed("t1")
    assert result.l0_blocks_created > 0, "剪不出余量就该压缩，而不是停在线下等着下轮再剪"
    assert engine.usage("t1").working_tokens <= settings.working_trigger
