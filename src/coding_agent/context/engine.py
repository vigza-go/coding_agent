from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
from itertools import pairwise

from sqlalchemy import select

from ..config import ContextSettings
from ..persistence.database import Database
from ..persistence.models import MemoryBlock, Message, WorkStateSnapshot
from ..persistence.repository import AgentRepository
from .cache import ThreadContextCache, ThreadContextState
from .compaction import (
    atomic_message_units,
    message_tokens,
    sanitize_transcript,
    split_atomic_units_token_balanced,
)
from .cover import ContextPiece
from .records import MemoryBlockSnapshot, MessageSnapshot, WorkStateView
from .summarizer import RetryingSummarizer, Summarizer
from .tokens import estimate_tokens
from .work_state import READ_OPS, apply_op


class CompressionInvariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompactionResult:
    l0_blocks_created: int = 0
    merges_completed: int = 0


@dataclass(frozen=True)
class ContextUsage:
    memory_levels: tuple[int, ...]
    memory_tokens: int
    working_messages: int
    working_tokens: int


class ContextEngine:
    def __init__(
        self, database: Database, settings: ContextSettings, summarizer: Summarizer
    ) -> None:
        self.database = database
        self.settings = settings
        self.summarizer = RetryingSummarizer(summarizer, settings.summary_max_attempts)
        self.cache = ThreadContextCache(database)

    def append_messages(self, thread_id: str, rows: Sequence[Message]) -> None:
        """Append rows only after their database transaction has committed."""

        try:
            self.cache.append_messages(thread_id, rows)
        except Exception:
            self.cache.invalidate(thread_id)
            raise

    def update_work_state(self, thread_id: str, row: WorkStateSnapshot) -> None:
        """Update the derived work-state view after the canonical row commits."""

        try:
            self.cache.update_work_state(thread_id, row)
        except Exception:
            self.cache.invalidate(thread_id)
            raise

    def current_work_state(self, thread_id: str) -> WorkStateView | None:
        return self.cache.current_work_state(thread_id)

    def mutate_work_state(
        self, thread_id: str, user_seq: int, op: str, key: str | None, value: str | None
    ) -> str:
        """在一次持锁期内完成「读最新→改→写回」，返回给模型的回执。

        框架会**并行执行同一批工具调用**，而 `apply_op` 是 read-modify-write：
        没有这把锁时两个并发写从同一基线出发、各插一行，后插那行整份覆盖前者——
        实测 8 线程 × 10 轮只活下 19/80 个键，且**零异常**（静默丢状态）。
        校验失败在 `apply_op` 内抛出，发生在写之前，所以坏参数不会留下半改状态。

        边界：这把锁是**进程内**的。多进程同写一个 thread 仍会丢，与派生缓存本身
        的已知限制一致（见 README「进程内缓存不支持多进程」）。
        """

        with self.cache.locked(thread_id):
            with self.database.session() as session:
                repo = AgentRepository(session)
                row = repo.latest_work_state(thread_id)
                state, receipt = apply_op(
                    row.state_json if row is not None else {}, op, key, value
                )
                saved = None
                if op not in READ_OPS:
                    saved = repo.save_work_state(thread_id, user_seq, state)
            # 只读操作不落库，也就不会产生新快照，派生缓存保持原样。
            if saved is not None:
                self.cache.update_work_state(thread_id, saved)
        return receipt

    def invalidate(self, thread_id: str) -> None:
        self.cache.invalidate(thread_id)

    def usage(self, thread_id: str) -> ContextUsage:
        """Return a read-only snapshot of the context currently selected for a thread."""

        with self.cache.locked(thread_id) as state:
            return ContextUsage(
                memory_levels=tuple(block.level for block in state.selected_blocks),
                memory_tokens=sum(block.token_count for block in state.selected_blocks),
                working_messages=len(state.working_messages),
                working_tokens=sum(message_tokens(message) for message in state.working_messages),
            )

    def compact_if_needed(self, thread_id: str) -> CompactionResult:
        with self.cache.locked(thread_id) as state:
            created = self._create_l0_if_needed(state)
            merges = self._merge_until_within_budget(state)
            return CompactionResult(created, merges)

    @staticmethod
    def _strip_reasoning(message: MessageSnapshot) -> MessageSnapshot | None:
        """Rebuild ``message`` without its chain-of-thought blocks, or None if it has none.

        Only reasoning blocks are touched, so ``tool_use`` blocks and their
        ``tool_result`` partners survive byte-for-byte by construction. A single plain
        marker replaces the first removed block so the model can tell that reasoning was
        withheld rather than never happening -- absence that looks like absence is the
        failure mode that lets a rejected idea come back as a fresh one.
        """

        trailer = "[较早的模型思维链已剪裁，不再回读。]"
        data = message.content_json.get("data")
        if not isinstance(data, dict):
            return None
        content = data.get("content")
        if not isinstance(content, list):
            return None
        reasoning = ("thinking", "reasoning", "redacted_thinking")
        rebuilt = []
        removed = False
        for block in content:
            if isinstance(block, dict) and block.get("type") in reasoning:
                removed = True
                already = any(
                    isinstance(b, dict) and b.get("type") == "text" and b.get("text") == trailer
                    for b in rebuilt
                )
                if not already:
                    rebuilt.append({"type": "text", "text": trailer})
                continue
            rebuilt.append(block)
        if not removed:
            return None
        return replace(message, content_json={**message.content_json, "data": {**data, "content": rebuilt}})

    @classmethod
    def _retain_reasoning_within_budget(
        cls, messages: list[MessageSnapshot], budget: int
    ) -> list[MessageSnapshot]:
        """Drop the coldest chain-of-thought first, newest-first, until ``budget`` tokens remain.

        The saving of each message is measured with the very estimator the trigger uses, on
        the projected result, so a message that costs nothing to trim is never charged for
        it. The newest reasoning segment is kept even when it alone exceeds ``budget``: a
        turn must never lose the thought that produced the call it is about to make.
        """

        stripped = [cls._strip_reasoning(message) for message in messages]
        savings = [
            0
            if trimmed is None
            else max(0, message_tokens(source) - message_tokens(trimmed))
            for source, trimmed in zip(messages, stripped, strict=True)
        ]
        spend = 0
        boundary = len(messages)
        newest = -1
        for index in range(len(messages) - 1, -1, -1):
            if stripped[index] is None:
                continue
            newest = index
            if spend + savings[index] > budget:
                break
            spend += savings[index]
            boundary = index
        boundary = min(boundary, newest)
        return [
            message if trimmed is None or index >= boundary else trimmed
            for index, (message, trimmed) in enumerate(zip(messages, stripped, strict=True))
        ]

    @staticmethod
    def _trim_old_tool_results(messages: list[MessageSnapshot], keep: int) -> list[MessageSnapshot]:
        tool_positions = [index for index, message in enumerate(messages) if message.type == "tool"]
        old_positions = set(tool_positions[:-keep]) if keep > 0 else set(tool_positions)
        result: list[MessageSnapshot] = []
        for index, message in enumerate(messages):
            if index in old_positions:
                content_json = {
                    **message.content_json,
                    "data": {
                        **message.content_json["data"],
                        "content": (
                            "[较早的工具结果已从近期模型输入中剪裁；如需细节，"
                            "请重新读取文件或执行查询。]"
                        ),
                    },
                }
                result.append(replace(message, content_json=content_json))
            else:
                result.append(message)
        return result

    def _create_l0_if_needed(self, state: ThreadContextState) -> int:
        working = state.working_messages
        working_tokens = sum(message_tokens(message) for message in working)
        if working_tokens <= self.settings.working_trigger:
            return 0

        # At the wall, shed cold chain-of-thought before reaching for the summarizer: it is
        # the only reduction available here that costs no API call and destroys no memory
        # block. Running it only at the wall -- rather than every turn -- is what keeps the
        # cached prefix alive, because cache cost is set by how early a change sits in the
        # prompt, not by how large the change is. See ContextSettings.reasoning_budget.
        working = self._retain_reasoning_within_budget(working, self.settings.reasoning_budget)
        state.working_messages = working
        working_tokens = sum(message_tokens(message) for message in working)
        if working_tokens <= self.settings.working_trigger:
            return 0

        # Once triggered, trim exactly once and then partition the resulting messages.
        working = self._trim_old_tool_results(working, self.settings.recent_tool_interactions)
        working_tokens = sum(message_tokens(message) for message in working)
        units = atomic_message_units(working)
        tail_target = math.ceil(working_tokens * self.settings.recent_tail_ratio)
        tail_tokens = 0
        split_at = len(units)
        while split_at > 0 and tail_tokens < tail_target:
            split_at -= 1
            tail_tokens += sum(message_tokens(message) for message in units[split_at])
        prefix_units = units[:split_at]
        chunks = split_atomic_units_token_balanced(prefix_units, self.settings.l0_block_count)
        if not chunks:
            state.working_messages = working
            return 0
        with ThreadPoolExecutor(
            max_workers=min(self.settings.summary_concurrency, len(chunks))
        ) as executor:
            # 每个任务出发前先复制一份当前上下文。新线程不继承 contextvar，而 LangChain
            # 的回调（进度里"正在压缩记忆 L0"就是它送出去的）正是挂在 contextvar 上往下
            # 传的；不带副本，整段压缩在界面上完全隐身（实测：同线程 4 条事件、线程池 0
            # 条）。副本必须一个任务一份 —— 同一个 Context 不能被多个线程并发进入。
            contexts = [copy_context() for _ in chunks]
            generated = list(
                executor.map(
                    lambda carried: carried[0].run(self._summarize_l0_chunk, carried[1]),
                    zip(contexts, chunks, strict=True),
                )
            )

        inserted: list[MemoryBlock] = []
        with self.database.session() as session:
            repo = AgentRepository(session)
            repo.get_or_create_conversation(state.thread_id, lock=True)
            for chunk, summary, token_count in generated:
                inserted.append(
                    repo.add_memory_block(
                        thread_id=state.thread_id,
                        text=summary,
                        begin_message_id=chunk[0].id,
                        end_message_id=chunk[-1].id,
                        level=0,
                        token_count=token_count,
                    )
                )

        try:
            state.selected_blocks.extend(
                MemoryBlockSnapshot.from_model(block) for block in inserted
            )
            consumed = sum(len(chunk) for chunk, _, _ in generated)
            state.working_messages = working[consumed:]
        except Exception:
            self.cache.invalidate(state.thread_id)
            raise
        return len(generated)

    def _summarize_l0_chunk(
        self, chunk: list[MessageSnapshot]
    ) -> tuple[list[MessageSnapshot], str, int]:
        source = sanitize_transcript(chunk)
        source_tokens = sum(message_tokens(message) for message in chunk)
        hard_limit = max(1, math.floor(source_tokens * self.settings.summary_target_ratio))
        summary = self.summarizer.summarize(source, hard_limit=hard_limit, level=0)
        return chunk, summary, estimate_tokens(summary)

    def _merge_until_within_budget(self, state: ThreadContextState) -> int:
        merge_count = 0
        while True:
            selected = state.selected_blocks
            if sum(block.token_count for block in selected) <= self.settings.compression_limit:
                return merge_count
            pair = next(
                (
                    (index, left, right)
                    for index, (left, right) in enumerate(pairwise(selected))
                    if left.level == right.level
                ),
                None,
            )
            if pair is None:
                levels = [block.level for block in selected]
                raise CompressionInvariantError(
                    "compression region exceeds its limit but has no adjacent "
                    f"same-level pair: {levels}"
                )
            index, left, right = pair
            source = f"{left.text}\n\n{right.text}"
            hard_limit = max(
                1,
                math.floor(
                    (left.token_count + right.token_count) * self.settings.summary_target_ratio
                ),
            )
            summary = self.summarizer.summarize(source, hard_limit=hard_limit, level=left.level + 1)

            with self.database.session() as session:
                repo = AgentRepository(session)
                repo.get_or_create_conversation(state.thread_id, lock=True)
                locked = list(
                    session.scalars(
                        select(MemoryBlock)
                        .where(MemoryBlock.id.in_([left.id, right.id]))
                        .with_for_update()
                    )
                )
                if len(locked) != 2 or any(not block.active for block in locked):
                    raise CompressionInvariantError("selected memory blocks became inactive")
                parent = session.scalar(
                    select(MemoryBlock).where(
                        MemoryBlock.thread_id == state.thread_id,
                        MemoryBlock.active.is_(True),
                        MemoryBlock.begin_message_id == left.begin_message_id,
                        MemoryBlock.end_message_id == right.end_message_id,
                        MemoryBlock.level == left.level + 1,
                    )
                )
                if parent is None:
                    parent = repo.add_memory_block(
                        thread_id=state.thread_id,
                        text=summary,
                        begin_message_id=left.begin_message_id,
                        end_message_id=right.end_message_id,
                        level=left.level + 1,
                        token_count=estimate_tokens(summary),
                    )

            try:
                state.selected_blocks[index : index + 2] = [MemoryBlockSnapshot.from_model(parent)]
            except Exception:
                self.cache.invalidate(state.thread_id)
                raise
            merge_count += 1

    def rebuild(self, thread_id: str) -> list[ContextPiece]:
        return self.cache.pieces(thread_id)
