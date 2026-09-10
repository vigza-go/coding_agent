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
from .work_state import READ_OPS, apply_op, render


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
        """等数据库事务真的提交了，才把这些行追加进来。"""

        try:
            self.cache.append_messages(thread_id, rows)
        except Exception:
            self.cache.invalidate(thread_id)
            raise

    def update_work_state(self, thread_id: str, row: WorkStateSnapshot) -> None:
        """权威行提交后，更新派生的工作状态视图。"""

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
        """返回某个线程当前所选上下文的只读快照。"""

        with self.cache.locked(thread_id) as state:
            return ContextUsage(
                memory_levels=tuple(block.level for block in state.selected_blocks),
                memory_tokens=sum(block.token_count for block in state.selected_blocks),
                working_messages=len(state.working_messages),
                working_tokens=sum(message_tokens(message) for message in state.working_messages),
            )

    def compact_if_needed(self, thread_id: str) -> CompactionResult:
        with self.cache.locked(thread_id) as state:
            created = self._slim_and_compact(state)
            merges = self._merge_until_within_budget(state)
            return CompactionResult(created, merges)

    @staticmethod
    def _strip_reasoning(message: MessageSnapshot) -> MessageSnapshot | None:
        """重建 ``message``，去掉它的思维链块；本来就没有就返回 None。

        只动 reasoning 块，所以 ``tool_use`` 块和与它配对的 ``tool_result`` 在构造上保证
        逐字节不变。第一个被删掉的块会留下一个纯文本标记，让模型知道“思考是被收起来的”，
        而不是“压根没想过”——看不见的缺失才是最坏的失效模式：一个已经被否掉的主意，会当成
        新想法重新冒出来。
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
        """从最冷的思维链开始丢，一路丢到只剩 ``budget`` 个 token。

        每条消息能省多少，用的是触发线那同一把尺子、在同一份投影结果上量的，所以一条“剪了
        也不省”的消息不会被记上一笔。最新那段思考一定保留，哪怕它自己就超了 ``budget``：
        一轮对话不能丢掉“即将发出的这次调用是怎么想出来的”。
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

    def _slim_and_compact(self, state: ThreadContextState) -> int:
        working = state.working_messages
        working_tokens = sum(message_tokens(message) for message in working)
        if working_tokens <= self.settings.working_trigger:
            return 0

        # 撑到线上才动，而且两种剪裁**一次做完**：剪裁会断前缀，触发一次付一次代价。分步碎剪
        # （这次剪思维链、下次剪工具结果）会把"断前缀"的次数翻倍——每次触发都改写投影，而缓存
        # 的代价取决于改动落在提示词多靠前的位置，不是改动大小。宁可在一次触发里多剪一点，也别
        # 把触发次数堆起来。见 ContextSettings.reasoning_budget / recent_tool_interactions。
        working = self._retain_reasoning_within_budget(working, self.settings.reasoning_budget)
        # 冷思维链的剪裁**立刻**写回 state：它不花 API、也不丢记忆块（账本里原文还在），所以
        # 万一后面摘要失败，这次剪裁留在缓存里不回退——回退只会让下一轮在同一位置重剪一次。
        state.working_messages = working

        # 旧工具结果不一样：它只在压缩成功时才写回 state。摘要失败时状态必须逐字节不动，
        # 否则模型会看到一批"这次没打算留下的"裁剪。
        working = self._trim_old_tool_results(working, self.settings.recent_tool_interactions)

        # 判据不是"剪到线下就算数"，而是"剪出了多少余量"。只剪到刚好压线，下一轮一个工具
        # 结果就能再顶过线，于是又剪一次、又断一次前缀——那等于拿剪裁当滑动窗口，每轮都改写
        # 投影。剪完还剩一大截，说明大头是对话正文本身，工具结果和冷草稿身上刮不出多少，
        # 该动的是压缩，不是继续刮。见 ContextSettings.trim_sufficient_ratio。
        if sum(message_tokens(message) for message in working) <= self.settings.trim_sufficient_line:
            state.working_messages = working
            created = 0
        else:
            created = self._compact_prefix(state, working)

        # 投影已经因为剪裁/压缩改变，这时候顺手把最新 work_state 贴成便签是搭便车，不额外多断
        # 一次前缀。没超线的那条路径不碰便签，投影逐字节不变、缓存全中。
        self._refresh_pin(state)
        return created

    def _refresh_pin(self, state: ThreadContextState) -> None:
        """把便签刷成最新 work_state 正文；没有内容就清空它。

        只在剪裁/压缩之后调用——这是唯一刷新便签的入口，保证两次剪裁之间便签逐字节不变。
        """

        view = state.work_state
        state.pin_work_state = render(view.state_json) if view and view.state_json else None

    def _compact_prefix(self, state: ThreadContextState, working: list[MessageSnapshot]) -> int:
        """把冷前缀压成 L0 记忆块，尾部按 recent_tail_ratio 原样保留。返回新建块数。"""

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
        # 配额的分母是"这一段历史在模型上下文里占多少"，也就是触发线、尾留、切块用的那把
        # 尺子（`message_tokens` 现在量的是可见内容，不是 JSON 包壳）。两条口径都不对：
        # 按 JSON 包壳算会把上限抬到比原文还长（等于没约束），按下面这份 source 算又只到
        # 真实的一半多 —— source 是喂给弱摘要器的安全副本，思考块与工具调用原文都被它抹掉了。
        # 合并路径本来就按左右两块入库文本的 token 算，两条路径同为「压缩掉多少、给一半名额」。
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
            pair = self._select_merge_pair(selected)
            if pair is None:
                levels = [block.level for block in selected]
                raise CompressionInvariantError(
                    "compression region exceeds its limit but has no adjacent "
                    f"same-level pair: {levels}"
                )
            index, left, right = pair
            self._merge_pair(state, index, left, right)
            merge_count += 1

    @staticmethod
    def _select_merge_pair(
        selected: list[MemoryBlockSnapshot],
    ) -> tuple[int, MemoryBlockSnapshot, MemoryBlockSnapshot] | None:
        return next(
            (
                (index, left, right)
                for index, (left, right) in enumerate(pairwise(selected))
                if left.level == right.level
            ),
            None,
        )

    def _merge_pair(
        self,
        state: ThreadContextState,
        index: int,
        left: MemoryBlockSnapshot,
        right: MemoryBlockSnapshot,
    ) -> None:
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

    def rebuild(self, thread_id: str) -> list[ContextPiece]:
        return self.cache.pieces(thread_id)
