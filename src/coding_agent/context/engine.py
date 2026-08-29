from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    render_message,
    split_atomic_units_token_balanced,
)
from .cover import ContextPiece
from .records import MemoryBlockSnapshot, MessageSnapshot, WorkStateView
from .summarizer import RetryingSummarizer, Summarizer
from .tokens import estimate_tokens


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

    def _create_l0_if_needed(self, state: ThreadContextState) -> int:
        working = state.working_messages
        working_tokens = sum(message_tokens(message) for message in working)
        if working_tokens <= self.settings.working_trigger:
            return 0

        units = atomic_message_units(working)
        tail_target = math.ceil(working_tokens * self.settings.recent_tail_ratio)
        tail_tokens = 0
        split_at = len(units)
        while split_at > 0 and tail_tokens < tail_target:
            split_at -= 1
            tail_tokens += sum(message_tokens(message) for message in units[split_at])
        prefix_units = units[:split_at]
        chunks = split_atomic_units_token_balanced(prefix_units, self.settings.l0_block_count)
        with ThreadPoolExecutor(
            max_workers=min(self.settings.summary_concurrency, len(chunks))
        ) as executor:
            generated = list(executor.map(self._summarize_l0_chunk, chunks))

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
            del state.working_messages[:consumed]
        except Exception:
            self.cache.invalidate(state.thread_id)
            raise
        return len(generated)

    def _summarize_l0_chunk(
        self, chunk: list[MessageSnapshot]
    ) -> tuple[list[MessageSnapshot], str, int]:
        source = "\n".join(render_message(message) for message in chunk)
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
