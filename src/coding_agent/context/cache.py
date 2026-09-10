from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock

from ..persistence.database import Database
from ..persistence.models import Message, WorkStateSnapshot
from ..persistence.repository import AgentRepository
from .cover import ContextPiece, greedy_cover
from .records import MemoryBlockSnapshot, MessageSnapshot, WorkStateView
from .work_state import render


class ContextCacheInvariantError(RuntimeError):
    pass


@dataclass
class ThreadContextState:
    thread_id: str
    selected_blocks: list[MemoryBlockSnapshot] = field(default_factory=list)
    working_messages: list[MessageSnapshot] = field(default_factory=list)
    work_state: WorkStateView | None = None
    # 便签：剪裁/压缩把投影弄断的那一刻，顺手把最新 work_state 正文贴在这里。
    # 它是**派生视图**（不落库、不进 messages/memory_blocks，也不是一条真消息），平日原样
    # 冻着，只有剪裁那条路径会刷新它——这样两次剪裁之间投影逐字节不变、缓存全中。存渲染好
    # 的正文而不是 MessageSnapshot：它没有真实 id，本就不是消息，别给它编字段。
    pin_work_state: str | None = None

    def pieces(self) -> list[ContextPiece]:
        pieces = [
            ContextPiece(
                kind="memory",
                begin_message_id=block.begin_message_id,
                end_message_id=block.end_message_id,
                text=block.text,
                level=block.level,
                block_id=block.id,
            )
            for block in self.selected_blocks
        ]
        if self.pin_work_state is not None:
            # 贴在压缩块之后、原文之前：投影是 [压缩历史] + [当前备忘] + [原文]。
            pieces.append(
                ContextPiece(
                    kind="work_state",
                    begin_message_id=0,
                    end_message_id=0,
                    text=self.pin_work_state,
                )
            )
        if self.working_messages:
            pieces.append(
                ContextPiece(
                    kind="raw",
                    begin_message_id=self.working_messages[0].id,
                    end_message_id=self.working_messages[-1].id,
                    messages=tuple(self.working_messages),
                )
            )
        return pieces


class ThreadContextCache:
    """进程内的派生上下文。权威数据仍以应用的数据库为准。"""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._entries: dict[str, ThreadContextState] = {}
        self._locks: defaultdict[str, RLock] = defaultdict(RLock)

    @contextmanager
    def locked(self, thread_id: str) -> Generator[ThreadContextState, None, None]:
        with self._locks[thread_id]:
            state = self._entries.get(thread_id)
            if state is None:
                state = self._load(thread_id)
                self._entries[thread_id] = state
            yield state

    def invalidate(self, thread_id: str) -> None:
        with self._locks[thread_id]:
            self._entries.pop(thread_id, None)

    def append_messages(self, thread_id: str, rows: Sequence[Message]) -> None:
        if not rows:
            return
        with self.locked(thread_id) as state:
            snapshots = sorted(
                (MessageSnapshot.from_model(row) for row in rows),
                key=lambda message: message.id,
            )
            covered_through = (
                state.selected_blocks[-1].end_message_id if state.selected_blocks else 0
            )
            for snapshot in snapshots:
                if snapshot.thread_id != thread_id:
                    raise ContextCacheInvariantError("cannot append a message from another thread")
                if snapshot.id <= covered_through:
                    continue
                message_ids = [message.id for message in state.working_messages]
                position = bisect_left(message_ids, snapshot.id)
                if (
                    position < len(state.working_messages)
                    and state.working_messages[position].id == snapshot.id
                ):
                    continue
                state.working_messages.insert(position, snapshot)

    def update_work_state(self, thread_id: str, row: WorkStateSnapshot) -> None:
        if row.thread_id != thread_id:
            raise ContextCacheInvariantError("cannot cache work state from another thread")
        with self.locked(thread_id) as state:
            if state.work_state is None or row.id >= state.work_state.id:
                state.work_state = WorkStateView.from_model(row)

    def current_work_state(self, thread_id: str) -> WorkStateView | None:
        with self.locked(thread_id) as state:
            if state.work_state is None:
                return None
            return WorkStateView(state.work_state.id, deepcopy(state.work_state.state_json))

    def pieces(self, thread_id: str) -> list[ContextPiece]:
        with self.locked(thread_id) as state:
            return state.pieces()

    def _load(self, thread_id: str) -> ThreadContextState:
        with self.database.session() as session:
            repo = AgentRepository(session)
            messages = [MessageSnapshot.from_model(row) for row in repo.active_messages(thread_id)]
            blocks = [
                MemoryBlockSnapshot.from_model(row)
                for row in repo.memory_blocks(thread_id, active_only=True)
            ]
            work_state_row = repo.latest_work_state(thread_id)

        pieces = greedy_cover(messages, blocks)
        blocks_by_id = {block.id: block for block in blocks}
        selected_blocks: list[MemoryBlockSnapshot] = []
        working_messages: list[MessageSnapshot] = []
        raw_seen = False
        for piece in pieces:
            if piece.kind == "raw":
                raw_seen = True
                working_messages.extend(piece.messages)
                continue
            if raw_seen:
                raise ContextCacheInvariantError(
                    "memory cover contains a memory block after a raw-message gap"
                )
            if piece.block_id is None or piece.block_id not in blocks_by_id:
                raise ContextCacheInvariantError("memory cover references an unknown block")
            selected_blocks.append(blocks_by_id[piece.block_id])

        return ThreadContextState(
            thread_id=thread_id,
            selected_blocks=selected_blocks,
            working_messages=working_messages,
            work_state=(
                WorkStateView.from_model(work_state_row) if work_state_row is not None else None
            ),
            # 冷启动（重启/缓存失效）后便签得自己长回来：只要历史上压过块、且 work_state
            # 有内容，就按最新快照重新渲染一版。冷启动本来就全量重编码，这一刻贴它不额外花钱。
            pin_work_state=(
                render(work_state_row.state_json)
                if selected_blocks and work_state_row is not None and work_state_row.state_json
                else None
            ),
        )
