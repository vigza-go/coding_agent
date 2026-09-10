from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import distinct, func, select, update

from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..persistence.database import Database
from ..persistence.models import (
    FileMutation,
    MemoryBlock,
    Message,
    TodoSnapshot,
    WorkStateSnapshot,
)
from ..persistence.repository import AgentRepository
from ..workspace.file_undo import FileMutationRecorder


@dataclass(frozen=True)
class RollbackResult:
    restored_files: int
    deactivated_messages: int
    rebuilt_pieces: tuple[ContextPiece, ...]
    work_state: dict | None


@dataclass(frozen=True)
class RollbackPreview:
    messages: int
    file_mutations: int
    files: int
    work_states: int
    todos: int = 0


class RollbackService:
    def __init__(
        self,
        database: Database,
        context_engine: ContextEngine,
        mutation_recorder: FileMutationRecorder,
    ) -> None:
        self.database = database
        self.context_engine = context_engine
        self.mutation_recorder = mutation_recorder

    def preview(self, thread_id: str, user_seq: int) -> RollbackPreview:
        if user_seq < 1:
            raise ValueError("user_seq must be >= 1")
        with self.database.session() as session:
            messages = session.scalar(
                select(func.count(Message.id)).where(
                    Message.thread_id == thread_id,
                    Message.user_seq >= user_seq,
                    Message.active.is_(True),
                )
            )
            mutations = session.scalar(
                select(func.count(FileMutation.id)).where(
                    FileMutation.thread_id == thread_id,
                    FileMutation.user_seq >= user_seq,
                    FileMutation.active.is_(True),
                )
            )
            files = session.scalar(
                select(func.count(distinct(FileMutation.path))).where(
                    FileMutation.thread_id == thread_id,
                    FileMutation.user_seq >= user_seq,
                    FileMutation.active.is_(True),
                )
            )
            work_states = session.scalar(
                select(func.count(WorkStateSnapshot.id)).where(
                    WorkStateSnapshot.thread_id == thread_id,
                    WorkStateSnapshot.user_seq >= user_seq,
                    WorkStateSnapshot.active.is_(True),
                )
            )
            todos = session.scalar(
                select(func.count(TodoSnapshot.id)).where(
                    TodoSnapshot.thread_id == thread_id,
                    TodoSnapshot.user_seq >= user_seq,
                    TodoSnapshot.active.is_(True),
                )
            )
        return RollbackPreview(
            messages=int(messages or 0),
            file_mutations=int(mutations or 0),
            files=int(files or 0),
            work_states=int(work_states or 0),
            todos=int(todos or 0),
        )

    def rollback(
        self,
        thread_id: str,
        user_seq: int,
        *,
        restore_files: bool = True,
    ) -> RollbackResult:
        """停用 ``user_seq`` 及之后的时间轴；``restore_files=False`` 时一个字都不碰工作树。

        不还原文件也不停用文件账：那些账留在自己的 ``user_seq`` 上，之后仍可按那轮 ``/undo``
        退回（``next_user_seq`` 只增不减，clear 之后的新轮次不会跟它们撞号）。
        """

        if user_seq < 1:
            raise ValueError("user_seq must be >= 1")

        restored = self.mutation_recorder.rollback(thread_id, user_seq) if restore_files else 0
        with self.database.session() as session:
            repo = AgentRepository(session)
            conversation = repo.get_or_create_conversation(thread_id, lock=True)
            revoked_ids = list(
                session.scalars(
                    select(Message.id).where(
                        Message.thread_id == thread_id,
                        Message.user_seq >= user_seq,
                        Message.active.is_(True),
                    )
                )
            )
            deactivation_result = session.execute(
                update(Message)
                .where(
                    Message.thread_id == thread_id,
                    Message.user_seq >= user_seq,
                    Message.active.is_(True),
                )
                .values(active=False)
            )
            deactivated = int(getattr(deactivation_result, "rowcount", 0))
            session.execute(
                update(WorkStateSnapshot)
                .where(
                    WorkStateSnapshot.thread_id == thread_id,
                    WorkStateSnapshot.user_seq >= user_seq,
                    WorkStateSnapshot.active.is_(True),
                )
                .values(active=False)
            )
            # 计划与工作状态同类：都挂在 user_seq 上，撤销时一起停用，恢复"最近一个有效版本"。
            session.execute(
                update(TodoSnapshot)
                .where(
                    TodoSnapshot.thread_id == thread_id,
                    TodoSnapshot.user_seq >= user_seq,
                    TodoSnapshot.active.is_(True),
                )
                .values(active=False)
            )
            if revoked_ids:
                first_revoked, last_revoked = min(revoked_ids), max(revoked_ids)
                session.execute(
                    update(MemoryBlock)
                    .where(
                        MemoryBlock.thread_id == thread_id,
                        MemoryBlock.active.is_(True),
                        MemoryBlock.end_message_id >= first_revoked,
                        MemoryBlock.begin_message_id <= last_revoked,
                    )
                    .values(active=False)
                )
            conversation.active_head_seq = repo.latest_active_user_seq(thread_id)

        self.context_engine.invalidate(thread_id)
        pieces = self.context_engine.rebuild(thread_id)
        latest_state = self.context_engine.current_work_state(thread_id)
        work_state = latest_state.state_json if latest_state else None

        # 应用数据库是 canonical source，文件恢复逐条幂等提交；不存在需要回写的派生快照。
        return RollbackResult(restored, deactivated or 0, tuple(pieces), work_state)

    def clear_context(self, thread_id: str) -> RollbackResult:
        """清空这个线程喂给模型的一切：历史消息、压缩块、工作状态，工作树原样不动。

        等于 ``rollback(1, restore_files=False)`` —— 复用同一套停用与重投影逻辑，不另发明
        "重置游标"之类的标记行。撤销的最小单位仍是 DESIGN 定的 ``user_seq``，这里就是 1。
        """

        return self.rollback(thread_id, 1, restore_files=False)
