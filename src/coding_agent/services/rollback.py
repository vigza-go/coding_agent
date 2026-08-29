from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select, update

from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..persistence.database import Database
from ..persistence.models import MemoryBlock, Message, WorkStateSnapshot
from ..persistence.repository import AgentRepository
from ..workspace.file_undo import FileMutationRecorder


@dataclass(frozen=True)
class RollbackResult:
    restored_files: int
    deactivated_messages: int
    rebuilt_pieces: tuple[ContextPiece, ...]
    work_state: dict | None


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

    def rollback(
        self,
        thread_id: str,
        user_seq: int,
        *,
        update_checkpoint: Callable[[list[ContextPiece], dict | None], None] | None = None,
    ) -> RollbackResult:
        if user_seq < 1:
            raise ValueError("user_seq must be >= 1")

        restored = self.mutation_recorder.rollback(thread_id, user_seq)
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
            conversation.active_head_seq = min(conversation.active_head_seq, user_seq - 1)

        self.context_engine.invalidate(thread_id)
        pieces = self.context_engine.rebuild(thread_id)
        latest_state = self.context_engine.current_work_state(thread_id)
        work_state = latest_state.state_json if latest_state else None

        # The application DB is canonical. A crash before this repair is handled by rebuilding
        # the checkpoint from the canonical rows on the next startup.
        if update_checkpoint is not None:
            update_checkpoint(pieces, work_state)
        return RollbackResult(restored, deactivated or 0, tuple(pieces), work_state)
