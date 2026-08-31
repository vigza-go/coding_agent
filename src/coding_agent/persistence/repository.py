from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import (
    Conversation,
    FileBlob,
    FileMutation,
    MemoryBlock,
    Message,
    WorkStateSnapshot,
)


class AgentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create_conversation(self, thread_id: str, *, lock: bool = False) -> Conversation:
        stmt: Select[tuple[Conversation]] = select(Conversation).where(
            Conversation.thread_id == thread_id
        )
        if lock:
            stmt = stmt.with_for_update()
        conversation = self.session.scalar(stmt)
        if conversation is None:
            conversation = Conversation(thread_id=thread_id)
            self.session.add(conversation)
            self.session.flush()
        return conversation

    def reserve_user_seq(self, thread_id: str) -> int:
        conversation = self.get_or_create_conversation(thread_id, lock=True)
        user_seq = conversation.next_user_seq
        conversation.next_user_seq += 1
        conversation.active_head_seq = user_seq
        self.session.flush()
        return user_seq

    def add_message(
        self,
        *,
        thread_id: str,
        user_seq: int,
        message_type: str,
        content_json: dict[str, Any],
        langchain_message_id: str | None = None,
    ) -> Message:
        if langchain_message_id:
            existing = self.session.scalar(
                select(Message).where(
                    Message.thread_id == thread_id,
                    Message.langchain_message_id == langchain_message_id,
                )
            )
            if existing is not None:
                return existing
        row = Message(
            thread_id=thread_id,
            user_seq=user_seq,
            type=message_type,
            content_json=content_json,
            langchain_message_id=langchain_message_id,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def active_messages(self, thread_id: str, *, limit: int | None = None) -> list[Message]:
        stmt = select(Message).where(
            Message.thread_id == thread_id,
            Message.active.is_(True),
        )
        if limit is None:
            return list(self.session.scalars(stmt.order_by(Message.id)))
        rows = list(self.session.scalars(stmt.order_by(Message.id.desc()).limit(limit)))
        rows.reverse()
        return rows

    def latest_active_user_seq(self, thread_id: str) -> int:
        value = self.session.scalar(
            select(func.max(Message.user_seq)).where(
                Message.thread_id == thread_id,
                Message.active.is_(True),
            )
        )
        return int(value or 0)

    def conversations(self, *, limit: int = 50) -> list[Conversation]:
        return list(
            self.session.scalars(
                select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
            )
        )

    def recent_usage_metadata(self, thread_id: str, *, limit: int) -> list[Any]:
        if limit < 1:
            raise ValueError("usage limit must be positive")
        # Include inactive history: undo does not refund API usage. Do not load message bodies.
        return list(
            self.session.scalars(
                select(Message.content_json["data"]["usage_metadata"])
                .where(Message.thread_id == thread_id, Message.type == "assistant")
                .order_by(Message.id.desc())
                .limit(limit)
            )
        )

    def memory_blocks(self, thread_id: str, *, active_only: bool = True) -> list[MemoryBlock]:
        stmt = select(MemoryBlock).where(MemoryBlock.thread_id == thread_id)
        if active_only:
            stmt = stmt.where(MemoryBlock.active.is_(True))
        return list(
            self.session.scalars(stmt.order_by(MemoryBlock.begin_message_id, MemoryBlock.id))
        )

    def add_memory_block(
        self,
        *,
        thread_id: str,
        text: str,
        begin_message_id: int,
        end_message_id: int,
        level: int,
        token_count: int,
    ) -> MemoryBlock:
        block = MemoryBlock(
            thread_id=thread_id,
            text=text,
            begin_message_id=begin_message_id,
            end_message_id=end_message_id,
            level=level,
            token_count=token_count,
            active=True,
        )
        self.session.add(block)
        self.session.flush()
        return block

    def save_work_state(
        self, thread_id: str, user_seq: int, state_json: dict[str, Any]
    ) -> WorkStateSnapshot:
        snapshot = WorkStateSnapshot(
            thread_id=thread_id,
            user_seq=user_seq,
            state_json=state_json,
            active=True,
        )
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def latest_work_state(self, thread_id: str) -> WorkStateSnapshot | None:
        return self.session.scalar(
            select(WorkStateSnapshot)
            .where(
                WorkStateSnapshot.thread_id == thread_id,
                WorkStateSnapshot.active.is_(True),
            )
            .order_by(WorkStateSnapshot.id.desc())
            .limit(1)
        )

    def blob_by_sha(self, sha256: str) -> FileBlob | None:
        return self.session.scalar(select(FileBlob).where(FileBlob.sha256 == sha256))

    def add_blob(self, *, sha256: str, content: bytes | None, storage_uri: str | None) -> FileBlob:
        values = {"sha256": sha256, "content": content, "storage_uri": storage_uri}
        dialect = self.session.get_bind().dialect.name
        if dialect in {"mysql", "mariadb"}:
            # A duplicate is normal: reuse the immutable blob without overwriting it.
            statement = (
                mysql_insert(FileBlob).values(**values).on_duplicate_key_update(id=FileBlob.id)
            )
        elif dialect == "sqlite":
            statement = (
                sqlite_insert(FileBlob)
                .values(**values)
                .on_conflict_do_nothing(index_elements=[FileBlob.sha256])
            )
        else:
            raise NotImplementedError(f"atomic blob insertion is not supported for {dialect}")
        self.session.execute(statement)
        # Use a current read, even if MySQL REPEATABLE READ already has an older snapshot.
        return self.session.scalars(
            select(FileBlob).where(FileBlob.sha256 == sha256).with_for_update()
        ).one()

    def mutations_to_rollback(self, thread_id: str, from_user_seq: int) -> list[FileMutation]:
        return list(
            self.session.scalars(
                select(FileMutation)
                .where(
                    FileMutation.thread_id == thread_id,
                    FileMutation.user_seq >= from_user_seq,
                    FileMutation.active.is_(True),
                )
                .order_by(FileMutation.id.desc())
            )
        )
