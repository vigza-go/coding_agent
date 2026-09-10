from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SQL_ID = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


class MessageType(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class MutationStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class OperationType(StrEnum):
    CREATE = "create"
    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False, unique=True)
    active_head_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    next_user_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_thread_active_id", "thread_id", "active", "id"),
        Index("ix_messages_thread_seq", "thread_id", "user_seq"),
        UniqueConstraint("thread_id", "langchain_message_id", name="uq_message_langchain_id"),
    )

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False)
    user_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    langchain_message_id: Mapped[str | None] = mapped_column(String(191), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class WorkStateSnapshot(Base):
    __tablename__ = "work_state_snapshots"
    __table_args__ = (Index("ix_work_state_thread_active", "thread_id", "active", "id"),)

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False)
    user_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TodoSnapshot(Base):
    """计划的不可变快照：每次写落一条新行，读-改-写整体在 engine 的 thread 锁里。

    与 ``work_state_snapshots`` 同构，只是载荷换成**有序**数组
    ``[{"id": ..., "title": ..., "status": ...}]``（顺序有语义，渲染不排序）。
    """

    __tablename__ = "todo_snapshots"
    __table_args__ = (Index("ix_todo_thread_active", "thread_id", "active", "id"),)

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False)
    user_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    items_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MemoryBlock(Base):
    __tablename__ = "memory_blocks"
    __table_args__ = (
        Index("ix_memory_thread_active_range", "thread_id", "active", "begin_message_id"),
    )

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    begin_message_id: Mapped[int] = mapped_column(SQL_ID, ForeignKey("messages.id"), nullable=False)
    end_message_id: Mapped[int] = mapped_column(SQL_ID, ForeignKey("messages.id"), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FileBlob(Base):
    __tablename__ = "file_blobs"

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    storage_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[bytes | None] = mapped_column(LargeBinary(length=16_777_215), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FileMutation(Base):
    __tablename__ = "file_mutations"
    __table_args__ = (Index("ix_mutations_rollback", "thread_id", "active", "user_seq", "id"),)

    id: Mapped[int] = mapped_column(SQL_ID, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(191), nullable=False)
    user_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tool_call_id: Mapped[str | None] = mapped_column(String(191), nullable=True)
    operation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    before_blob_id: Mapped[int | None] = mapped_column(
        SQL_ID, ForeignKey("file_blobs.id"), nullable=True
    )
    before_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    after_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=MutationStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
