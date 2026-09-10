from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from ..persistence.models import MemoryBlock, Message, TodoSnapshot, WorkStateSnapshot


@dataclass(frozen=True)
class MessageSnapshot:
    id: int
    thread_id: str
    user_seq: int
    type: str
    content_json: dict[str, Any]
    langchain_message_id: str | None

    @classmethod
    def from_model(cls, row: Message) -> MessageSnapshot:
        return cls(
            id=cast(int, row.id),
            thread_id=cast(str, row.thread_id),
            user_seq=cast(int, row.user_seq),
            type=cast(str, row.type),
            content_json=deepcopy(cast(dict[str, Any], row.content_json)),
            langchain_message_id=cast(str | None, row.langchain_message_id),
        )


@dataclass(frozen=True)
class MemoryBlockSnapshot:
    id: int
    thread_id: str
    text: str
    begin_message_id: int
    end_message_id: int
    level: int
    token_count: int

    @classmethod
    def from_model(cls, row: MemoryBlock) -> MemoryBlockSnapshot:
        return cls(
            id=cast(int, row.id),
            thread_id=cast(str, row.thread_id),
            text=cast(str, row.text),
            begin_message_id=cast(int, row.begin_message_id),
            end_message_id=cast(int, row.end_message_id),
            level=cast(int, row.level),
            token_count=cast(int, row.token_count),
        )


@dataclass(frozen=True)
class TodoView:
    """计划的最新有效版本（有序数组）。"""

    id: int
    items: list[dict[str, Any]]

    @classmethod
    def from_model(cls, row: TodoSnapshot) -> TodoView:
        return cls(
            id=cast(int, row.id),
            items=deepcopy(cast(list[dict[str, Any]], row.items_json)),
        )


@dataclass(frozen=True)
class WorkStateView:
    id: int
    state_json: dict[str, Any]

    @classmethod
    def from_model(cls, row: WorkStateSnapshot) -> WorkStateView:
        return cls(
            id=cast(int, row.id),
            state_json=deepcopy(cast(dict[str, Any], row.state_json)),
        )
