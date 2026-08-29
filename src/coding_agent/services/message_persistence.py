from __future__ import annotations

from dataclasses import replace
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from langgraph.types import Command

from ..context.engine import ContextEngine
from ..persistence.database import Database
from ..persistence.message_codec import encode_message, ensure_message_id, message_type
from ..persistence.repository import AgentRepository

MessageT = TypeVar("MessageT", bound=BaseMessage)


class MessagePersistenceService:
    def __init__(self, database: Database, context_engine: ContextEngine) -> None:
        self.database = database
        self.context_engine = context_engine

    def persist_messages(
        self, *, thread_id: str, user_seq: int, messages: list[MessageT]
    ) -> list[MessageT]:
        normalized: list[MessageT] = []
        rows = []
        with self.database.session() as session:
            repo = AgentRepository(session)
            for message in messages:
                current = ensure_message_id(message)
                normalized.append(current)
                rows.append(
                    repo.add_message(
                        thread_id=thread_id,
                        user_seq=user_seq,
                        message_type=message_type(current),
                        content_json=encode_message(current),
                        langchain_message_id=current.id,
                    )
                )
        self.context_engine.append_messages(thread_id, rows)
        return normalized

    def persist_command(
        self, *, thread_id: str, user_seq: int, command: Command[Any]
    ) -> Command[Any]:
        if not isinstance(command.update, dict):
            return command
        command_messages = command.update.get("messages")
        if not isinstance(command_messages, list):
            return command
        positions = [
            index
            for index, message in enumerate(command_messages)
            if isinstance(message, BaseMessage)
        ]
        if not positions:
            return command
        persisted = self.persist_messages(
            thread_id=thread_id,
            user_seq=user_seq,
            messages=[command_messages[index] for index in positions],
        )
        normalized_messages = list(command_messages)
        for index, message in zip(positions, persisted, strict=True):
            normalized_messages[index] = message
        return replace(command, update={**command.update, "messages": normalized_messages})
