from __future__ import annotations

from dataclasses import replace
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

from ..context.engine import ContextEngine
from ..persistence.database import Database
from ..persistence.message_codec import (
    decode_message,
    encode_message,
    ensure_message_id,
    message_type,
)
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

    def close_incomplete_tool_batch(self, *, thread_id: str, user_seq: int) -> int:
        """Close a trailing partial tool batch so provider protocol remains valid."""

        with self.database.session() as session:
            rows = [
                row
                for row in AgentRepository(session).active_messages(thread_id)
                if row.user_seq == user_seq
            ]
        messages = [decode_message(row) for row in rows]
        if not messages:
            return 0

        batch_start = len(messages)
        while batch_start > 0 and isinstance(messages[batch_start - 1], ToolMessage):
            batch_start -= 1
        assistant_index = batch_start - 1
        if assistant_index < 0:
            return 0
        assistant = messages[assistant_index]
        if not isinstance(assistant, AIMessage) or not assistant.tool_calls:
            return 0

        completed_ids = {
            message.tool_call_id
            for message in messages[batch_start:]
            if isinstance(message, ToolMessage)
        }
        missing = [
            tool_call
            for tool_call in assistant.tool_calls
            if tool_call.get("id") not in completed_ids
        ]
        synthetic = [
            ToolMessage(
                id=f"tool-{tool_call['id']}",
                content="工具调用因本轮提前结束，未产生可用结果。",
                name=str(tool_call.get("name", "tool")),
                tool_call_id=str(tool_call["id"]),
                status="error",
            )
            for tool_call in missing
            if tool_call.get("id")
        ]
        if synthetic:
            self.persist_messages(
                thread_id=thread_id,
                user_seq=user_seq,
                messages=synthetic,
            )
        return len(synthetic)
