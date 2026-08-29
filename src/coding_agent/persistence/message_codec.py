from __future__ import annotations

import uuid
from typing import Any, TypeVar, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

from .models import Message, MessageType

MessageT = TypeVar("MessageT", bound=BaseMessage)


def ensure_message_id(message: MessageT, fallback: str | None = None) -> MessageT:
    if message.id:
        return message
    if fallback:
        identifier = fallback
    elif isinstance(message, ToolMessage) and message.tool_call_id:
        identifier = f"tool-{message.tool_call_id}"
    else:
        identifier = str(uuid.uuid4())
    return cast(MessageT, message.model_copy(update={"id": identifier}))


def message_type(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return MessageType.USER
    if isinstance(message, AIMessage):
        return MessageType.ASSISTANT
    if isinstance(message, ToolMessage):
        return MessageType.TOOL
    if isinstance(message, SystemMessage):
        return MessageType.SYSTEM
    return message.type


def encode_message(message: BaseMessage) -> dict:
    return message_to_dict(message)


def decode_message_data(content_json: dict[str, Any]) -> BaseMessage:
    return messages_from_dict([content_json])[0]


def decode_message(row: Message) -> BaseMessage:
    return decode_message_data(row.content_json)
