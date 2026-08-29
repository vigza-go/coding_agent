from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse, ToolCallRequest
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from ..config import ContextSettings
from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..persistence.database import Database
from ..persistence.message_codec import (
    decode_message_data,
    encode_message,
    ensure_message_id,
    message_type,
)
from ..persistence.repository import AgentRepository
from ..workspace.artifacts import ArtifactStore

MessageT = TypeVar("MessageT", bound=BaseMessage)


@dataclass(frozen=True)
class RunContext:
    thread_id: str
    user_seq: int


def pieces_to_messages(pieces: list[ContextPiece]) -> list[BaseMessage]:
    projected: list[BaseMessage] = []
    for piece in pieces:
        if piece.kind == "memory":
            projected.append(
                HumanMessage(
                    id=f"memory-{piece.block_id}",
                    name="compressed_history",
                    content=(
                        f'<compressed_history level="{piece.level}" '
                        f'messages="{piece.begin_message_id}-{piece.end_message_id}">\n'
                        f"{piece.text}\n</compressed_history>"
                    ),
                )
            )
        else:
            projected.extend(decode_message_data(row.content_json) for row in piece.messages)
    return projected


def trim_old_tool_results(messages: list[BaseMessage], keep: int) -> list[BaseMessage]:
    tool_positions = [
        index for index, message in enumerate(messages) if isinstance(message, ToolMessage)
    ]
    old_positions = set(tool_positions[:-keep]) if keep > 0 else set(tool_positions)
    result: list[BaseMessage] = []
    for index, message in enumerate(messages):
        if index in old_positions and isinstance(message, ToolMessage):
            result.append(
                message.model_copy(
                    update={
                        "content": (
                            "[较早的工具结果已从近期模型输入中剪裁；如需细节，请重新读取文件或执行查询。]"
                        )
                    }
                )
            )
        else:
            result.append(message)
    return result


class CanonicalPersistenceMiddleware(AgentMiddleware):
    def __init__(self, database: Database, context_engine: ContextEngine) -> None:
        self.database = database
        self.context_engine = context_engine

    def _persist(self, context: RunContext, messages: list[MessageT]) -> list[MessageT]:
        normalized: list[MessageT] = []
        rows = []
        with self.database.session() as session:
            repo = AgentRepository(session)
            for message in messages:
                current = ensure_message_id(message)
                normalized.append(current)
                rows.append(
                    repo.add_message(
                        thread_id=context.thread_id,
                        user_seq=context.user_seq,
                        message_type=message_type(current),
                        content_json=encode_message(current),
                        langchain_message_id=current.id,
                    )
                )
        self.context_engine.append_messages(context.thread_id, rows)
        return normalized

    def _persist_command(self, context: RunContext, command: Command[Any]) -> Command[Any]:
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
        persisted = self._persist(context, [command_messages[index] for index in positions])
        normalized_messages = list(command_messages)
        for index, message in zip(positions, persisted, strict=True):
            normalized_messages[index] = message
        return replace(
            command,
            update={**command.update, "messages": normalized_messages},
        )

    def wrap_model_call(self, request, handler):
        response = handler(request)
        context = request.runtime.context
        if context is None or not isinstance(response, ModelResponse):
            return response
        result = self._persist(context, list(response.result))
        # Keep LangGraph checkpoint IDs identical to canonical database IDs.
        return ModelResponse(result=result, structured_response=response.structured_response)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        response = handler(request)
        context = request.runtime.context
        if context is None:
            return response
        if isinstance(response, ToolMessage):
            return self._persist(context, [response])[0]
        return self._persist_command(context, response)


class ArtifactOffloadMiddleware(AgentMiddleware):
    def __init__(self, store: ArtifactStore, settings: ContextSettings) -> None:
        self.store = store
        self.inline_limit = settings.tool_result_inline_tokens

    def wrap_tool_call(self, request, handler):
        response = handler(request)
        context = request.runtime.context
        if context is None or not isinstance(response, ToolMessage):
            return response
        if isinstance(response.content, str):
            content = response.content
        else:
            content = json.dumps(response.content, ensure_ascii=False, default=str)
        tool_call_id = response.tool_call_id or request.tool_call.get("id")
        if not tool_call_id:
            raise RuntimeError("tool result offloading requires a tool_call_id")
        shortened, path = self.store.offload_tool_result(
            thread_id=context.thread_id,
            user_seq=context.user_seq,
            tool_call_id=tool_call_id,
            content=content,
            inline_limit=self.inline_limit,
        )
        if path is None:
            return response
        return response.model_copy(update={"content": shortened})


class ContextProjectionMiddleware(AgentMiddleware):
    def __init__(self, context_engine: ContextEngine, settings: ContextSettings) -> None:
        self.context_engine = context_engine
        self.recent_tools = settings.recent_tool_interactions

    def before_model(self, state, runtime):
        del state
        context = runtime.context
        if context is None:
            return None
        self.context_engine.compact_if_needed(context.thread_id)
        pieces = self.context_engine.rebuild(context.thread_id)
        projected = pieces_to_messages(pieces)
        snapshot = self.context_engine.current_work_state(context.thread_id)
        if snapshot is not None:
            projected.append(
                HumanMessage(
                    id=f"work-state-{snapshot.id}",
                    name="work_state",
                    content=(
                        "<current_work_state>\n"
                        + json.dumps(snapshot.state_json, ensure_ascii=False)
                        + "\n</current_work_state>"
                    ),
                )
            )
        projected = trim_old_tool_results(projected, self.recent_tools)
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *projected]}
