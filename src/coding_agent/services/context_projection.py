from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ..config import ContextSettings
from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..persistence.message_codec import decode_message_data


class ContextProjectionService:
    def __init__(self, context_engine: ContextEngine, settings: ContextSettings) -> None:
        self.context_engine = context_engine
        self.recent_tools = settings.recent_tool_interactions

    @staticmethod
    def render_pieces(pieces: list[ContextPiece]) -> list[BaseMessage]:
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

    @staticmethod
    def _trim_old_tool_results(messages: list[BaseMessage], keep: int) -> list[BaseMessage]:
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
                                "[较早的工具结果已从近期模型输入中剪裁；如需细节，"
                                "请重新读取文件或执行查询。]"
                            )
                        }
                    )
                )
            else:
                result.append(message)
        return result

    def build(self, thread_id: str) -> list[BaseMessage]:
        self.context_engine.compact_if_needed(thread_id)
        pieces = self.context_engine.rebuild(thread_id)
        projected = self.render_pieces(pieces)
        snapshot = self.context_engine.current_work_state(thread_id)
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
        projected = self._trim_old_tool_results(projected, self.recent_tools)
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *projected]
