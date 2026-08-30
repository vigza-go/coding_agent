from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..persistence.message_codec import decode_message_data


class ContextProjectionService:
    def __init__(self, context_engine: ContextEngine) -> None:
        self.context_engine = context_engine

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
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *projected]
