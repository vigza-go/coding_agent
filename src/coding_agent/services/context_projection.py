from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..context.work_state import render, render_index
from ..persistence.message_codec import decode_message_data


class ContextProjectionService:
    def __init__(self, context_engine: ContextEngine) -> None:
        self.context_engine = context_engine
        # 进程内记录"上一轮发给模型的是哪个快照"。工作状态每次写入都是新的一行、
        # id 单调递增，所以比 id 就是精确判等，不需要 hash、也不加数据库列。
        # 缓存失效或重启只会让它退化成"重新给一次全文"，方向是安全的。
        self._sent_state_id: dict[str, int] = {}

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
            body = (
                render_index(snapshot.state_json)
                if self._sent_state_id.get(thread_id) == snapshot.id
                else render(snapshot.state_json)
            )
            self._sent_state_id[thread_id] = snapshot.id
            projected.append(
                HumanMessage(
                    id=f"work-state-{snapshot.id}",
                    name="work_state",
                    content=f"<current_work_state>\n{body}\n</current_work_state>",
                )
            )
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *projected]
