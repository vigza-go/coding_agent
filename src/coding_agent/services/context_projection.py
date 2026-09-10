from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ..context.cover import ContextPiece
from ..context.engine import ContextEngine
from ..persistence.message_codec import decode_message_data


class ContextProjectionService:
    """把引擎选出的 pieces 投影成给模型的一串消息。

    这里**不再每轮注入工作状态**：那会让一个频繁变化的东西占据提示词的最前面，一改就
    断掉整条历史的缓存前缀。工作状态改由引擎在剪裁/压缩时贴成一条"便签"
    （``ThreadContextState.pin_work_state``，见 ``context/engine.py``），随 pieces 一起
    流进来，位置固定在压缩块之后、原文之前。
    """

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
            elif piece.kind == "work_state":
                # 用 HumanMessage 而不是 SystemMessage：langchain_anthropic 会把
                # SystemMessage 提到请求最前（system 字段），那就又回到"一改就断全前缀"
                # 的老毛病了。这条只是系统侧背景，包在标签里、排在原文之前。
                projected.append(
                    HumanMessage(
                        name="work_state",
                        content=f"<current_work_state>\n{piece.text}\n</current_work_state>",
                    )
                )
            else:
                projected.extend(decode_message_data(row.content_json) for row in piece.messages)
        return projected

    def build(self, thread_id: str) -> list[BaseMessage]:
        self.context_engine.compact_if_needed(thread_id)
        pieces = self.context_engine.rebuild(thread_id)
        projected = self.render_pieces(pieces)
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *projected]
