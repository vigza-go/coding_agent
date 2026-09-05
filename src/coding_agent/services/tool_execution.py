from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command


class ToolExecutionService:
    """执行一次逻辑上的工具调用，重不重试由策略说了算。"""

    def __init__(self, *, max_retries: int, retryable_tools: Collection[str]) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.max_retries = max_retries
        self.retryable_tools = frozenset(retryable_tools)

    def execute(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call.get("name", "unknown"))
        retries = self.max_retries if tool_name in self.retryable_tools else 0

        for attempt in range(retries + 1):
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as error:
                if attempt == retries:
                    tool_call_id = request.tool_call.get("id")
                    if not tool_call_id:
                        raise
                    return ToolMessage(
                        content=(
                            f"Tool '{tool_name}' failed after {attempt + 1} attempts "
                            f"with {type(error).__name__}: {error}"
                        ),
                        name=tool_name,
                        tool_call_id=str(tool_call_id),
                        status="error",
                    )
        raise RuntimeError("unreachable retry state")
