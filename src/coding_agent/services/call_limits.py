from __future__ import annotations

from dataclasses import dataclass, field

from langchain.agents.middleware.types import ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class RunCallBudget:
    model_calls: int = 0
    tool_calls: int = 0
    blocked_tool_call_ids: set[str] = field(default_factory=set)

    def reserve_model_call(self, limit: int) -> bool:
        if self.model_calls >= limit:
            return False
        self.model_calls += 1
        return True

    def register_tool_calls(self, tool_call_ids: list[str], limit: int) -> None:
        remaining = max(0, limit - self.tool_calls)
        self.tool_calls += min(len(tool_call_ids), remaining)
        self.blocked_tool_call_ids.update(tool_call_ids[remaining:])


class CallLimitService:
    def __init__(self, *, model_limit: int, tool_limit: int) -> None:
        if model_limit < 1 or tool_limit < 1:
            raise ValueError("model_limit and tool_limit must be positive")
        self.model_limit = model_limit
        self.tool_limit = tool_limit

    def before_model(self, budget: RunCallBudget) -> ModelResponse | None:
        if budget.reserve_model_call(self.model_limit):
            return None
        return ModelResponse(
            result=[
                AIMessage(
                    content=f"本轮已达到模型调用上限（{self.model_limit} 次），代理循环已停止。"
                )
            ]
        )

    def observe_model_response(self, budget: RunCallBudget, response: ModelResponse) -> None:
        for message in response.result:
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_call_ids: list[str] = []
                for tool_call in message.tool_calls:
                    tool_call_id = tool_call.get("id")
                    if not tool_call_id:
                        raise RuntimeError("model tool call is missing an id")
                    tool_call_ids.append(tool_call_id)
                budget.register_tool_calls(tool_call_ids, self.tool_limit)

    def blocked_tool_result(
        self, budget: RunCallBudget, request: ToolCallRequest
    ) -> ToolMessage | None:
        tool_call_id = str(request.tool_call.get("id", "unknown"))
        if tool_call_id not in budget.blocked_tool_call_ids:
            return None
        tool_name = str(request.tool_call.get("name", "unknown"))
        return ToolMessage(
            content=f"本轮已达到工具调用上限（{self.tool_limit} 次），该工具未执行。",
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )
