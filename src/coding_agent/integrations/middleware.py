from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from ..services.call_limits import CallLimitService, RunCallBudget
from ..services.context_projection import ContextProjectionService
from ..services.message_persistence import MessagePersistenceService
from ..services.tool_execution import ToolExecutionService
from ..workspace.artifacts import ArtifactStore
from ..workspace.file_undo import FileMutationRecorder


@dataclass
class RunContext:
    thread_id: str
    user_seq: int
    budget: RunCallBudget = field(default_factory=RunCallBudget, init=False, repr=False)


class AgentRuntimeMiddleware(AgentMiddleware[Any, RunContext, Any]):
    """Orchestrate one model/tool lifecycle using explicit internal service order."""

    def __init__(
        self,
        *,
        persistence: MessagePersistenceService,
        context_projection: ContextProjectionService,
        call_limits: CallLimitService,
        tool_execution: ToolExecutionService,
        artifacts: ArtifactStore,
        file_mutations: FileMutationRecorder,
        tool_result_inline_tokens: int,
    ) -> None:
        self.persistence = persistence
        self.context_projection = context_projection
        self.call_limits = call_limits
        self.tool_execution = tool_execution
        self.artifacts = artifacts
        self.file_mutations = file_mutations
        self.tool_result_inline_tokens = tool_result_inline_tokens

    def before_model(self, state, runtime):
        del state
        context = runtime.context
        if context is None:
            return None
        return {"messages": self.context_projection.build(context.thread_id)}

    def wrap_model_call(self, request, handler):
        context = request.runtime.context
        if context is None:
            return handler(request)

        response = self.call_limits.before_model(context.budget)
        if response is None:
            response = handler(request)
        if not isinstance(response, ModelResponse):
            return response

        self.call_limits.observe_model_response(context.budget, response)
        result = self.persistence.persist_messages(
            thread_id=context.thread_id,
            user_seq=context.user_seq,
            messages=response.result,
        )
        return ModelResponse(result=result, structured_response=response.structured_response)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        context = request.runtime.context
        if context is None:
            return handler(request)

        response = self.call_limits.blocked_tool_result(context.budget, request)
        mutation_id: int | None = None
        if response is None:
            mutation_id = self.file_mutations.begin_for_tool(
                thread_id=context.thread_id,
                user_seq=context.user_seq,
                tool_call=request.tool_call,
            )
            try:
                response = self.tool_execution.execute(request, handler)
            except GraphBubbleUp:
                self._finish_interrupted_mutation(mutation_id)
                raise
            except Exception:
                self._finish_interrupted_mutation(mutation_id)
                raise
            if mutation_id is not None:
                self.file_mutations.finish(
                    mutation_id,
                    succeeded=self._tool_succeeded(response),
                )

        if isinstance(response, ToolMessage):
            response = self._offload_tool_result(context, request, response)
            return self.persistence.persist_messages(
                thread_id=context.thread_id,
                user_seq=context.user_seq,
                messages=[response],
            )[0]
        return self.persistence.persist_command(
            thread_id=context.thread_id,
            user_seq=context.user_seq,
            command=response,
        )

    def _finish_interrupted_mutation(self, mutation_id: int | None) -> None:
        if mutation_id is not None:
            self.file_mutations.finish(mutation_id, succeeded=False)

    @staticmethod
    def _tool_succeeded(response: ToolMessage | Command[Any]) -> bool:
        if isinstance(response, ToolMessage):
            return response.status != "error"
        if not isinstance(response.update, dict):
            return True
        messages = response.update.get("messages")
        if not isinstance(messages, list):
            return True
        return not any(
            isinstance(message, ToolMessage) and message.status == "error" for message in messages
        )

    def _offload_tool_result(
        self,
        context: RunContext,
        request: ToolCallRequest,
        response: ToolMessage,
    ) -> ToolMessage:
        if isinstance(response.content, str):
            content = response.content
        else:
            content = json.dumps(response.content, ensure_ascii=False, default=str)
        tool_call_id = response.tool_call_id or request.tool_call.get("id")
        if not tool_call_id:
            raise RuntimeError("tool result offloading requires a tool_call_id")
        shortened, path = self.artifacts.offload_tool_result(
            thread_id=context.thread_id,
            user_seq=context.user_seq,
            tool_call_id=tool_call_id,
            content=content,
            inline_limit=self.tool_result_inline_tokens,
        )
        if path is None:
            return response
        return response.model_copy(update={"content": shortened})
