from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class TurnEventKind(StrEnum):
    MODEL_STARTED = "model_started"
    MODEL_FINISHED = "model_finished"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"


@dataclass(frozen=True)
class TurnEvent:
    kind: TurnEventKind
    name: str | None = None
    detail: str | None = None


class ProgressCallbackHandler(BaseCallbackHandler):
    def __init__(self, emit: Callable[[TurnEvent], None]) -> None:
        self.emit = emit
        self._tool_names: dict[UUID, str] = {}
        self._lock = Lock()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        del serialized, messages, kwargs
        self.emit(TurnEvent(TurnEventKind.MODEL_STARTED))

    def on_llm_end(self, response, **kwargs):
        del response, kwargs
        self.emit(TurnEvent(TurnEventKind.MODEL_FINISHED))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del input_str, kwargs
        name = str(serialized.get("name", "tool"))
        with self._lock:
            self._tool_names[run_id] = name
        self.emit(TurnEvent(TurnEventKind.TOOL_STARTED, name, self._tool_detail(inputs)))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        del output, kwargs
        self.emit(TurnEvent(TurnEventKind.TOOL_FINISHED, self._pop_tool_name(run_id)))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        self.emit(
            TurnEvent(
                TurnEventKind.TOOL_FAILED,
                self._pop_tool_name(run_id),
                f"{type(error).__name__}: {error}",
            )
        )

    def _pop_tool_name(self, run_id: UUID) -> str:
        with self._lock:
            return self._tool_names.pop(run_id, "tool")

    @staticmethod
    def _tool_detail(inputs: dict[str, Any] | None) -> str | None:
        if not inputs:
            return None
        for key in ("file_path", "path", "pattern", "command"):
            value = inputs.get(key)
            if value:
                text = str(value).replace("\n", " ")
                return text[:120] + ("…" if len(text) > 120 else "")
        return None
