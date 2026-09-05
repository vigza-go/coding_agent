from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage


class TurnEventKind(StrEnum):
    MODEL_STARTED = "model_started"
    MODEL_FINISHED = "model_finished"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    SUMMARY_STARTED = "summary_started"
    SUMMARY_FINISHED = "summary_finished"


@dataclass(frozen=True)
class TurnEvent:
    kind: TurnEventKind
    name: str | None = None
    detail: str | None = None
    text: str | None = None


def _mid_turn_text(response: Any) -> str | None:
    """把模型自己写的正文捞出来，但只在它“还在接着干活”的时候。

    只有后面还会发起工具调用的消息才上报。`run_turn` 会单独把最终答案交回去（最后一条
    不带工具调用的 AI 消息），TUI 把它渲染成自己的面板，所以这里再报一遍就会印两次。
    同一道闸门也顺手把压缩摘要器的纯文本回复挡在对话记录外面。
    """
    parts: list[str] = []
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is None or not getattr(message, "tool_calls", None):
                continue
            text = (getattr(message, "text", "") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts) or None


class TurnEventGate:
    """一轮结束后就把回调丢掉，正在渲染的那个事件让它排空。"""

    def __init__(self, emit: Callable[[TurnEvent], None]) -> None:
        self._emit = emit
        self._lock = Lock()
        self._open = True

    def emit(self, event: TurnEvent) -> None:
        with self._lock:
            if self._open:
                self._emit(event)

    def close(self) -> None:
        with self._lock:
            self._open = False


class ProgressCallbackHandler(BaseCallbackHandler):
    # 压缩调用带这个 tag；命中时上报 SUMMARY_* 而不是 MODEL_*。
    SUMMARIZER_TAG = "summarizer"

    def __init__(self, emit: Callable[[TurnEvent], None]) -> None:
        self.emit = emit
        self._tool_names: dict[UUID, str] = {}
        self._summary_levels: dict[UUID, int] = {}
        self._lock = Lock()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        del serialized, messages
        tags = set(kwargs.get("tags") or [])
        if self.SUMMARIZER_TAG in tags:
            metadata = kwargs.get("metadata") or {}
            level = int(metadata.get("level", 0))
            run_id = kwargs.get("run_id")
            with self._lock:
                if run_id is not None:
                    self._summary_levels[run_id] = level
            self.emit(TurnEvent(TurnEventKind.SUMMARY_STARTED, f"L{level}"))
            return
        self.emit(TurnEvent(TurnEventKind.MODEL_STARTED))

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        with self._lock:
            level = self._summary_levels.pop(run_id, None) if run_id is not None else None
        if level is not None:
            self.emit(TurnEvent(TurnEventKind.SUMMARY_FINISHED, f"L{level}"))
            return
        self.emit(TurnEvent(TurnEventKind.MODEL_FINISHED, text=_mid_turn_text(response)))

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
        del kwargs
        name = self._pop_tool_name(run_id)
        if isinstance(output, ToolMessage) and output.status == "error":
            detail = "tool returned an error"
            if isinstance(output.artifact, dict) and "exit_code" in output.artifact:
                detail = f"exit_code={output.artifact['exit_code']}"
            self.emit(TurnEvent(TurnEventKind.TOOL_FAILED, name, detail))
            return
        self.emit(TurnEvent(TurnEventKind.TOOL_FINISHED, name))

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
