from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from coding_agent.application import AgentApplication, TurnExecutionError
from coding_agent.config import ContextSettings, Settings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.workspace.file_undo import FileMutationRecorder


class FailingAgent:
    def invoke(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("model unavailable")

    def update_state(self, *args, **kwargs):
        del args, kwargs


class PartiallyCompletedAgent:
    def __init__(self, app: AgentApplication | None = None) -> None:
        self.app = app

    def invoke(self, *args, **kwargs):
        del args, kwargs
        assert self.app is not None
        self.app.message_persistence.persist_messages(
            thread_id="t1",
            user_seq=1,
            messages=[
                AIMessage(
                    id="assistant-tools",
                    content="",
                    tool_calls=[
                        {"name": "bash", "args": {"command": "first"}, "id": "call-1"},
                        {"name": "bash", "args": {"command": "second"}, "id": "call-2"},
                    ],
                ),
                ToolMessage(
                    id="tool-call-1",
                    content="first completed",
                    name="bash",
                    tool_call_id="call-1",
                ),
            ],
        )
        raise KeyboardInterrupt

    def update_state(self, *args, **kwargs):
        del args, kwargs


def test_failed_turn_reports_persisted_user_seq(database, tmp_path):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    app = AgentApplication(
        Settings(workspace_root=tmp_path), database, FailingAgent(), engine, recorder
    )

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "hello")

    assert raised.value.user_seq == 1
    assert raised.value.thread_id == "t1"
    assert isinstance(raised.value.cause, RuntimeError)
    preview = app.rollback_preview("t1", 1)
    assert preview.messages == 1


def test_interrupted_turn_closes_partial_tool_batch(database, tmp_path):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    agent = PartiallyCompletedAgent()
    app = AgentApplication(Settings(workspace_root=tmp_path), database, agent, engine, recorder)
    agent.app = app

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "run both")

    assert raised.value.interrupted is True
    assert raised.value.closed_tool_results == 1
    assert raised.value.finalization_errors == ()
    assert app.rollback_preview("t1", 1).messages == 4


def test_finalization_error_does_not_hide_original_turn_error(database, tmp_path, monkeypatch):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    app = AgentApplication(
        Settings(workspace_root=tmp_path), database, FailingAgent(), engine, recorder
    )

    def fail_to_close(**kwargs):
        del kwargs
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        app.message_persistence,
        "close_incomplete_tool_batch",
        fail_to_close,
    )

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "hello")

    assert str(raised.value.cause) == "model unavailable"
    assert raised.value.finalization_errors == (
        "补齐工具结果失败：RuntimeError: database unavailable",
    )
