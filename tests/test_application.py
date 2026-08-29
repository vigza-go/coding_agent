from __future__ import annotations

import pytest

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
