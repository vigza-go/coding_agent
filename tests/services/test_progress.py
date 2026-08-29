from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import ToolMessage

from coding_agent.services.progress import (
    ProgressCallbackHandler,
    TurnEventKind,
)


def test_progress_handler_tracks_tool_name_and_safe_detail():
    events = []
    handler = ProgressCallbackHandler(events.append)
    run_id = uuid4()

    handler.on_tool_start(
        {"name": "read_file"},
        "ignored",
        run_id=run_id,
        inputs={"file_path": "/src/app.py", "content": "not displayed"},
    )
    handler.on_tool_end("contents", run_id=run_id)

    assert [event.kind for event in events] == [
        TurnEventKind.TOOL_STARTED,
        TurnEventKind.TOOL_FINISHED,
    ]
    assert events[0].name == "read_file"
    assert events[0].detail == "/src/app.py"
    assert events[1].name == "read_file"


def test_progress_handler_reports_error_tool_messages_as_failures():
    events = []
    handler = ProgressCallbackHandler(events.append)
    run_id = uuid4()
    handler.on_tool_start({"name": "bash"}, "ignored", run_id=run_id)

    handler.on_tool_end(
        ToolMessage(
            content="failed",
            tool_call_id="call-1",
            status="error",
            artifact={"exit_code": 7},
        ),
        run_id=run_id,
    )

    assert events[-1].kind == TurnEventKind.TOOL_FAILED
    assert events[-1].name == "bash"
    assert events[-1].detail == "exit_code=7"
