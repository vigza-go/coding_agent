from __future__ import annotations

from uuid import uuid4

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
