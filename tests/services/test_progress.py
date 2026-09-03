from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from coding_agent.services.progress import (
    ProgressCallbackHandler,
    TurnEvent,
    TurnEventGate,
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


def test_turn_event_gate_drops_late_tool_completion():
    events = []
    gate = TurnEventGate(events.append)
    started = TurnEvent(TurnEventKind.TOOL_STARTED, "bash")
    late = TurnEvent(TurnEventKind.TOOL_FAILED, "bash", "exit_code=124")

    gate.emit(started)
    gate.close()
    gate.emit(late)

    assert events == [started]


def test_turn_event_gate_drains_event_already_being_rendered():
    entered = Event()
    release = Event()
    close_started = Event()

    def slow_emit(event):
        del event
        entered.set()
        release.wait(timeout=1)

    gate = TurnEventGate(slow_emit)
    event = TurnEvent(TurnEventKind.TOOL_FINISHED, "bash")

    with ThreadPoolExecutor(max_workers=2) as pool:
        emit_future = pool.submit(gate.emit, event)
        assert entered.wait(timeout=1)

        def close_gate():
            close_started.set()
            gate.close()

        close_future = pool.submit(close_gate)
        assert close_started.wait(timeout=1)
        assert not close_future.done()
        release.set()
        emit_future.result(timeout=1)
        close_future.result(timeout=1)


def _result(message):
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_prose_written_alongside_tool_calls_is_carried_by_the_event():
    """The bug this guards: mid-turn text used to be dropped entirely, so the user only
    ever saw the final message of a turn and lost every explanation written next to a
    tool call. `thinking` must stay out of it."""
    events = []
    ProgressCallbackHandler(events.append).on_llm_end(
        _result(
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "内部推理，不该进对话"},
                    {"type": "text", "text": "我先看一下配置"},
                ],
                tool_calls=[
                    {"name": "bash", "args": {"command": "ls"}, "id": "c1", "type": "tool_call"}
                ],
            )
        )
    )

    assert events[0].kind == TurnEventKind.MODEL_FINISHED
    assert events[0].text == "我先看一下配置"


def test_final_answer_is_left_to_the_turn_result_so_it_prints_once():
    """`run_turn` returns the last tool-call-free message and the TUI renders it as its
    own panel; re-emitting it here would print the same text twice."""
    events = []
    ProgressCallbackHandler(events.append).on_llm_end(_result(AIMessage(content="最终答复")))

    assert events[0].kind == TurnEventKind.MODEL_FINISHED
    assert events[0].text is None


def test_tool_call_message_without_prose_carries_no_text():
    events = []
    ProgressCallbackHandler(events.append).on_llm_end(
        _result(
            AIMessage(
                content=[{"type": "thinking", "thinking": "只想了一下"}],
                tool_calls=[
                    {"name": "bash", "args": {"command": "ls"}, "id": "c1", "type": "tool_call"}
                ],
            )
        )
    )

    assert events[0].text is None
