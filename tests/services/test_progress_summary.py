from __future__ import annotations

from uuid import uuid4

from coding_agent.services.progress import (
    ProgressCallbackHandler,
    TurnEvent,
    TurnEventKind,
)


def _recorder():
    events: list[TurnEvent] = []
    handler = ProgressCallbackHandler(events.append)
    return events, handler


def test_summarizer_call_emits_summary_events_instead_of_model_events():
    events, handler = _recorder()
    run_id = uuid4()
    handler.on_chat_model_start(
        {}, [], tags=["summarizer"], metadata={"level": 3}, run_id=run_id
    )
    handler.on_llm_end(object(), run_id=run_id)

    kinds = [event.kind for event in events]
    assert kinds == [
        TurnEventKind.SUMMARY_STARTED,
        TurnEventKind.SUMMARY_FINISHED,
    ]
    assert events[0].name == "L3"
    assert events[1].name == "L3"
    assert TurnEventKind.MODEL_STARTED not in kinds


def test_plain_model_call_still_emits_model_events():
    events, handler = _recorder()
    run_id = uuid4()
    handler.on_chat_model_start({}, [], run_id=run_id)
    handler.on_llm_end(object(), run_id=run_id)

    kinds = [event.kind for event in events]
    assert kinds == [
        TurnEventKind.MODEL_STARTED,
        TurnEventKind.MODEL_FINISHED,
    ]
