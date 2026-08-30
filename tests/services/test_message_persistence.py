from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.persistence.message_codec import decode_message
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.message_persistence import MessagePersistenceService


def test_incomplete_tool_batch_is_closed_and_late_result_is_idempotent(database):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    persistence = MessagePersistenceService(database, engine)
    assistant = AIMessage(
        id="assistant-tools",
        content="",
        tool_calls=[
            {"name": "bash", "args": {"command": "first"}, "id": "call-1"},
            {"name": "bash", "args": {"command": "second"}, "id": "call-2"},
        ],
    )
    first_result = ToolMessage(
        id="tool-call-1",
        content="first completed",
        name="bash",
        tool_call_id="call-1",
    )
    persistence.persist_messages(
        thread_id="t1",
        user_seq=1,
        messages=[assistant, first_result],
    )

    closed = persistence.close_incomplete_tool_batch(thread_id="t1", user_seq=1)

    assert closed == 1
    with database.session() as session:
        rows = AgentRepository(session).active_messages("t1")
        messages = [decode_message(row) for row in rows]
    assert len(messages) == 3
    synthetic = messages[-1]
    assert isinstance(synthetic, ToolMessage)
    assert synthetic.id == "tool-call-2"
    assert synthetic.tool_call_id == "call-2"
    assert synthetic.status == "error"

    persistence.persist_messages(
        thread_id="t1",
        user_seq=1,
        messages=[
            ToolMessage(
                content="late real result",
                name="bash",
                tool_call_id="call-2",
            )
        ],
    )
    with database.session() as session:
        rows = AgentRepository(session).active_messages("t1")
        final = decode_message(rows[-1])
    assert len(rows) == 3
    assert isinstance(final, ToolMessage)
    assert final.status == "error"
    assert "提前结束" in str(final.content)
