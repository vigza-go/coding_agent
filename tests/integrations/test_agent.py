from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.langchain_agent import RuntimeState
from coding_agent.integrations.middleware import (
    CanonicalPersistenceMiddleware,
    ContextProjectionMiddleware,
    RunContext,
)
from coding_agent.persistence.message_codec import encode_message
from coding_agent.persistence.models import MessageType
from coding_agent.persistence.repository import AgentRepository


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


def test_projection_is_checkpointed_and_model_output_is_canonical(database):
    human = HumanMessage(id="human-1", content="hello")
    with database.session() as session:
        repo = AgentRepository(session)
        repo.get_or_create_conversation("t1").active_head_seq = 1
        repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = ToolAwareFakeModel(responses=[AIMessage(id="assistant-1", content="ok")])
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            CanonicalPersistenceMiddleware(database, engine),
            ContextProjectionMiddleware(engine, settings),
        ],
        state_schema=RuntimeState,
        context_schema=RunContext,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1", "user_seq": 1}}
    result = agent.invoke(
        {"messages": [human], "current_user_seq": 1},
        config=config,
        context=RunContext("t1", 1),
    )

    assert result["messages"][-1].content == "ok"
    checkpoint_messages = agent.get_state(config).values["messages"]
    assert [message.id for message in checkpoint_messages] == ["human-1", "assistant-1"]
    with database.session() as session:
        rows = AgentRepository(session).active_messages("t1")
        assert [row.langchain_message_id for row in rows] == ["human-1", "assistant-1"]


def test_command_messages_are_persisted_and_appended_to_context_cache(database):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    middleware = CanonicalPersistenceMiddleware(database, engine)
    command = Command(update={"messages": [ToolMessage(content="result", tool_call_id="call-1")]})

    persisted = middleware._persist_command(RunContext("t1", 1), command)

    tool_message = persisted.update["messages"][0]
    assert tool_message.id == "tool-call-1"
    pieces = engine.rebuild("t1")
    assert pieces[0].messages[0].langchain_message_id == "tool-call-1"
    with database.session() as session:
        rows = AgentRepository(session).active_messages("t1")
        assert [row.langchain_message_id for row in rows] == ["tool-call-1"]
