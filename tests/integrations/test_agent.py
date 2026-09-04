from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import PrivateAttr
from sqlalchemy import select

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.middleware import AgentRuntimeMiddleware, RunContext
from coding_agent.persistence.message_codec import encode_message
from coding_agent.persistence.models import FileBlob, FileMutation, MessageType, MutationStatus
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.call_limits import CallLimitService
from coding_agent.services.context_projection import ContextProjectionService
from coding_agent.services.message_persistence import MessagePersistenceService
from coding_agent.services.tool_execution import ToolExecutionService
from coding_agent.workspace.artifacts import ArtifactStore
from coding_agent.workspace.file_undo import FileMutationRecorder


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


class PromptRecordingModel(ToolAwareFakeModel):
    """记录每次真正送进模型的 message id 序列。

    应用不再配置 checkpointer，所以「模型看到的上下文 == 业务库投影」这条不变量只能直接
    在模型入口上断言——这本来就是我们关心的东西，比检查一份派生快照更贴近事实。
    """

    _seen: list[list[str | None]] = PrivateAttr(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append([getattr(message, "id", None) for message in messages])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_runtime_middleware(
    database,
    engine,
    root,
    *,
    model_limit: int = 20,
    tool_limit: int = 15,
    max_retries: int = 2,
    retryable_tools: list[str] | None = None,
) -> AgentRuntimeMiddleware:
    settings = ContextSettings()
    return AgentRuntimeMiddleware(
        persistence=MessagePersistenceService(database, engine),
        context_projection=ContextProjectionService(engine),
        call_limits=CallLimitService(model_limit=model_limit, tool_limit=tool_limit),
        tool_execution=ToolExecutionService(
            max_retries=max_retries,
            retryable_tools=retryable_tools or [],
        ),
        artifacts=ArtifactStore(root / "artifacts"),
        file_mutations=FileMutationRecorder(database, root),
        tool_result_inline_tokens=settings.tool_result_inline_tokens,
    )


@tool
def noop_tool() -> str:
    """Return a small deterministic result."""

    return "ok"


FAILURE_CALLS = 0
READ_FAILURE_CALLS = 0


@tool
def failing_write_tool() -> str:
    """Simulate an unexpected exception from a mutating tool."""

    global FAILURE_CALLS
    FAILURE_CALLS += 1
    raise RuntimeError("write failed")


@tool
def flaky_read_tool() -> str:
    """Fail once, then return a read result."""

    global READ_FAILURE_CALLS
    READ_FAILURE_CALLS += 1
    if READ_FAILURE_CALLS == 1:
        raise RuntimeError("temporary read failure")
    return "read ok"


def test_model_sees_the_projection_and_model_output_is_canonical(database, tmp_path):
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
        repo.save_work_state("t1", 1, {"step": "testing"})
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = PromptRecordingModel(responses=[AIMessage(id="assistant-1", content="ok")])
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[make_runtime_middleware(database, engine, tmp_path)],
        context_schema=RunContext,
    )
    config = {"configurable": {"thread_id": "t1", "user_seq": 1}}
    result = agent.invoke(
        {
            "messages": [human],
        },
        config=config,
        context=RunContext("t1", 1),
    )

    assert result["messages"][-1].content == "ok"
    final_messages = result["messages"]
    assert [message.id for message in final_messages] == [
        "work-state-1",
        "human-1",
        "assistant-1",
    ]
    assert final_messages[0].type == "system"  # 工作状态作为系统侧背景，置于用户提问之前
    # 没有 checkpointer，模型输入完全来自业务库投影：工作状态在前、以真实用户提问收尾。
    assert model._seen == [["work-state-1", "human-1"]]
    with database.session() as session:
        rows = AgentRepository(session).active_messages("t1")
        assert [row.langchain_message_id for row in rows] == ["human-1", "assistant-1"]


def _add_user_turn(database, engine: ContextEngine, user_seq: int, human: HumanMessage):
    """复刻 AgentApplication.run_turn 的落库前置：先写业务库，再同步引擎缓存。"""

    with database.session() as session:
        repo = AgentRepository(session)
        conversation = repo.get_or_create_conversation("t1")
        row = repo.add_message(
            thread_id="t1",
            user_seq=user_seq,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
        conversation.active_head_seq = user_seq
    engine.append_messages("t1", [row])
    return row


def test_cross_turn_history_comes_from_the_database_without_checkpointer(database, tmp_path):
    """不配 checkpointer 的前提：跨轮历史由业务库投影提供，而不是图的持久化状态。

    before_model 会先用 RemoveMessage(REMOVE_ALL) 清空 messages 再整体换成业务库投影，
    所以本轮输入必须已经落库，否则会被自己的投影抹掉——这条测试就是钉住那个方向。
    """

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    model = PromptRecordingModel(
        responses=[
            AIMessage(id="assistant-1", content="one"),
            AIMessage(id="assistant-2", content="two"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[make_runtime_middleware(database, engine, tmp_path)],
        context_schema=RunContext,
    )
    assert agent.checkpointer is None

    for user_seq, human in (
        (1, HumanMessage(id="human-1", content="first")),
        (2, HumanMessage(id="human-2", content="second")),
    ):
        _add_user_turn(database, engine, user_seq, human)
        result = agent.invoke(
            {
                "messages": [human],
            },
            config={"configurable": {"thread_id": "t1", "user_seq": user_seq}},
            context=RunContext("t1", user_seq),
        )
        last = result["messages"][-1]
        assert last.id == f"assistant-{user_seq}"

    # 第二轮送进模型的是第一轮的完整投影，并以本轮提问收尾
    assert model._seen == [["human-1"], ["human-1", "assistant-1", "human-2"]]


def test_command_messages_are_persisted_and_appended_to_context_cache(database):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    persistence = MessagePersistenceService(database, engine)
    command = Command(update={"messages": [ToolMessage(content="result", tool_call_id="call-1")]})

    persisted = persistence.persist_command(thread_id="t1", user_seq=1, command=command)

    tool_message = persisted.update["messages"][0]
    assert tool_message.id == "tool-call-1"
    pieces = engine.rebuild("t1")
    assert pieces[0].messages[0].langchain_message_id == "tool-call-1"
    with database.session() as session:
        rows = AgentRepository(session).active_messages("t1")
        assert [row.langchain_message_id for row in rows] == ["tool-call-1"]


def test_tool_limit_injected_result_is_persisted_before_next_model(database, tmp_path):
    human = HumanMessage(id="human-limit", content="run tools")
    with database.session() as session:
        AgentRepository(session).add_message(
            thread_id="limit-thread",
            user_seq=1,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="assistant-call-1",
                content="",
                tool_calls=[{"name": "noop_tool", "args": {}, "id": "call-1"}],
            ),
            AIMessage(
                id="assistant-call-2",
                content="",
                tool_calls=[{"name": "noop_tool", "args": {}, "id": "call-2"}],
            ),
            AIMessage(id="assistant-final", content="done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[noop_tool],
        middleware=[
            make_runtime_middleware(
                database,
                engine,
                tmp_path,
                model_limit=10,
                tool_limit=1,
            )
        ],
        context_schema=RunContext,
    )

    result = agent.invoke(
        {
            "messages": [human],
        },
        config={"configurable": {"thread_id": "limit-thread", "user_seq": 1}},
        context=RunContext("limit-thread", 1),
    )

    assert result["messages"][-1].content == "done"
    with database.session() as session:
        rows = AgentRepository(session).active_messages("limit-thread")
    assert [row.type for row in rows] == [
        MessageType.USER,
        MessageType.ASSISTANT,
        MessageType.TOOL,
        MessageType.ASSISTANT,
        MessageType.TOOL,
        MessageType.ASSISTANT,
    ]
    blocked_result = rows[-2].content_json["data"]
    assert blocked_result["tool_call_id"] == "call-2"
    assert blocked_result["status"] == "error"


def test_model_limit_injected_final_message_is_persisted(database, tmp_path):
    human = HumanMessage(id="human-model-limit", content="run")
    with database.session() as session:
        AgentRepository(session).add_message(
            thread_id="model-limit-thread",
            user_seq=1,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="assistant-before-limit",
                content="",
                tool_calls=[{"name": "noop_tool", "args": {}, "id": "model-call-1"}],
            )
        ]
    )
    agent = create_agent(
        model=model,
        tools=[noop_tool],
        middleware=[
            make_runtime_middleware(
                database,
                engine,
                tmp_path,
                model_limit=1,
                tool_limit=10,
            )
        ],
        context_schema=RunContext,
    )

    result = agent.invoke(
        {
            "messages": [human],
        },
        config={"configurable": {"thread_id": "model-limit-thread", "user_seq": 1}},
        context=RunContext("model-limit-thread", 1),
    )

    assert "模型调用上限" in result["messages"][-1].content
    with database.session() as session:
        rows = AgentRepository(session).active_messages("model-limit-thread")
    assert [row.type for row in rows] == [
        MessageType.USER,
        MessageType.ASSISTANT,
        MessageType.TOOL,
        MessageType.ASSISTANT,
    ]


def test_non_retryable_tool_returns_one_persisted_error(database, tmp_path):
    global FAILURE_CALLS
    FAILURE_CALLS = 0
    human = HumanMessage(id="human-tool-error", content="write")
    with database.session() as session:
        AgentRepository(session).add_message(
            thread_id="tool-error-thread",
            user_seq=1,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="assistant-tool-error",
                content="",
                tool_calls=[{"name": "failing_write_tool", "args": {}, "id": "failing-call"}],
            ),
            AIMessage(id="assistant-after-error", content="handled"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[failing_write_tool],
        middleware=[
            make_runtime_middleware(
                database,
                engine,
                tmp_path,
                retryable_tools=["flaky_read_tool"],
            )
        ],
        context_schema=RunContext,
    )

    result = agent.invoke(
        {
            "messages": [human],
        },
        config={"configurable": {"thread_id": "tool-error-thread", "user_seq": 1}},
        context=RunContext("tool-error-thread", 1),
    )

    assert result["messages"][-1].content == "handled"
    assert FAILURE_CALLS == 1
    with database.session() as session:
        rows = AgentRepository(session).active_messages("tool-error-thread")
    tool_result = rows[-2].content_json["data"]
    assert tool_result["tool_call_id"] == "failing-call"
    assert tool_result["status"] == "error"


def test_retryable_tool_succeeds_and_persists_one_result(database, tmp_path):
    global READ_FAILURE_CALLS
    READ_FAILURE_CALLS = 0
    human = HumanMessage(id="human-read-retry", content="read")
    with database.session() as session:
        AgentRepository(session).add_message(
            thread_id="read-retry-thread",
            user_seq=1,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="assistant-read-retry",
                content="",
                tool_calls=[{"name": "flaky_read_tool", "args": {}, "id": "read-call"}],
            ),
            AIMessage(id="assistant-after-read", content="done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[flaky_read_tool],
        middleware=[
            make_runtime_middleware(
                database,
                engine,
                tmp_path,
                retryable_tools=["flaky_read_tool"],
            )
        ],
        context_schema=RunContext,
    )

    result = agent.invoke(
        {
            "messages": [human],
        },
        config={"configurable": {"thread_id": "read-retry-thread", "user_seq": 1}},
        context=RunContext("read-retry-thread", 1),
    )

    assert result["messages"][-1].content == "done"
    assert READ_FAILURE_CALLS == 2
    with database.session() as session:
        rows = AgentRepository(session).active_messages("read-retry-thread")
    assert [row.type for row in rows] == [
        MessageType.USER,
        MessageType.ASSISTANT,
        MessageType.TOOL,
        MessageType.ASSISTANT,
    ]
    assert rows[-2].content_json["data"]["content"] == "read ok"


def test_retrying_write_uses_one_original_file_mutation(database, tmp_path):
    attempts = 0
    target = tmp_path / "retry.txt"
    target.write_text("old", encoding="utf-8")

    @tool("write_file")
    def retrying_write(file_path: str, content: str) -> str:
        """Overwrite a file and simulate a lost first response."""

        nonlocal attempts
        del file_path
        attempts += 1
        target.write_text(content, encoding="utf-8")
        if attempts == 1:
            raise RuntimeError("response lost after write")
        return "written"

    human = HumanMessage(id="human-write-retry", content="write")
    with database.session() as session:
        AgentRepository(session).add_message(
            thread_id="write-retry-thread",
            user_seq=1,
            message_type=MessageType.USER,
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    settings = ContextSettings()
    engine = ContextEngine(database, settings, DeterministicSummarizer())
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="assistant-write-retry",
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/retry.txt", "content": "new"},
                        "id": "write-call",
                    }
                ],
            ),
            AIMessage(id="assistant-after-write", content="done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[retrying_write],
        middleware=[
            make_runtime_middleware(
                database,
                engine,
                tmp_path,
                retryable_tools=["write_file"],
            )
        ],
        context_schema=RunContext,
    )

    result = agent.invoke(
        {
            "messages": [human],
        },
        config={"configurable": {"thread_id": "write-retry-thread", "user_seq": 1}},
        context=RunContext("write-retry-thread", 1),
    )

    assert result["messages"][-1].content == "done"
    assert attempts == 2
    assert target.read_text(encoding="utf-8") == "new"
    with database.session() as session:
        mutations = list(session.scalars(select(FileMutation).order_by(FileMutation.id)))
        assert len(mutations) == 1
        mutation = mutations[0]
        assert mutation.status == MutationStatus.SUCCEEDED
        assert mutation.before_blob_id is not None
        before_blob = session.get(FileBlob, mutation.before_blob_id)
        assert before_blob is not None
        assert before_blob.content == b"old"
