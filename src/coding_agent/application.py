from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .config import Settings
from .context.cover import ContextPiece
from .context.engine import ContextEngine
from .context.summarizer import LangChainSummarizer
from .integrations.langchain_agent import build_model, create_langchain_agent
from .integrations.middleware import RunContext
from .persistence.database import Database
from .persistence.message_codec import decode_message, encode_message
from .persistence.models import MessageType
from .persistence.repository import AgentRepository
from .services.context_projection import ContextProjectionService
from .services.progress import ProgressCallbackHandler, TurnEvent
from .services.rollback import RollbackPreview, RollbackResult, RollbackService
from .workspace.file_undo import FileMutationRecorder


@dataclass(frozen=True)
class HistoryEntry:
    user_seq: int
    message_type: str
    content: object


@dataclass(frozen=True)
class ThreadSummary:
    thread_id: str
    active_head_seq: int
    updated_at: datetime


@dataclass(frozen=True)
class ThreadStatus:
    thread_id: str
    active_head_seq: int
    next_user_seq: int
    memory_levels: tuple[int, ...]
    memory_tokens: int
    memory_limit: int
    working_messages: int
    working_tokens: int
    working_trigger: int
    work_state: dict[str, Any] | None


class TurnExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        thread_id: str,
        user_seq: int,
        cause: BaseException,
        interrupted: bool = False,
    ) -> None:
        self.thread_id = thread_id
        self.user_seq = user_seq
        self.cause = cause
        self.interrupted = interrupted
        label = "interrupted" if interrupted else "failed"
        super().__init__(f"turn {user_seq} {label}: {cause}")


class AgentApplication:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        agent: Any,
        context_engine: ContextEngine,
        recorder: FileMutationRecorder,
    ) -> None:
        self.settings = settings
        self.database = database
        self.agent = agent
        self.recorder = recorder
        self.context_engine = context_engine
        self.rollback_service = RollbackService(database, context_engine, recorder)

    @staticmethod
    def config(thread_id: str, user_seq: int | None = None) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if user_seq is not None:
            configurable["user_seq"] = user_seq
        return {"configurable": configurable}

    def run_turn(
        self,
        thread_id: str,
        text: str,
        *,
        on_event: Callable[[TurnEvent], None] | None = None,
    ) -> AIMessage | None:
        with self.database.session() as session:
            repo = AgentRepository(session)
            user_seq = repo.reserve_user_seq(thread_id)
            human = HumanMessage(id=str(uuid.uuid4()), content=text)
            row = repo.add_message(
                thread_id=thread_id,
                user_seq=user_seq,
                message_type=MessageType.USER,
                content_json=encode_message(human),
                langchain_message_id=human.id,
            )
        self.context_engine.append_messages(thread_id, [row])
        config = self.config(thread_id, user_seq)
        if on_event is not None:
            config["callbacks"] = [ProgressCallbackHandler(on_event)]
        try:
            response = self.agent.invoke(
                {"messages": [human], "current_user_seq": user_seq},
                config=config,
                context=RunContext(thread_id, user_seq),
            )
        except KeyboardInterrupt as error:
            raise TurnExecutionError(
                thread_id=thread_id,
                user_seq=user_seq,
                cause=error,
                interrupted=True,
            ) from error
        except Exception as error:
            raise TurnExecutionError(
                thread_id=thread_id,
                user_seq=user_seq,
                cause=error,
            ) from error
        return next(
            (
                message
                for message in reversed(response.get("messages", []))
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            None,
        )

    def rollback(self, thread_id: str, user_seq: int) -> RollbackResult:
        def repair(pieces: list[ContextPiece], work_state: dict | None) -> None:
            replacement = [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *ContextProjectionService.render_pieces(pieces),
            ]
            values: dict[str, Any] = {
                "messages": replacement,
                "work_state": work_state or {},
            }
            self.agent.update_state(self.config(thread_id), values)

        return self.rollback_service.rollback(thread_id, user_seq, update_checkpoint=repair)

    def rollback_preview(self, thread_id: str, user_seq: int) -> RollbackPreview:
        return self.rollback_service.preview(thread_id, user_seq)

    def active_head(self, thread_id: str) -> int:
        with self.database.session() as session:
            return AgentRepository(session).get_or_create_conversation(thread_id).active_head_seq

    def active_history(self, thread_id: str, *, limit: int | None = 20) -> list[HistoryEntry]:
        with self.database.session() as session:
            rows = AgentRepository(session).active_messages(thread_id, limit=limit)
            return [
                HistoryEntry(row.user_seq, row.type, decode_message(row).content) for row in rows
            ]

    def list_threads(self, *, limit: int = 50) -> list[ThreadSummary]:
        with self.database.session() as session:
            rows = AgentRepository(session).conversations(limit=limit)
            return [
                ThreadSummary(row.thread_id, row.active_head_seq, row.updated_at) for row in rows
            ]

    def thread_status(self, thread_id: str) -> ThreadStatus:
        usage = self.context_engine.usage(thread_id)
        with self.database.session() as session:
            conversation = AgentRepository(session).get_or_create_conversation(thread_id)
            snapshot = AgentRepository(session).latest_work_state(thread_id)
            return ThreadStatus(
                thread_id=thread_id,
                active_head_seq=conversation.active_head_seq,
                next_user_seq=conversation.next_user_seq,
                memory_levels=usage.memory_levels,
                memory_tokens=usage.memory_tokens,
                memory_limit=self.settings.context.compression_limit,
                working_messages=usage.working_messages,
                working_tokens=usage.working_tokens,
                working_trigger=self.settings.context.working_trigger,
                work_state=snapshot.state_json if snapshot is not None else None,
            )


@contextmanager
def create_application(settings: Settings) -> Generator[AgentApplication, None, None]:
    database = Database(settings.database_url)
    database.create_schema()
    model = build_model(settings)
    context_engine = ContextEngine(database, settings.context, LangChainSummarizer(model))
    recorder = FileMutationRecorder(database, settings.workspace_root)
    with create_langchain_agent(
        settings=settings,
        database=database,
        context_engine=context_engine,
        recorder=recorder,
        model=model,
    ) as agent:
        yield AgentApplication(settings, database, agent, context_engine, recorder)
