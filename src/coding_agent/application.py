from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
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
from .services.rollback import RollbackResult, RollbackService
from .workspace.file_undo import FileMutationRecorder


@dataclass(frozen=True)
class HistoryEntry:
    user_seq: int
    message_type: str
    content: object


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

    def run_turn(self, thread_id: str, text: str) -> AIMessage | None:
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
        response = self.agent.invoke(
            {"messages": [human], "current_user_seq": user_seq},
            config=self.config(thread_id, user_seq),
            context=RunContext(thread_id, user_seq),
        )
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

    def active_head(self, thread_id: str) -> int:
        with self.database.session() as session:
            return AgentRepository(session).get_or_create_conversation(thread_id).active_head_seq

    def active_history(self, thread_id: str) -> list[HistoryEntry]:
        with self.database.session() as session:
            rows = AgentRepository(session).active_messages(thread_id)
            return [
                HistoryEntry(row.user_seq, row.type, decode_message(row).content) for row in rows
            ]


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
