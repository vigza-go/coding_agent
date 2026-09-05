from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .config import Settings
from .context.engine import ContextEngine
from .context.summarizer import LangChainSummarizer
from .integrations.langchain_agent import build_model, build_summary_model, create_langchain_agent
from .integrations.middleware import RunContext
from .integrations.search import make_search_client
from .integrations.subagent import SubAgentJobs
from .persistence.database import Database
from .persistence.message_codec import decode_message, encode_message
from .persistence.models import MessageType
from .persistence.repository import AgentRepository
from .services.bash_execution import BashExecutionService
from .services.message_persistence import MessagePersistenceService
from .services.progress import ProgressCallbackHandler, TurnEvent
from .services.rollback import RollbackPreview, RollbackResult, RollbackService
from .services.usage import RecentUsage
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
    bash_enabled: bool
    search_enabled: bool
    work_state: dict[str, Any] | None


class TurnExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        thread_id: str,
        user_seq: int,
        cause: BaseException,
        interrupted: bool = False,
        closed_tool_results: int = 0,
        finalization_errors: tuple[str, ...] = (),
    ) -> None:
        self.thread_id = thread_id
        self.user_seq = user_seq
        self.cause = cause
        self.interrupted = interrupted
        self.closed_tool_results = closed_tool_results
        self.finalization_errors = finalization_errors
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
        bash_executor: BashExecutionService | None = None,
        subagent_jobs: Any = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.agent = agent
        self.recorder = recorder
        self.context_engine = context_engine
        self.bash_executor = bash_executor
        # 外派的子代理登记处；本轮收工前靠它把报告等回来，None 表示功能未开。
        self.subagent_jobs = subagent_jobs
        self.message_persistence = MessagePersistenceService(database, context_engine)
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
        """跑一轮。thread 的占用是会话级的（进入时取得，退出才放），所以这里只是
        幂等地确认一次：掉了就重取，已被别的会话接管就抛 :class:`ThreadBusyError`。
        """

        self.enter_thread(thread_id)
        return self._run_turn_locked(thread_id, text, on_event=on_event)

    def enter_thread(self, thread_id: str) -> None:
        """占用一条 thread，直到 :meth:`leave_thread`；正被别人占着就抛错。"""

        self.database.turn_guard.acquire(thread_id)

    def leave_thread(self) -> None:
        self.database.turn_guard.release()

    def _run_turn_locked(
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
        # 记下本轮 user_seq，之后每次写时间轴都用它验证"没人插过这条 thread"。
        self.database.turn_guard.bind(user_seq)
        self.context_engine.append_messages(thread_id, [row])
        if self.bash_executor is not None:
            self.bash_executor.prepare_turn()
        config = self.config(thread_id, user_seq)
        if on_event is not None:
            config["callbacks"] = [ProgressCallbackHandler(on_event)]
        try:
            # 整轮共用一个 RunContext：额度计数挂在它身上，所以"父说的那几步 + 为处理
            # 子代理报告而续跑的几步"合起来受一次 model_call_limit 封顶，不会续跑一次就
            # 重新领一份额度。
            run_context = RunContext(thread_id, user_seq)
            response = self.agent.invoke(
                {"messages": [human]},
                config=config,
                context=run_context,
            )
            # 父模型给出答案不等于本轮结束：它派出去、还没交回的子代理算本轮的一部分。
            # 等齐 → 报告作为一条注入消息再喂一步（模型据此继续干活，也可能又派新活），
            # 循环到没有待收的活儿为止。这期间输入框根本没在等输入，用户发不进新消息，
            # 所以 Ctrl-C 打的是"这一轮"，父子一起停。
            while (report := self._collect_subagent_reports(thread_id, user_seq)) is not None:
                response = self.agent.invoke(
                    {"messages": [report]},
                    config=config,
                    context=run_context,
                )
        except KeyboardInterrupt as error:
            closed_tool_results, finalization_errors = self._finalize_failed_turn(
                thread_id, user_seq
            )
            raise TurnExecutionError(
                thread_id=thread_id,
                user_seq=user_seq,
                cause=error,
                interrupted=True,
                closed_tool_results=closed_tool_results,
                finalization_errors=finalization_errors,
            ) from error
        except Exception as error:
            closed_tool_results, finalization_errors = self._finalize_failed_turn(
                thread_id, user_seq
            )
            raise TurnExecutionError(
                thread_id=thread_id,
                user_seq=user_seq,
                cause=error,
                closed_tool_results=closed_tool_results,
                finalization_errors=finalization_errors,
            ) from error
        return next(
            (
                message
                for message in reversed(response.get("messages", []))
                if isinstance(message, AIMessage) and not message.tool_calls
            ),
            None,
        )

    def _collect_subagent_reports(self, thread_id: str, user_seq: int) -> HumanMessage | None:
        """等齐本轮外派的子代理，把报告落成一条可投影的消息；没有待收的任务就返回 None。

        投影时 messages 通道是被整体覆盖的（见 ``AgentRuntimeMiddleware.before_model``），
        所以报告必须像用户那句话一样先入库，模型才看得见它 —— 不能只塞进 invoke 的入参。
        """

        if self.subagent_jobs is None:
            return None
        report = self.subagent_jobs.finish_turn()  # 阻塞点：Ctrl-C 会打在这里
        if report is None:
            return None
        # 包成 HumanMessage 但打上 name，理由跟压缩历史那条一样（见 ContextProjectionService）：
        # 它是"系统侧递进来的材料"，不是用户敲的话；靠 name 区分，不新增 role 或表。
        message = HumanMessage(
            id=str(uuid.uuid4()), name="subagent_report", content=f"【子代理回报】\n{report}"
        )
        with self.database.session() as session:
            row = AgentRepository(session).add_message(
                thread_id=thread_id,
                user_seq=user_seq,
                message_type=MessageType.USER,
                content_json=encode_message(message),
                langchain_message_id=message.id,
            )
        self.context_engine.append_messages(thread_id, [row])
        return message

    def _finalize_failed_turn(self, thread_id: str, user_seq: int) -> tuple[int, tuple[str, ...]]:
        errors: list[str] = []
        # 本轮派出去、还在飞的子代理跟着一起终止：它们记的账也挂在这一轮名下，
        # 留着只会往一个已经作废的回合里写文件。
        if self.subagent_jobs is not None:
            try:
                self.subagent_jobs.terminate_all()
            except Exception as error:  # noqa: BLE001 - preserve the original turn failure
                errors.append(f"终止子代理失败：{type(error).__name__}: {error}")
        if self.bash_executor is not None:
            try:
                self.bash_executor.interrupt_all()
            except Exception as error:  # noqa: BLE001 - preserve the original turn failure
                errors.append(f"终止 Bash 失败：{type(error).__name__}: {error}")
        try:
            closed = self.message_persistence.close_incomplete_tool_batch(
                thread_id=thread_id,
                user_seq=user_seq,
            )
        except Exception as error:  # noqa: BLE001 - preserve the original turn failure
            closed = 0
            errors.append(f"补齐工具结果失败：{type(error).__name__}: {error}")
        return closed, tuple(errors)

    def rollback(self, thread_id: str, user_seq: int) -> RollbackResult:
        # 不再回写 LangGraph checkpoint：应用库才是唯一真相源，每轮模型调用前
        # before_model 都会用投影结果整体覆盖 messages 通道（见 AgentRuntimeMiddleware）。
        return self.rollback_service.rollback(thread_id, user_seq)

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

    def recent_usage(self, thread_id: str, *, limit: int | None = None) -> RecentUsage:
        with self.database.session() as session:
            records = AgentRepository(session).recent_usage_metadata(
                thread_id,
                limit=self.settings.tui.usage_recent_messages if limit is None else limit,
            )
        return RecentUsage.from_metadata(records)

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
                bash_enabled=self.settings.agent.bash_enabled,
                search_enabled=self.settings.agent.search_enabled,
                work_state=snapshot.state_json if snapshot is not None else None,
            )


@contextmanager
def create_application(settings: Settings) -> Generator[AgentApplication, None, None]:
    database = Database(settings.database_url)
    database.create_schema()
    model = build_model(settings)
    summary_model = build_summary_model(settings)
    context_engine = ContextEngine(database, settings.context, LangChainSummarizer(summary_model))
    recorder = FileMutationRecorder(database, settings.workspace_root)
    bash_executor = (
        BashExecutionService(
            executable=settings.agent.bash_executable,
            workspace_root=settings.workspace_root,
            timeout_seconds=settings.agent.bash_timeout_seconds,
            max_output_bytes=settings.agent.bash_max_output_bytes,
        )
        if settings.agent.bash_enabled
        else None
    )
    search_client = make_search_client(settings.agent)
    jobs = SubAgentJobs(settings.artifact_dir)
    with create_langchain_agent(
        settings=settings,
        database=database,
        context_engine=context_engine,
        recorder=recorder,
        model=model,
        bash_executor=bash_executor,
        search_client=search_client,
        jobs=jobs,
    ) as agent:
        try:
            yield AgentApplication(
                settings,
                database,
                agent,
                context_engine,
                recorder,
                bash_executor,
                subagent_jobs=jobs,
            )
        finally:
            # 会话退出即放锁。进程被强杀时不用管：锁随连接被服务端收回。
            database.turn_guard.release()
            # 还活着的外派子代理跟着一起收：父进程都没了，报告没人取。
            jobs.terminate_all()
