"""thread 占用检查。

SQLite 上没有跨会话咨询锁，这里只测 seq 分歧检测与调用点；真正的跨进程互斥、
断连自动释放由 ``tests/integrations/test_thread_lock_mysql.py`` 在真 MySQL 上验证。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from coding_agent.application import AgentApplication, TurnExecutionError
from coding_agent.config import ContextSettings, Settings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.persistence.message_codec import encode_message
from coding_agent.persistence.models import Conversation, Message, MessageType
from coding_agent.persistence.repository import AgentRepository
from coding_agent.persistence.thread_lock import TurnAbortedError, lock_name
from coding_agent.workspace.file_undo import FileMutationRecorder


def test_lock_name_stays_within_mysql_limit() -> None:
    name = lock_name("t" * 191)

    assert name.startswith("ca:t:")
    assert len(name) <= 64


def test_acquire_is_idempotent_and_switching_moves_the_hold(database) -> None:
    """会话级占用：重复 acquire 同一条只是续期；换一条则让出旧的。"""

    guard = database.turn_guard
    guard.acquire("t1")
    guard.acquire("t1")
    assert guard.held_thread == "t1"

    guard.acquire("t2")
    assert guard.held_thread == "t2"
    guard.release()
    assert guard.held_thread is None


def _start_turn(database, thread_id: str = "t1") -> int:
    """占住 thread 并写下本轮的 user 消息，返回 user_seq。"""

    with database.session() as session:
        repo = AgentRepository(session)
        user_seq = repo.reserve_user_seq(thread_id)
        repo.add_message(
            thread_id=thread_id,
            user_seq=user_seq,
            message_type=MessageType.USER,
            content_json=encode_message(HumanMessage(id="h-1", content="问题")),
            langchain_message_id="h-1",
        )
    guard = database.turn_guard
    guard.acquire(thread_id)
    guard.bind(user_seq)
    return user_seq


def _conversation(session, thread_id: str) -> Conversation:
    return session.scalars(select(Conversation).where(Conversation.thread_id == thread_id)).one()


def _advance_seq(database, thread_id: str = "t1") -> None:
    """模拟另一条会话跑完一轮：两个计数器一起前进。"""

    with database.session() as session:
        row = _conversation(session, thread_id)
        row.active_head_seq += 1
        row.next_user_seq += 1


def test_seq_moved_by_someone_else_stops_writes(database) -> None:
    _start_turn(database)
    _advance_seq(database)

    guard = database.turn_guard
    with pytest.raises(TurnAbortedError, match="被别的会话推进或回滚"):
        guard.verify("t1")
    with database.session() as session:
        written = session.scalars(select(Message).where(Message.thread_id == "t1")).all()
        assert [m.langchain_message_id for m in written] == ["h-1"]  # 只有那条 user 消息
    guard.release()


def test_rolled_back_turn_stops_writes(database) -> None:
    """一次 /undo 把 head 退回去 → 本轮上下文已过期，拒绝再写。"""

    user_seq = _start_turn(database)
    with database.session() as session:
        _conversation(session, "t1").active_head_seq = user_seq - 1

    guard = database.turn_guard
    with pytest.raises(TurnAbortedError):
        guard.verify("t1")
    guard.release()


def test_verify_passes_while_the_turn_is_untouched(database) -> None:
    _start_turn(database)

    database.turn_guard.verify("t1")  # 不抛即通过
    database.turn_guard.release()


def test_turn_without_a_bound_seq_skips_the_check(database) -> None:
    """release 会清掉 bind；没 bind 的轮次不比 seq（user 消息还没落库时就是这个状态）。"""

    _start_turn(database)
    guard = database.turn_guard
    guard.release()
    guard.acquire("t1")

    _advance_seq(database)
    guard.verify("t1")
    guard.release()


class QuietAgent:
    """什么也不写的一轮，只用来看 turn 结束后占用还在不在。"""

    def __init__(self) -> None:
        self.app = None

    def invoke(self, *args, **kwargs):
        return {"messages": []}


class AdvancingAgent:
    """模型跑的过程中，另一条会话把 seq 推进了。"""

    def __init__(self) -> None:
        self.app: AgentApplication | None = None

    def invoke(self, *args, **kwargs):
        app = self.app
        assert app is not None
        _advance_seq(app.database)
        app.message_persistence.persist_messages(
            thread_id="t1",
            user_seq=1,
            messages=[AIMessage(id="a-1", content="不该落库")],
        )
        return {"messages": []}


def _app(database, tmp_path, agent):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    app = AgentApplication(
        Settings(workspace_root=tmp_path),
        database,
        agent,
        engine,
        FileMutationRecorder(database, tmp_path),
    )
    agent.app = app
    return app


def test_run_turn_keeps_the_thread_occupied_afterwards(database, tmp_path) -> None:
    """占用活过 turn：跑完一轮仍然占着，这正是"别的会话切不进来"的依据。"""

    app = _app(database, tmp_path, QuietAgent())
    app.run_turn("t1", "问题")

    assert database.turn_guard.held_thread == "t1"


def test_run_turn_aborts_when_another_session_writes_mid_turn(database, tmp_path) -> None:
    """调用点：turn 中途被别人写过，写时间轴那一步必须炸掉，而不是静默混写。"""

    app = _app(database, tmp_path, AdvancingAgent())

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "问题")

    assert isinstance(raised.value.cause, TurnAbortedError)
    # 本轮终止 ≠ 会话退出：占用还在，下一轮照样能跑（跨进程互斥由 MySQL 测试覆盖）。
    assert database.turn_guard.held_thread == "t1"
