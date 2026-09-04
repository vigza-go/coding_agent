"""同一条 thread 在同一时刻只允许一个会话待在里面。

锁不落数据层：``GET_LOCK`` 是 MySQL 的**连接级**咨询锁，挂在一条自己独占的连接上，
锁寿命等于连接寿命。会话**进入** thread 时取锁，切走或退出才放（不是 turn 结束就放），
所以空着不说话的终端也算占着。进程正常退出、``kill -9``、断电，服务端都会连会话带锁一起
收回 —— 没有 TTL、没有 holder 列、崩溃后不需要清锁。代价是必须单独用一个 ``NullPool`` 的
engine：业务池的连接退出只是归还、并不断开 TCP，锁会跟着池里的连接继续活着，那样
"锁随连接消失"就不成立了。

会话可能空闲很久，那条连接会被 ``wait_timeout`` 收走、锁随之消失，而自己并不知道。所以
``acquire`` 对已占用的 thread 是幂等的"续期确认"：掉了就重取一次。重取不到说明已被别的
会话接管，直接报错；空窗期里若真有人写过，紧随其后的 seq 比对也会拦住。turn 之内不重取
（那才是静默接管），只终止本轮。

一个进程同一时刻只待一条 thread，所以这里只支持"最多持有一把锁"。那条持锁连接只归拿锁的
线程用（进入、离开、每轮开头三处）；轮次中间由并行工具线程各自调的 ``verify`` 只比游标，
因为 SQLAlchemy 的连接不是线程安全的。SQLite 没有跨会话咨询锁，锁的部分直接不做（单进程
不会自己跟自己抢），但 seq 分歧检测照常生效。
"""

from __future__ import annotations

import contextlib
import hashlib

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from .models import Conversation

LOCK_WAIT_SECONDS = 3.0
_MYSQL_NAMES = {"mysql", "mariadb"}


def lock_name(thread_id: str) -> str:
    """MySQL 锁名上限 64 字符，而 thread_id 列宽 191，所以只能哈希。"""

    return "ca:t:" + hashlib.sha1(thread_id.encode("utf-8")).hexdigest()[:16]


class ThreadBusyError(RuntimeError):
    """该 thread 正被另一个会话占用，这个动作（进入 / 切换 / 本轮）没有发生。"""


class TurnAbortedError(RuntimeError):
    """本轮的独占前提已经不成立：锁没了，或空窗期这条 thread 被别的会话写过。"""


class TurnGuard:
    """一个会话对某条 thread 的占用凭证，外加每次写时间轴前的分歧检测。

    ``verify`` 只认 ``bind`` 记下的 ``user_seq``：本轮开头 ``reserve_user_seq`` 把
    conversations 行写成 ``(active_head_seq, next_user_seq) == (user_seq, user_seq + 1)``,
    本轮之内没有别的代码再动这两列。所以只要等式还成立，就证明空窗期没有别人写入或回滚。
    """

    def __init__(self, engine: Engine, session_factory: sessionmaker[Session]) -> None:
        # ``_pool is None`` 就是"这个后端没有咨询锁"的唯一依据，不再另存一个布尔。
        self._pool = (
            create_engine(engine.url, poolclass=NullPool)
            if engine.dialect.name in _MYSQL_NAMES
            else None
        )
        self._session_factory = session_factory
        self._connection: Connection | None = None
        self._thread_id: str | None = None
        self._user_seq = 0

    @property
    def held_thread(self) -> str | None:
        """本会话当前占着哪条 thread（SQLite 上只有这个，没有锁）。"""

        return self._thread_id

    def acquire(self, thread_id: str, timeout: float = LOCK_WAIT_SECONDS) -> None:
        """占住这条 thread，直到 :meth:`release`；抢不到抛 :class:`ThreadBusyError`。

        幂等：已经在自己手里就只做续期确认。切到另一条 thread 时先抢新的，
        抢不到就什么都不动 —— 不能"原来的座位丢了，新座位也没拿到"。
        """

        if self._thread_id == thread_id:
            self._rehold(thread_id)
            return
        pool = self._pool
        if pool is None:  # SQLite：单进程不会自己跟自己抢，改个名字就算占住了
            self._thread_id, self._user_seq = thread_id, 0
            return
        connection = self._take(pool, thread_id, timeout)
        previous = self._connection
        self._connection, self._thread_id, self._user_seq = connection, thread_id, 0
        if previous is not None:  # 旧锁随旧连接关闭被服务端收回
            self._close(previous)

    def bind(self, user_seq: int) -> None:
        """本轮的 seq 已确定，从这一刻起 verify 有可比对的基准。"""

        self._user_seq = user_seq

    def verify(self, thread_id: str) -> None:
        """每次写时间轴之前调用；前提不成立就抛 :class:`TurnAbortedError` 终止本轮。

        **只比游标，不碰持锁连接**：一轮里工具是并行跑的，各分支在自己的线程里调到这里，
        而那条连接不是线程安全的（真机踩过：两个线程同时 ``GET_LOCK`` 互相踩事务，好端端的
        锁被误判成"丢了"，把整条会话一路打死）。锁的存活交给每轮开头的 :meth:`_rehold`。
        真被人接管写过，游标不可能还吻合，所以这道判断并不比原来弱。
        """

        if self._thread_id != thread_id or self._user_seq <= 0:
            return
        head, nxt = self._seq_state(thread_id)
        if (head, nxt) == (self._user_seq, self._user_seq + 1):
            return
        raise TurnAbortedError(
            f"thread {thread_id!r} 在本轮期间被别的会话推进或回滚"
            f"（head {self._user_seq} → {head}）；本地上下文已过期，本轮终止，请重新发起。"
        )

    def release(self) -> None:
        """离开这条 thread。连接早就断了的话锁已随会话消失，这里怎么失败都不影响正确性。"""

        thread_id, connection = self._thread_id, self._connection
        self._thread_id, self._connection, self._user_seq = None, None, 0
        if connection is None:
            return
        if thread_id is not None:
            with contextlib.suppress(Exception):
                self._query_lock(connection, "RELEASE_LOCK", thread_id)
        self._close(connection)

    # --- 内部 ------------------------------------------------------------------

    def _take(self, pool: Engine, thread_id: str, timeout: float) -> Connection:
        """新建一条连接并取锁；拿不到就是别人正占着。"""

        connection = pool.connect()
        try:
            held = self._query_lock(connection, "GET_LOCK", thread_id, timeout=timeout)
        except Exception:
            self._close(connection)
            raise
        if not held:
            self._close(connection)
            raise ThreadBusyError(
                f"thread {thread_id!r} 正被另一个会话使用；等它退出，或用 /thread 换一条轨道。"
            )
        return connection

    def _rehold(self, thread_id: str) -> None:
        """会话空闲之后（往往是 ``wait_timeout`` 收走了连接）重新确认占用。"""

        pool, connection = self._pool, self._connection
        if pool is None or connection is None:
            return
        try:
            held = self._query_lock(connection, "GET_LOCK", thread_id)
        except Exception:  # noqa: BLE001 - 断连可能是任何形态的异常
            held = False
        if held:
            return  # 还在手上；这一问也顺带重置了空闲计时
        self._thread_id, self._connection = None, None
        self._close(connection)  # 别丢给垃圾回收：半死连接被回收时会甩一串 traceback
        self._connection = self._take(pool, thread_id, 0.0)

    @staticmethod
    def _close(connection: Connection) -> None:
        """关掉一条连接；它可能早就死了，关失败也不会有任何后果（锁本来就随会话消失）。"""

        with contextlib.suppress(Exception):
            connection.close()

    @staticmethod
    def _query_lock(
        connection: Connection, function: str, thread_id: str, *, timeout: float = 0.0
    ) -> bool:
        """在给定连接上执行一次锁函数。

        必须自己 ``begin()`` 收尾：留着未提交的事务，下一次 ``begin()`` 会直接报错，
        好端端的持锁连接看起来就像"锁丢了"。
        """

        with connection.begin():
            row = connection.execute(
                text(f"SELECT {function}(:name, :timeout)"),
                {"name": lock_name(thread_id), "timeout": timeout},
            ).scalar()
        return row == 1

    def _seq_state(self, thread_id: str) -> tuple[int, int]:
        # 普通读，不加 FOR UPDATE：要的是别人提交过的最新值，不是排他。
        with self._session_factory() as session:
            row = session.execute(
                select(Conversation.active_head_seq, Conversation.next_user_seq).where(
                    Conversation.thread_id == thread_id
                )
            ).first()
        return (0, 0) if row is None else (int(row[0]), int(row[1]))
