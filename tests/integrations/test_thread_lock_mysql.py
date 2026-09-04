"""真实 MySQL 上的 thread 占用锁。

``GET_LOCK`` 的意义全在"跨连接"，SQLite 单测覆盖不到锁语义本身，所以这里用一次性临时库
``coding_agent_test_<hex>`` 直接问服务端，不碰业务库。

只在设置 ``TEST_MYSQL_ADMIN_URL`` 时运行（需要 CREATE/DROP DATABASE 与 KILL 权限）：

    TEST_MYSQL_ADMIN_URL='mysql+pymysql://root:root@127.0.0.1:3306?charset=utf8mb4' \\
        uv run pytest -q tests/integrations/test_thread_lock_mysql.py
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from coding_agent.persistence.database import Database
from coding_agent.persistence.repository import AgentRepository
from coding_agent.persistence.thread_lock import ThreadBusyError, TurnAbortedError, lock_name


@pytest.fixture()
def mysql_url():
    admin_url = os.getenv("TEST_MYSQL_ADMIN_URL")
    if not admin_url:
        pytest.skip("set TEST_MYSQL_ADMIN_URL to test the MySQL lock")
    url = make_url(admin_url)
    schema = f"coding_agent_test_{uuid4().hex}"
    admin = create_engine(url.set(database=None), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.exec_driver_sql(f"CREATE DATABASE `{schema}`")
    try:
        yield url.set(database=schema).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.exec_driver_sql(f"DROP DATABASE `{schema}`")
        admin.dispose()


@pytest.fixture()
def database(mysql_url) -> Database:
    db = Database(mysql_url)
    db.create_schema()
    yield db
    db.engine.dispose()


def another_session(mysql_url) -> Database:
    """同 URL 的第二个引擎：在服务端就是另一条连接，因此是另一个候选持有者。"""

    db = Database(mysql_url)
    db.create_schema()
    return db


def reserve(db: Database, thread_id: str) -> int:
    with db.session() as session:
        return AgentRepository(session).reserve_user_seq(thread_id)


def holder_id(db: Database, thread_id: str) -> int | None:
    """服务端说了算：谁拿着这把锁（连接 id），没人拿就是 None。"""

    with db.session() as session:
        value = session.execute(
            text("SELECT IS_USED_LOCK(:n)"), {"n": lock_name(thread_id)}
        ).scalar()
    return None if value is None else int(value)


def kill(db: Database, connection_id: int) -> None:
    """服务端结束这条连接：等价于 wait_timeout 踢人 / 网络静默断，锁随之消失。"""

    with db.session() as session:
        session.execute(text(f"KILL {connection_id}"))


def test_only_one_session_can_hold_a_thread(mysql_url, database):
    thread_id = "mysql-busy"
    other = another_session(mysql_url)
    database.turn_guard.acquire(thread_id)
    try:
        held_by = holder_id(other, thread_id)

        assert held_by is not None
        with pytest.raises(ThreadBusyError) as raised:
            other.turn_guard.acquire(thread_id, timeout=0.5)
        assert "正被另一个会话使用" in str(raised.value)
        assert other.turn_guard.held_thread is None  # 抢锁失败的一方不留副作用
    finally:
        database.turn_guard.release()
        other.engine.dispose()

    assert holder_id(database, thread_id) is None  # release 真的把锁还给了服务端


def test_switching_threads_releases_the_previous(mysql_url, database):
    """切轨道：抢到新的才让出旧的，旧锁随旧连接关闭被服务端收回。"""

    other = another_session(mysql_url)
    database.turn_guard.acquire("mysql-from")
    database.turn_guard.acquire("mysql-to")

    assert database.turn_guard.held_thread == "mysql-to"
    assert holder_id(database, "mysql-from") is None
    try:
        other.turn_guard.acquire("mysql-from")  # 旧的那条已经空出来
    finally:
        other.turn_guard.release()
        other.engine.dispose()
    database.turn_guard.release()


def test_idle_holder_reclaims_its_thread_at_the_next_turn(database):
    """空闲期间连接被收走 → 下一轮开头重取；空窗期没人写过，所以照旧通过校验。"""

    thread_id = "mysql-idle"
    guard = database.turn_guard
    guard.acquire(thread_id)
    guard.bind(reserve(database, thread_id))
    before = holder_id(database, thread_id)

    kill(database, before)

    guard.acquire(thread_id)
    assert holder_id(database, thread_id) not in (None, before)
    guard.verify(thread_id)
    guard.release()


def test_idle_holder_never_silently_takes_over(mysql_url, database):
    """空闲期间被别人接管 → 重取失败报错，绝不换条连接假装还在自己手里。"""

    thread_id = "mysql-takeover"
    guard = database.turn_guard
    guard.acquire(thread_id)
    kill(database, holder_id(database, thread_id))

    other = another_session(mysql_url)
    other.turn_guard.acquire(thread_id)
    try:
        with pytest.raises(ThreadBusyError, match="正被另一个会话使用"):
            guard.acquire(thread_id)
        assert guard.held_thread is None
    finally:
        other.turn_guard.release()
        other.engine.dispose()
    guard.release()


def test_killed_lock_connection_does_not_stop_the_current_turn(database):
    """服务端结束这条连接（等价于 wait_timeout 踢人）：本轮照写，下一轮开头再重取。

    锁没了不等于会双写：接管者要写就得先占游标，游标随即对不上，被下面那道 seq 比对拦住。
    刻意**不**在轮次中间拿这条连接去复核锁 —— 它不是线程安全的，见
    :func:`test_parallel_tool_threads_keep_the_lock`。
    """

    thread_id = "mysql-kill"
    guard = database.turn_guard
    guard.acquire(thread_id)
    guard.bind(reserve(database, thread_id))
    kill(database, holder_id(database, thread_id))

    guard.verify(thread_id)  # 游标还吻合，本轮不必终止

    reserve(database, thread_id)  # 空窗期真有人写过一轮
    with pytest.raises(TurnAbortedError, match="被别的会话推进或回滚"):
        guard.verify(thread_id)
    guard.release()


def test_parallel_tool_threads_keep_the_lock(database):
    """一轮里工具并行执行，每个分支各自 verify：不得误判丢锁，也不得甩告警。

    回归：曾经两个线程同时用那条持锁连接 GET_LOCK，互相踩事务，把连接弄死、锁跟着没了，
    之后每一轮都以为"占用已失效"，整条会话再也跑不动。
    """

    thread_id = "mysql-parallel"
    guard = database.turn_guard
    guard.acquire(thread_id)
    guard.bind(reserve(database, thread_id))

    errors: list[BaseException] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(guard.verify, thread_id) for _ in range(40)]
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

    assert not errors, errors[0]
    assert [str(w.message) for w in caught if "deassociated" in str(w.message)] == []
    assert guard.held_thread == thread_id  # 锁没被自己搞丢
    guard.release()


def test_foreign_turn_during_the_gap_aborts_the_turn(database):
    """锁活着但空窗期有人写过一轮：seq 对不上，同样得终止。"""

    thread_id = "mysql-diverge"
    guard = database.turn_guard
    guard.acquire(thread_id)
    guard.bind(reserve(database, thread_id))
    guard.verify(thread_id)

    reserve(database, thread_id)  # 绕过锁又推进了一轮

    with pytest.raises(TurnAbortedError, match="被别的会话推进或回滚"):
        guard.verify(thread_id)
    guard.release()


def test_a_finished_turn_does_not_open_the_thread_to_others(mysql_url, database):
    """占用活过 turn：跑完一轮之后，别的会话仍然进不来。"""

    thread_id = "mysql-after-turn"
    database.turn_guard.acquire(thread_id)
    database.turn_guard.bind(reserve(database, thread_id))
    database.turn_guard.verify(thread_id)  # 这一轮到此为止，没有任何"放锁"动作

    other = another_session(mysql_url)
    try:
        with pytest.raises(ThreadBusyError):
            other.turn_guard.acquire(thread_id, timeout=0.5)
    finally:
        other.engine.dispose()
    database.turn_guard.release()
