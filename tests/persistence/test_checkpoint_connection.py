"""checkpointer 连接探活外壳的行为约束。

不连真实 MySQL：用假连接精确复现"服务端单方面断开"，验证两条设计约束——
① 事务之外探活失败必须换**新建**的连接（不是 ping(reconnect=True)，那个已 deprecated）；
② 事务中途绝不允许换连接，否则已经发出的 begin 会随旧 socket 丢掉、
   写入静默退化成 autocommit。
"""

from __future__ import annotations

import pymysql
import pytest

from coding_agent.persistence.checkpoint_connection import (
    PrePingedMySQLConnection,
    connect_kwargs,
)


class FakeConn:
    """最小 Connection 协议实现；dead=True 表示服务端已经把这条连接关了。"""

    def __init__(self, ident: int) -> None:
        self.ident = ident
        self.dead = False
        self.closed = False
        self.begun = False

    def ping(self, reconnect: bool = False) -> None:
        assert reconnect is False, "不得依赖已废弃的 reconnect 参数"
        if self.dead:
            raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server")

    def cursor(self, *args: object, **kwargs: object) -> str:
        if self.dead:
            raise pymysql.err.OperationalError(2006, "MySQL server has gone away")
        return f"cursor-on-{self.ident}"

    def begin(self) -> None:
        if self.dead:
            raise pymysql.err.OperationalError(2006, "MySQL server has gone away")
        self.begun = True

    def commit(self) -> None:
        self.begun = False

    def rollback(self) -> None:
        self.begun = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def pool() -> list[FakeConn]:
    return []


def make(pool: list[FakeConn]) -> PrePingedMySQLConnection:
    def connect() -> FakeConn:
        conn = FakeConn(len(pool) + 1)
        pool.append(conn)
        return conn

    return PrePingedMySQLConnection(connect)


def test_healthy_connection_is_reused(pool: list[FakeConn]) -> None:
    wrapper = make(pool)
    assert wrapper.cursor() == "cursor-on-1"
    assert wrapper.cursor() == "cursor-on-1"
    assert len(pool) == 1, "健康连接不该被换掉"


def test_dead_connection_is_replaced_before_next_use(pool: list[FakeConn]) -> None:
    wrapper = make(pool)
    assert wrapper.cursor() == "cursor-on-1"
    pool[0].dead = True  # 服务端 wait_timeout 到点，单方面关闭

    assert wrapper.cursor() == "cursor-on-2", "探活失败必须换新建连接"
    assert pool[0].closed is True, "旧连接要关掉，不能泄漏 socket"
    assert len(pool) == 2


def test_transaction_is_never_swapped_mid_flight(pool: list[FakeConn]) -> None:
    """事务中途换连接会丢掉 begin，让写入静默退化——绝不允许。"""
    wrapper = make(pool)
    wrapper.begin()
    assert wrapper._in_tx is True
    pool[0].dead = True

    # 事务内不再探活：底层该失败就失败（fail fast），而不是悄悄换一条把事务吞掉。
    with pytest.raises(pymysql.err.OperationalError):
        wrapper.cursor()
    assert len(pool) == 1, "事务中途不得新建连接"


def test_commit_and_rollback_clear_transaction_flag(pool: list[FakeConn]) -> None:
    wrapper = make(pool)
    wrapper.begin()
    wrapper.commit()
    assert wrapper._in_tx is False

    wrapper.begin()
    wrapper.rollback()
    assert wrapper._in_tx is False


def test_begin_on_dead_connection_reconnects_then_begins(pool: list[FakeConn]) -> None:
    wrapper = make(pool)
    pool[0].dead = True
    wrapper.begin()
    assert len(pool) == 2, "开事务前先换掉死连接"
    assert pool[1].begun is True
    assert wrapper._in_tx is True


def test_failed_begin_does_not_leave_wrapper_in_transaction() -> None:
    """连新连接都是死的：异常要抛出去，且不能把外壳卡在"事务中"状态。"""

    def broken_connect() -> FakeConn:
        conn = FakeConn(99)
        conn.dead = True
        return conn

    wrapper = PrePingedMySQLConnection(broken_connect)
    with pytest.raises(pymysql.err.OperationalError):
        wrapper.begin()
    assert wrapper._in_tx is False


def test_connect_kwargs_keep_charset_and_autocommit() -> None:
    kwargs = connect_kwargs("mysql://u:p@127.0.0.1:3307/cp?charset=utf8mb4")
    assert kwargs["charset"] == "utf8mb4", "parse_conn_string 会丢 charset，必须补回"
    assert kwargs["autocommit"] is True, "checkpointer 依赖 autocommit"
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 3307
    assert kwargs["user"] == "u" and kwargs["password"] == "p"
    assert kwargs["database"] == "cp"
    assert "unix_socket" not in kwargs, "None 的 unix_socket 不该传给 pymysql"


def test_connect_kwargs_default_charset_when_url_omits_it() -> None:
    assert connect_kwargs("mysql://u:p@h/d")["charset"] == "utf8mb4"
