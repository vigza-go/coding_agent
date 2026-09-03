"""给 LangGraph checkpointer 的连接补上存活保护（等价于业务库的 pool_pre_ping）。

背景（实测定位）：`PyMySQLSaver.from_conn_string()` 全程复用**一条裸 pymysql 连接**，
并且从不校验它是否还活着。MySQL 默认 `wait_timeout=28800`（8 小时）会单方面关掉
空闲连接；TUI 进程挂过一夜后再发一轮，第一条 checkpoint 写入就抛
`OperationalError (2006/2013, "MySQL server has gone away / Lost connection")`。
业务库那侧 `create_engine(..., pool_pre_ping=True)` 已经免疫，所以只有 checkpointer 中招。

修法的两个约束：
  1. `pymysql.ping(reconnect=True)` 已被上游标记 deprecated（源码：Create a new
     connection if you want to reconnect），所以这里探活失败**一律新建连接**替换。
  2. `BaseSyncMySQLSaver._cursor(pipeline=True)` 的顺序是 begin → cursor → commit。
     如果在事务中途探活并重建，已经发出的 begin 会随旧 socket 一起被丢弃，写入静默
     退化成 autocommit。因此只在**事务之外**才允许换连接。
"""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Any

import pymysql
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], Any]


class PrePingedMySQLConnection:
    """一个长期存活的连接外壳：交出游标/开事务之前先确认底层 socket 可用。

    只实现 `langgraph.checkpoint.mysql._internal.Connection` 协议用到的方法
    （cursor / begin / commit / rollback），外加 close 供装配点收尾。
    """

    def __init__(self, connect: ConnectionFactory) -> None:
        self._connect = connect
        self._conn: Any = connect()
        self._in_tx = False

    # -- 内部 --------------------------------------------------------------
    def _ensure(self) -> None:
        try:
            self._conn.ping(reconnect=False)
        except (pymysql.Error, OSError) as exc:
            # 只认这两类：pymysql.Error 覆盖服务端已关闭（2006/2013）与
            # ping 在 _sock is None 时抛的 "Already closed"；OSError 覆盖半开 socket
            # 写入直接 BrokenPipeError 的情况。其余异常（真 bug）必须原样抛出。
            logger.warning(
                "checkpoint DB connection is dead (%s: %s); opening a new one",
                type(exc).__name__,
                exc,
            )
            with suppress(pymysql.Error, OSError):
                self._conn.close()
            self._conn = self._connect()

    # -- Connection 协议 ---------------------------------------------------
    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        if not self._in_tx:
            self._ensure()
        return self._conn.cursor(*args, **kwargs)

    def begin(self) -> None:
        self._ensure()
        self._in_tx = True
        try:
            self._conn.begin()
        except (pymysql.Error, OSError):
            # begin 没成功就不能声称在事务里，否则 commit/rollback 会打到错误的连接上。
            self._in_tx = False
            raise

    def commit(self) -> None:
        self._in_tx = False
        self._conn.commit()

    def rollback(self) -> None:
        self._in_tx = False
        self._conn.rollback()

    def close(self) -> None:
        self._in_tx = False
        with suppress(Exception):
            self._conn.close()


def connect_kwargs(conn_string: str) -> dict[str, Any]:
    """把 `mysql://...` 连接串翻成 pymysql.connect 参数。

    `PyMySQLSaver.parse_conn_string` 会丢掉 URL 上的 charset，所以这里自己补一次
    （业务库用的是 utf8mb4，checkpoint 表里存的是序列化后的消息，编码不一致会踩坑）。
    """
    params = PyMySQLSaver.parse_conn_string(conn_string)
    if params.get("unix_socket") is None:
        params.pop("unix_socket", None)
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(conn_string).query))
    params["charset"] = query.get("charset", "utf8mb4")
    # 与 from_conn_string 保持一致：checkpointer 依赖 autocommit。
    params["autocommit"] = True
    return params


@contextmanager
def open_checkpointer(conn_string: str) -> Iterator[PyMySQLSaver]:
    """`PyMySQLSaver.from_conn_string` 的替代品：带探活重连，语义其余不变。"""
    kwargs = connect_kwargs(conn_string)
    conn = PrePingedMySQLConnection(lambda: pymysql.connect(**kwargs))
    try:
        # 协议兼容的外壳不是 pymysql.Connection 的子类，这里按 duck typing 交出去。
        yield PyMySQLSaver(conn)  # type: ignore[arg-type]
    finally:
        conn.close()
