from __future__ import annotations

from sqlalchemy import Engine, inspect

MEMORY_BLOCKS_TABLE = "memory_blocks"
LEGACY_FRONTIER_COLUMN = "is_frontier"
LEGACY_FRONTIER_INDEX = "ix_memory_thread_frontier"


def remove_legacy_memory_frontier(engine: Engine) -> bool:
    """删掉早先那套物化的 frontier 缓存表（如果还在的话）。

    这个迁移有意写成幂等的：老库可以在启动时顺手升级，新库的 schema 不受影响。
    """

    inspector = inspect(engine)
    if MEMORY_BLOCKS_TABLE not in inspector.get_table_names():
        return False
    columns = {column["name"] for column in inspector.get_columns(MEMORY_BLOCKS_TABLE)}
    indexes = {index["name"] for index in inspector.get_indexes(MEMORY_BLOCKS_TABLE)}
    has_column = LEGACY_FRONTIER_COLUMN in columns
    has_index = LEGACY_FRONTIER_INDEX in indexes
    if not has_column and not has_index:
        return False

    quote = engine.dialect.identifier_preparer.quote
    table = quote(MEMORY_BLOCKS_TABLE)
    column = quote(LEGACY_FRONTIER_COLUMN)
    index = quote(LEGACY_FRONTIER_INDEX)
    dialect = engine.dialect.name
    with engine.begin() as connection:
        if has_index:
            if dialect == "mysql":
                connection.exec_driver_sql(f"DROP INDEX {index} ON {table}")
            else:
                connection.exec_driver_sql(f"DROP INDEX {index}")
        if has_column:
            connection.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN {column}")
    return True
