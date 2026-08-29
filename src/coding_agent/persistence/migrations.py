from __future__ import annotations

from sqlalchemy import Engine, inspect

MEMORY_BLOCKS_TABLE = "memory_blocks"
LEGACY_FRONTIER_COLUMN = "is_frontier"
LEGACY_FRONTIER_INDEX = "ix_memory_thread_frontier"


def remove_legacy_memory_frontier(engine: Engine) -> bool:
    """Remove the former materialized frontier cache if it exists.

    The migration is intentionally idempotent so existing databases can be upgraded
    during startup while fresh schemas remain unchanged.
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
