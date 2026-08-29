from __future__ import annotations

from sqlalchemy import inspect


def test_create_schema_removes_legacy_frontier_cache(database):
    with database.engine.begin() as connection:
        connection.exec_driver_sql(
            "ALTER TABLE memory_blocks ADD COLUMN is_frontier BOOLEAN NOT NULL DEFAULT 1"
        )
        connection.exec_driver_sql(
            "CREATE INDEX ix_memory_thread_frontier "
            "ON memory_blocks (thread_id, active, is_frontier)"
        )

    database.create_schema()

    inspector = inspect(database.engine)
    columns = {column["name"] for column in inspector.get_columns("memory_blocks")}
    indexes = {index["name"] for index in inspector.get_indexes("memory_blocks")}
    assert "is_frontier" not in columns
    assert "ix_memory_thread_frontier" not in indexes
