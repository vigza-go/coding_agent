from __future__ import annotations

import pytest

from coding_agent.persistence.database import Database


@pytest.fixture
def database(tmp_path):
    db = Database(f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    db.create_schema()
    return db
