from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .migrations import remove_legacy_memory_frontier
from .models import Base


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.engine: Engine = create_engine(url, echo=echo, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        remove_legacy_memory_frontier(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with self.session_factory() as session, session.begin():
            yield session
