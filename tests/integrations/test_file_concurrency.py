from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolCall, ToolMessage
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import make_url

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.middleware import AgentRuntimeMiddleware, RunContext
from coding_agent.persistence.database import Database
from coding_agent.persistence.models import FileBlob, FileMutation, Message, MutationStatus
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.call_limits import CallLimitService
from coding_agent.services.context_projection import ContextProjectionService
from coding_agent.services.message_persistence import MessagePersistenceService
from coding_agent.services.tool_execution import ToolExecutionService
from coding_agent.workspace.artifacts import ArtifactStore
from coding_agent.workspace.file_undo import FileMutationRecorder, sha256_bytes


@pytest.fixture(params=["sqlite", "mysql"])
def concurrent_database(request, database):
    if request.param == "sqlite":
        yield database
        return
    admin_url = os.getenv("TEST_MYSQL_ADMIN_URL")
    if not admin_url:
        pytest.skip("set TEST_MYSQL_ADMIN_URL to test MySQL in a disposable database")
    # Never create tables or test rows in the application's configured database.
    url = make_url(admin_url)
    schema = f"coding_agent_test_{uuid4().hex}"
    admin = create_engine(url.set(database=None), isolation_level="AUTOCOMMIT")
    db = Database(url.set(database=schema).render_as_string(hide_password=False))
    created = False
    try:
        with admin.connect() as connection:
            connection.exec_driver_sql(f"CREATE DATABASE `{schema}`")
        created = True
        db.create_schema()
        yield db
    finally:
        db.engine.dispose()
        if created:
            with admin.connect() as connection:
                connection.exec_driver_sql(f"DROP DATABASE `{schema}`")
        admin.dispose()


def make_middleware(database, root):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    return AgentRuntimeMiddleware(
        persistence=MessagePersistenceService(database, engine),
        context_projection=ContextProjectionService(engine),
        call_limits=CallLimitService(model_limit=10, tool_limit=20),
        tool_execution=ToolExecutionService(max_retries=1, retryable_tools=[]),
        artifacts=ArtifactStore(root / "artifacts"),
        file_mutations=FileMutationRecorder(database, root),
        tool_result_inline_tokens=5000,
    )


def tool_request(identifier: str, path: str, *, name: str = "edit_file") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call=ToolCall(id=identifier, name=name, args={"file_path": path}),
        tool=None,
        state={},
        runtime=cast(Any, SimpleNamespace(context=RunContext("test", 1))),
    )


def test_concurrent_blob_inserts_reuse_one_row_and_leave_sessions_usable(concurrent_database):
    db = concurrent_database
    content = b"same file contents"
    digest = sha256_bytes(content)
    barrier = Barrier(4)

    def insert(index):
        with db.session() as session:
            repo = AgentRepository(session)
            # In MySQL this establishes a REPEATABLE READ snapshot without the row.
            assert repo.blob_by_sha(digest) is None
            barrier.wait(timeout=5)
            blob = repo.add_blob(sha256=digest, content=content, storage_uri=None)
            repo.add_message(
                thread_id="test", user_seq=index + 1, message_type="user", content_json={}
            )
            return blob.id

    with ThreadPoolExecutor(max_workers=4) as pool:
        identifiers = list(pool.map(insert, range(4)))
    assert len(set(identifiers)) == 1
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(FileBlob)) == 1
        assert session.scalar(select(func.count()).select_from(Message)) == 4
        blob = AgentRepository(session).add_blob(
            sha256=digest, content=content, storage_uri="must-not-replace-existing-storage"
        )
        assert blob.id == identifiers[0]
        assert blob.content == content
        assert blob.storage_uri is None


def test_blob_insert_is_rolled_back_with_its_outer_transaction(concurrent_database):
    db = concurrent_database
    digest = sha256_bytes(b"rollback")
    with pytest.raises(RuntimeError, match="abort"), db.session() as session:
        AgentRepository(session).add_blob(sha256=digest, content=b"rollback", storage_uri=None)
        raise RuntimeError("abort")
    with db.session() as session:
        assert AgentRepository(session).blob_by_sha(digest) is None


def test_same_file_lock_covers_snapshot_execution_and_finish(
    concurrent_database, tmp_path, monkeypatch
):
    db = concurrent_database
    middleware = make_middleware(db, tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    first_running, allow_write = Event(), Event()
    first_finishing, allow_finish = Event(), Event()
    second_attempted, second_running = Event(), Event()
    finish = middleware.file_mutations.finish

    def delayed_finish(mutation_id, *, succeeded):
        if target.read_text(encoding="utf-8") == "originalA":
            first_finishing.set()
            assert allow_finish.wait(5)
        finish(mutation_id, succeeded=succeeded)

    monkeypatch.setattr(middleware.file_mutations, "finish", delayed_finish)

    def handler(request):
        if request.tool_call["id"] == "first":
            first_running.set()
            assert allow_write.wait(5)
            suffix = "A"
        else:
            second_running.set()
            suffix = "B"
        target.write_text(target.read_text(encoding="utf-8") + suffix, encoding="utf-8")
        return ToolMessage(content="edited", tool_call_id=request.tool_call["id"])

    def second():
        second_attempted.set()
        return middleware.wrap_tool_call(tool_request("second", "sub/../target.txt"), handler)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            middleware.wrap_tool_call, tool_request("first", "/target.txt"), handler
        )
        try:
            assert first_running.wait(5)
            second_future = pool.submit(second)
            assert second_attempted.wait(5)
            assert not second_running.wait(0.1)
            with db.session() as session:
                assert session.scalar(select(func.count()).select_from(FileMutation)) == 1
            allow_write.set()
            assert first_finishing.wait(5)
            assert not second_running.wait(0.1)
            with db.session() as session:
                assert session.scalar(select(func.count()).select_from(FileMutation)) == 1
        finally:
            allow_write.set()
            allow_finish.set()
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    assert target.read_text(encoding="utf-8") == "originalAB"
    with db.session() as session:
        mutations = list(session.scalars(select(FileMutation).order_by(FileMutation.id)))
        assert len(mutations) == 2
        assert all(row.status == MutationStatus.SUCCEEDED for row in mutations)
        assert mutations[0].before_hash == sha256_bytes(b"original")
        assert mutations[0].after_hash == mutations[1].before_hash == sha256_bytes(b"originalA")
        assert mutations[1].after_hash == sha256_bytes(b"originalAB")
    assert middleware.file_mutations.rollback("test", 1) == 2
    assert target.read_text(encoding="utf-8") == "original"


def test_different_files_still_execute_in_parallel_and_share_blob(concurrent_database, tmp_path):
    db = concurrent_database
    middleware = make_middleware(db, tmp_path)
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("identical", encoding="utf-8")
    barrier = Barrier(2)

    def handler(request):
        barrier.wait(timeout=5)  # A global execution lock would break this test.
        path = tmp_path / request.tool_call["args"]["file_path"].lstrip("/")
        path.write_text("updated", encoding="utf-8")
        return ToolMessage(content="edited", tool_call_id=request.tool_call["id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(middleware.wrap_tool_call, tool_request(name, f"/{name}"), handler)
            for name in ("a.txt", "b.txt")
        ]
        results = [future.result(timeout=10) for future in futures]
    assert all(isinstance(result, ToolMessage) and result.status == "success" for result in results)
    with db.session() as session:
        mutations = list(session.scalars(select(FileMutation)))
        assert len(mutations) == 2
        assert len({row.before_blob_id for row in mutations}) == 1
    assert middleware.file_mutations.rollback("test", 1) == 2
    assert all((tmp_path / name).read_text() == "identical" for name in ("a.txt", "b.txt"))


@pytest.mark.parametrize("failure_stage", ["snapshot", "execution", "finish"])
def test_file_lock_is_released_after_failure(database, tmp_path, monkeypatch, failure_stage):
    middleware = make_middleware(database, tmp_path)
    recorder = middleware.file_mutations
    (tmp_path / "target.txt").write_text("original", encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    def handler(request):
        return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])

    with monkeypatch.context() as patch:
        if failure_stage != "execution":
            patch.setattr(recorder, "begin" if failure_stage == "snapshot" else "finish", fail)
        if failure_stage == "execution":
            result = middleware.wrap_tool_call(tool_request("fail", "/target.txt"), fail)
            assert isinstance(result, ToolMessage) and result.status == "error"
        else:
            with pytest.raises(RuntimeError, match="injected failure"):
                middleware.wrap_tool_call(tool_request("fail", "/target.txt"), handler)

    # Acquire from another thread: reacquiring an RLock in this thread could hide a leak.
    def acquire_again():
        path_lock = recorder._path_locks[(tmp_path / "target.txt").resolve()]
        assert path_lock.acquire(timeout=2)
        try:
            return middleware.wrap_tool_call(tool_request("ok", "/target.txt"), handler)
        finally:
            path_lock.release()

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(acquire_again).result(timeout=5)
    assert isinstance(result, ToolMessage) and result.status == "success"
