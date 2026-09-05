from __future__ import annotations

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from langchain_core.messages import ToolCall

from ..persistence.database import Database
from ..persistence.models import FileBlob, FileMutation, MutationStatus, OperationType
from ..persistence.repository import AgentRepository


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def resolve_workspace_path(workspace_root: Path, requested: str) -> Path:
    root = workspace_root.resolve()
    candidate = Path(requested)
    if candidate.is_absolute():
        resolved_absolute = candidate.resolve()
        if resolved_absolute == root or root in resolved_absolute.parents:
            resolved = resolved_absolute
        else:
            # FilesystemBackend 的虚拟路径里，/foo 指的是 <root>/foo。
            resolved = (root / requested.lstrip("/")).resolve()
    else:
        resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"file mutation escapes workspace root: {requested}")
    return resolved


class FileMutationRecorder:
    MUTATING_TOOLS = frozenset(
        {"write_file", "edit_file", "delete", "delete_file", "remove_file", "write", "edit"}
    )

    def __init__(self, database: Database, workspace_root: Path) -> None:
        self.database = database
        self.workspace_root = workspace_root.resolve()
        self._path_locks: dict[Path, RLock] = {}
        self._path_locks_guard = Lock()

    def begin(
        self,
        *,
        thread_id: str,
        user_seq: int,
        tool_call_id: str | None,
        tool_name: str,
        requested_path: str,
    ) -> int:
        path = resolve_workspace_path(self.workspace_root, requested_path)
        if path.exists() and not path.is_file():
            raise ValueError(f"V1 undo only supports regular files, not directories: {path}")
        before = path.read_bytes() if path.is_file() else None
        operation = self._operation(tool_name, before is not None)
        with self.database.session() as session:
            repo = AgentRepository(session)
            blob_id = None
            before_hash = None
            if before is not None:
                before_hash = sha256_bytes(before)
                blob_id = repo.add_blob(sha256=before_hash, content=before, storage_uri=None).id
            mutation = FileMutation(
                thread_id=thread_id,
                user_seq=user_seq,
                tool_call_id=tool_call_id,
                operation_type=operation,
                path=str(path),
                before_blob_id=blob_id,
                before_hash=before_hash,
                active=True,
                status=MutationStatus.PENDING,
            )
            session.add(mutation)
            session.flush()
            return mutation.id

    @contextmanager
    def mutation_for_tool(
        self,
        *,
        thread_id: str,
        user_seq: int,
        tool_call: ToolCall,
    ) -> Generator[int | None, None, None]:
        """从快照、执行/重试到收尾，全程持有这条文件的锁。"""

        tool_name = str(tool_call.get("name", ""))
        if tool_name not in self.MUTATING_TOOLS:
            yield None
            return
        args = tool_call.get("args", {})
        requested_path = args.get("file_path") or args.get("path")
        if not requested_path:
            yield None
            return
        path = resolve_workspace_path(self.workspace_root, str(requested_path))
        with self._path_locks_guard:
            path_lock = self._path_locks.setdefault(path, RLock())
        with path_lock:
            yield self.begin(
                thread_id=thread_id,
                user_seq=user_seq,
                tool_call_id=tool_call.get("id"),
                tool_name=tool_name,
                requested_path=str(path),
            )

    @staticmethod
    def _operation(tool_name: str, existed: bool) -> str:
        if "delete" in tool_name or "remove" in tool_name:
            return OperationType.DELETE
        if not existed:
            return OperationType.CREATE
        if "edit" in tool_name:
            return OperationType.EDIT
        return OperationType.WRITE

    def finish(self, mutation_id: int, *, succeeded: bool) -> None:
        with self.database.session() as session:
            mutation = session.get(FileMutation, mutation_id)
            if mutation is None:
                raise LookupError(f"unknown file mutation: {mutation_id}")
            if not succeeded:
                mutation.status = MutationStatus.FAILED
                return
            path = Path(mutation.path)
            mutation.after_hash = sha256_bytes(path.read_bytes()) if path.is_file() else None
            mutation.status = MutationStatus.SUCCEEDED

    def rollback(self, thread_id: str, from_user_seq: int) -> int:
        with self.database.session() as session:
            mutation_ids = [
                mutation.id
                for mutation in AgentRepository(session).mutations_to_rollback(
                    thread_id, from_user_seq
                )
            ]
        restored = 0
        for mutation_id in mutation_ids:
            with self.database.session() as session:
                mutation = session.get(FileMutation, mutation_id)
                if mutation is None or not mutation.active:
                    continue
                if mutation.status in (MutationStatus.SUCCEEDED, MutationStatus.PENDING):
                    self._restore(session, mutation)
                    mutation.status = MutationStatus.ROLLED_BACK
                    restored += 1
                mutation.active = False
        return restored

    @staticmethod
    def _restore(session: Any, mutation: FileMutation) -> None:
        path = Path(mutation.path)
        if mutation.before_blob_id is None:
            if path.exists() and path.is_file():
                path.unlink()
            return
        blob = session.get(FileBlob, mutation.before_blob_id)
        if blob is None:
            raise RuntimeError(f"missing before blob {mutation.before_blob_id}")
        if blob.content is not None:
            content = blob.content
        elif blob.storage_uri:
            content = Path(blob.storage_uri).read_bytes()
        else:
            raise RuntimeError(f"blob {blob.id} has neither content nor storage_uri")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
