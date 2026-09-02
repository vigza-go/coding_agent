from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from threading import Condition, Thread
from time import monotonic
from typing import cast

# How long to wait for the output reader to hit a real EOF after the command exits.
# A backgrounded process can keep the pipe's write end open indefinitely, so this is
# a bound on patience, not a guess about command duration.
_OUTPUT_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class BashResult:
    output: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False
    interrupted: bool = False


class BashExecutionService:
    """Run one non-interactive Bash command with bounded captured output."""

    def __init__(
        self,
        *,
        executable: str,
        workspace_root: Path,
        timeout_seconds: int,
        max_output_bytes: int,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout_seconds < 1:
            raise ValueError("bash timeout must be positive")
        if max_output_bytes < 1:
            raise ValueError("bash max output bytes must be positive")
        resolved_executable = Path(executable).expanduser().resolve()
        if not resolved_executable.is_file() or not os.access(resolved_executable, os.X_OK):
            raise ValueError(f"bash executable is not executable: {resolved_executable}")
        self.executable = resolved_executable
        self.workspace_root = workspace_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.env = dict(env) if env is not None else os.environ.copy()
        self._process_condition = Condition()
        self._active_processes: dict[int, subprocess.Popen[bytes]] = {}
        self._interrupted_processes: set[int] = set()
        self._interrupt_requested = False

    def prepare_turn(self) -> None:
        with self._process_condition:
            self._interrupt_requested = False

    def interrupt_all(self, *, wait_seconds: float = 5.0) -> int:
        with self._process_condition:
            self._interrupt_requested = True
            processes = list(self._active_processes.values())
            self._interrupted_processes.update(process.pid for process in processes)
        for process in processes:
            self._kill_process_group(process)

        deadline = monotonic() + wait_seconds
        with self._process_condition:
            while self._active_processes:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self._process_condition.wait(timeout=remaining)
        return len(processes)

    def execute(self, command: str, *, timeout_seconds: int | None = None) -> BashResult:
        if not command.strip():
            raise ValueError("bash command must not be empty")
        effective_timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if effective_timeout < 1:
            raise ValueError("bash timeout must be positive")
        if effective_timeout > self.timeout_seconds:
            raise ValueError(
                f"bash timeout {effective_timeout}s exceeds configured maximum "
                f"({self.timeout_seconds}s)"
            )

        started = monotonic()
        process = subprocess.Popen(
            [str(self.executable), "-lc", command],
            cwd=self.workspace_root,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._process_condition:
            self._active_processes[process.pid] = process
            interrupt_immediately = self._interrupt_requested
            if interrupt_immediately:
                self._interrupted_processes.add(process.pid)
        if interrupt_immediately:
            self._kill_process_group(process)
        captured = bytearray()
        truncated = False

        def drain_output() -> None:
            nonlocal truncated
            assert process.stdout is not None
            # Popen.stdout is annotated as IO[bytes], which has no read1(); with
            # stdout=PIPE the runtime object really is a BufferedReader, the only stream
            # type that offers a partial read. Restate it here instead of loosening the
            # read loop with a blanket ignore.
            stream = cast("BufferedReader", process.stdout)
            try:
                # read1() returns whatever has already arrived after a single raw read.
                # BufferedReader.read(n) instead blocks until it has n bytes or hits EOF,
                # which would strand already-produced output inside the buffer whenever a
                # backgrounded process keeps the pipe's write end open forever.
                while chunk := stream.read1(8192):
                    remaining = self.max_output_bytes - len(captured)
                    if remaining > 0:
                        captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True
            finally:
                # This thread is the only reader, so it also owns closing the pipe.
                # Closing it from the caller would deadlock: BufferedReader.close() needs
                with suppress(OSError):
                    stream.close()

        reader = Thread(target=drain_output, name="coding-agent-bash-output", daemon=True)
        reader.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_process_group(process)
            process.wait()
            exit_code = 124
        finally:
            # The command already returned, so read() can only still be blocked because
            # some process we deliberately left running inherited the pipe's write end.
            # Wait a bounded moment, then stop waiting: the daemon thread exits by itself
            # once every writer is gone, and it closes the fd on the way out.
            reader.join(timeout=_OUTPUT_GRACE_SECONDS)
            if reader.is_alive():
                # Later output belongs to a live background process. Surface it through
                # the existing flag instead of hanging this tool call forever.
                truncated = True
            with self._process_condition:
                interrupted = process.pid in self._interrupted_processes
                self._interrupted_processes.discard(process.pid)
                self._active_processes.pop(process.pid, None)
                self._process_condition.notify_all()

        if interrupted:
            exit_code = 130

        output = captured.decode("utf-8", errors="replace").rstrip()
        if not output:
            output = "<no output>"
        if truncated:
            output += f"\n\n[output truncated at {self.max_output_bytes:,} bytes]"
        if timed_out:
            output += f"\n\n[command timed out after {effective_timeout}s]"
        if interrupted:
            output += "\n\n[command interrupted by user]"
        return BashResult(
            output=output,
            exit_code=exit_code,
            duration_seconds=monotonic() - started,
            timed_out=timed_out,
            truncated=truncated,
            interrupted=interrupted,
        )

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
