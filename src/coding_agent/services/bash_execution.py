from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import monotonic


@dataclass(frozen=True)
class BashResult:
    output: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False


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
        captured = bytearray()
        truncated = False

        def drain_output() -> None:
            nonlocal truncated
            assert process.stdout is not None
            while chunk := process.stdout.read(8192):
                remaining = self.max_output_bytes - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated = True

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
            reader.join()
            if process.stdout is not None:
                process.stdout.close()

        output = captured.decode("utf-8", errors="replace").rstrip()
        if not output:
            output = "<no output>"
        if truncated:
            output += f"\n\n[output truncated at {self.max_output_bytes:,} bytes]"
        if timed_out:
            output += f"\n\n[command timed out after {effective_timeout}s]"
        return BashResult(
            output=output,
            exit_code=exit_code,
            duration_seconds=monotonic() - started,
            timed_out=timed_out,
            truncated=truncated,
        )

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
