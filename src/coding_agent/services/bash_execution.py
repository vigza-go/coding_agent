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

# 命令退出之后，输出读取线程最多再等多久去等到真正的 EOF。后台进程可以把管道的写端一直
# 挂着，所以这里限的是耐心，不是对命令时长的猜测。
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
    """跑一条非交互式 Bash 命令，捕获的输出有上限。"""

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
            # Popen.stdout 的类型标成 IO[bytes]，上面没有 read1()；而 stdout=PIPE 时运行时
            # 对象其实就是 BufferedReader，是唯一支持“读多少算多少”的流。在这里改述类型，
            # 比用一刷子 ignore 把整个读循环放松干净。
            stream = cast("BufferedReader", process.stdout)
            try:
                # read1() 一次裸读，已经到手的就返回。BufferedReader.read(n) 则会一直堵到攒
                # 够 n 字节或 EOF——一旦有后台进程把管道写端永远挂着，已经产出的输出就会被
                # 困在缓冲区里出不来。
                while chunk := stream.read1(8192):
                    remaining = self.max_output_bytes - len(captured)
                    if remaining > 0:
                        captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True
            finally:
                # 这个线程是唯一的读者，所以关管道也归它管。让调用方来关会死锁：
                # BufferedReader.close() 得先把缓冲区读完。
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
            # 命令都已经返回了，read() 还能堵着，只可能是我们有意留下的某个后台进程继承了
            # 管道写端。等有界的一会儿就不再等：守护线程在所有写者消失后自己退出，退出时
            # 顺手关掉 fd。
            reader.join(timeout=_OUTPUT_GRACE_SECONDS)
            if reader.is_alive():
                # 后面再冒出来的输出属于还活着的后台进程。用已有的标记把它露出来，
                # 而不是把这次工具调用永远挂在这儿。
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
