from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager

from ..config import TUISettings

_NOTIFICATION_SCRIPT = """on run argv
    display notification (item 1 of argv) with title "Coding Agent"
end run"""

_NOTIFICATION_SCRIPT_WITH_SOUND = """on run argv
    display notification (item 1 of argv) with title "Coding Agent" sound name (item 2 of argv)
end run"""


class DesktopService:
    """尽力而为的 macOS 辅助能力；绝不改动系统持久的电源/通知设置。"""

    def __init__(self, settings: TUISettings, warn: Callable[[str], None]) -> None:
        self.settings = settings
        self.warn = warn
        self._warned: set[str] = set()

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            self.warn(message)

    @contextmanager
    def keep_awake(self) -> Generator[None, None, None]:
        process: subprocess.Popen[bytes] | None = None
        if self.settings.prevent_sleep:
            executable = shutil.which("caffeinate") if sys.platform == "darwin" else None
            if executable is None:
                self._warn_once("防休眠不可用：当前仅支持带 caffeinate 的 macOS；任务仍会继续。")
            else:
                try:
                    process = subprocess.Popen(
                        [executable, "-i", "-w", str(os.getpid())],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    if process.poll() is not None:
                        self._warn_once("防休眠进程未能启动；任务仍会继续。")
                except OSError:
                    self._warn_once("无法启动防休眠；任务仍会继续。")
        try:
            yield
        finally:
            if process is not None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=self.settings.system_command_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=self.settings.system_command_timeout_seconds)
                except (OSError, subprocess.SubprocessError):
                    self._warn_once("防休眠进程清理失败；退出本应用时也会自动释放。")

    def notify(self, message: str) -> bool:
        if not self.settings.notifications_enabled:
            return False
        executable = shutil.which("osascript") if sys.platform == "darwin" else None
        if executable is None:
            return False
        # 声音名作为独立参数传给 AppleScript（item 2 of argv），不参与字符串拼接，
        # 与 message 一样不受注入影响。空串则退回无声音脚本。
        sound = (self.settings.notification_sound or "").strip()
        script = _NOTIFICATION_SCRIPT_WITH_SOUND if sound else _NOTIFICATION_SCRIPT
        arguments = [executable, "-e", script, message]
        if sound:
            arguments.append(sound)
        try:
            # 用户输入只当参数传进去，绝不拼进 AppleScript 或 shell。
            subprocess.run(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.settings.system_command_timeout_seconds,
                check=True,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            self._warn_once("桌面通知发送失败，已改用终端提示音。")
            return False
