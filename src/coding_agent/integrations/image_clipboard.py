"""Explicit macOS image paste; ordinary terminal text paste remains unchanged."""

import subprocess
import sys
import tempfile
from pathlib import Path


def read_clipboard_image() -> bytes:
    if sys.platform != "darwin":
        raise ValueError("图片粘贴目前仅支持 macOS。")
    script = """on run argv
set destination to POSIX file (item 1 of argv)
set imageData to the clipboard as «class PNGf»
set handle to open for access destination with write permission
try
    set eof handle to 0
    write imageData to handle
    close access handle
on error message number code
    close access handle
    error message number code
end try
end run"""
    with tempfile.TemporaryDirectory(prefix="coding-agent-image-") as temporary:
        path = Path(temporary) / "clipboard.png"
        try:
            subprocess.run(
                ["osascript", "-e", script, str(path)], check=True, capture_output=True, timeout=10
            )
            return path.read_bytes()
        except (subprocess.SubprocessError, OSError) as error:
            raise ValueError("无法读取剪贴板图片；请先将截图复制到剪贴板，再按 Ctrl+V。") from error
