from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from ..application import create_application
from ..config import load_settings

HELP = "命令：/undo [seq]、/list、/thread <id>、/help、/exit"


def _text(content: object) -> str:
    return content if isinstance(content, str) else str(content)


def run() -> None:
    parser = argparse.ArgumentParser(description="Layered-context coding agent")
    parser.add_argument("--thread", default="default", help="conversation thread id")
    parser.add_argument("--config", default=None, help="path to local JSON configuration")
    args = parser.parse_args()
    settings = load_settings(args.config)
    console = Console()
    thread_id = args.thread
    console.print(f"[bold]Coding Agent[/bold] · thread={thread_id}\n{HELP}")

    with create_application(settings) as app:
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]你[/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n已退出。")
                return
            if not user_input:
                continue
            if user_input == "/exit":
                return
            if user_input == "/help":
                console.print(HELP)
                continue
            if user_input.startswith("/thread "):
                thread_id = user_input.split(maxsplit=1)[1].strip()
                console.print(f"已切换到 thread={thread_id}")
                continue
            if user_input == "/list":
                for entry in app.active_history(thread_id):
                    console.print(f"[{entry.user_seq}:{entry.message_type}] {_text(entry.content)}")
                continue
            if user_input.startswith("/undo"):
                parts = user_input.split()
                seq = int(parts[1]) if len(parts) == 2 else app.active_head(thread_id)
                if seq < 1:
                    console.print("当前没有可撤销的用户轮次。")
                    continue
                result = app.rollback(thread_id, seq)
                console.print(
                    f"已撤销 user_seq >= {seq}：恢复 {result.restored_files} 个文件变更，"
                    f"停用 {result.deactivated_messages} 条消息。"
                )
                continue
            try:
                answer = app.run_turn(thread_id, user_input)
                if answer is None:
                    console.print("[yellow]本轮没有最终文本回复。[/yellow]")
                else:
                    console.print(Markdown(_text(answer.content)))
            except Exception as error:  # noqa: BLE001 - keep the interactive session alive
                console.print(f"[bold red]调用失败：[/bold red]{error}")


def main() -> None:
    run()
