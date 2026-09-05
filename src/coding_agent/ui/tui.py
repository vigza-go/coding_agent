from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from html import escape as html_escape
from time import monotonic, sleep

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.status import Status
from rich.table import Table
from rich.text import Text

from ..application import AgentApplication, TurnExecutionError, create_application
from ..config import load_settings
from ..integrations.desktop import DesktopService
from ..persistence.thread_lock import ThreadBusyError
from ..services.progress import TurnEvent, TurnEventGate, TurnEventKind
from .commands import CommandParseError, ParsedCommand, parse_command, positive_int
from .rendering import TUI_THEME, ContentRenderer

HELP = """可用命令：
  /history [N]   查看最近 N 条有效消息（默认 20）
  /threads       查看最近会话
  /thread ID     切换会话（别人正占着则拒绝）
  /status        查看当前会话状态
  /usage [N]     查询近期 N 条模型回复的 API 用量与缓存命中率
  /undo [SEQ]    预览并撤销 SEQ 及之后的历史
  /clear         清空当前线程的上下文（历史与工作状态，不改动文件）
  /help          显示帮助
  /exit          退出

输入：Enter 发送，Alt+Enter 插入换行，方向键浏览本次启动中的输入历史。"""


def _input_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


_TTY_POLL_SECONDS = 1.0
_TTY_SAMPLES = 2  # 连续几次同一结论才作数（也绕开 shell 换前台那一瞬返回 0）
_TTY_EXIT_GRACE_SECONDS = 8.0  # 够 bash 的 interrupt_all 等满它那 5 秒子进程收尾


def _tty_lost() -> bool:
    """我们还能不能使唤这块终端。两个问法都要，因为它们覆盖的坏状态不一样：

    - ``/dev/tty`` 打不开（ENXIO）= 本进程已经没有控制终端了。pty 的主端一关，内核就把终端
      从整个会话上摘掉（进程表里的 `e_tdev` 变 -1，实测），此刻 slave 还开着、`isatty` 还
      说 True，光看 fd 是看不出来的。
    - ``os.tcgetpgrp(0)`` 返回 0 或抛 ENOTTY = 终端已被撤销/销毁、没有前台作业组了。它不吞
      输入，比读 fd 安全；但只在终端彻底销毁时才可靠。
    """
    try:
        fd = os.open(os.ctermid(), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return True
    os.close(fd)
    try:
        return os.tcgetpgrp(0) == 0
    except OSError:
        return True


class TerminalUI:
    def __init__(self, app: AgentApplication, console: Console, *, thread_id: str, debug: bool):
        self.app = app
        self.console = console
        self.thread_id = thread_id
        self.debug = debug
        self.renderer = ContentRenderer()
        self.desktop = DesktopService(
            app.settings.tui,
            lambda message: self.console.print(Text(message, style="yellow")),
        )
        self.session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            multiline=True,
            key_bindings=_input_bindings(),
        )

    def run(self) -> None:
        self.console.print(
            f"[bold]Coding Agent[/bold] · "
            f"thread=[cyan]{markup_escape(self.thread_id)}[/cyan]\n{HELP}"
        )
        # 关掉标签页就是"我不想看了"：不管此刻是在等输入还是在跑 turn，终端没了就该停。
        # 之所以要自己发现：Terminal.app 关标签页只杀 shell、只撤销 pty，不给作业发挂断
        # （实测，见 _watch_terminal），所以在外面等不到任何人来通知我们。
        if sys.stdin.isatty():  # 管道喂输入、pytest 这类没有终端的场景不探测
            threading.Thread(target=self._watch_terminal, daemon=True).start()
        # 进入即占用：这条轨道里已经有别的会话（哪怕它正闲着不说话）就不启动。
        self.app.enter_thread(self.thread_id)
        while True:
            try:
                text = self.session.prompt(
                    HTML(
                        "<b><ansiblue>你</ansiblue></b>"
                        f" <ansibrightblack>· {html_escape(self.thread_id)}</ansibrightblack> › "
                    ),
                    prompt_continuation="… ",
                    bottom_toolbar=" Enter 发送 · Alt+Enter 换行 · Ctrl-D 退出 ",
                ).strip()
            except KeyboardInterrupt:
                self.console.print("[dim]已清空当前输入。[/dim]")
                continue
            except EOFError:
                self.console.print("\n已退出。")
                return
            if not text:
                continue

            try:
                command = parse_command(text)
                if command is not None:
                    if not self._handle_command(command):
                        return
                    continue
                self.console.rule(style="bright_black")
                self._run_turn(text)
            except CommandParseError as error:
                self._error(str(error))
            except ThreadBusyError as error:
                # 常态而非故障：另一会话正占着这条 thread，本轮一个字节都没写。
                self._error(f"本轮未开始：{error}")
            except Exception as error:  # noqa: BLE001 - keep the interactive session alive
                self._error(str(error))
                if self.debug:
                    self.console.print_exception(show_locals=False)

    def _watch_terminal(self) -> None:
        """终端没了就停手。整条会话期间都跑着，不区分"在等输入"还是"在跑 turn"——
        关标签页是用户的显式意图（"我不想看了"），那时候会话正在干什么不该由我们挑。

        实测：Terminal.app 关标签页只杀 shell、只撤销 pty，**不给前台作业组发挂断**，
        所以外部不会有任何信号来停我们；zsh 的 nohup/hup、进程组、Ctrl-Z 都与此无关。
        """
        gone = 0
        alive = 0
        armed = False  # 先确认"终端曾在"，之后失去它才算数
        while True:
            sleep(_TTY_POLL_SECONDS)
            if not _tty_lost():
                gone, alive = 0, alive + 1
                armed = armed or alive >= _TTY_SAMPLES
                continue
            alive, gone = 0, gone + 1
            if gone < _TTY_SAMPLES:
                continue
            if not armed:
                # 从来没见到过控制终端：这进程是被脚本直接拉起的（没有 shell 认领那块 pty，
                # tcgetpgrp 一上来就抛 ENOTTY），判据没资格说话，退出自检别去搅它。
                return
            # 先给自己一发 Ctrl-C，走 run_turn 既有的中断收尾：杀掉本轮 bash 的整个进程组、
            # 终止在飞的子代理、把半截工具批次结清在时间轴上，一路 unwind 顺带放出座位。
            os.kill(os.getpid(), signal.SIGINT)
            # 收尾不一定走得完（可能正堵在模型的网络读上），到点硬退。硬退也不脏：进程一死
            # socket 就断，未提交的事务由服务端回滚，这条 thread 的座位当秒收回。
            # 全程不往屏幕打一个字——终端已经没了，写它就是 OSError(EIO)，会把这个线程摔死。
            sleep(_TTY_EXIT_GRACE_SECONDS)
            os._exit(0)

    def _handle_command(self, command: ParsedCommand) -> bool:
        if command.name == "exit":
            return False
        if command.name == "help":
            self.console.print(HELP)
            return True
        if command.name == "thread":
            assert command.argument is not None
            target = command.argument
            try:
                self.app.enter_thread(target)
            except ThreadBusyError as error:
                self._error(f"不切换。{error}")
                return True
            self.thread_id = target
            self.console.print(f"已切换到 thread=[cyan]{markup_escape(self.thread_id)}[/cyan]")
            return True
        if command.name == "threads":
            self._show_threads()
            return True
        if command.name == "history":
            limit = (
                positive_int(command.argument, label="history 数量")
                if command.argument is not None
                else 20
            )
            self._show_history(limit)
            return True
        if command.name == "status":
            self._show_status()
            return True
        if command.name == "usage":
            limit = (
                positive_int(command.argument, label="usage 数量")
                if command.argument is not None
                else self.app.settings.tui.usage_recent_messages
            )
            self._show_usage(limit)
            return True
        if command.name == "undo":
            user_seq = (
                positive_int(command.argument, label="user_seq")
                if command.argument is not None
                else self.app.active_head(self.thread_id)
            )
            if user_seq < 1:
                self.console.print("当前没有可撤销的用户轮次。")
                return True
            self._confirm_and_rollback(user_seq)
            return True
        if command.name == "clear":
            self._confirm_and_clear()
            return True
        raise CommandParseError(f"未实现命令：/{command.name}")

    def _run_turn(self, text: str) -> None:
        started_at = monotonic()
        with self.console.status("[cyan]正在准备上下文…[/cyan]", spinner="dots") as status:
            event_gate = TurnEventGate(lambda event: self._render_event(status, event))
            failure: TurnExecutionError | None = None
            try:
                with self.desktop.keep_awake():
                    answer = self.app.run_turn(
                        self.thread_id,
                        text,
                        on_event=event_gate.emit,
                    )
            except TurnExecutionError as error:
                failure = error
                answer = None
            finally:
                event_gate.close()

            if failure is not None:
                status.stop()
                label = "已中断" if failure.interrupted else "执行失败"
                self._error(
                    f"本轮 {label}（user_seq={failure.user_seq}）："
                    f"{type(failure.cause).__name__}: {failure.cause}"
                )
                if failure.finalization_errors:
                    self.console.print(
                        "[red]中断收尾未完整完成："
                        + "；".join(failure.finalization_errors)
                        + "。继续前请撤销本轮。[/red]"
                    )
                elif failure.closed_tool_results:
                    self.console.print(
                        f"[yellow]已为 {failure.closed_tool_results} 个未完成工具调用补入中断结果，"
                        "工具协议已闭合。[/yellow]"
                    )
                else:
                    self.console.print("[yellow]中断收尾已完成，没有发现缺失的工具结果。[/yellow]")
                self._notify_turn(label, started_at)
                if Confirm.ask(f"现在撤销 user_seq >= {failure.user_seq} 吗？", default=True):
                    self._rollback(failure.user_seq)
                return

        if answer is None:
            self.console.print("[yellow]本轮没有最终文本回复。[/yellow]")
            return
        if self.renderer.text(answer.content):
            self._print_agent_text(answer.content)
        else:
            self.console.print("[yellow]模型回复中没有可显示的文本内容。[/yellow]")
        self._notify_turn("本轮完成", started_at)

    def _notify_turn(self, outcome: str, started_at: float) -> None:
        if not self.app.settings.tui.notifications_enabled:
            return
        elapsed = monotonic() - started_at
        if not self.desktop.notify(f"会话 {self.thread_id} · {outcome} · {elapsed:.1f} 秒"):
            self.console.bell()

    def _print_agent_text(self, content: object) -> None:
        """把一段模型正文渲染成 Agent 面板。

        轮中散文和最终答案共用这条路。以前只有最终答案有出口到控制台，所以模型在工具
        调用旁边写的每一条解释都被静默丢掉，用户只看到这一轮的最后那句话。
        """

        self.console.print(
            Panel(
                self.renderer.markdown(content),
                title="Agent",
                title_align="left",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def _render_event(self, status: Status, event: TurnEvent) -> None:
        if event.kind == TurnEventKind.MODEL_STARTED:
            status.update("[cyan]模型思考中…[/cyan]")
        elif event.kind == TurnEventKind.MODEL_FINISHED:
            if event.text:
                self._print_agent_text(event.text)
            status.update("[cyan]正在处理模型结果…[/cyan]")
        elif event.kind == TurnEventKind.TOOL_STARTED:
            detail = f" [dim]{markup_escape(event.detail)}[/dim]" if event.detail else ""
            self.console.print(f"[blue]→[/blue] {markup_escape(event.name or 'tool')}{detail}")
            status.update(f"[cyan]正在执行 {markup_escape(event.name or 'tool')}…[/cyan]")
        elif event.kind == TurnEventKind.TOOL_FINISHED:
            self.console.print(f"[green]✓[/green] {markup_escape(event.name or 'tool')}")
            status.update("[cyan]正在处理工具结果…[/cyan]")
        elif event.kind == TurnEventKind.TOOL_FAILED:
            self.console.print(
                f"[red]✗[/red] {markup_escape(event.name or 'tool')} "
                f"[dim]{markup_escape(event.detail or '')}[/dim]"
            )
        elif event.kind == TurnEventKind.SUMMARY_STARTED:
            status.update(f"[cyan]正在压缩记忆 {markup_escape(event.name or '')}…[/cyan]")
        elif event.kind == TurnEventKind.SUMMARY_FINISHED:
            status.update("[cyan]继续处理…[/cyan]")
            self.console.print(f"[dim]✓ 记忆 {markup_escape(event.name or '')} 压缩完成[/dim]")

    def _show_history(self, limit: int) -> None:
        entries = self.app.active_history(self.thread_id, limit=limit)
        if not entries:
            self.console.print("当前会话没有有效历史。")
            return
        for entry in entries:
            content = self.renderer.summary(entry.content)
            body = (
                self.renderer.markdown(content)
                if entry.message_type == "assistant"
                else Text(content)
            )
            role, border_style = {
                "user": ("你", "green"),
                "assistant": ("Agent", "blue"),
                "tool": ("Tool", "yellow"),
                "system": ("System", "magenta"),
            }.get(entry.message_type, (entry.message_type, "white"))
            self.console.print(
                Panel(
                    body,
                    title=f"{role} · user_seq={entry.user_seq}",
                    title_align="left",
                    border_style=border_style,
                    padding=(0, 1),
                )
            )

    def _show_threads(self) -> None:
        rows = self.app.list_threads()
        if not rows:
            self.console.print("还没有会话。")
            return
        table = Table("thread", "active head", "updated", box=None)
        for row in rows:
            marker = "* " if row.thread_id == self.thread_id else ""
            table.add_row(
                marker + row.thread_id,
                str(row.active_head_seq),
                row.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            )
        self.console.print(table)

    def _show_status(self) -> None:
        status = self.app.thread_status(self.thread_id)
        conversation = Table(show_header=False, box=None, padding=(0, 2))
        conversation.add_row("thread", status.thread_id)
        conversation.add_row("active head", str(status.active_head_seq))
        conversation.add_row("next user seq", str(status.next_user_seq))
        conversation.add_row(
            "Bash",
            Text("已启用", style="yellow") if status.bash_enabled else Text("未启用", style="dim"),
        )
        conversation.add_row(
            "搜索工具",
            Text("已启用", style="yellow") if status.search_enabled else Text("未启用", style="dim"),
        )
        self.console.print(Panel(conversation, title="会话", title_align="left"))

        context = Table(show_header=False, box=None, padding=(0, 2))
        context.add_row("记忆块", self._memory_timeline(status.memory_levels))
        context.add_row(
            "压缩区",
            self._token_usage(status.memory_tokens, status.memory_limit),
        )
        context.add_row(
            "工作区",
            Text.assemble(
                f"{status.working_messages} 条消息 · ",
                self._token_usage(status.working_tokens, status.working_trigger),
            ),
        )
        self.console.print(Panel(context, title="上下文", title_align="left"))
        if status.work_state is None:
            self.console.print("[dim]当前没有 work state。[/dim]")
        else:
            self.console.print(
                Panel(
                    json.dumps(status.work_state, ensure_ascii=False, indent=2),
                    title="work state",
                    title_align="left",
                )
            )

    def _show_usage(self, limit: int) -> None:
        usage = self.app.recent_usage(self.thread_id, limit=limit)
        if not usage.sampled_messages:
            self.console.print("当前会话还没有模型回复用量记录。")
            return
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("范围", f"当前会话最近 {usage.sampled_messages} 条模型回复（含撤销历史）")
        table.add_row("API 用量样本", f"{usage.usage_samples} / {usage.sampled_messages}")
        if usage.usage_samples:
            table.add_row("输入", f"{usage.input_tokens:,} tokens")
            table.add_row("输出", f"{usage.output_tokens:,} tokens")
        table.add_row("缓存统计样本", f"{usage.cache_samples} / {usage.sampled_messages}")
        rate = usage.cache_hit_rate
        if rate is None:
            table.add_row("缓存命中率", "暂无可计算数据（缺少字段或输入为 0）")
        else:
            table.add_row("缓存命中率", Text(f"{rate:.1%}", style="bold cyan"))
            table.add_row(
                "缓存命中 / 可统计输入",
                f"{usage.cache_read_tokens:,} / {usage.cache_input_tokens:,} tokens",
            )
            table.add_row(
                "未命中输入（含缓存创建）",
                f"{usage.cache_input_tokens - usage.cache_read_tokens:,} tokens",
            )
        self.console.print(Panel(table, title="近期 API 用量", title_align="left"))
        self.console.print(
            "[dim]命中率按输入 token 加权；缺失缓存字段不参与计算。\n"
            "仅统计已落库主模型响应，不含摘要及未记录的失败请求，不等同于完整账单。[/dim]"
        )

    @staticmethod
    def _memory_timeline(levels: tuple[int, ...]) -> Text:
        if not levels:
            return Text("尚无记忆块", style="dim")
        result = Text()
        for index, level in enumerate(levels):
            if index:
                result.append(" → ", style="dim")
            result.append(f"L{level}", style="bold cyan")
        result.append(f"  ({len(levels)} 块)", style="dim")
        return result

    @staticmethod
    def _token_usage(value: int, limit: int) -> Text:
        percentage = value / limit * 100 if limit else 0
        color = "red" if percentage > 100 else "yellow" if percentage >= 80 else "green"
        result = Text(f"{value:,} / {limit:,} tokens ")
        result.append(f"({percentage:.1f}%)", style=color)
        return result

    def _confirm_and_rollback(self, user_seq: int) -> None:
        preview = self.app.rollback_preview(self.thread_id, user_seq)
        if not any((preview.messages, preview.file_mutations, preview.work_states)):
            self.console.print(f"user_seq >= {user_seq} 没有可撤销的有效内容。")
            return
        table = Table("撤销范围", "数量", box=None)
        table.add_row("消息", str(preview.messages))
        table.add_row("文件 mutation", str(preview.file_mutations))
        table.add_row("不同文件", str(preview.files))
        table.add_row("work state", str(preview.work_states))
        self.console.print(table)
        if Confirm.ask(
            f"确认撤销 thread={self.thread_id} 中 user_seq >= {user_seq} 吗？",
            default=False,
        ):
            self._rollback(user_seq)

    def _rollback(self, user_seq: int) -> None:
        with self.console.status("[yellow]正在撤销…[/yellow]", spinner="dots"):
            result = self.app.rollback(self.thread_id, user_seq)
        self.console.print(
            f"已撤销 user_seq >= {user_seq}：恢复 {result.restored_files} 个文件变更，"
            f"停用 {result.deactivated_messages} 条消息。"
        )

    def _confirm_and_clear(self) -> None:
        preview = self.app.rollback_preview(self.thread_id, 1)
        if not any((preview.messages, preview.work_states)):
            self.console.print("当前线程没有可清空的上下文。")
            return
        table = Table("将被停用", "数量", box=None)
        table.add_row("消息", str(preview.messages))
        table.add_row("work state", str(preview.work_states))
        self.console.print(table)
        if preview.file_mutations:
            # 清空的是"模型下次看到什么"，不是"磁盘上有什么"。旧的文件账留在它自己那一轮上，
            # 想退回仍然可以按那一轮 /undo —— clear 不越这个界。
            self.console.print(
                f"[dim]文件改动 {preview.file_mutations} 处（{preview.files} 个文件）"
                f"保持原样、不退回；要退就用 /undo <那一轮的 seq>。[/dim]"
            )
        if Confirm.ask(f"确认清空 thread={self.thread_id} 的上下文吗？", default=False):
            with self.console.status("[yellow]正在清空…[/yellow]", spinner="dots"):
                result = self.app.clear_context(self.thread_id)
            self.console.print(
                f"已停用 {result.deactivated_messages} 条消息，工作状态一并清空；"
                "工作树未做任何改动。下一轮从零开始。"
            )

    def _error(self, message: str) -> None:
        self.console.print(Text(f"错误：{message}", style="bold red"))


def run() -> int:
    parser = argparse.ArgumentParser(description="Layered-context coding agent")
    parser.add_argument("--thread", default="default", help="conversation thread id")
    parser.add_argument("--config", default=None, help="path to local JSON configuration")
    parser.add_argument("--debug", action="store_true", help="show tracebacks for failures")
    args = parser.parse_args()
    console = Console(theme=TUI_THEME)

    try:
        settings = load_settings(args.config)
        with create_application(settings) as app:
            TerminalUI(app, console, thread_id=args.thread, debug=args.debug).run()
        return 0
    except ThreadBusyError as error:
        # 启动时抢不到占用是常态（别人停在那条 thread 里），不该刷 traceback。
        console.print(Text(f"启动失败：{error}", style="bold red"))
        return 1
    except KeyboardInterrupt:
        console.print("\n已退出。")
        return 130
    except Exception as error:  # noqa: BLE001 - convert startup failures into concise CLI errors
        console.print(Text(f"启动失败：{error}", style="bold red"))
        if args.debug:
            console.print_exception(show_locals=False)
        return 1


def main() -> None:
    raise SystemExit(run())
