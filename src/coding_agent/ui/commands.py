from __future__ import annotations

import shlex
from dataclasses import dataclass


class CommandParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    argument: str | None = None


def parse_command(text: str) -> ParsedCommand | None:
    if not text.startswith("/"):
        return None
    try:
        parts = shlex.split(text)
    except ValueError as error:
        raise CommandParseError(str(error)) from error
    if not parts:
        return None

    name = parts[0][1:].lower()
    if name == "list":
        name = "history"
    arity = {
        "clear": (0, 0),
        "exit": (0, 0),
        "help": (0, 0),
        "history": (0, 1),
        "status": (0, 0),
        "usage": (0, 1),
        "thread": (1, 1),
        "threads": (0, 0),
        "undo": (0, 1),
    }
    if name not in arity:
        raise CommandParseError(f"未知命令：/{name}。使用 /help 查看可用命令。")
    minimum, maximum = arity[name]
    arguments = parts[1:]
    if not minimum <= len(arguments) <= maximum:
        raise CommandParseError(f"命令 /{name} 的参数数量不正确。")
    return ParsedCommand(name=name, argument=arguments[0] if arguments else None)


def positive_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise CommandParseError(f"{label} 必须是整数。") from error
    if parsed < 1:
        raise CommandParseError(f"{label} 必须大于等于 1。")
    return parsed
