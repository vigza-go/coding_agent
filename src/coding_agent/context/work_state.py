"""工作状态：单个 `{key: markdown}` 字典的操作与渲染。

这里刻意不做的事，都是为了避开实测到的坑：

* **不做嵌套结构**。值一律是字符串（markdown）。旧数据里可能有嵌套，
  ``as_text`` 会把它稳定地摊平成 JSON 文本，所以读写不会炸。
* **不依赖存储的键序**。MySQL 的 JSON 列会把键规范化
  （插入 ``zz,a,mmm,b`` 读出 ``a,b,zz,mmm``），SQLite 则可能保留插入顺序。
  所以渲染一律走 :func:`ordered`：钉住键优先、其余按字典序，跨引擎确定。
* **不做预算硬闸门**。只回报体积，让人自己决定要不要删。
"""

from __future__ import annotations

import json
from typing import Any

from .tokens import estimate_tokens

PIN_PREFIX = "!"
"""键名前缀：被钉住的键每轮都注入正文，即使内容没变。

省略正文的前提是"上一轮已经给过、还能回看"，但投影每轮用
``RemoveMessage`` 整段重建，注入消息也不进 canonical 历史，所以"没变就只留
索引"等于让模型看不见它。钉住位是给"否决/边界"这类必须常驻的键留的口子。
"""

MAX_KEY_CHARS = 64
READ_OPS = frozenset({"list", "get"})
MUTATING_OPS = frozenset({"set", "append", "delete", "clear"})
OPS = READ_OPS | MUTATING_OPS


class WorkStateError(ValueError):
    """操作参数不合法；调用方据此返回错误文本，绝不写库。"""


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def flatten(state: dict[str, Any]) -> dict[str, str]:
    """把任意（可能嵌套的）快照摊平成 `{key: markdown}`。"""

    return {str(key): as_text(value) for key, value in state.items()}


def ordered(state: dict[str, Any]) -> list[tuple[str, str]]:
    """钉住的键在前，其余按键名字典序；与数据库返回顺序无关。"""

    flat = flatten(state)
    return sorted(flat.items(), key=lambda item: (not item[0].startswith(PIN_PREFIX), item[0]))


def render(state: dict[str, Any]) -> str:
    """完整正文渲染。"""

    return "\n\n".join(f"## {key}\n{text}" for key, text in ordered(state))


def render_index(state: dict[str, Any]) -> str:
    """只给键名与体积，用于"内容未变"时占位；钉住的键仍然给正文。"""

    lines = [
        f"{key} {estimate_tokens(text)}t" + (" (pinned)" if key.startswith(PIN_PREFIX) else "")
        for key, text in ordered(state)
    ]
    body = "\n\n".join(
        f"## {key}\n{text}" for key, text in ordered(state) if key.startswith(PIN_PREFIX)
    )
    head = "work state unchanged since the previous injection; keys: " + (
        ", ".join(lines) if lines else "(empty)"
    )
    return f"{head}\n{body}" if body else head


def _require_key(op: str, key: str | None) -> str:
    if key is None or not key.strip():
        raise WorkStateError(f"{op} needs a non-empty key")
    if len(key) > MAX_KEY_CHARS:
        raise WorkStateError(f"key is longer than {MAX_KEY_CHARS} chars: {key[:24]}…")
    if "\n" in key or "\r" in key:
        raise WorkStateError("key must be a single line")
    return key.strip()


def _require_value(op: str, value: str | None) -> str:
    if value is None:
        raise WorkStateError(f"{op} needs a value string (markdown); nested values are not used")
    if not isinstance(value, str):
        raise WorkStateError(f"{op} values must be markdown strings, got {type(value).__name__}")
    return value


def _summary(state: dict[str, str]) -> str:
    total = estimate_tokens(render(state))
    keys = ", ".join(key for key, _ in ordered(state)) or "(empty)"
    return f"keys: {keys} | ~{total}t"


def apply_op(
    state: dict[str, Any],
    op: str,
    key: str | None = None,
    value: str | None = None,
) -> tuple[dict[str, str], str]:
    """校验后路由到一个字典操作，返回（新状态, 给模型看的回执）。

    只读操作（``list``/``get``）原样返回同一份内容，调用方据此跳过写库。
    校验失败抛 :class:`WorkStateError`，不会留下半改状态。
    """

    if op not in OPS:
        raise WorkStateError(f"unknown op {op!r}; expected one of {', '.join(sorted(OPS))}")

    current = flatten(state)

    if op == "list":
        return current, _summary(current)
    if op == "clear":
        return {}, "cleared all keys | keys: (empty) | ~0t"

    name = _require_key(op, key)

    if op == "get":
        if name not in current:
            return current, f"no such key {name!r} | {_summary(current)}"
        return current, f"## {name}\n{current[name]}"

    if op == "delete":
        if current.pop(name, None) is None:
            return current, f"nothing to delete ({name!r} absent) | {_summary(current)}"
        return current, f"deleted {name} | {_summary(current)}"

    text = _require_value(op, value)

    if op == "set":
        current[name] = text
        return current, f"set {name} ({estimate_tokens(text)}t) | {_summary(current)}"

    # append：一条 bullet 的成本，而不是整份 state 的成本。
    existing = current.get(name)
    if existing is None:
        current[name] = text
        return current, f"created {name} ({estimate_tokens(text)}t) | {_summary(current)}"
    merged = f"{existing.rstrip()}\n- {text.lstrip('- ').rstrip()}"
    current[name] = merged
    return current, f"appended to {name} (now {estimate_tokens(merged)}t) | {_summary(current)}"
