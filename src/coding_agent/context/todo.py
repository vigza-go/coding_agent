"""计划（todo）：一个有序的 ``[{id, title, status}]`` 列表的校验、回执与渲染。

它装的是**计划**，不是笔记本。三处刻意与 :mod:`work_state` 相反：

* **顺序有语义**，渲染按提交顺序、**不排序**（work_state 按键名排序，因为字典序无所谓）。
  把计划排错序等于把要做的事搞乱。
* **回执带全表**：它短（几项），而且就是模型下一步要做的事，每步都该在它眼前；
  work_state 只回变更键，因为它可能很长、全量版本堆进历史会留下说不清哪个是当前的旧副本。
* **状态机在这里硬卡**：最多一个 ``in_progress``。机械可校验的约束才配硬闸门；
  "必须先标在做才能标完成"这类语义约束不卡——卡了只会逼模型说谎或者补一次假动作。

字段只有三样（``id`` / ``title`` / ``status``）：放弃一件事情的**原因写进 title**，不另开字段。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

MAX_ITEMS = 20
MAX_TITLE_CHARS = 200
STATUSES = ("pending", "in_progress", "completed", "cancelled")
OPEN_STATUSES = frozenset({"pending", "in_progress"})
CHECKBOX = {"pending": " ", "in_progress": "~", "completed": "x", "cancelled": "-"}


class TodoError(ValueError):
    """参数不合法；调用方据此返回错误文本，绝不写库。"""


def normalize(items: Any) -> list[dict[str, str]]:
    """校验整张表并规范化；任何一项不合法就整体拒绝，不留半改状态。"""

    if not isinstance(items, list):
        raise TodoError("items must be a list of {id, title, status} objects")
    if len(items) > MAX_ITEMS:
        raise TodoError(
            f"a plan holds at most {MAX_ITEMS} items, got {len(items)}; "
            "长内容是笔记，写 work_state"
        )

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    in_progress = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise TodoError(f"item #{index} must be an object with id/title/status")
        item_id = _single_line(item.get("id"), "id", index)
        title = _single_line(item.get("title"), "title", index)
        if len(title) > MAX_TITLE_CHARS:
            raise TodoError(
                f"item {item_id}: title is longer than {MAX_TITLE_CHARS} chars; "
                "计划写一行，细节写 work_state"
            )
        status = item.get("status")
        if status not in STATUSES:
            raise TodoError(
                f"item {item_id}: status must be one of {', '.join(STATUSES)}, got {status!r}"
            )
        if item_id in seen:
            raise TodoError(f"duplicate id {item_id!r}")
        if status == "in_progress":
            in_progress += 1
        seen.add(item_id)
        normalized.append({"id": item_id, "title": title, "status": status})

    if in_progress > 1:
        raise TodoError(
            f"at most one item can be in_progress, got {in_progress}; "
            "其余排成 pending，做完一项再开下一项"
        )
    return normalized


def _single_line(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TodoError(f"item #{index}: {field} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise TodoError(f"item #{index}: {field} must be a single line")
    return value.strip()


def render(items: Sequence[Any]) -> str:
    """渲染成勾选清单：``[ ]`` 待做、``[~]`` 在做、``[x]`` 完成、``[-]`` 放弃。

    对库里读出来的旧数据保持宽容（缺字段也不炸），所以取值全走 ``.get``。
    """

    lines = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mark = CHECKBOX.get(str(item.get("status")), "?")
        lines.append(f"- [{mark}] {item.get('id')} {item.get('title')}")
    return "\n".join(lines)


def receipt(before: Sequence[Any], after: Sequence[Any]) -> str:
    """给模型的回执：变了什么 + 全表现状。"""

    head = f"plan updated | {_counts_text(after)}" if _changes(before, after) else (
        f"plan unchanged | {_counts_text(after)}"
    )
    lines = [head]
    if changes := _changes(before, after):
        lines.append("changes: " + ", ".join(changes))
    if table := render(after):
        lines.extend(["", table])
    return "\n".join(lines)


def _changes(before: Sequence[Any], after: Sequence[Any]) -> list[str]:
    old = {str(item.get("id")): item for item in before if isinstance(item, dict)}
    new = {str(item.get("id")): item for item in after if isinstance(item, dict)}
    changes: list[str] = []
    for item_id, item in new.items():
        title, status = item.get("title"), str(item.get("status"))
        previous = old.get(item_id)
        if previous is None:
            changes.append(f"added {item_id} {title}")
        elif str(previous.get("status")) != status:
            changes.append(f"{status} {item_id} {title}")
        elif previous.get("title") != title:
            changes.append(f"reworded {item_id} {title}")
    for item_id, item in old.items():
        if item_id not in new:
            changes.append(f"dropped {item_id} {item.get('title')}")
    return changes


def _counts_text(items: Sequence[Any]) -> str:
    if not items:
        return "plan cleared (no items)"
    counts = {status: 0 for status in STATUSES}
    for item in items:
        status = str(item.get("status")) if isinstance(item, dict) else ""
        if status in counts:
            counts[status] += 1
    detail = ", ".join(f"{count} {status}" for status, count in counts.items() if count)
    return f"{len(items)} items: {detail}"


def open_items(items: Sequence[Any]) -> list[Any]:
    """还没交代的项（待做 + 在做）；收尾软提醒用它判断该不该拉一把。"""

    return [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("status")) in OPEN_STATUSES
    ]
