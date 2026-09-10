"""便签：剪裁/压缩把投影弄断的那一刻，顺手贴在压缩块之后的一张"当前状态"。

它是**渲染期派生视图**：不落库、不进 ``messages``、不进 ``memory_blocks``，也不是一条真消息。
平日原样冻着，只有剪裁那条路径（``context/engine.py`` 的 ``_refresh_pin``）会刷新它——这样两次
剪裁之间投影逐字节不变、缓存全中。

里面放两样东西，顺序固定：**计划在前**，工作状态在后。计划是每一步都要照它走的指针，工作状态是
细节、按需 ``get``。两样都空就返回 ``None``：不贴一张空标签。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .todo import render as render_todo
from .work_state import render as render_work_state

PLAN_HEADING = "# 计划（todo）"
STATE_HEADING = "# 工作状态"


def render_pin(todos: Sequence[Any] | None, work_state: dict[str, Any] | None) -> str | None:
    sections: list[str] = []
    if todos and (table := render_todo(todos)):
        sections.append(f"{PLAN_HEADING}\n{table}")
    if work_state and (body := render_work_state(work_state)):
        sections.append(f"{STATE_HEADING}\n{body}")
    return "\n\n".join(sections) if sections else None
