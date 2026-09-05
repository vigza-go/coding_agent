from __future__ import annotations

import json
from collections.abc import Sequence

from .records import MessageSnapshot
from .tokens import estimate_tokens


def render_message(message: MessageSnapshot) -> str:
    return json.dumps(message.content_json, ensure_ascii=False, separators=(",", ":"))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _summarize_arguments(arguments: object) -> str:
    """把工具调用的参数对象压成一句短而安全的描述。"""
    if isinstance(arguments, str):
        return _truncate(arguments, 200)
    if isinstance(arguments, dict):
        summary = ", ".join(f"{key}={_truncate(str(value), 60)}" for key, value in arguments.items())
        return _truncate(summary, 300)
    return _truncate(str(arguments), 200)


def sanitize_transcript(messages: Sequence[MessageSnapshot]) -> str:
    """把消息渲染成一份“安全、好消化”的转录，交给摘要器。

    逐条消息的原始 JSON 不能直接丢给（有时很弱的）摘要模型：里面混着助手第一人称的正文、
    工具调用的线协议结构、字面量 ``<...>`` 协议标签，这些都会诱发角色错乱——模型会“接着把
    活干完”，吐出一份假的 ``<tool_call>`` 块，而不是做压缩。所以这里把转录重建成带角色
    前缀的纯文本，跳过 reasoning 块，把工具调用换成占位描述，并将残留的尖括号中和掉，
    压缩器就没法把活的协议标签原样复述出来。
    """
    lines: list[str] = []
    for message in messages:
        data = message.content_json.get("data") or message.content_json
        role = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}.get(
            message.type, message.type
        )
        content = data.get("content")
        if isinstance(content, str):
            if content.strip():
                lines.append(f"[{role}] {content.strip()}")
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type in ("tool_use", "tool_call"):
                    name = block.get("name") or block.get("tool_name") or "tool"
                    args = block.get("input", block.get("arguments", {}))
                    parts.append(f"[调用工具 {name}({_summarize_arguments(args)})]")
                # thinking / reasoning / redacted_thinking 这几种块是有意丢掉的。
            body = "\n".join(part for part in parts if part).strip()
            if body:
                lines.append(f"[{role}] {body}")
        if content is None or isinstance(content, str):
            tool_calls = data.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    name = call.get("name") or call.get("tool_name") or "tool"
                    args = call.get("args", call.get("arguments", {}))
                    lines.append(f"[{role}] [调用工具 {name}({_summarize_arguments(args)})]")
    rendered = "\n".join(lines)
    # 漏网的字面标签，按 & 优先的顺序中和掉。
    return rendered.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")



def visible_text(message: MessageSnapshot) -> str:
    """这条消息在模型上下文里真正能被读到的字。

    算预算要问的是「模型打开上下文能看见什么」，不是「库里存了什么」「喂给摘要器的
    那份副本有多长」。所以三类都要算进来：思考块（模型看得到自己想过的话，占全库
    约 32%）、工具调用参数（write_file 那种几千字的正文就在这儿，占约 18%）、工具消息的
    结果内容（约 27%）。反过来，两样东西不算：JSON 包壳的那些键名（id、type、
    usage_metadata、response_metadata 之类，约 21%，是存储与线协议格式，不是内容），
    以及 ToolMessage.artifact（LangChain 侧的附带物，从来不进模型）。

    注意与 `sanitize_transcript()` 的区别：那个函数是给弱摘要器看的"安全副本"，会主动
    抹掉思考块、把工具调用压成一行占位文字，实测只有这里量的 37%。拿它当预算分母，
    等于把被压缩掉的东西忘掉三分之二。
    """

    data = message.content_json.get("data") or message.content_json
    parts: list[str] = []
    content = data.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "thinking":
                parts.append(str(block.get("thinking", "")))
            elif block_type == "reasoning":
                parts.append(json.dumps(block.get("summary", ""), ensure_ascii=False))
            elif block_type == "redacted_thinking":
                parts.append(str(block.get("data", "")))
            elif block_type in ("tool_use", "tool_call"):
                name = block.get("name") or block.get("tool_name") or "tool"
                arguments = block.get("input", block.get("arguments", {}))
                parts.append(f"{name}{json.dumps(arguments, ensure_ascii=False)}")
    # 与 sanitize_transcript 同一条规矩：content 是块列表时，调用信息已经在 tool_use 块里了，
    # data["tool_calls"] 装的是同一批（实测 2,635/2,635 条 id 完全重合），再数一遍就是重计。
    if not isinstance(content, list):
        for call in data.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("tool_name") or "tool"
            arguments = call.get("args", call.get("arguments", {}))
            parts.append(f"{name}{json.dumps(arguments, ensure_ascii=False)}")
    return "\n".join(part for part in parts if part)


def message_tokens(message: MessageSnapshot) -> int:
    """一条消息占多少 token —— 全局只有这一把尺子（触发线、尾留、切块、摘要配额共用）。"""

    return estimate_tokens(visible_text(message))


def split_token_balanced(
    messages: Sequence[MessageSnapshot], parts: int
) -> list[list[MessageSnapshot]]:
    if not messages:
        return []
    parts = min(parts, len(messages))
    weights = [message_tokens(message) for message in messages]
    total = sum(weights)
    result: list[list[MessageSnapshot]] = []
    start = 0
    consumed = 0
    for part_index in range(parts - 1):
        remaining_parts = parts - part_index
        target = consumed + (total - consumed) / remaining_parts
        end = start
        running = consumed
        max_end = len(messages) - (remaining_parts - 1)
        while end < max_end:
            next_running = running + weights[end]
            if end > start and abs(running - target) <= abs(next_running - target):
                break
            running = next_running
            end += 1
        if end == start:
            running += weights[end]
            end += 1
        result.append(list(messages[start:end]))
        start = end
        consumed = running
    result.append(list(messages[start:]))
    return result


def _tool_call_ids(message: MessageSnapshot) -> set[str]:
    if message.type != "assistant":
        return set()
    data = message.content_json.get("data", {})
    return {
        str(call["id"])
        for call in data.get("tool_calls", [])
        if isinstance(call, dict) and call.get("id")
    }


def _tool_result_id(message: MessageSnapshot) -> str | None:
    if message.type != "tool":
        return None
    value = message.content_json.get("data", {}).get("tool_call_id")
    return str(value) if value else None


def atomic_message_units(messages: Sequence[MessageSnapshot]) -> list[list[MessageSnapshot]]:
    """助手发起的工具调用，和紧随其后的所有结果，必须留在同一个单元里。"""

    units: list[list[MessageSnapshot]] = []
    cursor = 0
    while cursor < len(messages):
        unit = [messages[cursor]]
        pending = _tool_call_ids(messages[cursor])
        cursor += 1
        while pending and cursor < len(messages):
            tool_call_id = _tool_result_id(messages[cursor])
            if tool_call_id is None:
                break
            unit.append(messages[cursor])
            pending.discard(tool_call_id)
            cursor += 1
        units.append(unit)
    return units


def split_atomic_token_balanced(
    messages: Sequence[MessageSnapshot], parts: int
) -> list[list[MessageSnapshot]]:
    return split_atomic_units_token_balanced(atomic_message_units(messages), parts)


def split_atomic_units_token_balanced(
    units: Sequence[Sequence[MessageSnapshot]], parts: int
) -> list[list[MessageSnapshot]]:
    """按 token 均衡切分已经算好的原子单元，不再拆散单元内部的消息。"""

    if not units:
        return []
    parts = min(parts, len(units))
    unit_weights = [sum(message_tokens(message) for message in unit) for unit in units]
    total = sum(unit_weights)
    result: list[list[MessageSnapshot]] = []
    start = 0
    consumed = 0
    for part_index in range(parts - 1):
        remaining_parts = parts - part_index
        target = consumed + (total - consumed) / remaining_parts
        end = start
        running = consumed
        max_end = len(units) - (remaining_parts - 1)
        while end < max_end:
            next_running = running + unit_weights[end]
            if end > start and abs(running - target) <= abs(next_running - target):
                break
            running = next_running
            end += 1
        if end == start:
            running += unit_weights[end]
            end += 1
        result.append([message for unit in units[start:end] for message in unit])
        start = end
        consumed = running
    result.append([message for unit in units[start:] for message in unit])
    return result
