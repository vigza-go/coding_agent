from __future__ import annotations

import json
from collections.abc import Sequence

from .records import MessageSnapshot
from .tokens import estimate_tokens


def render_message(message: MessageSnapshot) -> str:
    return json.dumps(message.content_json, ensure_ascii=False, separators=(",", ":"))


def message_tokens(message: MessageSnapshot) -> int:
    return estimate_tokens(message.content_json)


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
    """Keep an assistant tool request and all immediately following results together."""

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
    """Split precomputed atomic units without regrouping their messages."""

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
