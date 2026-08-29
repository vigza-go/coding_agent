from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .records import MemoryBlockSnapshot, MessageSnapshot


@dataclass(frozen=True)
class ContextPiece:
    kind: str
    begin_message_id: int
    end_message_id: int
    text: str | None = None
    messages: tuple[MessageSnapshot, ...] = ()
    level: int | None = None
    block_id: int | None = None


def greedy_cover(
    messages: Sequence[MessageSnapshot], blocks: Sequence[MemoryBlockSnapshot]
) -> list[ContextPiece]:
    if not messages:
        return []
    position = {message.id: index for index, message in enumerate(messages)}
    by_begin: dict[int, list[MemoryBlockSnapshot]] = defaultdict(list)
    for block in blocks:
        if (
            block.begin_message_id in position
            and block.end_message_id in position
            and position[block.begin_message_id] <= position[block.end_message_id]
        ):
            by_begin[block.begin_message_id].append(block)
    for candidates in by_begin.values():
        candidates.sort(
            key=lambda block: (position[block.end_message_id], block.level), reverse=True
        )

    pieces: list[ContextPiece] = []
    cursor = 0
    while cursor < len(messages):
        current_id = messages[cursor].id
        candidates = by_begin.get(current_id, [])
        if candidates:
            block = candidates[0]
            pieces.append(
                ContextPiece(
                    kind="memory",
                    begin_message_id=block.begin_message_id,
                    end_message_id=block.end_message_id,
                    text=block.text,
                    level=block.level,
                    block_id=block.id,
                )
            )
            cursor = position[block.end_message_id] + 1
            continue

        raw_start = cursor
        cursor += 1
        while cursor < len(messages) and not by_begin.get(messages[cursor].id):
            cursor += 1
        raw = tuple(messages[raw_start:cursor])
        pieces.append(
            ContextPiece(
                kind="raw",
                begin_message_id=raw[0].id,
                end_message_id=raw[-1].id,
                messages=raw,
            )
        )
    return pieces
