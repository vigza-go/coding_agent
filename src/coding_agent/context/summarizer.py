from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from .tokens import estimate_tokens


class SummaryError(RuntimeError):
    pass


class Summarizer(Protocol):
    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str: ...


class RetryingSummarizer:
    def __init__(self, delegate: Summarizer, max_attempts: int) -> None:
        self.delegate = delegate
        self.max_attempts = max_attempts

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int = 1) -> str:
        del attempt
        last_count = 0
        for current_attempt in range(1, self.max_attempts + 1):
            output = self.delegate.summarize(
                text,
                hard_limit=hard_limit,
                level=level,
                attempt=current_attempt,
            ).strip()
            last_count = estimate_tokens(output) if output else 0
            if output and last_count <= hard_limit:
                return output
        raise SummaryError(
            f"summary remained empty or above {hard_limit} tokens after "
            f"{self.max_attempts} attempts (last={last_count})"
        )


class LangChainSummarizer:
    def __init__(
        self, model: object, token_counter: Callable[[str], int] = estimate_tokens
    ) -> None:
        self.model = model
        self.token_counter = token_counter

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str:
        aggression = "严格压缩，只保留目标、决定、约束、事实、错误与未完成事项。"
        if attempt > 1:
            aggression += " 上次超限；删除解释、重复、寒暄和可从代码重新获得的细节。"
        if attempt > 2:
            aggression += " 使用极短条目和紧凑符号，不得超过硬上限。"
        response = self.model.invoke(  # type: ignore[attr-defined]
            [
                SystemMessage(
                    content=(
                        "你是编码代理的上下文压缩器。摘要必须自包含，不得虚构。"
                        f"这是 L{level} 输入，输出硬上限约 {hard_limit} tokens。{aggression}"
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        content = response.content
        if isinstance(content, str):
            return content
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )


class DeterministicSummarizer:
    """Offline/test fallback; not intended for production semantic compression."""

    def summarize(self, text: str, *, hard_limit: int, level: int, attempt: int) -> str:
        del level, attempt
        max_chars = max(1, hard_limit * 3)
        return text[:max_chars]
