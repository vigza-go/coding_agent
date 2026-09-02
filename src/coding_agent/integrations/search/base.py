"""搜索抽象层：与任何供应商无关。

配置契约只提供开关、凭证和通用行为参数；具体引擎由实现决定。要换搜索引擎，
就在这里之外新增一个 client 实现，并改 `make_search_client` 里的一行——
调用方与配置面都不需要动。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class SearchError(RuntimeError):
    """搜索没有完成。抛出后由 ToolExecutionService 统一转成 error ToolMessage。"""


@dataclass(frozen=True)
class SearchResult:
    answer: str
    sources: tuple[Mapping[str, str], ...]

    def render(self) -> str:
        lines = [
            f"answer: {self.answer}" if self.answer else "answer: (远端未给出回答)",
            "",
            "sources（仅标题与 URL，接口不返回正文片段）:",
        ]
        for index, source in enumerate(self.sources, start=1):
            site = source.get("site_name") or "unknown"
            lines.append(f"{index}. {source.get('title') or '(无标题)'}  [{site}]")
            lines.append(f"   {source.get('url') or ''}")
        lines.append("")
        lines.append("注意：answer 来自远端搜索增强模型，重要事实请按 sources 自行核验。")
        return "\n".join(lines)


@runtime_checkable
class SearchClient(Protocol):
    """一个可替换的搜索引擎。"""

    def search(self, query: str, max_results: int = 3) -> SearchResult:
        """返回 answer 与来源；无法确认「真的搜到了」时必须抛 SearchError。"""
        ...
