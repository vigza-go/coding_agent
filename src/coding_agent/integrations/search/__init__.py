"""联网搜索的可插拔入口。

配置面只提供 `search_enabled` / `search_api_key` / `search_timeout_seconds` /
`search_max_results_limit` 这四项与供应商无关的内容；provider 专有细节（端点、模型名、
payload 形状）全部留在各实现内部。

新增一个搜索引擎 = 新增一个实现文件 + 在 `make_search_client` 里加一个分支，
调用方与配置契约都不动。
"""

from __future__ import annotations

from ...config import AgentSettings
from .aliyun import DEFAULT_MODEL, AliSearchClient
from .base import SearchClient, SearchError, SearchResult

__all__ = [
    "AliSearchClient",
    "SearchClient",
    "SearchError",
    "SearchResult",
    "make_search_client",
]


def make_search_client(agent: AgentSettings) -> SearchClient | None:
    """按配置装配搜索客户端；搜索关闭时返回 None，工具就不会注册。"""
    if not agent.search_enabled:
        return None
    return AliSearchClient(
        api_key=agent.search_api_key,
        timeout_seconds=agent.search_timeout_seconds,
        max_results_limit=agent.search_max_results_limit,
        model=DEFAULT_MODEL,
    )
