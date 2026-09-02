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
    """按配置装配搜索客户端；搜索关闭时返回 None，工具就不会注册。

    凭证检查放在这里而不是 AgentSettings.__post_init__：构造期没有凭证是合法的
    （测试、局部覆盖），但"开关打开却搜不了"必须是装配错误。
    """
    if not agent.search_enabled:
        return None
    if not agent.search_api_key.strip():
        raise ValueError(
            "agent.search_enabled is on but no search credential is available: set "
            "AGENT_SEARCH_API_KEY / DASHSCOPE_API_KEY (or llm.api_key / agent.search_api_key "
            "in the config file), or disable search with AGENT_SEARCH_ENABLED=false"
        )
    return AliSearchClient(
        api_key=agent.search_api_key,
        timeout_seconds=agent.search_timeout_seconds,
        max_results_limit=agent.search_max_results_limit,
        model=DEFAULT_MODEL,
    )
