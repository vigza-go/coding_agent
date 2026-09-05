"""阿里云百炼（DashScope）联网搜索实现。

该服务**没有独立的搜索 API**：联网检索是生成请求上的一个插件，官方 SDK 已把
`enable_search` / `search_options` 暴露成 `Generation.call` 的一等参数，所以这里走
SDK 而不是手搓 HTTP —— 手搓等于自己维护一份随时可能被上游改掉的私有协议。

实测约束（2026-09，qwen-plus，逐条本地验证过）：

- 结果条目只有 `title` / `url` / `site_name` / `icon` / `index`，**接口不返回正文片段**，
  所以本实现给不出 snippet；真要正文得换搜索服务或再抓取。
- `search_options.enable_source` 是**唯一决定 `search_results` 是否出现**的参数。不传它时
  搜索照样执行（`usage.plugins.search.count=1`）、HTTP 照样返回 200、回答照样流畅，但来源
  为空 —— 这是最危险的静默形状：等于把模型自身知识包装成"搜索结果"喂回上下文。
- `search_options.max_results` 服务端**不生效**（传 1 / 2 / 10 都稳定返回 9 条），条数只能在
  本地截断；`forced_search` 在这组实验里没有可观测差异，仅作为"即使模型自认为知道也强制查"
  的保险保留，真正 load-bearing 的是 `enable_source`。
- 一次搜索固定注入约 4.3k **input** tokens（4380 vs 不搜索时 17），且不随 `max_results` 变化。
  这是搜索工具的真实预算成本，会直接挤占工作区、影响压缩触发点。
- 网络与超时错误由 SDK **抛异常**（默认超时 300 秒），必须显式传 `request_timeout`
  才有超时保护；失败不会以错误对象的形式返回。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dashscope import Generation

from .base import SearchError, SearchResult

DEFAULT_MODEL = "qwen-plus"
HTTP_OK = 200

SearchCall = Callable[..., Any]


class AliSearchClient:
    """发起一次强制联网检索的生成调用，返回答案 + 来源。"""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: int = 20,
        max_results_limit: int = 10,
        model: str = DEFAULT_MODEL,
        search_call: SearchCall | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("search_api_key must be provided")
        if timeout_seconds < 1:
            raise ValueError("search timeout must be positive")
        if max_results_limit < 1:
            raise ValueError("search max_results_limit must be positive")
        if not model.strip():
            raise ValueError("search model must not be empty")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_results_limit = max_results_limit
        self._api_key = api_key
        # 注入点：测试里传 fake，生产走 SDK。
        self._call = search_call or Generation.call

    def search(self, query: str, max_results: int = 3) -> SearchResult:
        if not query.strip():
            raise SearchError("query must not be empty")
        # 模型可以要求更少，但不能突破配置上限——和 Bash 的 timeout_seconds 同一策略。
        wanted = max(1, min(int(max_results), self.max_results_limit))
        try:
            response = self._call(
                model=self.model,
                api_key=self._api_key,
                messages=[{"role": "user", "content": query}], # type: ignore
                result_format="message",
                enable_search=True,
                # enable_source 缺省时服务端会静默返回 200 + 流畅回答但零来源，绝不能省。
                # 故意不传 max_results：实测服务端忽略它，写了只会误导后来的维护者。
                search_options={"forced_search": True, "enable_source": True},
                request_timeout=self.timeout_seconds,
            )
        except (OSError, ValueError) as error:
            # OSError 覆盖 SDK 的连接失败与读超时（requests 异常都是 IOError 子类）；
            # ValueError 覆盖响应解析失败。
            raise SearchError(f"搜索请求失败：{type(error).__name__}: {error}") from error
        status_code = getattr(response, "status_code", None)
        if status_code != HTTP_OK:
            code = getattr(response, "code", "") or ""
            message = getattr(response, "message", "") or ""
            raise SearchError(f"搜索服务返回 status={status_code} code={code!r} {message}")
        return self._parse(getattr(response, "output", None), wanted)

    @staticmethod
    def _parse(output: Any, wanted: int) -> SearchResult:
        if not isinstance(output, Mapping):
            raise SearchError("搜索响应缺少 output 字段")
        search_info = output.get("search_info")
        raw_results = search_info.get("search_results") if isinstance(search_info, Mapping) else None
        if not isinstance(raw_results, list) or not raw_results:
            raise SearchError("搜索未返回任何来源；模型自身知识不得当作搜索结果返回")
        sources: list[Mapping[str, str]] = []
        for item in raw_results[:wanted]:
            if not isinstance(item, Mapping):
                continue
            sources.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "site_name": str(item.get("site_name") or ""),
                }
            )
        if not sources:
            raise SearchError("搜索来源解析后为空")
        choices = output.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        answer = str(message.get("content") or "").strip() if isinstance(message, Mapping) else ""
        return SearchResult(answer=answer, sources=tuple(sources))


