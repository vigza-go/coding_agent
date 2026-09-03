from __future__ import annotations

import json

import pytest

from coding_agent.config import AgentSettings, Settings, load_settings
from coding_agent.integrations.langchain_agent import make_search_tool
from coding_agent.integrations.search import (
    AliSearchClient,
    SearchClient,
    SearchError,
    make_search_client,
)


def make_output(titles: tuple[str, ...], answer: str = "2023 年 10 月") -> dict:
    """按 2026-09 实测到的 DashScope 响应结构构造（条目里没有正文片段）。"""
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": answer, "role": "assistant"}}
        ],
        "search_info": {
            "extra_tool_info": [],
            "search_results": [
                {
                    "icon": "https://example.com/favicon.ico",
                    "site_name": f"site-{index}",
                    "index": index,
                    "title": title,
                    "url": f"https://example.com/{index}",
                }
                for index, title in enumerate(titles, start=1)
            ],
        },
    }


class FakeResponse:
    """模仿 SDK 的 GenerationResponse：属性访问，不是 dict。"""

    def __init__(self, output, status_code=200, code="", message=""):
        self.output = output
        self.status_code = status_code
        self.code = code
        self.message = message


class RecordingCall:
    """AliSearchClient 的注入点：不联网也能锁住请求形状与解析行为。"""

    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.response = response if response is not None else FakeResponse(
            make_output(("标题一", "标题二"))
        )
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def make_client(search_call, **overrides) -> AliSearchClient:
    options = {"api_key": "test-key", "timeout_seconds": 20, "max_results_limit": 3}
    options.update(overrides)
    return AliSearchClient(search_call=search_call, **options)


@pytest.fixture(autouse=True)
def _isolate_search_environment(monkeypatch):
    """配置层断言不能被开发机上的 export 污染，先一律清空。"""
    for name in (
        "AGENT_SEARCH_ENABLED",
        "AGENT_SEARCH_API_KEY",
        "AGENT_SEARCH_TIMEOUT_SECONDS",
        "AGENT_SEARCH_MAX_RESULTS_LIMIT",
        "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_search_returns_answer_and_sources():
    call = RecordingCall()

    result = make_client(call).search("Python 3.13 发布年份", max_results=2)

    assert result.answer == "2023 年 10 月"
    assert [(s["title"], s["url"]) for s in result.sources] == [
        ("标题一", "https://example.com/1"),
        ("标题二", "https://example.com/2"),
    ]
    assert result.sources[0]["site_name"] == "site-1"


def test_search_sends_the_load_bearing_options_and_bounds_the_request_timeout():
    call = RecordingCall()

    make_client(call, timeout_seconds=23).search("任意查询")

    kwargs = call.calls[0]
    assert kwargs["enable_search"] is True
    assert kwargs["result_format"] == "message"
    assert kwargs["api_key"] == "test-key"
    # SDK 默认超时是 300 秒，不显式传就等于没有超时保护。
    assert kwargs["request_timeout"] == 23


def test_enable_source_is_sent_because_its_absence_is_silent():
    """实测：不传 enable_source 时搜索仍执行、HTTP 仍 200，但 search_results 为空。

    那个形状下模型会给出一段"像是查过"的回答，却没有可核验出处。所以这个参数是
    load-bearing 的，用测试钉住它不被当成"看着多余的字段"清理掉。
    """
    call = RecordingCall()

    make_client(call).search("q")

    assert call.calls[0]["search_options"]["enable_source"] is True


def test_max_results_is_not_forwarded_because_the_server_ignores_it():
    """实测传 1 / 2 / 10 都稳定返回 9 条：条数只能本地截断，不伪造远程参数。"""
    call = RecordingCall()

    make_client(call).search("q", max_results=2)

    assert "max_results" not in call.calls[0]["search_options"]


def test_search_clamps_max_results_to_configured_limit():
    call = RecordingCall(FakeResponse(make_output(tuple(f"标题{i}" for i in range(10)))))

    result = make_client(call, max_results_limit=3).search("q", max_results=999)

    assert len(result.sources) == 3


def test_answer_without_sources_is_rejected():
    """实测：没搜索时模型也会凭空编造实时信息。没有来源就必须判失败，不能返回 answer。"""
    hallucinated = FakeResponse({"choices": [{"message": {"content": "今天晴，气象局数据"}}]})

    with pytest.raises(SearchError):
        make_client(RecordingCall(hallucinated)).search("今天天气")


def test_empty_search_results_are_rejected():
    output = make_output(("甲",))
    output["search_info"]["search_results"] = []

    with pytest.raises(SearchError):
        make_client(RecordingCall(FakeResponse(output))).search("q")


@pytest.mark.parametrize("bad_output", [None, "not-a-dict", {}])
def test_malformed_response_becomes_search_error(bad_output):
    with pytest.raises(SearchError):
        make_client(RecordingCall(FakeResponse(bad_output))).search("q")


def test_non_ok_status_becomes_search_error():
    failure = FakeResponse(None, status_code=401, code="InvalidApiKey", message="invalid key")

    with pytest.raises(SearchError, match="401"):
        make_client(RecordingCall(failure)).search("q")


def test_sdk_network_error_becomes_search_error():
    """SDK 把连接失败和读超时作为异常抛出（不是错误对象），必须收口成 SearchError。"""
    with pytest.raises(SearchError, match="搜索请求失败"):
        make_client(RecordingCall(error=ConnectionError("read timeout"))).search("q")


def test_missing_answer_keeps_sources_with_empty_answer():
    """远端只给来源不给回答时不该失败：来源本身才是这个工具的核心数据。"""
    output = make_output(("甲",))
    output["choices"] = "unexpected-shape"

    result = make_client(RecordingCall(FakeResponse(output))).search("q")

    assert result.answer == ""
    assert result.sources[0]["title"] == "甲"


def test_blank_query_is_rejected_before_any_request():
    call = RecordingCall()

    with pytest.raises(SearchError):
        make_client(call).search("   ")

    assert call.calls == []


def test_client_requires_an_api_key():
    with pytest.raises(ValueError, match="search_api_key"):
        make_client(RecordingCall(), api_key="   ")


def test_ali_client_satisfies_the_search_protocol():
    assert isinstance(make_client(RecordingCall()), SearchClient)


def test_tool_renders_sources_and_flags_that_answer_is_remote():
    tool = make_search_tool(make_client(RecordingCall(FakeResponse(make_output(("标题甲",))))))

    rendered = tool.invoke({"query": "q", "max_results": 3})

    assert "标题甲" in rendered
    assert "https://example.com/1" in rendered
    assert "不返回正文片段" in rendered
    assert "自行核验" in rendered


def test_tool_propagates_search_error_instead_of_returning_error_text():
    """失败要抛出，交给中间件转成 error ToolMessage。

    返回 `"搜索失败：..."` 这类字符串会被包成 status=success 的工具结果，
    UI 显示成功、模型也当成一次正常查询——这是最坏的静默降级。
    """
    hallucinated = FakeResponse({"choices": [{"message": {"content": "编的回答"}}]})
    tool = make_search_tool(make_client(RecordingCall(hallucinated)))

    with pytest.raises(SearchError):
        tool.invoke({"query": "q", "max_results": 3})


def test_factory_returns_none_when_search_is_disabled():
    assert make_search_client(AgentSettings(search_enabled=False)) is None


def test_factory_builds_the_configured_engine():
    client = make_search_client(
        AgentSettings(
            search_enabled=True,
            search_api_key="k",
            search_timeout_seconds=7,
            search_max_results_limit=4,
        )
    )

    assert isinstance(client, AliSearchClient)
    assert client.timeout_seconds == 7
    assert client.max_results_limit == 4


def test_provider_details_stay_out_of_the_config_contract():
    """换搜索引擎时配置契约不动：端点/模型名属于实现。"""
    fields = AgentSettings.__dataclass_fields__
    assert "search_endpoint" not in fields
    assert "search_model" not in fields
    assert {name for name in fields if name.startswith("search_")} == {
        "search_enabled",
        "search_api_key",
        "search_timeout_seconds",
        "search_max_results_limit",
    }


def test_search_api_key_falls_back_to_llm_api_key(tmp_path, monkeypatch):
    for name in ("AGENT_SEARCH_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"llm": {"api_key": "shared-dashscope-key"}, "agent": {"search_enabled": True}}
        ),
        encoding="utf-8",
    )

    settings = load_settings(config)

    # 开搜索但没抄 key：同一厂商下直接复用模型 key。
    assert settings.agent.search_enabled is True
    assert settings.agent.search_api_key == "shared-dashscope-key"


def test_settings_construct_fine_without_any_credential(tmp_path, monkeypatch):
    """构造期不得索要凭证——否则每个默认构造 Settings() 的测试都会被拖下水。

    这条同时是那次 16 个 tui 测试集体失败的回归守卫：跨字段校验一旦搬回
    AgentSettings.__post_init__，这里就会红。
    """
    for name in ("AGENT_SEARCH_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    settings = load_settings(config)

    assert settings.agent.search_api_key == ""
    assert Settings()  # 默认构造必须无条件成立
    assert make_search_client(AgentSettings(search_enabled=True, search_api_key="k"))


def test_factory_raises_actionably_when_enabled_without_key():
    """开关打开却没凭证：在装配点报错，并直接说出怎么关掉。"""
    with pytest.raises(ValueError, match="no search credential"):
        make_search_client(AgentSettings(search_enabled=True, search_api_key="   "))


def test_search_env_overrides_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_ENABLED", "true")
    monkeypatch.setenv("AGENT_SEARCH_API_KEY", "from-env")
    monkeypatch.setenv("AGENT_SEARCH_MAX_RESULTS_LIMIT", "5")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agent": {"search_api_key": "from-file"}}), encoding="utf-8")

    agent = load_settings(config).agent

    assert agent.search_enabled is True
    assert agent.search_api_key == "from-env"
    assert agent.search_max_results_limit == 5


def test_disabled_search_needs_no_key():
    """关掉搜索就不该要凭证：工厂安静返回 None，工具不注册。"""
    assert make_search_client(AgentSettings(search_enabled=False, search_api_key="")) is None
