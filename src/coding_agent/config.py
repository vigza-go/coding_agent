from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContextSettings:
    total_tokens: int = 1_000_000
    compression_ratio: float = 0.25
    working_trigger_ratio: float = 0.50
    recent_tail_ratio: float = 0.20
    l0_block_count: int = 4
    summary_concurrency: int = 4
    summary_target_ratio: float = 0.50
    summary_max_attempts: int = 3
    recent_tool_interactions: int = 10
    tool_result_inline_tokens: int = 5_000
    reasoning_retain_ratio: float = 0.15

    def __post_init__(self) -> None:
        ratios = {
            "compression_ratio": self.compression_ratio,
            "working_trigger_ratio": self.working_trigger_ratio,
            "recent_tail_ratio": self.recent_tail_ratio,
            "summary_target_ratio": self.summary_target_ratio,
        }
        for name, value in ratios.items():
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.compression_ratio + self.working_trigger_ratio > 0.75:
            raise ValueError("compression and working regions must leave at least 25% headroom")
        if self.l0_block_count < 1:
            raise ValueError("l0_block_count must be positive")
        if self.summary_concurrency < 1:
            raise ValueError("summary_concurrency must be positive")
        if self.summary_max_attempts < 1:
            raise ValueError("summary_max_attempts must be positive")
        if self.recent_tool_interactions < 0:
            raise ValueError("recent_tool_interactions must be non-negative")
        if not 0 < self.reasoning_retain_ratio < 1:
            raise ValueError("reasoning_retain_ratio must be between 0 and 1")

    @property
    def compression_limit(self) -> int:
        return int(self.total_tokens * self.compression_ratio)

    @property
    def working_trigger(self) -> int:
        return int(self.total_tokens * self.working_trigger_ratio)

    @property
    def reasoning_budget(self) -> int:
        return int(self.working_trigger * self.reasoning_retain_ratio)


@dataclass(frozen=True)
class LLMSettings:
    model: str = "deepseek-v4-flash[1M]"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/anthropic"
    max_output_tokens: int = 2_000


@dataclass(frozen=True)
class AgentSettings:
    tool_retry_max: int = 2
    tool_call_limit: int = 200
    model_call_limit: int = 200
    filesystem_max_file_size_mb: int = 10
    bash_enabled: bool = True
    bash_executable: str = "/bin/bash"
    bash_timeout_seconds: int = 120
    bash_max_output_bytes: int = 100_000
    # 配置面只暴露「开关 + 凭证 + 与供应商无关的行为参数」。endpoint / model 属于
    # 具体实现，留在 SearchClient 内部，将来换搜索服务时不用动配置契约。
    # 默认关闭：搜索需要外部凭证，不像 Bash 那样开箱可用。
    search_enabled: bool = True
    search_api_key: str = ""
    search_timeout_seconds: int = 500
    search_max_results_limit: int = 10
    # 子代理：另起进程跑一套完整 agent，默认开。工具/额度/超时等一律沿用父的
    # model_call_limit / tool_call_limit / tool_retry_max，不再复制一份改数。
    subagent_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.subagent_enabled, bool):
            raise TypeError("subagent_enabled must be a boolean")
        positive_values = {
            "tool_retry_max": self.tool_retry_max,
            "tool_call_limit": self.tool_call_limit,
            "model_call_limit": self.model_call_limit,
            "filesystem_max_file_size_mb": self.filesystem_max_file_size_mb,
            "bash_timeout_seconds": self.bash_timeout_seconds,
            "bash_max_output_bytes": self.bash_max_output_bytes,
            "search_timeout_seconds": self.search_timeout_seconds,
            "search_max_results_limit": self.search_max_results_limit,
        }
        for name, value in positive_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.bash_enabled, bool):
            raise TypeError("bash_enabled must be a boolean")
        if not self.bash_executable.strip():
            raise ValueError("bash_executable must not be empty")
        if not isinstance(self.search_enabled, bool):
            raise TypeError("search_enabled must be a boolean")
        # 「开了搜索就要有 key」是跨字段不变量，而且凭证在构造期常常还不存在
        # （测试、局部覆盖、只设了环境变量的场景），所以交给 make_search_client 判定。


@dataclass(frozen=True)
class TUISettings:
    notifications_enabled: bool = True
    prevent_sleep: bool = True
    usage_recent_messages: int = 20
    system_command_timeout_seconds: int = 3
    # macOS `display notification` 默认不发声；填系统声音名（如 "Glass"）即带提示音，
    # 置空字符串可退回纯横幅。非 macOS 上本字段无效果。
    notification_sound: str = "Glass"

    def __post_init__(self) -> None:
        for name in ("notifications_enabled", "prevent_sleep"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"tui.{name} must be a boolean")
        if not isinstance(self.notification_sound, str):
            raise TypeError("tui.notification_sound must be a string")
        for name in ("usage_recent_messages", "system_command_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"tui.{name} must be a positive integer")


@dataclass(frozen=True)
class SummaryLLMSettings:
    """Dedicated model used for context compression/summarization.

    Kept separate from the main agent LLM because compression should be fast, cheap
    and deterministic: extended thinking only burns budget and delays the turn. Leave
    ``model`` empty to reuse the main LLM; ``thinking`` is ``"disabled"`` by default.
    """

    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_output_tokens: int = 0
    thinking: str = "disabled"  # "disabled" | "auto"

    def __post_init__(self) -> None:
        if self.thinking not in ("disabled", "auto"):
            raise ValueError("summary_llm.thinking must be 'disabled' or 'auto'")
        if self.max_output_tokens < 0:
            raise ValueError("summary_llm.max_output_tokens must be >= 0")


@dataclass(frozen=True)
class Settings:
    database_url: str = "mysql+pymysql://root:root@127.0.0.1:3306/langchain?charset=utf8mb4"
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    artifact_dir: Path = field(default_factory=lambda: Path.cwd() / ".artifacts")
    # 解析后的配置文件绝对路径。子代理是独立进程、cwd 可能不同，必须按这个绝对路径回读
    # 同一份配置，否则会像“在 workspace 里找不到 config.json”那样静默读成空配置。
    config_path: Path = field(default_factory=lambda: Path("config.json").resolve())
    llm: LLMSettings = field(default_factory=LLMSettings)
    summary_llm: SummaryLLMSettings = field(default_factory=SummaryLLMSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    tui: TUISettings = field(default_factory=TUISettings)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"configuration root must be an object: {path}")
    return value


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("AGENT_CONFIG", "config.json"))
    resolved_config_path = config_path.resolve()
    raw = _read_json(config_path)
    llm_raw = raw.get("llm", {})
    context_raw = raw.get("context", {})
    agent_raw = raw.get("agent", {})
    tui_values = dict(raw.get("tui", {}))
    for name, environment in (
        ("notifications_enabled", "AGENT_NOTIFICATIONS_ENABLED"),
        ("prevent_sleep", "AGENT_PREVENT_SLEEP"),
    ):
        default = tui_values.get(name, getattr(TUISettings, name))
        if not isinstance(default, bool):
            raise TypeError(f"tui.{name} must be a boolean")
        tui_values[name] = _environment_bool(environment, default)
    if "AGENT_USAGE_RECENT_MESSAGES" in os.environ:
        tui_values["usage_recent_messages"] = int(os.environ["AGENT_USAGE_RECENT_MESSAGES"])
    if "AGENT_NOTIFICATION_SOUND" in os.environ:
        tui_values["notification_sound"] = os.environ["AGENT_NOTIFICATION_SOUND"]
    agent_values = dict(agent_raw)
    for field_name, environment_name in (
        ("bash_enabled", "AGENT_BASH_ENABLED"),
        ("search_enabled", "AGENT_SEARCH_ENABLED"),
        ("subagent_enabled", "AGENT_SUBAGENT_ENABLED"),
    ):
        default = agent_values.get(field_name, getattr(AgentSettings, field_name))
        if not isinstance(default, bool):
            raise TypeError(f"agent.{field_name} must be a boolean")
        agent_values[field_name] = _environment_bool(environment_name, default)
    string_agent_fields = frozenset({"bash_executable"})
    environment_agent_fields = {
        "bash_executable": "AGENT_BASH_EXECUTABLE",
        "bash_timeout_seconds": "AGENT_BASH_TIMEOUT_SECONDS",
        "bash_max_output_bytes": "AGENT_BASH_MAX_OUTPUT_BYTES",
        "search_timeout_seconds": "AGENT_SEARCH_TIMEOUT_SECONDS",
        "search_max_results_limit": "AGENT_SEARCH_MAX_RESULTS_LIMIT",
    }
    for field_name, environment_name in environment_agent_fields.items():
        environment_value = os.getenv(environment_name)
        if environment_value is not None:
            agent_values[field_name] = (
                environment_value if field_name in string_agent_fields else int(environment_value)
            )

    workspace = Path(os.getenv("WORKSPACE_ROOT", raw.get("workspace_root", "."))).resolve()
    artifact_dir = Path(
        os.getenv("ARTIFACT_DIR", raw.get("artifact_dir", workspace / ".artifacts"))
    ).resolve()
    llm = LLMSettings(
        model=os.getenv("LLM_MODEL", llm_raw.get("model", LLMSettings.model)),
        api_key=os.getenv("LLM_API_KEY", llm_raw.get("api_key", "")),
        base_url=os.getenv("LLM_BASE_URL", llm_raw.get("base_url", LLMSettings.base_url)),
        max_output_tokens=int(
            os.getenv(
                "LLM_MAX_OUTPUT_TOKENS",
                llm_raw.get("max_output_tokens", llm_raw.get("max_tokens", 2_000)),
            )
        ),
    )
    # DashScope 的模型与联网搜索共用同一把 key，所以允许回落到 llm.api_key，
    # 省掉在配置里把同一个值抄两遍。
    agent_values["search_api_key"] = (
        os.getenv("AGENT_SEARCH_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or agent_raw.get("search_api_key", "")
        or llm.api_key
    )
    summary_raw = raw.get("summary_llm", {})
    if not isinstance(summary_raw, dict):
        raise TypeError("summary_llm must be an object")
    summary_llm = SummaryLLMSettings(
        model=os.getenv("SUMMARY_LLM_MODEL", summary_raw.get("model", "")),
        api_key=os.getenv("SUMMARY_LLM_API_KEY", summary_raw.get("api_key", "")),
        base_url=os.getenv("SUMMARY_LLM_BASE_URL", summary_raw.get("base_url", "")),
        max_output_tokens=int(
            os.getenv(
                "SUMMARY_LLM_MAX_OUTPUT_TOKENS",
                summary_raw.get("max_output_tokens", 0),
            )
        ),
        thinking=os.getenv("SUMMARY_LLM_THINKING", summary_raw.get("thinking", "disabled")),
    )
    return Settings(
        database_url=os.getenv("DATABASE_URL", raw.get("database_url", Settings.database_url)),
        workspace_root=workspace,
        artifact_dir=artifact_dir,
        config_path=resolved_config_path,
        llm=llm,
        summary_llm=summary_llm,
        context=ContextSettings(**context_raw),
        agent=AgentSettings(**agent_values),
        tui=TUISettings(**tui_values),
    )
