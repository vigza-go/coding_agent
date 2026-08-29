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

    @property
    def compression_limit(self) -> int:
        return int(self.total_tokens * self.compression_ratio)

    @property
    def working_trigger(self) -> int:
        return int(self.total_tokens * self.working_trigger_ratio)


@dataclass(frozen=True)
class LLMSettings:
    model: str = "deepseek-v4-flash[1M]"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/anthropic"
    max_output_tokens: int = 2_000


@dataclass(frozen=True)
class AgentSettings:
    tool_retry_max: int = 2
    tool_call_limit: int = 15
    model_call_limit: int = 20
    filesystem_max_file_size_mb: int = 10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Settings:
    database_url: str = "mysql+pymysql://root:root@127.0.0.1:3306/langchain?charset=utf8mb4"
    checkpoint_database_url: str = "mysql://root:root@127.0.0.1:3306/langchain?charset=utf8mb4"
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    artifact_dir: Path = field(default_factory=lambda: Path.cwd() / ".artifacts")
    llm: LLMSettings = field(default_factory=LLMSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"configuration root must be an object: {path}")
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("AGENT_CONFIG", "config.json"))
    raw = _read_json(config_path)
    llm_raw = raw.get("llm", {})
    context_raw = raw.get("context", {})
    agent_raw = raw.get("agent", {})

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
    return Settings(
        database_url=os.getenv("DATABASE_URL", raw.get("database_url", Settings.database_url)),
        checkpoint_database_url=os.getenv(
            "CHECKPOINT_DATABASE_URL",
            raw.get("checkpoint_database_url", Settings.checkpoint_database_url),
        ),
        workspace_root=workspace,
        artifact_dir=artifact_dir,
        llm=llm,
        context=ContextSettings(**context_raw),
        agent=AgentSettings(**agent_raw),
    )
