from __future__ import annotations

import json

import pytest

from coding_agent.config import SummaryLLMSettings, load_settings
from coding_agent.integrations import langchain_agent


def test_summary_llm_defaults_to_main_llm_with_thinking_disabled(monkeypatch, tmp_path):
    for name in ("SUMMARY_LLM_MODEL", "SUMMARY_LLM_MAX_OUTPUT_TOKENS", "SUMMARY_LLM_THINKING"):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(tmp_path / "missing.json")

    assert settings.summary_llm == SummaryLLMSettings()
    assert settings.summary_llm.model == ""  # 空 => 回落主模型
    assert settings.summary_llm.thinking == "disabled"


def test_summary_llm_from_json(monkeypatch, tmp_path):
    for name in ("SUMMARY_LLM_MODEL", "SUMMARY_LLM_MAX_OUTPUT_TOKENS", "SUMMARY_LLM_THINKING"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "summary_llm": {
                    "model": "qwen3.7-flash",
                    "max_output_tokens": 4000,
                    "thinking": "disabled",
                }
            }
        )
    )
    settings = load_settings(path)
    assert settings.summary_llm.model == "qwen3.7-flash"
    assert settings.summary_llm.max_output_tokens == 4000
    assert settings.summary_llm.thinking == "disabled"


def test_summary_llm_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SUMMARY_LLM_MODEL", "qwen-turbo")
    monkeypatch.setenv("SUMMARY_LLM_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.setenv("SUMMARY_LLM_THINKING", "auto")

    settings = load_settings(tmp_path / "missing.json")

    assert settings.summary_llm.model == "qwen-turbo"
    assert settings.summary_llm.max_output_tokens == 2000
    assert settings.summary_llm.thinking == "auto"


def test_invalid_summary_thinking_fails():
    with pytest.raises(ValueError, match="thinking"):
        SummaryLLMSettings(thinking="sometimes")


class _FakeChat:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_build_summary_model_requests_disabled_thinking(monkeypatch):
    monkeypatch.setattr(langchain_agent, "ChatAnthropic", _FakeChat)

    settings = load_settings("config.json")
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "llm": settings.llm.__class__(
                model="agent-main", api_key="k", base_url="https://x", max_output_tokens=1_000_000
            ),
            "summary_llm": SummaryLLMSettings(
                model="qwen3.7-flash", api_key="k", base_url="https://x",
                max_output_tokens=4000, thinking="disabled",
            ),
        }
    )
    model = langchain_agent.build_summary_model(settings)

    assert model.kwargs["model"] == "qwen3.7-flash"
    assert model.kwargs["max_tokens"] == 4000
    assert model.kwargs["thinking"] == {"type": "disabled"}


def test_build_summary_model_falls_back_to_main_llm(monkeypatch):
    monkeypatch.setattr(langchain_agent, "ChatAnthropic", _FakeChat)

    settings = load_settings("config.json")
    settings = settings.__class__(
        **{
            **settings.__dict__,
            "llm": settings.llm.__class__(
                model="agent-main", api_key="k", base_url="https://x", max_output_tokens=5000
            ),
            "summary_llm": SummaryLLMSettings(model="", thinking="disabled"),
        }
    )
    model = langchain_agent.build_summary_model(settings)

    assert model.kwargs["model"] == "agent-main"
    assert model.kwargs["max_tokens"] == 5000
    assert model.kwargs["thinking"] == {"type": "disabled"}
