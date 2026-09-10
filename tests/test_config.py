from __future__ import annotations

import json

import pytest

from coding_agent.config import ContextSettings, TUISettings, load_settings


@pytest.mark.parametrize("budget", [0, -1])
def test_the_pin_budget_must_be_positive(budget):
    """便签里工作状态那一段的上限：0 或负数会让便签永远只剩一行提示，等于把笔记本关了。"""

    with pytest.raises(ValueError, match="pin_state_budget_tokens"):
        ContextSettings(pin_state_budget_tokens=budget)


def test_bash_is_enabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_BASH_ENABLED", raising=False)

    settings = load_settings(tmp_path / "missing.json")

    assert settings.agent.bash_enabled is True


def test_bash_settings_can_be_loaded_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_BASH_ENABLED", "true")
    monkeypatch.setenv("AGENT_BASH_EXECUTABLE", "/bin/bash")
    monkeypatch.setenv("AGENT_BASH_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("AGENT_BASH_MAX_OUTPUT_BYTES", "2048")

    settings = load_settings(tmp_path / "missing.json")

    assert settings.agent.bash_enabled is True
    assert settings.agent.bash_executable == "/bin/bash"
    assert settings.agent.bash_timeout_seconds == 45
    assert settings.agent.bash_max_output_bytes == 2048


def test_tui_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_NOTIFICATIONS_ENABLED", "false")
    monkeypatch.setenv("AGENT_PREVENT_SLEEP", "false")
    monkeypatch.setenv("AGENT_USAGE_RECENT_MESSAGES", "50")
    settings = load_settings(tmp_path / "missing.json")
    assert settings.tui.notifications_enabled is False
    assert settings.tui.prevent_sleep is False
    assert settings.tui.usage_recent_messages == 50


@pytest.mark.parametrize(
    "values",
    [
        {"notifications_enabled": "false"},
        {"prevent_sleep": 1},
        {"usage_recent_messages": 0},
        {"system_command_timeout_seconds": -1},
    ],
)
def test_invalid_tui_settings_fail_early(values):
    with pytest.raises((ValueError, TypeError)):
        TUISettings(**values)


def test_tui_settings_from_json(monkeypatch, tmp_path):
    for name in (
        "AGENT_NOTIFICATIONS_ENABLED",
        "AGENT_PREVENT_SLEEP",
        "AGENT_USAGE_RECENT_MESSAGES",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "tui": {
                    "notifications_enabled": False,
                    "prevent_sleep": False,
                    "usage_recent_messages": 12,
                    "system_command_timeout_seconds": 5,
                }
            }
        )
    )
    settings = load_settings(path)
    assert settings.tui == TUISettings(False, False, 12, 5)
