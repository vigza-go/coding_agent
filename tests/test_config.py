from __future__ import annotations

from coding_agent.config import load_settings


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
