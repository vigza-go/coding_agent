from __future__ import annotations

import os
import subprocess
from unittest.mock import Mock

import pytest

from coding_agent.config import TUISettings
from coding_agent.integrations import desktop
from coding_agent.integrations.desktop import DesktopService


@pytest.fixture
def mac_helpers(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: f"/usr/bin/{name}")
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    run = Mock()
    monkeypatch.setattr(desktop.subprocess, "Popen", popen)
    monkeypatch.setattr(desktop.subprocess, "run", run)
    return process, popen, run


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_sleep_assertion_is_scoped_to_turn_even_on_failure(mac_helpers, failure):
    process, popen, _ = mac_helpers
    service = DesktopService(TUISettings(), Mock())

    def turn():
        with service.keep_awake():
            process.terminate.assert_not_called()
            if failure:
                raise failure()

    if failure:
        with pytest.raises(failure):
            turn()
    else:
        turn()
    assert popen.call_args.args[0] == ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]
    assert popen.call_args.kwargs["start_new_session"] is True
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=3)


def test_stuck_sleep_helper_is_killed(mac_helpers):
    process, _, _ = mac_helpers
    process.wait.side_effect = [subprocess.TimeoutExpired("caffeinate", 3), 0]
    with DesktopService(TUISettings(), Mock()).keep_awake():
        pass
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_disabled_helpers_do_not_start_processes(mac_helpers):
    _, popen, run = mac_helpers
    service = DesktopService(TUISettings(prevent_sleep=False, notifications_enabled=False), Mock())
    with service.keep_awake():
        pass
    assert service.notify("finished") is False
    popen.assert_not_called()
    run.assert_not_called()


def test_missing_platform_support_warns_once_without_blocking(monkeypatch, mac_helpers):
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    warn = Mock()
    service = DesktopService(TUISettings(), warn)
    for _ in range(2):
        with service.keep_awake():
            pass
    assert warn.call_count == 1
    assert service.notify("finished") is False
    mac_helpers[1].assert_not_called()
    mac_helpers[2].assert_not_called()


def test_failed_sleep_helper_is_best_effort(mac_helpers):
    _, popen, _ = mac_helpers
    popen.side_effect = OSError("cannot start")
    warn = Mock()
    with DesktopService(TUISettings(), warn).keep_awake():
        pass
    warn.assert_called_once()


def test_notification_passes_text_as_data_not_script(mac_helpers):
    _, _, run = mac_helpers
    message = 'thread "x"; $(touch /not-a-command)\n任务已完成'
    assert DesktopService(TUISettings(), Mock()).notify(message)
    arguments = run.call_args.args[0]
    assert arguments[-1] == message
    assert message not in arguments[2]
    assert run.call_args.kwargs["timeout"] == 3
    assert not run.call_args.kwargs.get("shell", False)


def test_notification_failure_returns_fallback_signal(mac_helpers):
    _, _, run = mac_helpers
    run.side_effect = subprocess.TimeoutExpired("osascript", 3)
    warn = Mock()
    service = DesktopService(TUISettings(), warn)
    assert service.notify("finished") is False
    assert service.notify("finished") is False
    warn.assert_called_once()
