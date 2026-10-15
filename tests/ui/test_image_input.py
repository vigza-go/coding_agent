from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from prompt_toolkit.keys import Keys

from coding_agent.config import Settings
from coding_agent.integrations import image_clipboard
from coding_agent.ui.commands import parse_command
from coding_agent.ui.tui import _input_bindings
from tests.test_images import png_bytes
from tests.ui.test_tui_lifecycle import make_ui


def test_paste_detach_and_image_only_submission(monkeypatch, tmp_path):
    monkeypatch.setattr("coding_agent.ui.tui.read_clipboard_image", png_bytes)
    ui, app, _, _ = make_ui(monkeypatch, Settings(workspace_root=tmp_path, artifact_dir=tmp_path))
    ui._paste_image()
    assert len(ui.attachments) == 1
    attachment = ui.attachments[0]
    assert "待发送图片 1 张" in "".join(text for _, text in ui._toolbar())
    app.run_turn.return_value = AIMessage(content="seen")
    ui.session = Mock()
    ui.session.prompt.side_effect = ["", EOFError()]
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    ui.run()
    assert app.run_turn.call_args.args == ("t1", "")
    assert app.run_turn.call_args.kwargs["attachments"] == [attachment]
    assert not ui.attachments
    ui._paste_image()
    ui._handle_command(parse_command("/detach 1"))
    assert not ui.attachments


def test_switch_and_cancel_clear_pending_images_but_prestart_failure_keeps_them(
    monkeypatch, tmp_path
):
    ui, app, _, _ = make_ui(monkeypatch, Settings(artifact_dir=tmp_path))
    image = ui.images.import_bytes(png_bytes())
    ui.attachments = [image]
    app.run_turn.side_effect = RuntimeError("before persistence")
    with pytest.raises(RuntimeError):
        ui._run_turn("look")
    assert ui.attachments == [image]
    ui._handle_command(parse_command("/thread other"))
    assert not ui.attachments
    ui.attachments = [image]
    ui.session = Mock()
    ui.session.prompt.side_effect = [KeyboardInterrupt(), EOFError()]
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    ui.run()
    assert not ui.attachments


def test_image_paste_binding_and_clipboard_adapter(monkeypatch, tmp_path):
    assert _input_bindings(lambda: None).get_bindings_for_keys((Keys.ControlV,))
    monkeypatch.setattr(image_clipboard.sys, "platform", "darwin")
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        assert kwargs["timeout"] == 10
        Path(args[-1]).write_bytes(png_bytes())

    monkeypatch.setattr(image_clipboard.subprocess, "run", run)
    assert image_clipboard.read_clipboard_image() == png_bytes()
    assert seen[0][0] == "osascript"
    assert not Path(seen[0][-1]).exists()
    monkeypatch.setattr(image_clipboard.sys, "platform", "linux")
    with pytest.raises(ValueError, match="macOS"):
        image_clipboard.read_clipboard_image()


def test_failed_paste_keeps_existing_text_and_attachments(monkeypatch, tmp_path):
    ui, _, output, _ = make_ui(monkeypatch, Settings(artifact_dir=tmp_path))
    image = ui.images.import_bytes(png_bytes())
    ui.attachments = [image]
    monkeypatch.setattr(
        "coding_agent.ui.tui.read_clipboard_image",
        Mock(side_effect=ValueError("clipboard unavailable")),
    )
    ui._paste_image()
    assert ui.attachments == [image]
    assert "clipboard unavailable" in output.getvalue()
