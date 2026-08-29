from __future__ import annotations

from coding_agent.workspace.artifacts import ArtifactStore


def test_large_tool_result_is_offloaded(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    rendered, path = store.offload_tool_result(
        thread_id="thread/a",
        user_seq=2,
        tool_call_id="call:1",
        content="x" * 1000,
        inline_limit=20,
    )
    assert path is not None
    assert path.read_text(encoding="utf-8") == "x" * 1000
    assert str(path) in rendered
    assert len(rendered) < 1000
