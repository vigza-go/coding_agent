from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from coding_agent.integrations.bash_tool import make_bash_tool
from coding_agent.services.bash_execution import BashExecutionService


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


def make_executor(tmp_path, *, timeout: int = 2, max_output_bytes: int = 1_000):
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is not installed")
    return BashExecutionService(
        executable=executable,
        workspace_root=tmp_path,
        timeout_seconds=timeout,
        max_output_bytes=max_output_bytes,
        env={"PATH": "/usr/bin:/bin"},
    )


def test_bash_runs_with_bash_semantics_from_workspace_root(tmp_path):
    (tmp_path / "marker.txt").write_text("ok", encoding="utf-8")
    executor = make_executor(tmp_path)

    result = executor.execute('[[ -f marker.txt ]] && printf "%s" "${BASH_VERSION%%.*}"')

    assert result.exit_code == 0
    assert result.output.isdigit()
    assert result.timed_out is False


def test_bash_bounds_output_and_kills_timed_out_process(tmp_path):
    executor = make_executor(tmp_path, timeout=1, max_output_bytes=5)

    truncated = executor.execute("printf 123456789")
    timed_out = executor.execute("sleep 5")

    assert truncated.output.startswith("12345")
    assert truncated.truncated is True
    assert timed_out.exit_code == 124
    assert timed_out.timed_out is True
    assert "timed out" in timed_out.output
    with pytest.raises(ValueError, match="timeout must be positive"):
        executor.execute("printf should-not-run", timeout_seconds=0)


def test_bash_tool_returns_nonzero_exit_as_error_message(tmp_path):
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "printf failure; exit 7"},
                        "id": "bash-call",
                    }
                ],
            ),
            AIMessage(content="handled"),
        ]
    )
    agent = create_agent(model=model, tools=[make_bash_tool(make_executor(tmp_path))])

    result = agent.invoke({"messages": [{"role": "user", "content": "run"}]})

    message = result["messages"][-2]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert message.tool_call_id == "bash-call"
    assert message.artifact["exit_code"] == 7
    assert "failure" in message.content


def test_bash_interrupt_kills_the_running_process_group(tmp_path):
    executor = make_executor(tmp_path, timeout=10)
    executor.prepare_turn()
    started = monotonic()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, "sleep 10")
        sleep(0.1)
        executor.interrupt_all()
        result = future.result(timeout=2)

    assert result.exit_code == 130
    assert result.interrupted is True
    assert "interrupted by user" in result.output
    assert monotonic() - started < 2
