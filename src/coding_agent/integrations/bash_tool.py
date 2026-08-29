from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ..services.bash_execution import BashExecutionService


class BashToolInput(BaseModel):
    command: str = Field(description="Bash command to execute from the workspace root")
    timeout_seconds: int | None = Field(
        default=None,
        description="Optional timeout no greater than the configured Bash maximum",
    )


def make_bash_tool(executor: BashExecutionService) -> BaseTool:
    @tool("bash", args_schema=BashToolInput)
    def bash(
        command: str,
        runtime: ToolRuntime[None, Any],
        timeout_seconds: int | None = None,
    ) -> ToolMessage:
        """Execute a non-interactive Bash command from the workspace root.

        Relative paths such as `src/app.py` resolve from the workspace root. Absolute
        paths are real host paths, not filesystem-tool virtual paths. Commands cannot
        read stdin, are never retried, and file changes made here cannot be undone by
        /undo. Do not batch this tool with file operations that depend on its effects.
        """

        tool_call_id = str(runtime.tool_call_id or "unknown")
        try:
            result = executor.execute(command, timeout_seconds=timeout_seconds)
        except ValueError as error:
            return ToolMessage(
                content=f"Error: {error}",
                name="bash",
                tool_call_id=tool_call_id,
                status="error",
            )
        status = "success" if result.exit_code == 0 and not result.timed_out else "error"
        content = (
            f"{result.output}\n\n"
            f"[exit_code={result.exit_code}; duration={result.duration_seconds:.2f}s]"
        )
        return ToolMessage(
            content=content,
            name="bash",
            tool_call_id=tool_call_id,
            status=status,
            artifact={
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "truncated": result.truncated,
            },
        )

    return bash
