"""子代理运行时的「薄片」中间件、子图装配，以及独立进程入口。

设计要点（都验证过）：子代理另起一个进程，跑一套**完整**的 agent 循环，只把中间件换成
这里的 :class:`SubAgentMiddleware` —— 相对正常中间件只改两件事：

- ``before_model`` 不再把父会话历史投影进来（返回 None，子代理只看自己那一句任务）；
- 消息不落 ``messages`` 表（:class:`NoopPersistence`），因此永远不触发 :class:`TurnGuard`
  的 ``verify`` —— 而会话锁只在写时间轴时才起作用，子代理不写，父子就互不抢占；子进程还
  自带独立连接、独立 ``TurnGuard``，物理上也是两把锁。

但**文件改动的记账照常走** ``FileMutationRecorder``：子图的 ``RunContext`` 复用父传进来的
``(thread_id, user_seq)``，所以子代理写过的文件挂在父这一轮名下，父 ``/undo`` 顺带退回。
``work_state`` 走另一条通道（``config.configurable`` 里一个合成的隐形色 thread），既不建
``conversations`` 行、也不进 ``/thread`` 列表，对用户完全不可见、不可能撞号。

作为进程入口运行：``python -m coding_agent.integrations.subagent``，参数见 :func:`main`。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from contextlib import suppress
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ..config import Settings, load_settings
from .middleware import AgentRuntimeMiddleware, RunContext


class NoopPersistence:
    """子代理用的假存库：原样返回，什么都不写。

    只需满足 :class:`AgentRuntimeMiddleware` 用到的两个方法签名。因为不写时间轴，它永远
    不会触发 ``turn_guard.verify``，也就不参与会话锁；子的中间消息因此不进主时间轴。
    """

    def persist_messages(
        self, *, thread_id: str, user_seq: int, messages: list[Any]
    ) -> list[Any]:
        del thread_id, user_seq
        return list(messages)

    def persist_command(self, *, thread_id: str, user_seq: int, command: Any) -> Any:
        del thread_id, user_seq
        return command


class SubAgentMiddleware(AgentRuntimeMiddleware):
    """子代理用：不投影父对话、不落消息，只保留额度检查、工具执行与文件记账。"""

    def before_model(self, state, runtime):  # type: ignore[override]
        """返回 None = 不动 messages 通道；子代理从任务那一句开始，看不到父的历史。

        若改成返回 ``{"messages": ...}``（哪怕重排），就把父上下文拽进子代理。落库同样
        由 ``NoopPersistence`` 挡掉。
        """

        del state, runtime


SUBAGENT_SYSTEM_PROMPT = (
    "你是被主代理派出来的子代理，独立执行一份被明确委托的任务。你看不到用户，也无法追问；"
    "任务不清楚时按最合理的理解去做，并把假设写进报告。每条结论都要给出出处（文件路径+行号，"
    "或命令与其输出）。不要递归：你没有再派子代理的能力。"
    "最后必须用三段式收尾：先『结论』（能直接引用的答案），再『证据』（逐条带出处），"
    "最后『未确认项』（没查到或存疑的东西）。结论放最前——报告过长会被截断，别把答案埋在最后。"
)


def build_subagent_graph(settings: Settings):
    """装配一棵子代理图。工具面与父代理一致（bash、工作状态、搜索都在），仅少派子代理。

    与父侧共用同一份文件后端根目录、同一套服务，不搞两套实现。
    """

    # 延迟导入：父侧 langchain_agent 会 spawn 本模块，避免模块级循环依赖。
    from ..context.engine import ContextEngine
    from ..context.summarizer import LangChainSummarizer
    from ..persistence.database import Database
    from ..services.bash_execution import BashExecutionService
    from ..services.call_limits import CallLimitService
    from ..services.context_projection import ContextProjectionService
    from ..services.tool_execution import ToolExecutionService
    from ..workspace.artifacts import ArtifactStore
    from ..workspace.file_undo import FileMutationRecorder
    from .langchain_agent import (
        build_model,
        build_summary_model,
        shared_agent_tools,
        shared_filesystem_middleware,
    )
    from .search import make_search_client

    database = Database(settings.database_url)
    database.create_schema()
    model = build_model(settings)
    summary_model = build_summary_model(settings)
    context_engine = ContextEngine(database, settings.context, LangChainSummarizer(summary_model))
    recorder = FileMutationRecorder(database, settings.workspace_root)
    bash_executor = (
        BashExecutionService(
            executable=settings.agent.bash_executable,
            workspace_root=settings.workspace_root,
            timeout_seconds=settings.agent.bash_timeout_seconds,
            max_output_bytes=settings.agent.bash_max_output_bytes,
        )
        if settings.agent.bash_enabled
        else None
    )
    tools, prompt_suffix = shared_agent_tools(
        settings=settings,
        context_engine=context_engine,
        bash_executor=bash_executor,
        search_client=make_search_client(settings.agent),
    )
    runtime = SubAgentMiddleware(
        persistence=NoopPersistence(),  # type: ignore[arg-type]
        context_projection=ContextProjectionService(context_engine),
        call_limits=CallLimitService(
            tool_limit=settings.agent.tool_call_limit,
            model_limit=settings.agent.model_call_limit,
        ),
        tool_execution=ToolExecutionService(
            max_retries=settings.agent.tool_retry_max,
            retryable_tools=["ls", "read_file", "glob", "grep", "write_file", "search_tool"],
        ),
        artifacts=ArtifactStore(settings.artifact_dir),
        file_mutations=recorder,
        tool_result_inline_tokens=settings.context.tool_result_inline_tokens,
    )
    # 复用整棵父图的模型/工具/文件后端，只有中间件是薄片、提示词是子代理版。
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    # 必须显式标成 AgentMiddleware[Any, Any, Any]：AgentMiddleware 的 ContextT 是不变
    # （invariant）参数，而 FilesystemMiddleware 的是 None、薄片的是 RunContext，裸列表推
    # 断合不到 create_agent 的重载签名上（Pylance 会报"No overloads match"）。父侧同理。
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        runtime,
        shared_filesystem_middleware(settings),
    ]
    graph = create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        context_schema=RunContext,
        system_prompt=SUBAGENT_SYSTEM_PROMPT + prompt_suffix,
    )
    return graph


def _final_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.text if isinstance(message.text, str) else str(message.text)
    return ""


class SubAgentJobs:
    """父进程里的派遣登记处：任务号 -> 子进程 + 报告落盘位置。

    只存内存，不建表 —— 父进程一退，这些任务号就再没人认，落库反而是骗人的持久化。
    子进程的 stdout 直接重定向进 ``<报告目录>/<号>.md``（stderr 一并并进去），所以父侧
    不必守着管道，回来读文件就是报告本身。
    """

    def __init__(self, report_root: Path) -> None:
        self.root = Path(report_root) / "subagents"
        self._lock = Lock()
        self._jobs: dict[int, dict[str, Any]] = {}
        self._next = 0

    def spawn(self, task: str, build_command: Any, *, cwd: str) -> int:
        """落任务文件 → 起子进程 → 登记。``build_command(task_path)`` 拼出命令行。"""

        with self._lock:
            self._next += 1
            task_id = self._next
        self.root.mkdir(parents=True, exist_ok=True)
        task_path = self.root / f"{task_id}.task"
        report_path = self.root / f"{task_id}.md"
        task_path.write_text(task, encoding="utf-8")
        handle = report_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                build_command(task_path),
                cwd=cwd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except BaseException:
            handle.close()
            raise
        with self._lock:
            self._jobs[task_id] = {
                "process": process,
                "handle": handle,
                "report": report_path,
                "collected": False,
            }
        return task_id

    def finish_turn(self) -> str | None:
        """把本轮派出、还没交回的活儿**等齐**，攒成一段报告；没有待收就立刻返回 None。

        在这里阻塞是设计而不是妥协：父模型说完话不等于这一轮结束，本轮收齐之前用户发不进
        新消息，所以一次 Ctrl-C 就能把父子一起停掉（Ctrl-C 落进这个 sleep 小块 → 往上抛 →
        上层 except 里 terminate_all）。分批 ``poll`` 而非一口闷 ``wait``，就是为了让 Ctrl-C
        有落点。
        """

        with self._lock:
            pending = [(tid, job) for tid, job in self._jobs.items() if not job["collected"]]
        for _, job in pending:
            while job["process"].poll() is None:
                sleep(0.2)
        blocks: list[str] = []
        with self._lock:
            for task_id, job in pending:
                job["handle"].close()  # 重复 close 是空操作，不必再记一个标志
                body = job["report"].read_text(encoding="utf-8").strip()
                code = job["process"].returncode
                job["collected"] = True
                blocks.append(f"#{task_id}（退出码 {code}）\n{body or '（无报告产出）'}")
        return "\n\n".join(blocks) if blocks else None

    def terminate_all(self) -> None:
        """父进程收摊：还在跑的一起终止，并等到真咽气——别留个半死的娃在后台写文件。"""

        with self._lock:
            for job in self._jobs.values():
                if job["process"].poll() is None:
                    job["process"].terminate()
                    with suppress(subprocess.TimeoutExpired):
                        job["process"].wait(timeout=5)
                job["handle"].close()
                job["collected"] = True  # 本轮已作废，别拿它的残报告再喂模型


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coding-agent-subagent")
    parser.add_argument("--parent-thread", required=True)
    parser.add_argument("--parent-seq", required=True, type=int)
    parser.add_argument("--config", default=None)
    parser.add_argument("--task-file", required=True)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    with open(args.task_file, encoding="utf-8") as handle:
        task = handle.read()
    # 任务书是给子进程的一次性输入，读完就该没了：报告目录只留报告，不堆暂存物。
    Path(args.task_file).unlink(missing_ok=True)
    # 子代理自己的身份只用于 work_state：合成一个绝不会被用户选中的 id，且不建会话行。
    child_thread = f"subagent:{uuid.uuid4().hex}"
    graph = build_subagent_graph(settings)
    result = graph.invoke(
        {"messages": [HumanMessage(content=task)]},
        # 不传 recursion_limit —— 跟主代理一致：步数上限用 LangGraph 默认（10007），
        # 真正的闸是中间件里的 model/tool 额度。自己按额度倒推步数上限只会先撞崩。
        config={"configurable": {"thread_id": child_thread, "user_seq": 1}},
        # 文件记账挂到父这一轮：/undo 才罩得住子的改动。
        context=RunContext(args.parent_thread, args.parent_seq),
    )
    report = _final_text(result)
    sys.stdout.write(report or "（子代理未产出文本结论）")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
