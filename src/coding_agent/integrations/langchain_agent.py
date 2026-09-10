from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from deepagents import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.config import get_config

from ..config import Settings
from ..context.engine import ContextEngine
from ..context.work_state import WorkStateError
from ..persistence.database import Database
from ..services.bash_execution import BashExecutionService
from ..services.call_limits import CallLimitService
from ..services.context_projection import ContextProjectionService
from ..services.message_persistence import MessagePersistenceService
from ..services.tool_execution import ToolExecutionService
from ..workspace.artifacts import ArtifactStore
from ..workspace.file_undo import FileMutationRecorder
from .bash_tool import make_bash_tool
from .middleware import AgentRuntimeMiddleware, RunContext
from .search import SearchClient
from .subagent import SubAgentJobs


def build_model(settings: Settings) -> ChatAnthropic:
    if not settings.llm.api_key:
        raise RuntimeError("LLM_API_KEY is empty; set it in the environment or local config.json")
    return ChatAnthropic(
        model=settings.llm.model,  # type: ignore
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        max_tokens=settings.llm.max_output_tokens,  # type: ignore
    )


def build_summary_model(settings: Settings) -> ChatAnthropic:
    """给上下文压缩单开一个便宜、稳定的模型。

    摘要不该把延迟和 token 花在 extended thinking 上，所以默认给摘要模型请求
    ``thinking={"type": "disabled"}``（想保留 provider 的默认行为，就在配置里写
    ``summary_llm.thinking="auto"``）。``summary_llm.model`` 留空时退回主 Agent 模型。
    """
    summary = settings.summary_llm
    llm = settings.llm
    api_key = summary.api_key or llm.api_key
    if not api_key:
        raise RuntimeError(
            "summary LLM has no api_key; set summary_llm.api_key or LLM_API_KEY"
        )
    kwargs: dict[str, Any] = {
        "model": summary.model or llm.model,
        "api_key": api_key,
        "base_url": summary.base_url or llm.base_url,
        "max_tokens": summary.max_output_tokens or llm.max_output_tokens,  # type: ignore
    }
    if summary.thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}
    return ChatAnthropic(**kwargs)  # type: ignore[call-arg]


def make_work_state_tool(context_engine: ContextEngine):
    @tool("work_state", parse_docstring=True)
    def work_state(op: str, key: str = "", value: str = "") -> str:
        """我的备忘录：一个 {键: markdown 字符串} 的扁平字典，用来跨轮记住东西。

        适合放用户偏好、已确认的事实、踩过的坑、临时约束这类跨轮还要用的信息。
        内容**不会每轮自动出现**，需要回顾时用 ``list`` / ``get`` 主动翻；一旦历史被
        剪裁或压缩，框架会把最新一版作为便签贴到上下文最前面提醒，所以正常情况下写完
        不用担心它丢。一次调用只付一个键的成本，不要为了改一行而重打整份状态；值用
        markdown，不要传嵌套结构。常见用法是 ``append`` 追加一条 bullet。保持克制——
        它是备忘录，不是所有东西的家。

        Args:
            op: 操作名，取 list / get / set / append / delete / clear。
            key: 目标键名；list 与 clear 之外都必填。
            value: markdown 正文；仅 set / append 需要。
        """

        configurable = get_config().get("configurable", {})
        thread_id = str(configurable["thread_id"])
        user_seq = int(configurable["user_seq"])
        # RMW 必须整体跑在 engine 的 thread 锁里：框架并行执行同一批工具，
        # 无锁时实测 8 线程 × 10 轮只活下 19/80 个键，且零异常。
        try:
            return context_engine.mutate_work_state(
                thread_id, user_seq, op, key or None, value or None
            )
        except WorkStateError as error:
            return f"Error: {error}"

    return work_state


def make_search_tool(client: SearchClient):
    @tool("search_tool", parse_docstring=True)
    def search_tool(query: str, max_results: int = 3) -> str:
        """联网搜索实时信息，返回远端回答与来源链接列表。

        用于查询实时信息、最新版本号、或本地记忆未覆盖的知识。注意该接口只返回标题和
        URL，不返回网页正文；answer 由远端搜索增强模型给出，重要事实需按 sources 自行核验。

        Args:
            query: 搜索关键词，应简洁明确。
            max_results: 期望返回的来源条数，超过服务端配置上限会被收敛。
        """

        return client.search(query, max_results=max_results).render()

    return search_tool


def shared_filesystem_middleware(settings: Settings) -> FilesystemMiddleware:
    return FilesystemMiddleware(
        backend=FilesystemBackend(
            root_dir=settings.workspace_root,
            # 关掉虚拟路径：开启时以 "/" 开头的路径会被当成"虚拟绝对路径"（剥掉前导斜杠
            # 再拼到 root_dir 下），于是传宿主机真实绝对路径会被静默写到
            # root_dir/<原路径> 这种嵌套目录里——实机踩过一次（写到了 ~/Users/apple/...）。
            # 关闭后绝对路径按真实路径解释、相对路径仍基于 root_dir，与 bash 的路径语义一致。
            # 代价是失去 root_dir 沙箱；本地个人使用、且 bash 本就全权在，沙箱没有意义。
            virtual_mode=False,
            max_file_size_mb=settings.agent.filesystem_max_file_size_mb,
        )
    )


def shared_agent_tools(
    *,
    settings: Settings,
    context_engine: ContextEngine,
    bash_executor: BashExecutionService | None,
    search_client: SearchClient | None,
    extra_tools: list[Any] | None = None,
) -> tuple[list[Any], str]:
    """组装父/子共用的工具面：work_state +（可选）bash +（可选）search + 额外工具。

    返回工具表和一段拼进 system_prompt 的 bash 说明。抽出来是为了父子走同一份实现，
    不搞两套；子代理只是不传 ``extra_tools``（拿不到 ``run_subagent``，防止再套娃）。
    """

    tools: list[Any] = [make_work_state_tool(context_engine)]
    if extra_tools:
        tools.extend(extra_tools)
    bash_prompt = ""
    if settings.agent.bash_enabled:
        if bash_executor is None:
            raise RuntimeError("bash is enabled but no BashExecutionService was provided")
        tools.append(make_bash_tool(bash_executor))
        bash_prompt = (
            "\n\nBash 从工作区根目录运行；相对路径基于该目录，绝对路径表示宿主机真实路径，"
            "文件工具同一套规则。Bash 非交互、不会自动重试，且它造成的文件变化无法通过 "
            "/undo 恢复。不要把存在读写依赖的 Bash 和文件操作放进同一批工具调用。"
        )
    if settings.agent.search_enabled:
        if search_client is None:
            raise RuntimeError("search is enabled but no SearchClient was provided")
        tools.append(make_search_tool(search_client))
    return tools, bash_prompt


def _make_run_subagent_tool(settings: Settings, jobs: SubAgentJobs) -> Any:
    """父侧的 ``run_subagent``：派活不等，但**本轮结束前会等它交卷**。

    派遣走独立子进程：父子之间不共享进程内的任何东西（子进程自建房的一整套服务），只通过
    命令行传 ``(parent_thread, parent_seq)``、任务文件与配置绝对路径。子进程把报告写到
    stdout（已重定向成文件），父进程在本轮收尾时按任务号取回。
    """

    @tool("run_subagent", parse_docstring=True)
    def run_subagent(task: str) -> str:
        """把一份可独立完成的活儿交给子代理进程去做，立刻返回任务号，不在这儿等。

        适合"要读很多文件/跑很多命令、但不必占用主对话上下文"的调研或批量改动。派出去就
        继续干你自己的事；等你说完了，这一轮会等它交卷并把报告给你，你据此继续干活或再派活
        ——所以不必自己写"稍后再查"之类的话。子代理看不到用户也看不到主对话，``task`` 必须
        自包含：目标、范围、要交什么、何时该停。它的文件改动会记在当前这一轮名下，主代理
        ``/undo`` 可一并撤销。

        Args:
            task: 交给子代理的完整任务说明书。
        """

        configurable = get_config().get("configurable", {})
        parent_thread = str(configurable.get("thread_id", ""))
        parent_seq = int(configurable.get("user_seq", 0))

        def build(task_path: Path) -> list[str]:
            return [
                sys.executable,
                "-m",
                "coding_agent.integrations.subagent",
                "--parent-thread",
                parent_thread,
                "--parent-seq",
                str(parent_seq),
                # 无条件透传父实际用的那份配置绝对路径：子进程 cwd 是 workspace，
                # 若只靠默认 "config.json" 会在别处读成空配置（真机踩过一次）。
                "--config",
                str(settings.config_path),
                "--task-file",
                str(task_path),
            ]

        task_id = jobs.spawn(task, build, cwd=str(settings.workspace_root))
        return (
            f"已派遣子代理 #{task_id}，不必等它，继续手上的事。"
            "等你这一轮说完，系统会在收工前把它的报告取回来给你。"
        )

    return run_subagent


@contextmanager
def create_langchain_agent(
    *,
    settings: Settings,
    database: Database,
    context_engine: ContextEngine,
    recorder: FileMutationRecorder,
    model: ChatAnthropic,
    bash_executor: BashExecutionService | None,
    search_client: SearchClient | None,
    jobs: SubAgentJobs,
) -> Generator[Any, None, None]:
    runtime_middleware = AgentRuntimeMiddleware(
        persistence=MessagePersistenceService(database, context_engine),
        context_projection=ContextProjectionService(context_engine),
        call_limits=CallLimitService(
            tool_limit=settings.agent.tool_call_limit,
            model_limit=settings.agent.model_call_limit,
        ),
        tool_execution=ToolExecutionService(
            max_retries=settings.agent.tool_retry_max,
            # search_tool 是只读幂等的远端查询，和网络抖动都值得重试；
            # edit_file / delete / bash 有副作用，仍然只执行一次。
            retryable_tools=["ls", "read_file", "glob", "grep", "write_file", "search_tool"],
        ),
        artifacts=ArtifactStore(settings.artifact_dir),
        file_mutations=recorder,
        tool_result_inline_tokens=settings.context.tool_result_inline_tokens,
    )
    extra_tools = (
        [_make_run_subagent_tool(settings, jobs)] if settings.agent.subagent_enabled else []
    )
    tools, bash_prompt = shared_agent_tools(
        settings=settings,
        context_engine=context_engine,
        bash_executor=bash_executor,
        search_client=search_client,
        extra_tools=extra_tools,
    )

    # 不配置 checkpointer：应用库的 messages 表才是唯一真相源。AgentRuntimeMiddleware.before_model
    # 在每次模型调用前用业务库投影整体覆盖 messages 通道，所以 checkpoint 里的快照既不进 prompt、
    # 也没人读，却会按「每个图步骤一份全量 messages 快照」无界增长（实测把单线程推到 4.27GB，
    # 而读一次要 10s）。跨步骤状态由 Pregel 在进程内持有，一轮之内的执行不依赖持久化。
    # 需要 LangGraph 原生 interrupt() 式人工审批时再开回来，届时应选 Shallow/SQLite 这类
    # 「每线程只留最新」的 saver，别再按图步骤写全量快照。
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        runtime_middleware,
        shared_filesystem_middleware(settings),
    ]
    yield create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        context_schema=RunContext,
        system_prompt=(
            "你是编码代理。合理使用文件与工作状态工具，不要过度设计。如无必要，勿增实体。保持回答简洁，不要把简单的事情复杂化。"
            """
              请用通俗的语言表达，不要过度使用术语，
              注意模仿用户的表达风格。
            """
            "文件工具路径与 bash 同规则：以 / 开头表示宿主机真实绝对路径，相对路径基于工作区根目录。"
            f"{bash_prompt}"
        ),
    )
