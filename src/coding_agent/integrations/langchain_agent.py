from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
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
    """A dedicated, cheap, deterministic model for context compression.

    Summarization must not burn latency or tokens on extended thinking, so by default
    ``thinking={"type": "disabled"}`` is requested for the summary model (use
    ``summary_llm.thinking="auto"`` in config to keep the provider default instead).
    Falls back to the main agent LLM when ``summary_llm.model`` is empty.
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
        """对当前工作状态做一次字典操作；状态是 {键: markdown 字符串} 的扁平结构。

        一次调用只付一个键的成本，不要为了改一行而重打整份状态。键名以 ``!``
        开头表示钉住（常驻注入）。值用 markdown，不要传嵌套结构。常见用法是
        ``append`` 往一个键追加一条 bullet。

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
) -> Generator[Any, None, None]:
    filesystem = FilesystemMiddleware(
        backend=FilesystemBackend(
            root_dir=settings.workspace_root,
            virtual_mode=True,
            max_file_size_mb=settings.agent.filesystem_max_file_size_mb,
        )
    )
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
    tools = [make_work_state_tool(context_engine)]
    bash_prompt = ""
    if settings.agent.bash_enabled:
        if bash_executor is None:
            raise RuntimeError("bash is enabled but no BashExecutionService was provided")
        tools.append(make_bash_tool(bash_executor))
        bash_prompt = (
            "Bash 从工作区根目录运行；相对路径基于该目录，绝对路径表示宿主机真实路径，不是"
            "文件工具的虚拟路径。Bash 非交互、不会自动重试，且它造成的文件变化无法通过 "
            "/undo 恢复。不要把存在读写依赖的 Bash 和文件操作放进同一批工具调用。"
        )
    if settings.agent.search_enabled:
        if search_client is None:
            raise RuntimeError("search is enabled but no SearchClient was provided")
        tools.append(make_search_tool(search_client))

    # 不配置 checkpointer：应用库的 messages 表才是唯一真相源。AgentRuntimeMiddleware.before_model
    # 在每次模型调用前用业务库投影整体覆盖 messages 通道，所以 checkpoint 里的快照既不进 prompt、
    # 也没人读，却会按「每个图步骤一份全量 messages 快照」无界增长（实测把单线程推到 4.27GB，
    # 而读一次要 10s）。跨步骤状态由 Pregel 在进程内持有，一轮之内的执行不依赖持久化。
    # 需要 LangGraph 原生 interrupt() 式人工审批时再开回来，届时应选 Shallow/SQLite 这类
    # 「每线程只留最新」的 saver，别再按图步骤写全量快照。
    middleware: list[AgentMiddleware[Any, Any, Any]] = [runtime_middleware, filesystem]
    yield create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        context_schema=RunContext,
        system_prompt=(
            "你是编码代理。合理使用文件与工作状态工具，保持回答简洁。"
            "所有文件工具路径使用以 / 开头、相对于工作区根目录的虚拟路径。"
            f"{bash_prompt}"
        ),
    )
