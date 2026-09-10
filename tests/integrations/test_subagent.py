from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from sqlalchemy import func, select

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.middleware import RunContext
from coding_agent.integrations.subagent import NoopPersistence, SubAgentJobs, SubAgentMiddleware
from coding_agent.persistence.models import Conversation, FileMutation, Message
from coding_agent.services.call_limits import CallLimitService
from coding_agent.services.context_projection import ContextProjectionService
from coding_agent.services.tool_execution import ToolExecutionService
from coding_agent.workspace.artifacts import ArtifactStore
from coding_agent.workspace.file_undo import FileMutationRecorder


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


def make_subagent_middleware(
    database,
    engine,
    root,
    *,
    model_limit: int = 30,
    tool_limit: int = 60,
) -> SubAgentMiddleware:
    """和父用同一套服务，只把持久化换成 Noop、中间件换成薄片。"""

    settings = ContextSettings()
    return SubAgentMiddleware(
        persistence=NoopPersistence(),  # type: ignore[arg-type]
        context_projection=ContextProjectionService(engine),
        call_limits=CallLimitService(model_limit=model_limit, tool_limit=tool_limit),
        tool_execution=ToolExecutionService(max_retries=0, retryable_tools=[]),
        artifacts=ArtifactStore(root / "artifacts"),
        file_mutations=FileMutationRecorder(database, root),
        tool_result_inline_tokens=settings.tool_result_inline_tokens,
    )


def _guard_must_not_fire(database):
    """把会话锁的三个动作换成抛错：子代理路径只要调了它们，测试就会炸。"""

    guard = database.turn_guard
    guard.acquire = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("子代理不该碰 enter_thread/acquire")
    )
    guard.verify = lambda *a, **k: (_ for _ in ()).throw(AssertionError("子代理不该碰 verify"))
    guard.bind = lambda *a, **k: (_ for _ in ()).throw(AssertionError("子代理不该碰 bind"))


def test_subagent_writes_are_undoable_under_parent_without_polluting_timeline(database, tmp_path):
    """子代理改的文件记在父那一轮名下（/undo 罩得住），但一条消息都不进、不建会话行、不碰锁。"""

    parent_thread, parent_seq = "parent-turn", 7
    child_thread = "subagent:deadbeef"
    target = tmp_path / "notes.md"

    @tool("write_file")
    def write_file(file_path: str, content: str) -> str:
        """Write content to a workspace file."""

        del file_path
        target.write_text(content, encoding="utf-8")
        return f"wrote {content!r}"

    _guard_must_not_fire(database)
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="sub-a1",
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/notes.md", "content": "hi"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(id="sub-a2", content="结论：已写入。证据：notes.md=hi。未确认项：无。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[write_file],
        middleware=[make_subagent_middleware(database, engine, tmp_path)],
        context_schema=RunContext,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(id="sub-task", content="把 hi 写进 notes.md")]},
        config={"configurable": {"thread_id": child_thread, "user_seq": 1}},
        # 关键：context 用父的坐标 → 文件记账挂父这一轮。
        context=RunContext(parent_thread, parent_seq),
    )

    assert target.read_text(encoding="utf-8") == "hi"
    assert "结论" in result["messages"][-1].content
    with database.session() as session:
        mutations = list(session.scalars(select(FileMutation)))
        assert len(mutations) == 1
        # 记账落在父的 (thread_id, user_seq)，于是父 /undo 能一并退回。
        assert mutations[0].thread_id == parent_thread
        assert mutations[0].user_seq == parent_seq
        # 主时间轴零污染：一条消息都没有，也没给隐形色子会话建 conversations 行。
        assert session.scalar(select(func.count()).select_from(Message)) == 0
        visible = session.scalars(select(Conversation.thread_id)).all()
        assert child_thread not in list(visible)


def test_subagent_model_never_sees_parent_history(database, tmp_path):
    """before_model 返回 None：子代理第一步只看得到自己那句任务，父历史不倒灌。"""

    seen: list[int] = []

    class RecordingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            del tools, kwargs
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(len(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    model = RecordingModel(responses=[AIMessage(id="sub-only", content="只看到任务一句")])
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[make_subagent_middleware(database, engine, tmp_path)],
        context_schema=RunContext,
    )
    # 先在库里造一条“父历史”，投影若误开就会把它灌进来。
    agent.invoke(
        {"messages": [HumanMessage(id="t", content="任务")]},
        config={"configurable": {"thread_id": "subagent:x", "user_seq": 1}},
        context=RunContext("parent-turn", 1),
    )
    assert seen == [1]  # 模型第一次见到 1 条消息，正是任务那句


def test_child_file_change_is_reverted_by_parent_undo(database, tmp_path):
    """父 /undo 顺带把子代理写的文件退回：证明记账确实挂在父这一轮、且撤销罩得住。"""

    parent_thread, parent_seq = "parent-undo", 3
    target = tmp_path / "made_by_child.txt"

    @tool("write_file")
    def write_file(file_path: str, content: str) -> str:
        """Write content to a workspace file."""

        del file_path
        target.write_text(content, encoding="utf-8")
        return "written"

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    middleware = SubAgentMiddleware(
        persistence=NoopPersistence(),  # type: ignore[arg-type]
        context_projection=ContextProjectionService(engine),
        call_limits=CallLimitService(model_limit=30, tool_limit=60),
        tool_execution=ToolExecutionService(max_retries=0, retryable_tools=[]),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        file_mutations=recorder,
        tool_result_inline_tokens=ContextSettings().tool_result_inline_tokens,
    )
    model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="a",
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/made_by_child.txt", "content": "x"},
                        "id": "w",
                    }
                ],
            ),
            AIMessage(id="b", content="done"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[write_file],
        middleware=[middleware],
        context_schema=RunContext,
    )
    agent.invoke(
        {"messages": [HumanMessage(id="t", content="写文件")]},
        config={"configurable": {"thread_id": "subagent:u", "user_seq": 1}},
        context=RunContext(parent_thread, parent_seq),
    )
    assert target.exists()

    reverted = recorder.rollback(parent_thread, parent_seq)

    assert reverted == 1
    assert not target.exists()  # 子代理新建的文件被撤掉


def _settings_with_subagent():
    from coding_agent.config import AgentSettings, Settings

    return Settings(
        workspace_root="/tmp",
        artifact_dir="/tmp/art",
        agent=AgentSettings(bash_enabled=False, search_enabled=False, subagent_enabled=True),
    )


def test_run_subagent_tool_dispatches_without_waiting(monkeypatch, tmp_path):
    """派活这段：命令里带父坐标与配置绝对路径，任务落成文件，且**不等**子进程。"""

    from pathlib import Path

    from coding_agent.integrations import langchain_agent

    monkeypatch.setattr(
        langchain_agent, "get_config", lambda: {"configurable": {"thread_id": "pth", "user_seq": 9}}
    )

    class RecordingJobs:
        def __init__(self) -> None:
            self.seen = []

        def spawn(self, task, build_command, *, cwd):
            self.seen.append((task, build_command, cwd))
            return len(self.seen)

    jobs = RecordingJobs()
    tool = langchain_agent._make_run_subagent_tool(_settings_with_subagent(), jobs)

    out = tool.invoke({"task": "去做这件事"})

    assert "#1" in out  # 立刻拿到任务号
    task, build, cwd = jobs.seen[0]
    assert task == "去做这件事" and cwd == "/tmp"
    command = build(Path("/tmp/job.task"))
    assert command[:3] == [
        langchain_agent.sys.executable,
        "-m",
        "coding_agent.integrations.subagent",
    ]
    assert command[command.index("--parent-thread") + 1] == "pth"
    assert command[command.index("--parent-seq") + 1] == "9"
    assert command[command.index("--task-file") + 1] == "/tmp/job.task"
    # 真机踩过：子进程 cwd 不在仓库里时，相对 config 会被静默读成空 → 必须传绝对路径
    assert command[command.index("--config") + 1] == str(_settings_with_subagent().config_path)


def _jobs(tmp_path):
    from coding_agent.integrations.subagent import SubAgentJobs

    return SubAgentJobs(tmp_path / "art")


def _echo_task(path_arg: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text(), end='')",
        path_arg,
    ]


def test_jobs_collect_report_once_and_increment_ids(tmp_path):
    """报告落进文件、本轮收齐；收过的不再重复上报。"""

    jobs = _jobs(tmp_path)
    first = jobs.spawn("任务甲", lambda p: _echo_task(str(p)), cwd=str(tmp_path))
    second = jobs.spawn("任务乙", lambda p: _echo_task(str(p)), cwd=str(tmp_path))
    assert (first, second) == (1, 2)

    text = jobs.finish_turn()  # 阻塞到两个都跑完
    assert text is not None, "子进程没在期限内跑完"
    assert "任务甲" in text and "任务乙" in text
    assert "退出码 0" in text
    assert jobs.finish_turn() is None  # 收过就不再重复上报


def test_jobs_surface_failure_code(tmp_path):
    """子进程崩了也要出声：非零退出码与 stderr 尾巴一起带回来（并进同一个文件）。"""

    jobs = _jobs(tmp_path)
    jobs.spawn(
        "会崩的任务",
        lambda p: [
            sys.executable,
            "-c",
            "import sys;print('炸在这', file=sys.stderr); sys.exit(3)",
        ],
        cwd=str(tmp_path),
    )
    text = jobs.finish_turn()
    assert text is not None
    assert "退出码 3" in text and "炸在这" in text


def test_terminate_all_stops_children_without_reporting(tmp_path):
    """父进程收摊：还在跑的娃一起终止，且不再拿它的残报告当"本轮成果"（那是我们自己杀的）。"""

    jobs = _jobs(tmp_path)
    pid_file = tmp_path / "alive.pid"
    jobs.spawn(
        "长任务",
        lambda p: [
            sys.executable,
            "-c",
            (
                f"import os,pathlib,time;pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
                "time.sleep(30)"
            ),
        ],
        cwd=str(tmp_path),
    )
    deadline = time.time() + 10
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "子进程没起来"

    jobs.terminate_all()

    child_pid = int(pid_file.read_text())
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("terminate_all 之后子进程还活着")
    assert jobs.finish_turn() is None


def test_looping_child_brakes_at_model_limit_not_recursion_crash(database, tmp_path):
    """回归：模型在打转时，子代理要在额度处优雅刹车（回一句上限提示），而不是崩掉。

    故意**不设 recursion_limit**：步数上限就交给 LangGraph 默认，跟主代理完全一样，闸只有
    中间件里的额度。曾经在子代理里按额度倒推过一个 ``3 × model_limit`` 的步数上限，结果正好
    差几步，每次到额度都先抛 GraphRecursionError —— 那个倒推本身就是多余的。
    """

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class Loop(FakeListChatModel):
        _seen: int = PrivateAttr(default=0)

        def bind_tools(self, tools, **kw):
            del tools, kw
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self._seen += 1
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/x", "content": "y"},
                        "id": f"c{self._seen}",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

    @tool("write_file")
    def write_file(file_path: str, content: str) -> str:
        """Write a file."""

        del file_path, content
        return "ok"

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    model_limit = 4
    agent = create_agent(
        model=Loop(responses=["unused"]),
        tools=[write_file],
        middleware=[make_subagent_middleware(database, engine, tmp_path, model_limit=model_limit)],
        context_schema=RunContext,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="go")]},
        config={"configurable": {"thread_id": "subagent:loop", "user_seq": 1}},
        context=RunContext("parent", 1),
    )
    assert "模型调用上限" in result["messages"][-1].content


def test_child_tool_surface_has_no_run_subagent(tmp_path):
    """防套娃：子代理工具面里绝不含 run_subagent。"""

    from coding_agent.config import AgentSettings, Settings
    from coding_agent.integrations.langchain_agent import shared_agent_tools

    # 显式指向一个不存在的规则文件：装配会读 AGENTS.md，不该去读开发机自己那份。
    settings = Settings(
        agent=AgentSettings(bash_enabled=False, search_enabled=False),
        agents_md_path=tmp_path / "AGENTS.md",
    )
    tools, _ = shared_agent_tools(
        settings=settings,
        context_engine=None,  # 只建 work_state 工具，用不到 engine 实例
        bash_executor=None,
        search_client=None,
    )
    assert "work_state" in {t.name for t in tools}
    assert "run_subagent" not in {t.name for t in tools}


def test_main_entrypoint_end_to_end_without_api(monkeypatch, tmp_path, capsys):
    """子进程真正会走的整条装配线（load_settings→build_subagent_graph→真实文件工具）跑通，
    只把最外层网络模型换成假模型——证明入口没写错，且不花一分钱 API。"""

    import json

    from coding_agent.integrations import langchain_agent, subagent

    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {"agent": {"bash_enabled": False, "search_enabled": False, "subagent_enabled": False}}
        ),
        encoding="utf-8",
    )
    db_file = tmp_path / "e2e.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{db_file}")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    child_model = ToolAwareFakeModel(
        responses=[
            AIMessage(
                id="c1",
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/from_child.txt", "content": "child said hi"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(
                id="c2",
                content="结论：已写 from_child.txt。证据：内容=child said hi。未确认项：无。",
            ),
        ]
    )
    monkeypatch.setattr(langchain_agent, "build_model", lambda settings: child_model)
    monkeypatch.setattr(langchain_agent, "build_summary_model", lambda settings: child_model)

    task_file = tmp_path / "task.txt"
    task_file.write_text("把 hi 写进 from_child.txt", encoding="utf-8")

    code = subagent.main(
        [
            "--parent-thread",
            "parent-e2e",
            "--parent-seq",
            "5",
            "--task-file",
            str(task_file),
            "--config",
            str(config),
        ]
    )

    assert code == 0
    assert not task_file.exists()  # 任务书读完即删，产物目录只留报告
    out = capsys.readouterr().out
    assert "结论" in out
    assert (tmp_path / "from_child.txt").read_text(encoding="utf-8") == "child said hi"

    from coding_agent.persistence.database import Database
    from coding_agent.persistence.models import Message as MessageModel

    probe = Database(f"sqlite+pysqlite:///{db_file}")
    with probe.session() as session:
        mutations = list(session.scalars(select(FileMutation)))
        assert [(m.thread_id, m.user_seq) for m in mutations] == [("parent-e2e", 5)]
        assert session.scalar(select(func.count()).select_from(MessageModel)) == 0


def test_report_file_lands_in_artifacts(tmp_path: Path) -> None:
    """报告只落一个文件：stdout 和 stderr 都在里面，父侧只拿正文。"""

    jobs = SubAgentJobs(tmp_path / "art")
    task_id = jobs.spawn(
        "写报告",
        lambda p: [
            sys.executable,
            "-c",
            "import sys; print('正文'); print('警告', file=sys.stderr)",
        ],
        cwd=str(tmp_path),
    )
    text = jobs.finish_turn()
    assert text is not None and "正文" in text
    report = jobs.root / f"{task_id}.md"
    assert "正文" in report.read_text(encoding="utf-8")
    assert "警告" in report.read_text(encoding="utf-8")  # stderr 一并进了同一个文件
