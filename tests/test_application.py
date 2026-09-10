from __future__ import annotations

import signal
import sys
import threading
import time

import pytest
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from sqlalchemy import select

from coding_agent.application import AgentApplication, TurnExecutionError
from coding_agent.config import ContextSettings, Settings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.persistence.message_codec import decode_message
from coding_agent.persistence.models import Message
from coding_agent.services.context_projection import ContextProjectionService
from coding_agent.workspace.file_undo import FileMutationRecorder


class FailingAgent:
    def invoke(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("model unavailable")

    def update_state(self, *args, **kwargs):
        del args, kwargs


class PartiallyCompletedAgent:
    def __init__(self, app: AgentApplication | None = None) -> None:
        self.app = app

    def invoke(self, *args, **kwargs):
        del args, kwargs
        assert self.app is not None
        self.app.message_persistence.persist_messages(
            thread_id="t1",
            user_seq=1,
            messages=[
                AIMessage(
                    id="assistant-tools",
                    content="",
                    tool_calls=[
                        {"name": "bash", "args": {"command": "first"}, "id": "call-1"},
                        {"name": "bash", "args": {"command": "second"}, "id": "call-2"},
                    ],
                ),
                ToolMessage(
                    id="tool-call-1",
                    content="first completed",
                    name="bash",
                    tool_call_id="call-1",
                ),
            ],
        )
        raise KeyboardInterrupt

    def update_state(self, *args, **kwargs):
        del args, kwargs


def test_failed_turn_reports_persisted_user_seq(database, tmp_path):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    app = AgentApplication(
        Settings(workspace_root=tmp_path), database, FailingAgent(), engine, recorder
    )

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "hello")

    assert raised.value.user_seq == 1
    assert raised.value.thread_id == "t1"
    assert isinstance(raised.value.cause, RuntimeError)
    preview = app.rollback_preview("t1", 1)
    assert preview.messages == 1


def test_interrupted_turn_closes_partial_tool_batch(database, tmp_path):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    agent = PartiallyCompletedAgent()
    app = AgentApplication(Settings(workspace_root=tmp_path), database, agent, engine, recorder)
    agent.app = app

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "run both")

    assert raised.value.interrupted is True
    assert raised.value.closed_tool_results == 1
    assert raised.value.finalization_errors == ()
    assert app.rollback_preview("t1", 1).messages == 4


def test_finalization_error_does_not_hide_original_turn_error(database, tmp_path, monkeypatch):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, tmp_path)
    app = AgentApplication(
        Settings(workspace_root=tmp_path), database, FailingAgent(), engine, recorder
    )

    def fail_to_close(**kwargs):
        del kwargs
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        app.message_persistence,
        "close_incomplete_tool_batch",
        fail_to_close,
    )

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "hello")

    assert str(raised.value.cause) == "model unavailable"
    assert raised.value.finalization_errors == (
        "补齐工具结果失败：RuntimeError: database unavailable",
    )


class ScriptedAgent:
    """假代理：按脚本回答，并把每次 invoke 的入参记下来供断言。"""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[object, object]] = []

    def invoke(self, state, *, config, context):
        self.calls.append((state, context))
        answer = self.answers.pop(0) if self.answers else "收尾"
        return {"messages": [AIMessage(id=f"answer-{len(self.calls)}", content=answer)]}

    def update_state(self, *args, **kwargs):
        del args, kwargs


class StubJobs:
    """假登记处：第一次收齐给一份报告，之后没有待收的活儿。"""

    def __init__(self, reports=("\n#1（退出码 0）\n子代理报告\n",)) -> None:
        self.reports = list(reports)
        self.terminated = 0

    def finish_turn(self):
        return self.reports.pop(0) if self.reports else None

    def terminate_all(self):
        self.terminated += 1


def _app(database, tmp_path, agent, jobs=None):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    return AgentApplication(
        Settings(workspace_root=tmp_path),
        database,
        agent,
        engine,
        FileMutationRecorder(database, tmp_path),
        subagent_jobs=jobs,
    )


def test_subagent_reports_are_collected_inside_the_same_turn(database, tmp_path):
    """本轮派出去的子代理，报告必须在**同一个 user_seq** 里收回来并续跑，绝不拖到下次按键。"""

    agent = ScriptedAgent("派完活就说半句")
    jobs = StubJobs()
    app = _app(database, tmp_path, agent, jobs)

    app.run_turn("t1", "去做调研")

    assert len(agent.calls) == 2, "报告回来之后必须再喂模型一步，否则它看不到结论"
    injected, context = agent.calls[1]
    assert context is agent.calls[0][1], "整轮要共用一个 RunContext，额度才会被统一封住"
    assert "子代理报告" in injected["messages"][0].content
    # 报告得进时间轴，且挂在当前这一轮名下（投影只认库里的行，不认函数入参）
    with database.session() as session:
        rows = session.execute(select(Message).where(Message.user_seq == 1)).scalars().all()
    decoded = [decode_message(row) for row in rows]
    reports = [
        message for message in decoded if getattr(message, "name", None) == "subagent_report"
    ]
    assert len(reports) == 1
    assert reports[0].content.startswith("【子代理回报】")


def test_turn_waits_until_every_report_is_drained(database, tmp_path):
    """一次收齐多份、模型又派了活：要把队列掏干净才允许本轮结束。"""

    agent = ScriptedAgent()
    jobs = StubJobs(reports=("#1 报告\n", "#2 报告\n"))
    app = _app(database, tmp_path, agent, jobs)

    app.run_turn("t1", "去做调研")

    assert len(agent.calls) == 3
    assert "子代理回报" in agent.calls[1][0]["messages"][0].content
    assert "子代理回报" in agent.calls[2][0]["messages"][0].content


def test_interrupted_turn_terminates_running_children(database, tmp_path):
    """Ctrl-C 打的是"这一轮"：正在飞的子代理跟着一起停，不给作废的回合继续写文件。"""

    class InterruptingAgent:
        def invoke(self, *args, **kwargs):
            del args, kwargs
            raise KeyboardInterrupt

        def update_state(self, *args, **kwargs):
            del args, kwargs

    jobs = StubJobs()
    app = _app(database, tmp_path, InterruptingAgent(), jobs)

    with pytest.raises(TurnExecutionError):
        app.run_turn("t1", "去做调研")

    assert jobs.terminated == 1


def test_turn_blocks_until_real_child_exits(database, tmp_path):
    """真起子进程：父模型说完话之后，本轮要**真的等**到它退出，报告才进得了时间轴。

    用 ``sleep`` 假装有活儿，不碰 API；测的是"等待发生在本轮之内"这个新边界。
    """

    from coding_agent.integrations.subagent import SubAgentJobs

    jobs = SubAgentJobs(tmp_path / "art")

    slow = [sys.executable, "-c", "import time;time.sleep(2);print('跑完了')"]

    class SpawningAgent(ScriptedAgent):
        def invoke(self, state, *, config, context):
            if not self.calls:
                jobs.spawn("慢活儿", lambda p: slow, cwd=str(tmp_path))
            return super().invoke(state, config=config, context=context)

    agent = SpawningAgent("我先说半句")
    app = _app(database, tmp_path, agent, jobs)
    started = time.monotonic()

    app.run_turn("t1", "去做调研")

    assert time.monotonic() - started >= 1.5, "本轮没等住子进程"
    assert len(agent.calls) == 2, "报告回来应当只续跑一步，父模型不再派活就该收工"
    injected = agent.calls[1][0]["messages"][0].content
    assert "#1" in injected and "退出码 0" in injected and "跑完了" in injected


def test_ctrl_c_while_waiting_kills_parent_and_child(database, tmp_path):
    """等子代理的这段时间里按 Ctrl-C：这一下打的是"这一轮"，父子一起停。

    真发一个 SIGINT 给主线程（等价于用户在终端按键），不看返回值编故事。
    """

    from coding_agent.integrations.subagent import SubAgentJobs

    jobs = SubAgentJobs(tmp_path / "art")
    spawned = []

    class SpawningAgent(ScriptedAgent):
        def invoke(self, state, *, config, context):
            if not self.calls:
                pid = jobs.spawn(
                    "永远跑不完",
                    lambda p: [sys.executable, "-c", "import time;time.sleep(60)"],
                    cwd=str(tmp_path),
                )
                spawned.append(pid)
            return super().invoke(state, config=config, context=context)

    agent = SpawningAgent("我还在等子代理")
    app = _app(database, tmp_path, agent, jobs)

    def interrupt_later():
        time.sleep(1.0)
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    threading.Thread(target=interrupt_later, daemon=True).start()

    with pytest.raises(TurnExecutionError) as raised:
        app.run_turn("t1", "去做调研")

    assert raised.value.interrupted is True
    assert len(agent.calls) == 1, "被打断了就不该再续跑，报告也不该被当成用户输入"
    process = jobs._jobs[spawned[0]]["process"]
    process.wait(timeout=10)
    assert process.poll() is not None, "父轮停了，子进程还活着 —— 那它会往作废的回合里写文件"


def projected_history(projection, thread_id):
    """模型下一轮真会看到的对话内容（去掉那条"整段覆盖"的 RemoveMessage 标记）。"""

    return [
        str(message.content)
        for message in projection.build(thread_id)
        if not isinstance(message, RemoveMessage)
    ]


class SilentAgent:
    """只回一句话、不自己落库的假代理：AI 消息由图的中间件负责，这里没有图。"""

    def __init__(self) -> None:
        self.prompts: list[list] = []

    def invoke(self, payload, *args, **kwargs):
        self.prompts.append(list(payload["messages"]))
        # 图返回的是 state dict，不是单条消息。
        return {"messages": [AIMessage(id="ai-reply", content="ok")]}

    def update_state(self, *args, **kwargs):
        del args, kwargs


def test_clear_context_leaves_nothing_for_the_model_to_see(database, tmp_path):
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    agent = SilentAgent()
    app = AgentApplication(
        Settings(workspace_root=tmp_path),
        database,
        agent,
        engine,
        FileMutationRecorder(database, tmp_path),
    )
    projection = ContextProjectionService(engine)

    app.run_turn("t1", "第一句")
    app.run_turn("t1", "第二句")
    assert projected_history(projection, "t1") == ["第一句", "第二句"]

    result = app.clear_context("t1")
    assert result.restored_files == 0
    # 库和进程内缓存都得空：只翻数据库、缓存还留着旧行，模型照样看得见历史。
    assert projected_history(projection, "t1") == []

    app.run_turn("t1", "第三句")
    assert projected_history(projection, "t1") == ["第三句"]


def test_open_plan_items_bring_the_model_back_for_one_more_round(database, tmp_path):
    """收工前还有没交代的计划项：递一条提醒再走一步——软提醒，不是硬闸门。"""

    agent = ScriptedAgent("我干完了")
    app = _app(database, tmp_path, agent)
    app.context_engine.mutate_todos(
        "t1",
        1,
        [
            {"id": "1", "title": "写代码", "status": "completed"},
            {"id": "2", "title": "补测试", "status": "pending"},
        ],
    )

    app.run_turn("t1", "做那件事")

    assert len(agent.calls) == 2, "计划没交代完就得再喂一步"
    injected = agent.calls[1][0]["messages"][0]
    assert getattr(injected, "name", None) == "todo_reminder"
    assert "补测试" in injected.content
    # 提醒得进时间轴：投影只认库里的行，只塞进 invoke 入参模型看不见
    with database.session() as session:
        rows = session.execute(select(Message).where(Message.user_seq == 1)).scalars().all()
    assert any(getattr(decode_message(row), "name", None) == "todo_reminder" for row in rows)


def test_the_reminder_is_given_only_once(database, tmp_path):
    """第二次还想收工就放行：不听就不再啰嗦，免得陷进死循环或逼它改状态作弊。"""

    agent = ScriptedAgent("还是不做")
    app = _app(database, tmp_path, agent)
    app.context_engine.mutate_todos("t1", 1, [{"id": "1", "title": "补测试", "status": "pending"}])

    app.run_turn("t1", "做那件事")

    assert len(agent.calls) == 2


def test_a_finished_plan_does_not_trigger_a_reminder(database, tmp_path):
    agent = ScriptedAgent("做完了")
    app = _app(database, tmp_path, agent)
    app.context_engine.mutate_todos(
        "t1",
        1,
        [
            {"id": "1", "title": "写代码", "status": "completed"},
            {"id": "2", "title": "老方案", "status": "cancelled"},
        ],
    )

    app.run_turn("t1", "做那件事")

    assert len(agent.calls) == 1, "全部交代清楚（完成或明确放弃）就别多嘴"
