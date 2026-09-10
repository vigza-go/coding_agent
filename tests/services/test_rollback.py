from __future__ import annotations

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
from coding_agent.context.pin import render_pin
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.persistence.models import MemoryBlock, MessageType
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.rollback import RollbackService
from coding_agent.workspace.file_undo import FileMutationRecorder


def test_file_mutations_roll_back_in_reverse_global_id_order(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("original", encoding="utf-8")
    recorder = FileMutationRecorder(database, workspace)

    first = recorder.begin(
        thread_id="t1",
        user_seq=2,
        tool_call_id="call-1",
        tool_name="write_file",
        requested_path="/value.txt",
    )
    target.write_text("first", encoding="utf-8")
    recorder.finish(first, succeeded=True)

    second = recorder.begin(
        thread_id="t1",
        user_seq=3,
        tool_call_id="call-2",
        tool_name="edit_file",
        requested_path="/value.txt",
    )
    target.write_text("second", encoding="utf-8")
    recorder.finish(second, succeeded=True)

    assert recorder.rollback("t1", 2) == 2
    assert target.read_text(encoding="utf-8") == "original"


def test_failed_file_mutation_is_not_restored(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    recorder = FileMutationRecorder(database, workspace)
    mutation_id = recorder.begin(
        thread_id="t1",
        user_seq=1,
        tool_call_id="failed",
        tool_name="edit_file",
        requested_path="/value.txt",
    )
    target.write_text("external", encoding="utf-8")
    recorder.finish(mutation_id, succeeded=False)

    assert recorder.rollback("t1", 1) == 0
    assert target.read_text(encoding="utf-8") == "external"


def test_rollback_preview_reports_scope_without_changing_state(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    recorder = FileMutationRecorder(database, workspace)
    with database.session() as session:
        repo = AgentRepository(session)
        repo.add_message(
            thread_id="t1",
            user_seq=2,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "change"}},
            langchain_message_id="preview-message",
        )
        repo.save_work_state("t1", 2, {"step": "change"})
    mutation_id = recorder.begin(
        thread_id="t1",
        user_seq=2,
        tool_call_id="preview-call",
        tool_name="write_file",
        requested_path="/value.txt",
    )
    target.write_text("after", encoding="utf-8")
    recorder.finish(mutation_id, succeeded=True)
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())

    preview = RollbackService(database, engine, recorder).preview("t1", 2)

    assert preview.messages == 1
    assert preview.file_mutations == 1
    assert preview.files == 1
    assert preview.work_states == 1
    assert target.read_text(encoding="utf-8") == "after"


def test_context_rollback_invalidates_by_message_range_and_reuses_child(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with database.session() as session:
        repo = AgentRepository(session)
        rows = []
        for seq in (1, 1, 2, 2, 3, 3):
            rows.append(
                repo.add_message(
                    thread_id="t1",
                    user_seq=seq,
                    message_type=MessageType.USER,
                    content_json={"type": "human", "data": {"content": f"seq-{seq}"}},
                    langchain_message_id=f"m-{len(rows)}",
                )
            )
        child1 = repo.add_memory_block(
            thread_id="t1",
            text="seq 1",
            begin_message_id=rows[0].id,
            end_message_id=rows[1].id,
            level=0,
            token_count=2,
        )
        child2 = repo.add_memory_block(
            thread_id="t1",
            text="seq 2",
            begin_message_id=rows[2].id,
            end_message_id=rows[3].id,
            level=0,
            token_count=2,
        )
        parent = repo.add_memory_block(
            thread_id="t1",
            text="seq 1-2",
            begin_message_id=rows[0].id,
            end_message_id=rows[3].id,
            level=1,
            token_count=2,
        )
        repo.save_work_state("t1", 1, {"step": 1})
        repo.save_work_state("t1", 2, {"step": 2})
        conversation = repo.get_or_create_conversation("t1")
        conversation.active_head_seq = 3
        conversation.next_user_seq = 4

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    assert engine.rebuild("t1")[0].block_id == parent.id
    recorder = FileMutationRecorder(database, workspace)
    result = RollbackService(database, engine, recorder).rollback("t1", 2)

    assert result.deactivated_messages == 4
    assert result.work_state == {"step": 1}
    # 撤销后有块的线程重新载入，投影是 [记忆块] + [便签]：便签跟着回滚后的状态走。
    assert [piece.kind for piece in result.rebuilt_pieces] == ["memory", "work_state"]
    assert result.rebuilt_pieces[0].block_id == child1.id
    # 便签里放的是「计划 + 工作状态」，这条线程没有计划，所以就只剩工作状态那一段。
    assert result.rebuilt_pieces[1].text == render_pin(None, result.work_state)
    with database.session() as session:
        assert session.get(MemoryBlock, child1.id).active is True
        assert session.get(MemoryBlock, child2.id).active is False
        assert session.get(MemoryBlock, parent.id).active is False
        conversation = AgentRepository(session).get_or_create_conversation("t1")
        assert conversation.active_head_seq == 1
        assert conversation.next_user_seq == 4


def test_rollback_head_skips_already_inactive_sequence(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with database.session() as session:
        repo = AgentRepository(session)
        repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "first"}},
            langchain_message_id="head-first",
        )
        second = repo.add_message(
            thread_id="t1",
            user_seq=2,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "second"}},
            langchain_message_id="head-second",
        )
        repo.add_message(
            thread_id="t1",
            user_seq=3,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "third"}},
            langchain_message_id="head-third",
        )
        second.active = False
        conversation = repo.get_or_create_conversation("t1")
        conversation.active_head_seq = 3
        conversation.next_user_seq = 4

    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    recorder = FileMutationRecorder(database, workspace)
    RollbackService(database, engine, recorder).rollback("t1", 3)

    with database.session() as session:
        repo = AgentRepository(session)
        conversation = repo.get_or_create_conversation("t1")
        assert conversation.active_head_seq == 1
        assert repo.latest_active_user_seq("t1") == 1
        assert [row.user_seq for row in repo.active_messages("t1")] == [1]


def _seed_turn(database, workspace, *, user_seq, before, after, ids):
    """造一轮：一条提问、一条回答、一份工作状态、一处文件改动。"""

    target = workspace / "value.txt"
    target.write_text(before, encoding="utf-8")
    recorder = FileMutationRecorder(database, workspace)
    with database.session() as session:
        repo = AgentRepository(session)
        repo.add_message(
            thread_id="t1",
            user_seq=user_seq,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "change"}},
            langchain_message_id=f"{ids}-user",
        )
        repo.add_message(
            thread_id="t1",
            user_seq=user_seq,
            message_type=MessageType.ASSISTANT,
            content_json={"type": "ai", "data": {"content": "done"}},
            langchain_message_id=f"{ids}-ai",
        )
        repo.save_work_state("t1", user_seq, {"step": "change"})
    mutation_id = recorder.begin(
        thread_id="t1",
        user_seq=user_seq,
        tool_call_id=f"{ids}-call",
        tool_name="write_file",
        requested_path="/value.txt",
    )
    target.write_text(after, encoding="utf-8")
    recorder.finish(mutation_id, succeeded=True)
    return recorder, target


def test_clear_context_wipes_timeline_but_never_the_worktree(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder, target = _seed_turn(
        database, workspace, user_seq=2, before="original", after="after", ids="clear"
    )
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    assert engine.current_work_state("t1") is not None  # 先让缓存热起来

    result = RollbackService(database, engine, recorder).clear_context("t1")

    assert result.restored_files == 0
    assert result.deactivated_messages == 2
    assert target.read_text(encoding="utf-8") == "after"  # 磁盘上一个字都没动
    assert engine.current_work_state("t1") is None  # 工作状态也是上下文，一起清
    with database.session() as session:
        repo = AgentRepository(session)
        assert repo.active_messages("t1") == []
        conversation = repo.get_or_create_conversation("t1")
        assert conversation.active_head_seq == 0
        assert conversation.next_user_seq == 1  # 取号器只增不减，clear 不动它


def test_after_clear_old_turn_can_still_be_undone_by_seq(database, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    recorder, target = _seed_turn(
        database, workspace, user_seq=2, before="original", after="old-turn", ids="old"
    )
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    service = RollbackService(database, engine, recorder)
    service.clear_context("t1")
    # 清空之后又走了一轮（只留时间轴，没碰文件）。
    with database.session() as session:
        AgentRepository(session).add_message(
            thread_id="t1",
            user_seq=3,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "next"}},
            langchain_message_id="next-user",
        )

    fresh = service.rollback("t1", 3)
    assert fresh.restored_files == 0  # 退新轮不该顺手把 clear 之前的改动也退了
    assert target.read_text(encoding="utf-8") == "old-turn"

    # 旧账没被 clear 停用，按它自己那一轮仍然退得回去。
    old = service.rollback("t1", 2)
    assert old.restored_files == 1
    assert target.read_text(encoding="utf-8") == "original"
