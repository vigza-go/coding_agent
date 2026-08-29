from __future__ import annotations

from coding_agent.config import ContextSettings
from coding_agent.context.engine import ContextEngine
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
    assert len(result.rebuilt_pieces) == 1
    assert result.rebuilt_pieces[0].block_id == child1.id
    with database.session() as session:
        assert session.get(MemoryBlock, child1.id).active is True
        assert session.get(MemoryBlock, child2.id).active is False
        assert session.get(MemoryBlock, parent.id).active is False
        conversation = AgentRepository(session).get_or_create_conversation("t1")
        assert conversation.active_head_seq == 1
        assert conversation.next_user_seq == 4
