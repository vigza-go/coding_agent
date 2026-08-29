from __future__ import annotations

from coding_agent.persistence.models import MessageType
from coding_agent.persistence.repository import AgentRepository


def test_user_seq_is_never_reused_after_head_moves_back(database):
    with database.session() as session:
        repo = AgentRepository(session)
        assert repo.reserve_user_seq("t1") == 1
        assert repo.reserve_user_seq("t1") == 2
        conversation = repo.get_or_create_conversation("t1")
        conversation.active_head_seq = 0
    with database.session() as session:
        assert AgentRepository(session).reserve_user_seq("t1") == 3


def test_message_persistence_is_idempotent_by_langchain_id(database):
    with database.session() as session:
        repo = AgentRepository(session)
        first = repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "hello"}},
            langchain_message_id="same-id",
        )
        second = repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type=MessageType.USER,
            content_json={"type": "human", "data": {"content": "hello"}},
            langchain_message_id="same-id",
        )
        assert first.id == second.id
