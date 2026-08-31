from __future__ import annotations

import pytest

from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.usage import RecentUsage


def metadata(inputs: object = 1000, outputs: int = 10, cached: object = 900):
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "input_token_details": {"cache_read": cached, "cache_creation": 50},
    }


def test_cache_hit_rate_is_token_weighted_and_does_not_double_count_cache():
    usage = RecentUsage.from_metadata([metadata(), metadata(inputs=100, cached=0)])
    assert usage.input_tokens == 1100
    assert usage.output_tokens == 20
    assert usage.cache_input_tokens == 1100
    assert usage.cache_read_tokens == 900
    assert usage.cache_hit_rate == pytest.approx(900 / 1100)
    assert usage.cache_samples == usage.usage_samples == 2


def test_missing_or_invalid_cache_fields_are_not_counted_as_zero_hits():
    usage = RecentUsage.from_metadata(
        [
            metadata(),
            {"input_tokens": 1000, "output_tokens": 20},
            metadata(cached=None),
            metadata(cached=1001),
            metadata(cached=-1),
            None,
            {},
        ]
    )
    assert usage.sampled_messages == 7
    assert usage.usage_samples == 5
    assert usage.input_tokens == 5000
    assert usage.cache_samples == 1
    assert usage.cache_hit_rate == 0.9


@pytest.mark.parametrize("records", [[], [None], [{}], [metadata(inputs=0, cached=0)]])
def test_no_computable_input_has_no_hit_rate(records):
    assert RecentUsage.from_metadata(records).cache_hit_rate is None


def test_zero_cache_reads_is_a_real_zero_rate():
    assert RecentUsage.from_metadata([metadata(cached=0)]).cache_hit_rate == 0


def test_recent_usage_query_is_thread_scoped_and_includes_undone_calls(database):
    with database.session() as session:
        repo = AgentRepository(session)
        for kind, value, active, thread_id in [
            ("assistant", metadata(inputs=10, cached=5), True, "t1"),
            ("assistant", metadata(), False, "t1"),
            ("tool", metadata(), True, "t1"),
            ("assistant", None, True, "t1"),
            ("assistant", metadata(), True, "other"),
        ]:
            row = repo.add_message(
                thread_id=thread_id,
                user_seq=1,
                message_type=kind,
                content_json={"data": {"content": "not needed", "usage_metadata": value}},
            )
            row.active = active
    with database.session() as session:
        repo = AgentRepository(session)
        records = repo.recent_usage_metadata("t1", limit=2)
        assert records == [None, metadata()]
        assert repo.recent_usage_metadata("missing", limit=2) == []
        with pytest.raises(ValueError):
            repo.recent_usage_metadata("t1", limit=0)


@pytest.mark.parametrize("value", [True, -1, "100", 1.5, None])
def test_invalid_api_counts_are_not_estimated(value):
    assert RecentUsage.from_metadata([metadata(inputs=value)]).usage_samples == 0
