"""单题评测：真实数据 + 生产 ContextEngine + 滚动摘要对照。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from coding_agent.config import ContextSettings, load_settings
from coding_agent.context.compaction import (
    atomic_message_units,
    message_tokens,
    sanitize_transcript,
)
from coding_agent.context.engine import CompressionInvariantError, ContextEngine
from coding_agent.context.records import MemoryBlockSnapshot
from coding_agent.context.summarizer import (
    DeterministicSummarizer,
    LangChainSummarizer,
    SummaryError,
)
from coding_agent.context.tokens import estimate_tokens
from coding_agent.integrations.langchain_agent import build_summary_model
from coding_agent.persistence.database import Database
from coding_agent.persistence.message_codec import encode_message, message_type
from coding_agent.persistence.models import MemoryBlock, Message
from coding_agent.persistence.repository import AgentRepository

EXP = Path(__file__).resolve().parent
VERSION = 5
DATA_FILE = EXP / "_data" / "lme_s.json"
CASE_SLICE = slice(166, 167)
MODES = ("hier", "roll")
BUDGET = 32_000
STEP = 1_000_000
FAKE = False


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class Case:
    qid: str
    sessions: list[list[Turn]]
    total_tokens: int
    tokens_before_evidence: int = 0
    has_evidence: bool = False


def make_message(qid: str, session: int, turn: int, role: str, content: str):
    cls = HumanMessage if role == "user" else AIMessage
    return cls(content=content).model_copy(update={"id": f"lme:{qid}:{session}:{turn}"})


def build_case(item: dict) -> Case:
    qid = item["question_id"]
    sessions, total, evidence = [], 0, 0
    for session in item["haystack_sessions"]:
        turns = []
        for turn in session:
            message = make_message(qid, 0, 0, turn["role"], turn["content"])
            total += estimate_tokens(encode_message(message))
            if turn.get("has_answer") and not evidence:
                evidence = total
            turns.append(Turn(turn["role"], turn["content"]))
        sessions.append(turns)
    return Case(qid, sessions, total, evidence, bool(evidence))


class RollingEngine(ContextEngine):
    """传统滚动摘要：旧压缩区和新工作区前缀一起重写成一个摘要。"""

    def _create_l0_if_needed(self, state) -> int:
        working = state.working_messages
        if sum(message_tokens(message) for message in working) <= self.settings.working_trigger:
            return 0
        working = self._retain_reasoning_within_budget(working, self.settings.reasoning_budget)
        state.working_messages = working
        working_tokens = sum(message_tokens(message) for message in working)
        if working_tokens <= self.settings.working_trigger:
            return 0
        working = self._trim_old_tool_results(working, self.settings.recent_tool_interactions)
        working_tokens = sum(message_tokens(message) for message in working)
        units = atomic_message_units(working)
        tail_target = working_tokens * self.settings.recent_tail_ratio
        split_at, tail_tokens = len(units), 0
        while split_at > 0 and tail_tokens < tail_target:
            split_at -= 1
            tail_tokens += sum(message_tokens(message) for message in units[split_at])
        prefix = [message for unit in units[:split_at] for message in unit]
        if not prefix:
            return 0

        old_text = "\n\n".join(block.text for block in state.selected_blocks)
        new_text = sanitize_transcript(prefix)
        source = "\n\n".join(part for part in (old_text, new_text) if part)
        source_tokens = sum(block.token_count for block in state.selected_blocks)
        source_tokens += sum(message_tokens(message) for message in prefix)
        summary = self.summarizer.summarize(
            source,
            hard_limit=max(1, int(source_tokens * self.settings.summary_target_ratio)),
            level=0,
        )
        begin = state.selected_blocks[0].begin_message_id if state.selected_blocks else prefix[0].id
        with self.database.session() as session:
            repo = AgentRepository(session)
            repo.get_or_create_conversation(state.thread_id, lock=True)
            block = repo.add_memory_block(
                thread_id=state.thread_id,
                text=summary,
                begin_message_id=begin,
                end_message_id=prefix[-1].id,
                level=0,
                token_count=estimate_tokens(summary),
            )
        state.selected_blocks[:] = [MemoryBlockSnapshot.from_model(block)]
        state.working_messages = working[len(prefix) :]
        return 1

    def _merge_until_within_budget(self, state) -> int:
        # 滚动摘要已经在上一步把旧压缩区和新前缀合成一个块，不再做分层块合并。
        return 0


class Counting:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = self.input_tokens = self.output_tokens = 0

    def summarize(self, text, *, hard_limit, level, attempt=1, feedback=""):
        result = self.inner.summarize(
            text, hard_limit=hard_limit, level=level, attempt=attempt, feedback=feedback
        )
        self.calls += 1
        self.input_tokens += estimate_tokens(text)
        self.output_tokens += estimate_tokens(result)
        return result


@dataclass
class Sample:
    version: int
    session: int
    finished: bool
    error: str
    top_level: int
    press_max: int
    calls: int
    input_tokens: int
    output_tokens: int


def _metrics(db: Database, thread: str) -> tuple[int, int]:
    with db.session() as session:
        ids = list(session.scalars(select(Message.id).where(Message.thread_id == thread)))
        blocks = list(session.scalars(select(MemoryBlock).where(MemoryBlock.thread_id == thread)))
    counts = [sum(b.begin_message_id <= mid <= b.end_message_id for b in blocks) for mid in ids]
    return max(counts, default=0), max((b.level for b in blocks), default=-1)


class Run:
    def __init__(self, case: Case, budget: int, mode: str, settings, fake: bool) -> None:
        self.case, self.budget, self.mode = case, budget, mode
        self.thread = f"eval-{case.qid}-{mode}-{uuid.uuid4().hex[:8]}"
        self.db = Database(
            f"sqlite+pysqlite:///file:{self.thread}?mode=memory&cache=shared&uri=true"
        )
        self.db.create_schema()
        model = build_summary_model(settings).bind(temperature=0) if not fake else None
        summarizer = DeterministicSummarizer() if fake else LangChainSummarizer(model)
        self.counter = Counting(summarizer)
        engine = RollingEngine if mode == "roll" else ContextEngine
        self.engine = engine(self.db, ContextSettings(total_tokens=budget), self.counter)

    def add_session(self, index: int) -> None:
        added = []
        with self.db.session() as session:
            repo = AgentRepository(session)
            for turn_index, turn in enumerate(self.case.sessions[index]):
                message = make_message(self.case.qid, index, turn_index, turn.role, turn.content)
                added.append(
                    repo.add_message(
                        thread_id=self.thread,
                        user_seq=repo.reserve_user_seq(self.thread),
                        message_type=message_type(message),
                        content_json=encode_message(message),
                        langchain_message_id=message.id,
                    )
                )
        self.engine.append_messages(self.thread, added)
        self.engine.compact_if_needed(self.thread)

    def snapshot(self, session: int, finished: bool, error: str = "") -> Sample:
        top, level = _metrics(self.db, self.thread)
        return Sample(
            VERSION,
            session,
            finished,
            error,
            level,
            top,
            self.counter.calls,
            self.counter.input_tokens,
            self.counter.output_tokens,
        )


def run_case(
    item: dict, budget: int, mode: str, settings, step: int, fake: bool, out: Path | None = None
):
    case, run = build_case(item), None
    run = Run(case, budget, mode, settings, fake)
    samples, error, done = [], "", 0
    for index in range(len(case.sessions)):
        try:
            run.add_session(index)
        except (CompressionInvariantError, SummaryError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:100]}"
            break
        done = index + 1
        if step > 0 and done % step == 0:
            samples.append(run.snapshot(done, False))
    samples.append(run.snapshot(done, not error, error))
    path = out or output_path(mode, case.qid, budget, fake)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in samples), encoding="utf-8"
    )
    print(
        f"[{mode}] {case.qid}｜{case.total_tokens:,} tok｜{done}/{len(case.sessions)} session｜"
        f"调用 {samples[-1].calls}｜输入 {samples[-1].input_tokens:,}｜最高 L{samples[-1].top_level}",
        flush=True,
    )
    return samples


def output_path(mode: str, qid: str, budget: int, fake: bool) -> Path:
    return EXP / "_out" / f"lme_tl_{mode}_{qid}_{budget}_{'fake' if fake else 'real'}.jsonl"


def main() -> int:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    items = data[CASE_SLICE]
    settings = load_settings()
    for item in items:
        for mode in MODES:
            run_case(item, BUDGET, mode, settings, STEP, FAKE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
