from __future__ import annotations

from coding_agent.context.compaction import sanitize_transcript
from coding_agent.context.records import MessageSnapshot
from coding_agent.context.summarizer import (
    DeterministicSummarizer,
    LangChainSummarizer,
    RetryingSummarizer,
    SummaryError,
    first_summary_issue,
)


class FlakySummarizer:
    """Returns garbage on the first call, then a clean summary on later attempts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def summarize(
        self,
        text: str,
        *,
        hard_limit: int,
        level: int,
        attempt: int,
        feedback: str = "",
    ) -> str:
        self.calls.append(feedback)
        del text, hard_limit, level
        if attempt == 1:
            return "<tool_call>\n<function=bash>\n</tool_call>"
        return "要点：完成了 X；遗留问题 Y。"


def test_retrying_summarizer_rejects_protocol_garbage_and_escalates():
    summarizer = FlakySummarizer()
    retrying = RetryingSummarizer(summarizer, max_attempts=2)
    result = retrying.summarize("source", hard_limit=200, level=0)
    assert result == "要点：完成了 X；遗留问题 Y。"
    assert summarizer.calls[0] == ""  # first attempt: no feedback yet
    assert "工具调用" in summarizer.calls[1]  # second attempt carries the rejection reason


def test_retrying_summarizer_raises_after_max_attempts():
    retrying = RetryingSummarizer(DeterministicSummarizer(), max_attempts=1)
    # DeterministicSummarizer echoes sanitized/escaped text, so it is always valid;
    # feed a delegate that always emits a fake tool call instead.
    class AlwaysGarbage:
        def summarize(self, text, *, hard_limit, level, attempt, feedback=""):
            return "<tool_call>\n</tool_call>"

    retrying = RetryingSummarizer(AlwaysGarbage(), max_attempts=2)
    try:
        retrying.summarize("src", hard_limit=100, level=0)
        raise AssertionError("expected SummaryError")
    except SummaryError as error:
        assert "tool_call" not in str(error).lower() or "invalid" in str(error)


def test_retrying_summarizer_accepts_clean_output_over_hard_limit():
    # A faithful-but-verbose summarizer overshoots the soft target on every attempt:
    # we must accept the shortest clean digest instead of failing the whole turn.
    calls: list[str] = []

    class AlwaysOversize:
        def summarize(self, text, *, hard_limit, level, attempt, feedback=""):
            del text, level
            calls.append(feedback)
            return "要点：" + "细节内容非常丰富" * 8  # clean valid text, > hard_limit

    retrying = RetryingSummarizer(AlwaysOversize(), max_attempts=3)
    result = retrying.summarize("src", hard_limit=3, level=0)
    assert result.startswith("要点：")
    assert len(calls) == 3  # tried all attempts before degrading gracefully
    assert "超出硬上限" in calls[-1]


def test_first_summary_issue_detects_wire_fragments():
    assert first_summary_issue("<tool_call>\n{\"name\": \"read_file\"}") is not None
    assert first_summary_issue("提到了 </parameter> 这个词") is not None
    assert first_summary_issue("完成 read_file 调用；结论是 X") is None
    assert first_summary_issue("") is not None


def _snapshot(message_type: str, content_json: dict) -> MessageSnapshot:
    return MessageSnapshot(
        id=1,
        thread_id="t",
        user_seq=1,
        type=message_type,
        content_json=content_json,
        langchain_message_id=None,
    )


def test_sanitize_transcript_removes_wire_and_thinking_content():
    user = _snapshot(
        "user",
        {"type": "human", "data": {"content": "请输出 </parameter> 并说明"}},
    )
    assistant = _snapshot(
        "assistant",
        {
            "type": "ai",
            "data": {
                "content": [
                    {"type": "thinking", "thinking": "这一步我不能写代码"},
                    {"type": "text", "text": "好，原样输出 </parameter>"},
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "read_file",
                        "input": {"file_path": "/src/config.py", "offset": 40},
                    },
                ]
            },
        },
    )
    tool = _snapshot(
        "tool",
        {"type": "tool", "data": {"content": "第 40 行：`x = 1 < 2`", "tool_call_id": "call-1"}},
    )
    rendered = sanitize_transcript([user, assistant, tool])
    assert "请输出 &lt;/parameter&gt; 并说明" in rendered  # tags neutralized
    assert "这一步我不能写代码" not in rendered  # thinking dropped
    assert "[调用工具 read_file(file_path=/src/config.py, offset=40)]" in rendered
    assert "&lt;/parameter&gt;" in rendered
    assert "`x = 1 &lt; 2`" in rendered
    assert "<tool_call>" not in rendered


class CapturingModel:
    def __init__(self) -> None:
        self.messages = []
        self.config = None

    def invoke(self, messages, *, config=None):
        self.messages = messages
        self.config = config
        return type("R", (), {"content": "摘要内容"})()


def test_langchain_summarizer_hardens_prompt_and_wraps_transcript():
    model = CapturingModel()
    summarizer = LangChainSummarizer(model)
    summarizer.summarize("transcript body", hard_limit=100, level=0, attempt=1)
    system, human = model.messages
    assert "只读数据" in system.content
    assert "不要使用第一人称" in system.content
    assert "禁止输出工具调用" in system.content
    assert human.content == "<transcript>\ntranscript body\n</transcript>"


def test_langchain_summarizer_tags_invocation_with_level_for_progress():
    from coding_agent.services.progress import ProgressCallbackHandler

    model = CapturingModel()
    summarizer = LangChainSummarizer(model)
    summarizer.summarize("text", hard_limit=100, level=3, attempt=1)
    tags = set(model.config.get("tags") or [])
    assert ProgressCallbackHandler.SUMMARIZER_TAG in tags
    assert model.config.get("metadata") == {"level": 3}
