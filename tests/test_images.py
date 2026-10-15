import io
import json
from unittest.mock import Mock

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image
from sqlalchemy import select

from coding_agent.application import AgentApplication, TurnExecutionError
from coding_agent.config import ContextSettings, Settings
from coding_agent.context.compaction import message_tokens, sanitize_transcript
from coding_agent.context.cover import ContextPiece
from coding_agent.context.engine import ContextEngine
from coding_agent.context.records import MessageSnapshot
from coding_agent.context.summarizer import DeterministicSummarizer
from coding_agent.integrations.langchain_agent import make_read_image_tool
from coding_agent.integrations.middleware import RunContext
from coding_agent.persistence.message_codec import encode_message
from coding_agent.persistence.models import Message
from coding_agent.persistence.repository import AgentRepository
from coding_agent.services.context_projection import ContextProjectionService
from coding_agent.ui.rendering import ContentRenderer
from coding_agent.workspace.file_undo import FileMutationRecorder
from coding_agent.workspace.images import ImageStore, preserve_image_references
from tests.integrations.test_agent import ToolAwareFakeModel, make_runtime_middleware


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "red").save(stream, format="PNG")
    return stream.getvalue()


def snapshot(message):
    return MessageSnapshot(1, "t1", 1, message.type, encode_message(message), message.id)


def test_import_is_durable_and_request_expansion_does_not_change_history(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(png_bytes())
    store = ImageStore(tmp_path / "artifacts")
    image = store.import_file(source)
    source.unlink()
    message = HumanMessage(content="look", additional_kwargs={"attachments": [image]})
    restarted = ImageStore(tmp_path / "artifacts")
    request = restarted.materialize(message)
    assert request.content[1]["source"]["media_type"] == "image/png"
    assert request.content[1]["source"]["data"]
    assert "base64" not in json.dumps(encode_message(message))
    assert restarted.lookup(image["id"]) == image
    assert store.import_bytes(png_bytes()) == image


def test_bad_images_and_missing_references_fail_explicitly(tmp_path):
    store = ImageStore(tmp_path)
    with pytest.raises(ValueError, match="损坏"):
        store.import_bytes(b"not an image")
    with pytest.raises(ValueError, match="10 MiB"):
        store.import_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="格式"):
        store.lookup("../../config.json")
    image = store.import_bytes(png_bytes())
    (store.root / f"{image['id']}.png").unlink()
    with pytest.raises(ValueError, match="不存在"):
        store.materialize(HumanMessage(content="", additional_kwargs={"attachments": [image]}))


def test_image_budget_summary_and_trimmed_tool_result(tmp_path):
    image = ImageStore(tmp_path).import_bytes(png_bytes())
    message = HumanMessage(content="look", additional_kwargs={"attachments": [image]})
    record = snapshot(message)
    assert message_tokens(record) > message_tokens(snapshot(HumanMessage(content="look")))
    transcript = sanitize_transcript([record])
    assert image["id"] in transcript
    assert "base64" not in transcript
    first = preserve_image_references(transcript, "summary without references")
    second = preserve_image_references(first, "higher summary without references")
    assert image["id"] in second
    assert preserve_image_references(second, second).count(image["id"]) == 1
    tool = ToolMessage(
        content="look", tool_call_id="call", additional_kwargs={"attachments": [image]}
    )
    trimmed = ContextEngine._trim_old_tool_results([snapshot(tool)], keep=0)[0]
    assert image["id"] in trimmed.content_json["data"]["content"]
    assert "attachments" not in trimmed.content_json["data"]["additional_kwargs"]


def test_image_only_turn_persists_restores_and_undo_removes_projection(database, tmp_path):
    store = ImageStore(tmp_path / "artifacts")
    image = store.import_bytes(png_bytes())
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    failing_agent = Mock()
    failing_agent.invoke.side_effect = RuntimeError("offline")
    app = AgentApplication(
        Settings(workspace_root=tmp_path, artifact_dir=tmp_path / "artifacts"),
        database,
        failing_agent,
        engine,
        FileMutationRecorder(database, tmp_path),
    )
    with pytest.raises(TurnExecutionError):
        app.run_turn("t1", "", attachments=[image])
    restored_engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    messages = ContextProjectionService(restored_engine).build("t1")[1:]
    assert messages[0].additional_kwargs["attachments"][0]["id"] == image["id"]
    assert store.materialize(messages[0]).content[0]["type"] == "image"
    assert image["id"] in ContentRenderer().text(app.active_history("t1")[0].content)
    app.rollback("t1", 1)
    after_undo = ContextProjectionService(engine).build("t1")[1:]
    assert not any(m.additional_kwargs.get("attachments") for m in after_undo)
    assert store.lookup(image["id"]) == image
    app.leave_thread()


def test_l0_summary_keeps_reference_when_summarizer_omits_it(database, tmp_path):
    image = ImageStore(tmp_path).import_bytes(png_bytes())
    record = snapshot(HumanMessage(content="look", additional_kwargs={"attachments": [image]}))
    summary = Mock()
    summary.summarize.return_value = "simple summary"
    engine = ContextEngine(database, ContextSettings(), summary)
    _, text, _ = engine._summarize_l0_chunk([record])
    assert image["id"] in text
    assert "base64" not in text
    projected = ContextProjectionService.render_pieces(
        [ContextPiece("memory", 1, 1, text=text, level=0, block_id=1)]
    )
    assert not projected[0].additional_kwargs.get("attachments")


def test_real_agent_middleware_expands_user_and_reread_images_only_on_request(database, tmp_path):
    store = ImageStore(tmp_path / "artifacts")
    image = store.import_bytes(png_bytes())
    human = HumanMessage(id="human", content="look", additional_kwargs={"attachments": [image]})
    with database.session() as session:
        repo = AgentRepository(session)
        repo.get_or_create_conversation("t1").active_head_seq = 1
        repo.add_message(
            thread_id="t1",
            user_seq=1,
            message_type="user",
            content_json=encode_message(human),
            langchain_message_id=human.id,
        )
    seen = []

    class RecordingModel(ToolAwareFakeModel):
        def _generate(self, messages, **kwargs):
            seen.append(messages)
            return super()._generate(messages, **kwargs)

    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_image",
                        "args": {"image_id": image["id"]},
                        "id": "image-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="red"),
        ]
    )
    engine = ContextEngine(database, ContextSettings(), DeterministicSummarizer())
    agent = create_agent(
        model=model,
        tools=[make_read_image_tool(store)],
        middleware=[make_runtime_middleware(database, engine, tmp_path)],
        context_schema=RunContext,
    )
    agent.invoke(
        {"messages": [human]},
        config={"configurable": {"thread_id": "t1", "user_seq": 1}},
        context=RunContext("t1", 1),
    )
    assert seen[0][0].content[1]["type"] == "image"
    tool_result = next(m for m in seen[1] if isinstance(m, ToolMessage))
    assert isinstance(tool_result.content, str)
    assert seen[1][-1].content[1]["type"] == "image"
    with database.session() as session:
        rows = list(session.scalars(select(Message)))
        assert all("base64" not in json.dumps(row.content_json) for row in rows)
        assert any(
            row.type == "tool" and row.content_json["data"]["additional_kwargs"].get("attachments")
            for row in rows
        )


def test_renderer_never_prints_image_payload():
    assert (
        ContentRenderer().text([{"type": "image", "source": {"data": "secret-payload"}}])
        == "[图片内容]"
    )


def test_parallel_tool_results_stay_together_and_images_are_separate_on_wire(tmp_path):
    from langchain_anthropic import ChatAnthropic

    store = ImageStore(tmp_path)
    image = store.import_bytes(png_bytes())
    calls = [
        {"name": "read_image", "args": {"image_id": image["id"]}, "id": identifier}
        for identifier in ("first", "second")
    ]
    messages = [HumanMessage(content="look"), AIMessage(content="", tool_calls=calls)]
    messages.extend(
        ToolMessage(
            content=image["id"], tool_call_id=call["id"], additional_kwargs={"attachments": [image]}
        )
        for call in calls
    )
    expanded = store.materialize_messages(messages)
    assert isinstance(expanded[2], ToolMessage) and isinstance(expanded[3], ToolMessage)
    model = ChatAnthropic(model="test", api_key="test")
    payload = model._get_request_payload(expanded)
    image_messages = [
        m
        for m in payload["messages"]
        if isinstance(m["content"], list) and any(b.get("type") == "image" for b in m["content"])
    ]
    assert len(image_messages) == 1
    assert image_messages[0]["role"] == "user"
    assert sum(b["type"] == "image" for b in image_messages[0]["content"]) == 2
    assert all(b["type"] != "tool_result" for b in image_messages[0]["content"])
