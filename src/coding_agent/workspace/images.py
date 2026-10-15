"""Durable image attachments; binary content is expanded only for model requests."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import warnings
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 8
MAX_PIXELS = 25_000_000
REFERENCE = re.compile(r"\[图片附件：(img_[0-9a-f]{64})\]")


def image_references(attachments: list[dict]) -> str:
    return "\n".join(f"[图片附件：{item['id']}]" for item in attachments)


def preserve_image_references(source: str, summary: str) -> str:
    """Keep references even when a summary model omits them, including higher levels."""
    missing = sorted(set(REFERENCE.findall(source)) - set(REFERENCE.findall(summary)))
    suffix = image_references([{"id": identifier} for identifier in missing])
    return f"{summary}\n{suffix}".strip() if suffix else summary


def estimated_image_tokens(attachments: list[dict]) -> int:
    # Conservative local heuristic, not a provider billing formula.
    return sum(
        256 + math.ceil(item["width"] / 512) * math.ceil(item["height"] / 512) * 512
        for item in attachments
    )


class ImageStore:
    def __init__(self, root: Path):
        self.root = root.resolve() / "attachments"

    def import_file(self, path: Path) -> dict:
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("图片不能超过 10 MiB。")
        return self.import_bytes(path.read_bytes())

    def import_bytes(self, data: bytes) -> dict:
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("图片不能超过 10 MiB。")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise ValueError("目前仅支持 PNG、JPEG 图片。")
                    if image.width * image.height > MAX_PIXELS:
                        raise ValueError("图片不能超过 2500 万像素。")
                    image.load()
                    # Normalize EXIF orientation and remove unrelated metadata.
                    normalized = ImageOps.exif_transpose(image)
                    mode = (
                        "RGBA"
                        if "A" in normalized.getbands() or "transparency" in image.info
                        else "RGB"
                    )
                    normalized = normalized.convert(mode)
                    stream = io.BytesIO()
                    normalized.save(stream, format="PNG")
                    payload = stream.getvalue()
                    width, height = normalized.size
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as error:
            raise ValueError("图片损坏或无法读取。") from error
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("转换后的图片超过 10 MiB，请缩小图片后再添加。")
        identifier = "img_" + hashlib.sha256(payload).hexdigest()
        metadata = {
            "id": identifier,
            "mime_type": "image/png",
            "storage_key": f"attachments/{identifier}.png",
            "width": width,
            "height": height,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        # Content-addressed files are immutable; concurrent imports contain identical bytes.
        self._write_once(self.root / f"{identifier}.png", payload)
        self._write_once(self.root / f"{identifier}.json", json.dumps(metadata).encode())
        return metadata

    @staticmethod
    def _write_once(path: Path, payload: bytes) -> None:
        import os
        import tempfile

        fd, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def lookup(self, identifier: str) -> dict:
        if not re.fullmatch(r"img_[0-9a-f]{64}", identifier):
            raise ValueError("图片引用格式不正确。")
        try:
            metadata = json.loads((self.root / f"{identifier}.json").read_text())
            payload = (self.root / f"{identifier}.png").read_bytes()
        except (OSError, ValueError) as error:
            raise ValueError(f"图片附件不存在或损坏：{identifier}") from error
        if "img_" + hashlib.sha256(payload).hexdigest() != identifier:
            raise ValueError(f"图片附件校验失败：{identifier}")
        return metadata

    def materialize(self, message: BaseMessage) -> BaseMessage:
        attachments = message.additional_kwargs.get("attachments", [])
        if not attachments:
            return message
        blocks = (
            [{"type": "text", "text": message.content}]
            if message.content and isinstance(message.content, str)
            else list(message.content or [])
        )
        for attachment in attachments:
            info = self.lookup(attachment["id"])
            data = (self.root / f"{info['id']}.png").read_bytes()
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": info["mime_type"],
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        return message.model_copy(
            update={
                "content": blocks,
                "additional_kwargs": {
                    key: value
                    for key, value in message.additional_kwargs.items()
                    if key != "attachments"
                },
            }
        )

    def materialize_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Project tool images as user image blocks for the configured gateway.

        Its user-image path works, but images nested in tool results are not
        reliably visible. Finish the entire tool-result batch before appending
        a derived user message, preserving tool call/result ordering.
        """
        result: list[BaseMessage] = []
        pending: list[dict] = []

        def flush() -> None:
            if pending:
                # LangChain merges adjacent user/tool-result messages. This marker
                # keeps the image in a separate user request after the tool batch;
                # the gateway otherwise discards images beside tool_result blocks.
                result.append(
                    AIMessage(content="[附件传递：下一条消息提供 read_image 对应的图片。]")
                )
                result.append(
                    self.materialize(
                        HumanMessage(
                            content="以下是 read_image 返回的图片，顺序与引用一致：\n"
                            + image_references(pending),
                            additional_kwargs={"attachments": list(pending)},
                        )
                    )
                )
                pending.clear()

        for message in messages:
            if not isinstance(message, ToolMessage):
                flush()
                result.append(self.materialize(message))
                continue
            pending.extend(message.additional_kwargs.get("attachments", []))
            result.append(
                message.model_copy(
                    update={
                        "additional_kwargs": {
                            key: value
                            for key, value in message.additional_kwargs.items()
                            if key != "attachments"
                        }
                    }
                )
            )
        flush()
        return result
