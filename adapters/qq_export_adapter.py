"""QQ 聊天记录导出（QQ Chat Exporter, QCE）文件适配器。

来源与适配说明（第 1 步已核对）
------------------------------
- 输入是**用户自己上传的 JSON 文件**（QCE 导出格式），不涉及 QCE 源码、
  不启动其服务、不调用其 API。QCE 本体为 GPL-3.0，本项目**不复制其代码**，
  只解析它导出的数据文件。
- 公开文档（README / 官方文档站）只说明支持 ``TXT / JSON / HTML / EXCEL /
  STREAMING_JSONL`` 导出，**未暴露 JSON 字段级 schema**；消息是否以
  segment 形式出现、segment 的 type 取值、发送者字段名，全部**需本地确认**。
  因此本文件是「容错适配器」：
  * 顶层消息数组用 ``find_message_list`` 递归定位（不假设层级）；
  * 发送者/时间/内容/segment 全部走候选字段表 + ``.get()``；
  * 按 segment type 分发：文本拼接，非文本替换为可读占位符（如 ``[图片]``），
    绝不让非文本消息静默消失；
  * 取不到内容时抛 ``AdapterFieldError``，异常内含该条消息的原始 JSON 片段（截断）。
- 默认关闭：仅当 ``st.session_state["enable_qq_upload"]`` 为 True 且用户上传文件后才调用。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from adapters.base import (
    CONTENT_KEYS,
    SEGMENT_KEYS,
    SEGMENT_TEXT_KEYS,
    SEGMENT_TYPE_KEYS,
    SENDER_KEYS,
    TIME_KEYS,
    AdapterFieldError,
    AdapterFormatError,
    find_message_list,
    pick_optional,
    snippet_of,
)
from parser import Utterance

#: 需要渲染成占位符的非文本 segment 类型（小写）。
PLACEHOLDER_BY_TYPE: dict[str, str] = {
    "image": "[图片]",
    "img": "[图片]",
    "pic": "[图片]",
    "photo": "[图片]",
    "sticker": "[表情]",
    "face": "[表情]",
    "mface": "[表情]",
    "emoji": "[表情]",
    "voice": "[语音]",
    "audio": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "reply": "[引用消息]",
    "quote": "[引用消息]",
    "forward": "[合并转发]",
    "json": "[卡片消息]",
    "xml": "[卡片消息]",
    "ark": "[卡片消息]",
    "system": "[系统消息]",
    "notice": "[系统消息]",
    "recall": "[撤回提示]",
}

#: 纯文本 segment 类型；其余按占位符处理。
TEXT_SEGMENT_TYPES: frozenset[str] = frozenset({"text", "plain", "plaintext", "markdown", "md", "longtext"})

#: 「@某人」这类提及：不进正文，避免污染辩论文本。
MENTION_TYPES: frozenset[str] = frozenset({"at", "mention"})

UNKNOWN_TEXT: str = "[非文本消息]"


def _decode(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw.replace("\ufeff", "")
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AdapterFormatError("encoding", "无法解码文件内容（已尝试 utf-8-sig / utf-8 / gb18030）")


def _sender_of(message: Mapping[str, Any]) -> str | None:
    value = pick_optional(message, SENDER_KEYS)
    if value is None:
        return None
    if isinstance(value, Mapping):
        nested = pick_optional(
            value,
            ("name", "nickname", "nickName", "remark", "displayName", "uin", "uid", "qq", "user"),
        )
        if isinstance(nested, Mapping):
            nested = pick_optional(nested, ("name", "nickname", "uin", "uid", "qq"))
        return None if nested is None else str(nested)
    return str(value)


def _time_of(message: Mapping[str, Any]) -> str | None:
    value = pick_optional(message, TIME_KEYS)
    return None if value is None else str(value)


def _segment_text(segment: Mapping[str, Any]) -> str:
    """从一个 segment 中取出文本（data 可能是嵌套 dict）。"""
    for container in (segment.get("data"), segment):
        if isinstance(container, Mapping):
            for key in SEGMENT_TEXT_KEYS:
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return ""


def _segments_to_text(segments: Sequence[Any], *, index: int, raw: str) -> str:
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            parts.append(segment)
            continue
        if not isinstance(segment, Mapping):
            parts.append(UNKNOWN_TEXT)
            continue

        raw_type = pick_optional(segment, SEGMENT_TYPE_KEYS)
        segment_type = str(raw_type).strip().lower() if raw_type is not None else ""

        if segment_type in MENTION_TYPES:
            continue
        if not segment_type or segment_type in TEXT_SEGMENT_TYPES:
            text = _segment_text(segment)
            if text:
                parts.append(text)
            elif not segment_type:
                parts.append(UNKNOWN_TEXT)
            continue
        parts.append(PLACEHOLDER_BY_TYPE.get(segment_type, UNKNOWN_TEXT))

    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise AdapterFieldError("qq-content", f"第 {index} 条消息的所有分段都没有可读文本", raw)
    return text


def _message_text(message: Mapping[str, Any], *, index: int) -> str:
    raw = snippet_of(message)

    # 1) 优先按 segment 数组处理（QCE 可能把消息拆成 文本/图片/引用 等分段）。
    for key in SEGMENT_KEYS:
        if key not in message:
            continue
        value = message.get(key)
        if isinstance(value, list) and value:
            return _segments_to_text(value, index=index, raw=raw)
        if isinstance(value, str) and value.strip():
            return value

    # 2) 退回普通内容字段（content/text/message...）。
    for key in CONTENT_KEYS:
        if key not in message:
            continue
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            for text_key in SEGMENT_TEXT_KEYS:
                inner = value.get(text_key)
                if isinstance(inner, str) and inner.strip():
                    return inner

    raise AdapterFieldError(
        "qq-content",
        f"第 {index} 条消息无法解析出内容；请用 scripts/probe_schema.py 确认该文件的字段名",
        raw,
    )


def load_qq_export(raw: bytes | str, *, allow_missing_sender: bool = False) -> list[Utterance]:
    """解析 QCE 导出的 JSON，规范为 ``Utterance`` 列表。

    Raises:
        AdapterFormatError: 编码/JSON/整体结构不可识别。
        AdapterFieldError: 单条消息关键字段缺失（带原始片段）。
    """
    text = _decode(raw)
    if not text.strip():
        raise AdapterFormatError("qq-empty", "上传的文件为空")

    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterFormatError(
            "qq-json",
            f"文件不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
            text,
        ) from exc

    messages = find_message_list(payload, snippet=snippet_of(payload))
    if not messages:
        raise AdapterFormatError("qq-empty", "文件中没有找到任何消息", snippet_of(payload))

    utterances: list[Utterance] = []
    for index, message in enumerate(messages, start=1):
        sender = _sender_of(message)
        if sender is None and not allow_missing_sender:
            raise AdapterFieldError(
                "qq-sender",
                f"第 {index} 条消息缺少发送者字段；如为系统消息可将 allow_missing_sender 设为 True",
                snippet_of(message),
            )
        text_content = _message_text(message, index=index)
        utterances.append(
            Utterance(sender=sender, side=None, text=text_content, ts=_time_of(message))
        )

    return utterances
