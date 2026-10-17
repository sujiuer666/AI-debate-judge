"""适配器公共契约与容错取值工具。

硬约束对应
----------
- 约束 5：适配器**不得假设固定 JSON 字段名**。因此本文件提供
  ``pick`` / ``pick_nested`` / ``find_message_list``：字段名以「候选列表」形式声明，
  逐个 ``.get()`` 尝试；全部缺失时抛 ``AdapterFieldError``，异常内含截断的原始片段。
- 约束 4：所有异常都带 ``stage``（失败环节）与 ``snippet``（原始片段），
  由 app.py 转成 st.error，用户能看到「哪个环节、原文长什么样」。
- 约束 3：本模块不发起任何请求、不读环境变量；是否调用由 app.py 的
  session_state 开关决定。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, Sequence

from parser import Utterance

SNIPPET_LIMIT: int = 400

# 字段名候选表：**故意不写死单一名字**，按顺序尝试。
# 具体真实字段名需用 scripts/probe_schema.py 在本地确认后，把确认结果补到列表最前面。
SENDER_KEYS: tuple[str, ...] = (
    "sender",
    "senderName",
    "sender_name",
    "from",
    "speaker",
    "nickname",
    "nickName",
    "name",
    "remark",
    "talker",
    "uin",
    "uid",
    "qq",
)
CONTENT_KEYS: tuple[str, ...] = (
    "content",
    "text",
    "message",
    "msg",
    "plainText",
    "plain_text",
    "raw_message",
    "body",
)
TIME_KEYS: tuple[str, ...] = (
    "time",
    "timestamp",
    "ts",
    "createTime",
    "create_time",
    "msgTime",
    "msg_time",
    "date",
    "sendTime",
)
LIST_KEYS: tuple[str, ...] = (
    "messages",
    "data",
    "items",
    "list",
    "records",
    "rows",
    "msgList",
    "messageList",
    "chatRecords",
    "elements",
)
SEGMENT_KEYS: tuple[str, ...] = ("segments", "elements", "content", "message", "parts")
SEGMENT_TYPE_KEYS: tuple[str, ...] = ("type", "msgType", "messageType", "subType", "kind")
SEGMENT_TEXT_KEYS: tuple[str, ...] = ("text", "content", "value", "title", "desc")


class AdapterError(RuntimeError):
    """适配器失败的基类：带失败环节与原始片段。"""

    def __init__(self, stage: str, detail: str, snippet: str = "") -> None:
        self.stage = stage
        self.detail = detail
        self.snippet = truncate(str(snippet or ""), SNIPPET_LIMIT)
        message = f"[{stage}] {detail}"
        if self.snippet:
            message = f"{message} | 原始片段：{self.snippet}"
        super().__init__(message)


class AdapterConfigError(AdapterError):
    """配置缺失（如未填 API URL / 未选择正方发言人）。"""


class AdapterHTTPError(AdapterError):
    """网络或 HTTP 状态异常。"""


class AdapterFormatError(AdapterError):
    """响应/文件不是合法 JSON 或整体结构不可识别。"""


class AdapterFieldError(AdapterError):
    """结构可识别，但关键字段缺失。"""


def truncate(text: str, limit: int = SNIPPET_LIMIT) -> str:
    text = text or ""
    return text if len(text) <= limit else f"{text[:limit]}…（已截断，共 {len(text)} 字符）"


def snippet_of(value: Any, limit: int = SNIPPET_LIMIT) -> str:
    """把任意对象转成可展示的截断片段（不抛异常）。"""
    try:
        dumped = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        dumped = repr(value)
    return truncate(dumped, limit)


def pick(mapping: Mapping[str, Any], candidates: Sequence[str], *, what: str, snippet: str) -> Any:
    """按候选字段名依次 ``.get()``，返回第一个非 ``None`` 值。

    全部缺失时抛 ``AdapterFieldError``，错误信息包含「已尝试的字段名」，方便对照
    probe_schema 的输出立即修正候选表。
    """
    for key in candidates:
        if key in mapping:
            value = mapping.get(key)
            if value is not None:
                return value
    raise AdapterFieldError(
        "field-missing",
        f"无法取得{what}；已尝试字段名 {list(candidates)}",
        snippet,
    )


def pick_optional(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    """同 ``pick``，但全部缺失时返回 ``None`` 而不抛异常（用于可选字段）。"""
    for key in candidates:
        if key in mapping:
            value = mapping.get(key)
            if value is not None:
                return value
    return None


def pick_nested_optional(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    """同 ``pick_nested``，但全部缺失时返回 ``None``。"""
    for path in paths:
        current: Any = mapping
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return None


def pick_nested(
    mapping: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    *,
    what: str,
    snippet: str,
) -> Any:
    """按候选「路径」逐层 ``.get()``，适合 ``{"sender": {"name": ...}}`` 这类嵌套结构。"""
    for path in paths:
        current: Any = mapping
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    pretty = [".".join(path) for path in paths]
    raise AdapterFieldError("field-missing", f"无法取得{what}；已尝试路径 {pretty}", snippet)


def as_display_text(value: Any, *, what: str, snippet: str) -> str:
    """把可能是 str / dict / list 的内容字段转成可读文本。

    - ``str``：原样返回；
    - ``dict``：尝试 text/content/title/desc，失败则返回 ``[非文本消息]``；
    - ``list``：逐项转换后用换行拼接；
    - 其它：``[非文本消息]``。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in SEGMENT_TEXT_KEYS:
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner
        return "[非文本消息]"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = [as_display_text(item, what=what, snippet=snippet) for item in value]
        text = "\n".join(part for part in parts if part)
        return text or "[非文本消息]"
    if value is None:
        raise AdapterFieldError("field-missing", f"{what}为空", snippet)
    return "[非文本消息]"


def find_message_list(
    payload: Any,
    *,
    list_keys: Sequence[str] = LIST_KEYS,
    snippet: str = "",
    max_depth: int = 3,
) -> list[Mapping[str, Any]]:
    """在未知层级的 JSON 中找到「消息数组」。

    策略（全部基于 ``.get()``，不假设固定层级）：
    1. 顶层就是数组 -> 直接用；
    2. 顶层是对象 -> 按 ``list_keys`` 找值为数组的键；
    3. 找到的值是对象 -> 递归一层（如 ``{"data": {"messages": [...]}}``）；
    4. 兜底：扫描任意「元素为对象的数组」或可递归对象（深度上限 ``max_depth``）。

    失败抛 ``AdapterFormatError``，附带截断的原始响应。
    """

    def _search(node: Any, depth: int) -> list[Mapping[str, Any]] | None:
        if isinstance(node, list):
            items = [item for item in node if isinstance(item, Mapping)]
            return items or None
        if not isinstance(node, Mapping) or depth > max_depth:
            return None

        for key in list_keys:
            if key not in node:
                continue
            value = node.get(key)
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, Mapping)]
                if items:
                    return items
            elif isinstance(value, Mapping):
                found = _search(value, depth + 1)
                if found:
                    return found

        for value in node.values():
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, Mapping)]
                if items and any(
                    any(candidate in items[0] for candidate in (*SENDER_KEYS, *CONTENT_KEYS))
                    for _ in (0,)
                ):
                    return items
            elif isinstance(value, Mapping):
                found = _search(value, depth + 1)
                if found:
                    return found
        return None

    found = _search(payload, 0)
    if found is None:
        raise AdapterFormatError(
            "structure",
            "未能在响应/文件中定位消息数组；请用 scripts/probe_schema.py 确认结构",
            snippet or snippet_of(payload),
        )
    return found


class ChatlogFetcher(Protocol):
    """chatlog 适配器的最小接口（便于注入假实现做离线测试）。"""

    def __call__(self, url: str, params: dict[str, Any], timeout: float) -> Any: ...


class QQExportLoader(Protocol):
    """QQ 导出适配器的最小接口。"""

    def __call__(self, raw: bytes | str) -> list[Utterance]: ...
