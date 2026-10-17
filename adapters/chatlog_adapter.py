"""chatlog（Go 二进制）HTTP API 适配器。

来源与适配说明（第 1 步已核对）
------------------------------
- 仅消费 ``xlight/chatlog``（镜像 ``sjzar/chatlog``）README 中公开的 HTTP 契约：
  ``GET /api/v1/chatlog``，参数 ``time``（``YYYY-MM-DD`` 或 ``YYYY-MM-DD~YYYY-MM-DD``）、
  ``talker``、``limit``、``offset``、``format``（json / csv / 纯文本）。
  默认服务地址 ``http://127.0.0.1:5030``（README 原文）。
- **不涉及** Go 源码、不 clone、不调用其 TUI/解密流程；本文件是纯 ``requests`` 客户端。
- **响应 JSON 的具体字段名未在公开文档中暴露**（第 1 步标为需本地确认），因此：
  * 字段名全部走 ``adapters.base`` 的候选表 + ``.get()`` 链；
  * 缺失时抛 ``AdapterFieldError``，异常携带截断后的原始响应；
  * 本地确认后，把真实字段名补到 ``adapters/base.py`` 的候选表最前面即可，无需改本文件。
- 默认关闭：本模块不被 app.py 自动调用，只有在 ``st.session_state["enable_chatlog"]``
  为 True 且 URL/talker 已填时才触发。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from adapters.base import (
    CONTENT_KEYS,
    SENDER_KEYS,
    TIME_KEYS,
    AdapterConfigError,
    AdapterFieldError,
    AdapterFormatError,
    AdapterHTTPError,
    as_display_text,
    find_message_list,
    pick,
    pick_nested_optional,
    pick_optional,
    snippet_of,
)
from parser import Utterance

ENDPOINT_PATH: str = "/api/v1/chatlog"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ChatlogQuery:
    """一次查询的参数。"""

    base_url: str
    talker: str
    start_date: str
    end_date: str | None = None
    limit: int = 200
    offset: int = 0
    timeout: float = 20.0
    allow_missing_sender: bool = False

    def validate(self) -> None:
        if not self.base_url or not self.base_url.startswith(("http://", "https://")):
            raise AdapterConfigError("chatlog-config", "chatlog API URL 必须以 http:// 或 https:// 开头")
        if not self.talker.strip():
            raise AdapterConfigError("chatlog-config", "talker（聊天对象）不能为空")
        if not _DATE_RE.match(self.start_date):
            raise AdapterConfigError("chatlog-config", f"start_date 必须是 YYYY-MM-DD，实际为 {self.start_date!r}")
        if self.end_date is not None:
            if not _DATE_RE.match(self.end_date):
                raise AdapterConfigError("chatlog-config", f"end_date 必须是 YYYY-MM-DD，实际为 {self.end_date!r}")
            if self.end_date < self.start_date:
                raise AdapterConfigError("chatlog-config", "end_date 不能早于 start_date")
        if not 1 <= self.limit <= 5000:
            raise AdapterConfigError("chatlog-config", f"limit 必须在 1-5000 之间，实际为 {self.limit}")


def build_params(query: ChatlogQuery) -> dict[str, Any]:
    """构造 chatlog 查询参数（字段名来自 README 公开契约）。"""
    time_range = query.start_date if not query.end_date else f"{query.start_date}~{query.end_date}"
    return {
        "time": time_range,
        "talker": query.talker,
        "limit": query.limit,
        "offset": query.offset,
        "format": "json",
    }


def _extract_sender(message: dict[str, Any]) -> str | None:
    value = pick_optional(message, SENDER_KEYS)
    if value is None:
        return None
    if isinstance(value, dict):
        nested = pick_nested_optional(
            value,
            (
                ("name",),
                ("nickname",),
                ("nickName",),
                ("remark",),
                ("displayName",),
                ("uin",),
                ("uid",),
                ("user", "name"),
                ("user", "uin"),
            ),
        )
        return None if nested is None else str(nested)
    return str(value)


def _extract_time(message: dict[str, Any]) -> str | None:
    value = pick_optional(message, TIME_KEYS)
    return None if value is None else str(value)


def _map_message(message: dict[str, Any], index: int, *, allow_missing_sender: bool) -> Utterance:
    raw = snippet_of(message)

    sender = _extract_sender(message)
    if sender is None and not allow_missing_sender:
        raise AdapterFieldError(
            "chatlog-sender",
            f"第 {index} 条消息缺少发送者字段（可能是系统消息）；"
            "确认字段名后请补进 SENDER_KEYS，或将 allow_missing_sender 设为 True",
            raw,
        )

    content_value = pick(message, CONTENT_KEYS, what=f"第 {index} 条消息的内容", snippet=raw)
    text = as_display_text(content_value, what=f"第 {index} 条消息的内容", snippet=raw).strip()
    if not text:
        raise AdapterFieldError("chatlog-content", f"第 {index} 条消息内容为空", raw)

    return Utterance(sender=sender, side=None, text=text, ts=_extract_time(message))


def fetch_chatlog(query: ChatlogQuery, *, http_get: Any | None = None) -> list[Utterance]:
    """请求 chatlog 并规范为 ``Utterance`` 列表。

    Args:
        query: 查询参数。
        http_get: 可注入的 HTTP 函数，签名 ``(url, params, timeout) -> response``。
            仅用于离线测试；生产路径下用 ``requests``。

    Raises:
        AdapterConfigError / AdapterHTTPError / AdapterFormatError / AdapterFieldError
    """
    query.validate()

    if http_get is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确提示
            raise AdapterConfigError("chatlog-dependency", "缺少 requests，请先 pip install -r requirements.txt") from exc
        session = requests.Session()

        def http_get(url: str, params: dict[str, Any], timeout: float) -> Any:  # type: ignore[misc]
            return session.get(url, params=params, timeout=timeout)

    url = query.base_url.rstrip("/") + ENDPOINT_PATH
    try:
        response = http_get(url, build_params(query), query.timeout)
    except Exception as exc:  # noqa: BLE001 - 网络异常统一包装
        raise AdapterHTTPError(
            "chatlog-request",
            f"请求 {url} 失败：{type(exc).__name__}: {exc}",
        ) from exc

    status = getattr(response, "status_code", None)
    body_text = getattr(response, "text", "") or ""
    if status is not None and status != 200:
        raise AdapterHTTPError(
            "chatlog-http",
            f"HTTP {status}；请确认 chatlog server 已启动、地址与 talker 正确",
            body_text,
        )

    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - 非 JSON 响应
        raise AdapterFormatError(
            "chatlog-json",
            "响应不是合法 JSON；请确认请求带上 format=json",
            body_text,
        ) from exc

    messages = find_message_list(payload, snippet=snippet_of(payload))
    if not messages:
        raise AdapterFormatError(
            "chatlog-empty",
            f"时间范围 {query.start_date} ~ {query.end_date or query.start_date} 内没有消息",
            snippet_of(payload),
        )

    return [
        _map_message(message, index, allow_missing_sender=query.allow_missing_sender)
        for index, message in enumerate(messages, start=1)
    ]
