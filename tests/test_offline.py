"""离线测试：不联网、不需要 API Key、不发真实 LLM 请求。

运行（从 debate_judge/ 根目录）：::

    python -m pytest tests -q

覆盖：
- ``parser.parse_text`` 的格式/配对/错误定位；
- ``parser.group_utterances`` 的发送者归属与严格交替校验；
- ``adapters.chatlog_adapter`` 的字段候选、HTTP/JSON/字段缺失三类错误；
- ``adapters.qq_export_adapter`` 的分段分发、非文本占位符、嵌套结构定位。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.base import AdapterFieldError, AdapterFormatError, AdapterHTTPError
from adapters.chatlog_adapter import ChatlogQuery, build_params, fetch_chatlog
from adapters.qq_export_adapter import load_qq_export
from parser import ParseError, Utterance, group_utterances, parse_text

FIXTURE = Path(__file__).parent / "fixtures" / "qq_export_synthetic.json"

DEBATE_TEXT = """\
辩题：是否应全面禁止一次性塑料制品

正方：我方认为应禁止一次性塑料。
欧盟2021年一次性塑料指令数据显示，塑料餐具用量下降约30%。
反方：对方混淆了禁令与回收，我方主张押金回收体系。

正方：第二，替代材料已成熟。
反方：替代材料的成本被低估。
"""


# ---------------------------------------------------------------------------
# parser.parse_text
# ---------------------------------------------------------------------------


def test_parse_text_structure() -> None:
    debate = parse_text(DEBATE_TEXT)

    assert debate.motion == "是否应全面禁止一次性塑料制品"
    assert debate.total_rounds == 2
    assert "欧盟2021年" in debate.rounds[0].pro_text
    assert debate.rounds[0].con_text.startswith("对方混淆了禁令与回收")
    assert debate.rounds[1].pro_text == "第二，替代材料已成熟。"
    assert debate.warnings == ()


def test_parse_text_accepts_halfwidth_colon_and_pro_alias() -> None:
    debate = parse_text("pro: A\ncon: B")
    assert debate.total_rounds == 1
    assert debate.rounds[0].pro_text == "A"
    assert debate.rounds[0].con_text == "B"


def test_parse_text_requires_pro_first() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_text("反方：B")
    assert "正方" in str(excinfo.value)


def test_parse_text_rejects_unpaired_last_segment() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_text("正方：A\n反方：B\n正方：C")
    message = str(excinfo.value)
    assert "第 2 轮" in message and "反方" in message


def test_parse_text_rejects_two_pro_in_a_row() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_text("正方：A\n正方：B")
    assert "第 1 轮" in str(excinfo.value)


def test_parse_text_rejects_unattributed_preamble() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_text("随便说点什么\n正方：A\n反方：B")
    assert "第 1 行" in str(excinfo.value)


def test_parse_text_rejects_empty_input() -> None:
    with pytest.raises(ParseError):
        parse_text("   \n  ")


# ---------------------------------------------------------------------------
# parser.group_utterances
# ---------------------------------------------------------------------------


def test_group_utterances_merges_consecutive_and_pairs() -> None:
    utterances = [
        Utterance(sender="甲", side=None, text="A1", ts="2026-01-01 10:00:00"),
        Utterance(sender="乙", side=None, text="B1", ts="2026-01-01 10:00:30"),
        Utterance(sender="乙", side=None, text="B1补充", ts="2026-01-01 10:00:40"),
        Utterance(sender="甲", side=None, text="A2", ts="2026-01-01 10:01:00"),
        Utterance(sender="乙", side=None, text="B2", ts="2026-01-01 10:01:30"),
    ]

    debate = group_utterances(utterances, pro_sender="甲")

    assert debate.total_rounds == 2
    assert debate.rounds[0].pro_text == "A1"
    assert debate.rounds[0].con_text == "B1\nB1补充"
    assert debate.rounds[1].pro_text == "A2"
    assert debate.motion is None


def test_group_utterances_ignores_senderless_with_warning() -> None:
    utterances = [
        Utterance(sender=None, side=None, text="[系统] 某人加入群聊"),
        Utterance(sender="甲", side=None, text="A1"),
        Utterance(sender="乙", side=None, text="B1"),
    ]

    debate = group_utterances(utterances, pro_sender="甲")

    assert debate.total_rounds == 1
    assert debate.warnings and "1 条消息" in debate.warnings[0]


def test_group_utterances_rejects_three_senders() -> None:
    utterances = [
        Utterance(sender="甲", side=None, text="A"),
        Utterance(sender="乙", side=None, text="B"),
        Utterance(sender="丙", side=None, text="C"),
    ]

    with pytest.raises(ParseError) as excinfo:
        group_utterances(utterances, pro_sender="甲")
    assert "3 个发言人" in str(excinfo.value)


def test_group_utterances_rejects_unknown_pro_sender() -> None:
    utterances = [Utterance(sender="甲", side=None, text="A")]
    with pytest.raises(ParseError) as excinfo:
        group_utterances(utterances, pro_sender="不存在的人")
    assert "不在消息中" in str(excinfo.value)


# ---------------------------------------------------------------------------
# chatlog adapter（注入假 HTTP，不联网）
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        text: str | None = None,
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)

    def json(self) -> Any:
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def _client(payload: Any = None, **kwargs: Any):
    calls: list[tuple[str, dict[str, Any], float]] = []

    def http_get(url: str, params: dict[str, Any], timeout: float) -> FakeResponse:
        calls.append((url, params, timeout))
        return FakeResponse(payload, **kwargs)

    http_get.calls = calls  # type: ignore[attr-defined]
    return http_get


def _query(**overrides: Any) -> ChatlogQuery:
    base = {
        "base_url": "http://127.0.0.1:5030",
        "talker": "wxid_demo",
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
        "limit": 50,
    }
    base.update(overrides)
    return ChatlogQuery(**base)


def test_chatlog_build_params() -> None:
    params = build_params(_query())
    assert params["time"] == "2026-01-01~2026-01-02"
    assert params["format"] == "json"
    assert params["talker"] == "wxid_demo"

    single = build_params(_query(end_date=None))
    assert single["time"] == "2026-01-01"


def test_chatlog_alternative_field_names() -> None:
    payload = {
        "data": {
            "messages": [
                {"senderName": "甲", "text": "A1", "time": "2026-01-01 10:00:00"},
                {"sender_name": "乙", "content": "B1", "timestamp": 1767225630},
            ]
        }
    }
    client = _client(payload)

    utterances = fetch_chatlog(_query(), http_get=client)

    assert [item.sender for item in utterances] == ["甲", "乙"]
    assert [item.text for item in utterances] == ["A1", "B1"]
    assert utterances[1].ts == "1767225630"

    url, params, _timeout = client.calls[0]  # type: ignore[attr-defined]
    assert url.endswith("/api/v1/chatlog")
    assert params["limit"] == 50


def test_chatlog_missing_content_raises_with_candidates_and_snippet() -> None:
    payload = {"messages": [{"senderName": "甲", "unknownField": "x"}]}
    client = _client(payload)

    with pytest.raises(AdapterFieldError) as excinfo:
        fetch_chatlog(_query(), http_get=client)

    message = str(excinfo.value)
    assert "已尝试字段名" in message
    assert "原始片段" in message
    assert "unknownField" in message


def test_chatlog_http_error() -> None:
    client = _client({"error": "boom"}, status_code=500, text='{"error":"boom"}')
    with pytest.raises(AdapterHTTPError) as excinfo:
        fetch_chatlog(_query(), http_get=client)
    assert "HTTP 500" in str(excinfo.value)


def test_chatlog_non_json_response() -> None:
    client = _client(None, json_error=True, text="<html>oops</html>")
    with pytest.raises(AdapterFormatError) as excinfo:
        fetch_chatlog(_query(), http_get=client)
    assert "format=json" in str(excinfo.value)


def test_chatlog_empty_message_list_raises_format_error() -> None:
    client = _client({"data": {"messages": []}})
    with pytest.raises(AdapterFormatError):
        fetch_chatlog(_query(), http_get=client)


# ---------------------------------------------------------------------------
# QQ 导出 adapter（合成 fixture，非真实 schema）
# ---------------------------------------------------------------------------


def test_qq_fixture_segments_and_placeholders() -> None:
    utterances = load_qq_export(FIXTURE.read_bytes())

    assert len(utterances) == 5
    assert [item.sender for item in utterances[:2]] == ["正方甲", "反方乙"]

    reply_message = utterances[1].text
    assert "[引用消息]" in reply_message
    assert "对方混淆了禁令与回收。" in reply_message
    assert "[图片]" in reply_message

    assert utterances[0].ts == "2026-01-01 10:00:00"


def test_qq_fixture_groups_into_rounds() -> None:
    utterances = load_qq_export(FIXTURE.read_bytes())

    debate = group_utterances(utterances, pro_sender="正方甲")

    assert debate.total_rounds == 2
    assert debate.rounds[0].pro_text.startswith("我方认为应禁止一次性塑料")
    assert "押金回收体系" in debate.rounds[0].con_text
    assert debate.rounds[1].pro_text == "第二，替代材料已成熟。"
    assert debate.rounds[1].con_text == "替代材料的成本被低估。"


def test_qq_alternative_shape() -> None:
    payload = {
        "data": {
            "items": [
                {"messageType": "text", "senderName": "X", "text": "a"},
                {"messageType": "text", "senderName": "Y", "text": "b"},
            ]
        }
    }
    utterances = load_qq_export(json.dumps(payload, ensure_ascii=False))
    assert [(item.sender, item.text) for item in utterances] == [("X", "a"), ("Y", "b")]


def test_qq_non_text_only_message_becomes_placeholder() -> None:
    payload = {"messages": [{"senderName": "甲", "content": [{"type": "image", "data": {"file": "x.jpg"}}]}]}
    utterances = load_qq_export(json.dumps(payload, ensure_ascii=False))
    assert utterances[0].text == "[图片]"


def test_qq_missing_content_raises_with_raw_snippet() -> None:
    payload = {"messages": [{"senderName": "甲", "type": "text"}]}
    with pytest.raises(AdapterFieldError) as excinfo:
        load_qq_export(json.dumps(payload, ensure_ascii=False))
    message = str(excinfo.value)
    assert "第 1 条消息" in message
    assert "原始片段" in message


def test_qq_bad_json_raises() -> None:
    with pytest.raises(AdapterFormatError) as excinfo:
        load_qq_export("{not json")
    assert "合法 JSON" in str(excinfo.value)
