"""离线测试：评判引擎（假 LLM，不联网、不需要 Key）。

运行（从 debate_judge/ 根目录）：::

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import pytest
from tenacity import wait_none

from config import ENV_API_KEY
from judge import (
    DebateJudge,
    _clamp,
    _support_pct,
    _winner_from_scores,
)
from llm_client import ChatResult, LLMConfig, LLMTransientError, MissingAPIKeyError
from parser import Debate, Round
from schemas import JudgeValidationError

CONFIG = LLMConfig(api_key="test-key-not-a-real-secret", model="fake-model")

PRO_TEXT = "我方认为应禁止一次性塑料。欧盟2021年一次性塑料指令数据显示，塑料餐具用量下降约30%。"
CON_TEXT = "对方混淆了禁令与回收。我方主张押金回收体系，德国2022年PET瓶回收率达98%。"

QUOTES: dict[str, dict[str, str]] = {
    "pro": {
        "logic": "我方认为应禁止一次性塑料",
        "evidence": "塑料餐具用量下降约30%",
        "persuasion": "我方认为应禁止一次性塑料",
        "rebuttal": "欧盟2021年一次性塑料指令",
    },
    "con": {
        "logic": "对方混淆了禁令与回收",
        "evidence": "PET瓶回收率达98%",
        "persuasion": "我方主张押金回收体系",
        "rebuttal": "对方混淆了禁令与回收",
    },
}

SIDE_ORDER = ("logic", "evidence", "persuasion", "rebuttal")

DEFAULT_PRO = {"logic": 8, "evidence": 8, "persuasion": 7, "rebuttal": 7}
DEFAULT_CON = {"logic": 6, "evidence": 7, "persuasion": 6, "rebuttal": 6}


def make_round_json(
    round_no: int = 1,
    pro: dict[str, int] | None = None,
    con: dict[str, int] | None = None,
    *,
    quote_override: tuple[str, str, str] | None = None,
) -> str:
    """构造一份合法（或按需破坏 quote 的）LLM 输出。"""
    pro_scores = {**DEFAULT_PRO, **(pro or {})}
    con_scores = {**DEFAULT_CON, **(con or {})}

    def build(side: str, scores: dict[str, int]) -> dict[str, Any]:
        dimensions = []
        for dimension in SIDE_ORDER:
            score = scores[dimension]
            quote = "" if score == 0 else QUOTES[side][dimension]
            if quote_override and side == quote_override[0] and dimension == quote_override[1]:
                quote = quote_override[2]
            dimensions.append(
                {"dimension": dimension, "score": score, "quote": quote}
                if score != 0
                else {
                    "dimension": dimension,
                    "score": 0,
                    "quote": "",
                    "missing_evidence": "本轮该维度找不到可引用的原文依据",
                }
            )
        return {"side": side, "dimensions": dimensions}

    return json.dumps(
        {"round_no": round_no, "pro": build("pro", pro_scores), "con": build("con", con_scores)},
        ensure_ascii=False,
    )


def round_no_from(messages: Sequence[dict[str, str]]) -> int:
    combined = "\n".join(message.get("content", "") for message in messages)
    match = re.search(r"`round_no` 填 (\d+)", combined)
    return int(match.group(1)) if match else 1


class FakeClient:
    """按顺序返回预设响应；也可抛预设异常。"""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, config: LLMConfig, messages: Sequence[dict[str, str]]) -> ChatResult:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("出现了未预设的额外 LLM 调用")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if callable(nxt):
            nxt = nxt(round_no_from(messages))
        return ChatResult(content=str(nxt), model="fake-model")


def make_judge(client: FakeClient, **kwargs: Any) -> DebateJudge:
    return DebateJudge(
        CONFIG,
        client=client,
        retry_wait=wait_none(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_clamp_and_support_helpers() -> None:
    assert _clamp(12) == 10
    assert _clamp(-3) == 0
    assert _clamp(7.5) == 7.5
    assert _support_pct(30, 25) == pytest.approx(54.5454, rel=1e-3)
    assert _support_pct(0, 0) == 50.0


def test_winner_from_scores() -> None:
    assert _winner_from_scores(7.0, 7.0) == "tie"
    assert _winner_from_scores(7.2, 7.0) == "tie"  # 差值 <= 0.5
    assert _winner_from_scores(8.0, 7.0) == "pro"
    assert _winner_from_scores(6.0, 7.0) == "con"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(MissingAPIKeyError):
        LLMConfig.from_env()


# ---------------------------------------------------------------------------
# 单轮：对调平均、引用选取、胜方重算
# ---------------------------------------------------------------------------


def test_round_average_and_winner() -> None:
    # 第一遍 pro.logic=8，第二遍（对调呈现）pro.logic=7 -> 平均 7.5
    client = FakeClient([make_round_json(pro={"logic": 8}), make_round_json(pro={"logic": 7})])
    judgement = make_judge(client).judge_round(
        motion="是否应全面禁止一次性塑料制品",
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )

    pro_logic = judgement.pro.dimensions[0]
    assert pro_logic.dimension == "logic"
    assert pro_logic.score == 7.5
    assert pro_logic.quote in PRO_TEXT  # 引用始终来自正方原文
    assert judgement.pro.total == 29.5  # 7.5 + 8 + 7 + 7
    assert judgement.con.total == 25.0  # 6 + 7 + 6 + 6
    assert judgement.winner == "pro"
    assert judgement.support_pct == 54.1
    assert len(client.calls) == 2


def test_ab_swap_disabled_uses_single_call() -> None:
    client = FakeClient([make_round_json()])
    judgement = make_judge(client, ab_swap=False).judge_round(
        motion=None,
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )
    assert len(client.calls) == 1
    assert judgement.pro.dimensions[0].score == 8.0


def test_zero_score_dimension_keeps_missing_evidence() -> None:
    client = FakeClient(
        [make_round_json(pro={"rebuttal": 0}), make_round_json(pro={"rebuttal": 0})]
    )
    judgement = make_judge(client).judge_round(
        motion=None,
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )
    rebuttal = judgement.pro.dimensions[3]
    assert rebuttal.score == 0
    assert rebuttal.quote == ""
    assert rebuttal.missing_evidence
    assert any("反驳质量" in note for note in judgement.notes)


def test_round_winner_is_recomputed_not_taken_from_llm() -> None:
    # 让「反方」分数更高，胜方必须从数值算出来
    client = FakeClient([make_round_json(pro={"logic": 3, "evidence": 3})])
    judgement = make_judge(client, ab_swap=False).judge_round(
        motion=None,
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )
    assert judgement.winner == "con"


# ---------------------------------------------------------------------------
# 修复重试 / tenacity
# ---------------------------------------------------------------------------


def test_repair_retry_recovers_from_bad_json() -> None:
    client = FakeClient(["这不是 JSON", make_round_json()])
    judgement = make_judge(client, ab_swap=False).judge_round(
        motion=None,
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )

    assert judgement.pro.total == 30.0
    assert len(client.calls) == 2
    repair_messages = client.calls[1]
    assert len(repair_messages) == 4  # system + user + assistant(坏输出) + user(修复指令)
    assert repair_messages[2]["role"] == "assistant"
    assert "未通过结构校验" in repair_messages[3]["content"]


def test_repair_retry_on_non_verbatim_quote() -> None:
    bad = make_round_json(quote_override=("pro", "logic", "我方强烈认为必须禁止"))
    client = FakeClient([bad, make_round_json()])
    judgement = make_judge(client, ab_swap=False).judge_round(
        motion=None,
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )
    assert judgement.pro.total == 30.0
    assert len(client.calls) == 2


def test_repair_exhausted_raises_with_stage() -> None:
    client = FakeClient(["坏输出1", "坏输出2"])
    with pytest.raises(JudgeValidationError) as excinfo:
        make_judge(client, repair_attempts=1).judge_round(
            motion=None,
            round_no=1,
            total_rounds=1,
            pro_text=PRO_TEXT,
            con_text=CON_TEXT,
        )
    assert "repair-exhausted" in str(excinfo.value)
    assert len(client.calls) == 2


def test_transient_error_is_retried_by_tenacity() -> None:
    client = FakeClient(
        [
            LLMTransientError("llm-http", "HTTP 429"),
            LLMTransientError("llm-http", "HTTP 503"),
            make_round_json(),
        ]
    )
    judgement = make_judge(client, ab_swap=False, max_attempts=3).judge_round(
        motion=None,
        round_no=1,
        total_rounds=1,
        pro_text=PRO_TEXT,
        con_text=CON_TEXT,
    )
    assert judgement.pro.total == 30.0
    assert len(client.calls) == 3


def test_permanent_error_is_not_retried() -> None:
    client = FakeClient(["坏输出", "坏输出"])
    with pytest.raises(JudgeValidationError):
        make_judge(client, repair_attempts=0).judge_round(
            motion=None,
            round_no=1,
            total_rounds=1,
            pro_text=PRO_TEXT,
            con_text=CON_TEXT,
        )
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# 整场：累计、支持度、最终胜方、进度回调
# ---------------------------------------------------------------------------


def test_judge_debate_two_rounds() -> None:
    debate = Debate(
        motion="是否应全面禁止一次性塑料制品",
        rounds=(
            Round(round_no=1, pro_text=PRO_TEXT, con_text=CON_TEXT),
            Round(round_no=2, pro_text=PRO_TEXT + "第二点。", con_text=CON_TEXT + "回应。"),
        ),
    )
    client = FakeClient(
        [
            make_round_json(1),
            make_round_json(1),
            make_round_json(2, pro={"evidence": 6}),
            make_round_json(2, pro={"evidence": 6}),
        ]
    )
    progress: list[tuple[int, int]] = []

    judgement = make_judge(client).judge_debate(debate, progress=lambda done, total: progress.append((done, total)))

    assert judgement.total_rounds == 2
    assert judgement.pro_cum == pytest.approx(30.0 + 28.0)
    assert judgement.con_cum == pytest.approx(50.0)
    assert judgement.support_pct == pytest.approx(53.7, rel=1e-2)
    assert judgement.winner == "pro"
    assert judgement.ab_swapped is True
    assert progress == [(1, 2), (2, 2)]
    assert len(client.calls) == 4


def test_judge_debate_rejects_empty_rounds() -> None:
    with pytest.raises(ValueError):
        make_judge(FakeClient([])).judge_debate(Debate(motion=None, rounds=()))
