"""Pydantic 数据契约 + LLM 输出解析。

来源与适配说明（对应第 1 / 2 步清单）
------------------------------------
- 数据模型结构与「严格校验」思路提取自 syed-waleed-ahmed/LLM-as-Judge
  ``src/llm_judge/models.py``（``Criterion`` / ``CriterionScore`` / ``JudgeResult``）
  与 Debatrix ``debatrix/panel/interface/common/verdict.py``
  （``Score`` / ``JudgmentAndScore`` / ``Winner``）。
- 适配点：
  * 字段 ``output_a / output_b`` -> ``pro / con``（本项目是正反方，不是两个模型输出）；
  * 新增 ``quote`` 与 ``missing_evidence``：Debatrix 与 LLM-as-Judge **都没有**这两个字段，
    它们是本项目的硬要求（每维一句原文引用；无引用则 0 分），属新建；
  * 不再保留 LLM 产出的 winner：胜方一律由 judge.py 从数值重算（LLM-as-Judge 的
    ``_winner_from_scores`` 思路），避免 verdict 与分数矛盾。
- JSON 提取用 Debatrix ``panel/interface/common/parser.py::JSONParser.parse``
  的正则抠取 + Pydantic 校验思路（正则 ``\\{.*\\}``，DOTALL）。
- 置信度：高（两个仓库的模型定义均为源码实读）；本项目字段为新建，无编造风险。

运行环境假设：从项目根目录运行（``streamlit run app.py`` / ``pytest``），
因此使用绝对导入 ``from config import ...``，项目根不是包。
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from config import DIMENSION_KEYS, SCORE_MAX, SCORE_MIN

SNIPPET_LIMIT: int = 200


class JudgeValidationError(ValueError):
    """LLM 输出不符合数据契约。

    携带「失败环节 + 截断后的原始片段」，供 app.py 直接 st.error 给用户看；
    异常信息里不包含 API Key 等敏感内容。
    """

    def __init__(self, stage: str, detail: str, snippet: str = "") -> None:
        self.stage = stage
        self.detail = detail
        self.snippet = (snippet or "")[:SNIPPET_LIMIT]
        message = f"[{stage}] {detail}"
        if self.snippet:
            message = f"{message} | 原始片段：{self.snippet}"
        super().__init__(message)


class DimensionScore(BaseModel):
    """某一方在某一维度上的得分，以及该分数的唯一依据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: Literal["logic", "evidence", "persuasion", "rebuttal"]
    score: int = Field(ge=SCORE_MIN, le=SCORE_MAX, description="0-10 整数")
    quote: str = Field(default="", description="本轮该方原文的逐字引用，score>0 时必填")
    missing_evidence: str | None = Field(
        default=None,
        description="score=0 时必填：说明为何找不到可引用原文",
    )

    @model_validator(mode="after")
    def _evidence_rule(self) -> "DimensionScore":
        quote = self.quote.strip()
        missing = (self.missing_evidence or "").strip()

        if self.score == SCORE_MIN:
            if not missing:
                raise ValueError(
                    f"维度 {self.dimension} 得 0 分时必须填写 missing_evidence 说明原因"
                )
        else:
            if not quote:
                raise ValueError(
                    f"维度 {self.dimension} 得 {self.score} 分时必须在 quote 中给出原文引用"
                )
            if missing:
                raise ValueError(
                    f"维度 {self.dimension} 得 {self.score} 分时 missing_evidence 必须为 null"
                )
        return self


class SideScores(BaseModel):
    """一方在本轮的四个维度得分。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    side: Literal["pro", "con"]
    dimensions: list[DimensionScore] = Field(
        min_length=len(DIMENSION_KEYS),
        max_length=len(DIMENSION_KEYS),
    )

    @model_validator(mode="after")
    def _dimensions_complete(self) -> "SideScores":
        keys = [item.dimension for item in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError(f"side={self.side} 的 dimensions 存在重复维度：{keys}")
        if set(keys) != set(DIMENSION_KEYS):
            raise ValueError(
                f"side={self.side} 的 dimensions 必须恰好是 {list(DIMENSION_KEYS)}，实际为 {keys}"
            )
        return self

    def by_key(self) -> dict[str, DimensionScore]:
        return {item.dimension: item for item in self.dimensions}


class LLMRoundResponse(BaseModel):
    """LLM 针对「一轮」必须返回的完整结构（这是 prompt 里注入的 schema 来源）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_no: int = Field(ge=1)
    pro: SideScores
    con: SideScores

    @model_validator(mode="after")
    def _sides_labelled(self) -> "LLMRoundResponse":
        if self.pro.side != "pro":
            raise ValueError('pro 字段内的 side 必须为 "pro"')
        if self.con.side != "con":
            raise ValueError('con 字段内的 side 必须为 "con"')
        return self


# ---------------------------------------------------------------------------
# 解析与二次校验
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL | re.IGNORECASE)


def parse_llm_round_payload(raw: str, *, expected_round_no: int | None = None) -> LLMRoundResponse:
    """把 LLM 原始文本解析为 ``LLMRoundResponse``。

    - 先用正则抠出最外层 JSON 对象（模型常带 markdown 围栏或前后闲话）；
    - 再用 Pydantic 严格校验；
    - 任何失败都转成 ``JudgeValidationError``，附带截断后的原始片段。
    """
    if not raw or not raw.strip():
        raise JudgeValidationError("json-parse", "LLM 返回了空内容")

    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise JudgeValidationError("json-parse", "未在返回内容中找到 JSON 对象", raw)

    try:
        data: Any = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise JudgeValidationError(
            "json-parse", f"JSON 语法错误：{exc.msg}（位置 {exc.pos}）", match.group()
        ) from exc

    try:
        response = LLMRoundResponse.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
        raise JudgeValidationError(
            "schema-validate",
            f"结构校验失败（{exc.error_count()} 处）；首个问题在 {location}：{first.get('msg', '')}",
            match.group(),
        ) from exc

    if expected_round_no is not None and response.round_no != expected_round_no:
        raise JudgeValidationError(
            "round-check",
            f"LLM 回报的轮次为 {response.round_no}，与请求的第 {expected_round_no} 轮不一致",
            match.group(),
        )

    return response


def assert_quotes_verbatim(response: LLMRoundResponse, *, pro_text: str, con_text: str) -> None:
    """校验每一处 quote 确实是该方本轮原文的子串。

    Debatrix 与 LLM-as-Judge 都没有这一步；它把「引用必须是原文」从 prompt 的
    口头约定变成可执行断言，是本项目新增的防幻觉措施。
    """
    texts = {"pro": pro_text, "con": con_text}

    for side_scores in (response.pro, response.con):
        source = texts[side_scores.side]
        for item in side_scores.dimensions:
            if not item.quote:
                continue
            if item.quote not in source:
                raise JudgeValidationError(
                    "quote-check",
                    f"第 {response.round_no} 轮 {side_scores.side} 的 {item.dimension} 维度"
                    f"引用并非原文逐字子串：{item.quote[:80]!r}",
                    source,
                )


def public_schema() -> dict[str, Any]:
    """对外暴露的 JSON Schema（prompt.py 用它注入，避免手写 schema 与模型漂移）。"""
    return LLMRoundResponse.model_json_schema()
