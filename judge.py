"""评判引擎：Pydantic 校验 + A/B 对调平均 + 裁剪 + 胜方重算 + tenacity 重试。

来源与适配说明（第 1 / 2 步清单）
--------------------------------
- 核心算法语义直接沿用 syed-waleed-ahmed/LLM-as-Judge ``src/llm_judge/judge.py``：
  * ``_clamp``：把分数夹到区间内（本项目的 schema 已用 ``ge/le`` 约束，这里是防御性二层）；
  * ``_winner_from_scores``：胜方一律从数值重算，不信 LLM 自己报的 winner；
  * ``swapped`` 对调两遍后逐维平均，缓解位置偏见；
  * ``_TIE_THRESHOLD`` 平局阈值；
  * tenacity：``retry_if_exception_type`` + ``wait_exponential(multiplier=0.5, max=8)`` +
    ``stop_after_attempt`` + ``reraise=True``。
- 适配点：
  * 该仓库是「一对模型输出、整场一次判」；本项目是「正/反两方、**每一轮**各判一次」，
    因此对调发生在轮级，平均后按 side 标签回填（schema 里分数挂在 pro/con 下，不靠位置）；
  * 平局阈值作用在 **0-10 的四维加权均值** 上（与该仓库 criterion 分数同尺度），
    而不是直接作用在 0-40 的加权总分上；
  * 新增「修复重试」：输出不合契约时，把错误信息 + 原文喂回去要求重写
    （Debatrix ``JSONParser.parse`` 的 ``fix_prompt_template`` 思路），
    这在本项目是必要的——quote 必须逐字命中原文，比数值越界更容易失败。
- 硬约束 2：不写日志、不落盘；重试与中间结果只存在于内存。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import prompt as prompt_builder
from config import (
    AB_SWAP,
    DIMENSION_BY_KEY,
    DIMENSIONS,
    LLM_MAX_RETRIES,
    MAX_REPAIR_ATTEMPTS,
    SCORE_MAX,
    SCORE_MIN,
    TIE_THRESHOLD,
)
from llm_client import ChatResult, LLMConfig, LLMTransientError, chat_completion
from parser import Debate, SIDE_TEXT
from schemas import (
    JudgeValidationError,
    LLMRoundResponse,
    assert_quotes_verbatim,
    parse_llm_round_payload,
)

Winner = Literal["pro", "con", "tie"]

WEIGHT_SUM: float = sum(dimension.weight for dimension in DIMENSIONS)


# ---------------------------------------------------------------------------
# 纯函数（便于离线单测，不依赖网络）
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = SCORE_MIN, high: float = SCORE_MAX) -> float:
    """夹到 [low, high]（LLM-as-Judge ``_clamp`` 的同语义实现）。"""
    return max(low, min(high, value))


def _winner_from_scores(pro: float, con: float, *, threshold: float = TIE_THRESHOLD) -> Winner:
    """由数值决定胜方；差值在阈值内视为平局（不信 LLM 的 winner 字段）。"""
    if abs(pro - con) <= threshold:
        return "tie"
    return "pro" if pro > con else "con"


def _support_pct(pro: float, con: float) -> float:
    """正方支持度百分比；双方均为 0 时按 50% 处理（由调用方附加说明）。"""
    total = pro + con
    if total <= 0:
        return 50.0
    return pro / total * 100.0


# ---------------------------------------------------------------------------
# 结果结构（给 app.py 直接渲染）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DimensionResult:
    dimension: str
    label: str
    score: float
    quote: str
    missing_evidence: str | None = None


@dataclass(frozen=True, slots=True)
class SideResult:
    side: Literal["pro", "con"]
    dimensions: tuple[DimensionResult, ...]
    total: float  # 四维加权总分（默认等权，满分 40）

    @property
    def average(self) -> float:
        """折算到 0-10 的加权均值，用于跨轮比较与平局判定。"""
        return self.total / WEIGHT_SUM


@dataclass(frozen=True, slots=True)
class RoundJudgement:
    round_no: int
    pro: SideResult
    con: SideResult
    winner: Winner
    support_pct: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DebateJudgement:
    motion: str | None
    rounds: tuple[RoundJudgement, ...]
    pro_cum: float
    con_cum: float
    support_pct: float
    winner: Winner
    model: str
    ab_swapped: bool

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class DebateJudge:
    """按轮评判，并对每轮做 A/B 对调平均。"""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: Callable[[LLMConfig, Sequence[dict[str, str]]], ChatResult] | None = None,
        ab_swap: bool = AB_SWAP,
        max_attempts: int = LLM_MAX_RETRIES,
        retry_wait: Any | None = None,
        repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    ) -> None:
        """
        Args:
            config: 运行时配置；默认从环境变量读取（缺 Key 立即抛错）。
            client: 可注入的调用函数（命令行/测试用），默认 ``chat_completion``。
            ab_swap: 是否每轮做两遍对调平均。
            max_attempts: tenacity 总尝试次数（仅对 ``LLMTransientError``）。
            retry_wait: 覆盖 tenacity 的 wait 策略（测试用 ``wait_none()``）。
            repair_attempts: 输出不合契约时「喂回错误重写」的次数。
        """
        self.config = config or LLMConfig.from_env()
        self._client = client or chat_completion
        self.ab_swap = ab_swap
        self._repair_attempts = max(0, repair_attempts)
        self._model = self.config.model

        wait = retry_wait if retry_wait is not None else wait_exponential(multiplier=0.5, max=8)
        self._invoke = retry(
            reraise=True,
            retry=retry_if_exception_type(LLMTransientError),
            stop=stop_after_attempt(max(1, max_attempts)),
            wait=wait,
        )(self._invoke_once)

    # -- 对外 API ---------------------------------------------------------

    def judge_debate(
        self,
        debate: Debate,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> DebateJudgement:
        """逐轮评判并汇总。

        Args:
            debate: 统一格式的辩论。
            progress: 进度回调 ``(已完成轮数, 总轮数)``，供 Streamlit 进度条使用。
        """
        if not debate.rounds:
            raise ValueError("没有可评判的轮次")

        round_results: list[RoundJudgement] = []
        history: list[tuple[int, str, str]] = []

        for round_item in debate.rounds:
            round_results.append(
                self.judge_round(
                    motion=debate.motion,
                    round_no=round_item.round_no,
                    total_rounds=debate.total_rounds,
                    pro_text=round_item.pro_text,
                    con_text=round_item.con_text,
                    history=tuple(history),
                )
            )
            history.append((round_item.round_no, round_item.pro_text, round_item.con_text))
            if progress is not None:
                progress(len(round_results), debate.total_rounds)

        pro_cum = round(sum(item.pro.total for item in round_results), 2)
        con_cum = round(sum(item.con.total for item in round_results), 2)
        pro_avg = pro_cum / (WEIGHT_SUM * len(round_results))
        con_avg = con_cum / (WEIGHT_SUM * len(round_results))

        return DebateJudgement(
            motion=debate.motion,
            rounds=tuple(round_results),
            pro_cum=pro_cum,
            con_cum=con_cum,
            support_pct=round(_support_pct(pro_cum, con_cum), 1),
            winner=_winner_from_scores(pro_avg, con_avg),
            model=self._model,
            ab_swapped=self.ab_swap,
        )

    def judge_round(
        self,
        *,
        motion: str | None,
        round_no: int,
        total_rounds: int,
        pro_text: str,
        con_text: str,
        history: Sequence[tuple[int, str, str]] = (),
    ) -> RoundJudgement:
        """评判单轮：可选两遍（顺序对调）→ 逐维平均 → 重算胜方。"""
        orders: list[tuple[str, str]] = [("pro", "con")]
        if self.ab_swap:
            orders.append(("con", "pro"))

        responses: list[LLMRoundResponse] = []
        for order in orders:
            messages = prompt_builder.build_messages(
                motion=motion,
                round_no=round_no,
                total_rounds=total_rounds,
                pro_text=pro_text,
                con_text=con_text,
                order=order,  # type: ignore[arg-type]
                history=history,
            )
            responses.append(
                self._request_round(
                    messages,
                    round_no=round_no,
                    pro_text=pro_text,
                    con_text=con_text,
                )
            )

        return self._merge_round(
            responses,
            round_no=round_no,
            pro_text=pro_text,
            con_text=con_text,
        )

    # -- 内部实现 ---------------------------------------------------------

    def _invoke_once(self, messages: list[dict[str, str]]) -> str:
        result = self._client(self.config, messages)
        self._model = result.model or self._model
        return result.content

    def _request_round(
        self,
        messages: list[dict[str, str]],
        *,
        round_no: int,
        pro_text: str,
        con_text: str,
    ) -> LLMRoundResponse:
        """调用 + 解析 + 引用校验；不合契约时按 ``repair_attempts`` 喂回错误重写。"""
        conversation = list(messages)
        last_error: JudgeValidationError | None = None

        for attempt in range(self._repair_attempts + 1):
            raw = self._invoke(conversation)
            try:
                response = parse_llm_round_payload(raw, expected_round_no=round_no)
                assert_quotes_verbatim(response, pro_text=pro_text, con_text=con_text)
                return response
            except JudgeValidationError as exc:
                last_error = exc
                if attempt >= self._repair_attempts:
                    break
                conversation = list(messages) + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": prompt_builder.build_repair_message(str(exc))},
                ]

        assert last_error is not None
        raise JudgeValidationError(
            "repair-exhausted",
            f"连续 {self._repair_attempts + 1} 次输出不符合数据契约；最后一次错误：{last_error.detail}",
            last_error.snippet,
        ) from last_error

    @staticmethod
    def _pick_quote(
        responses: Sequence[LLMRoundResponse],
        *,
        side: Literal["pro", "con"],
        dimension: str,
        target: float,
    ) -> str:
        """取「分数最接近平均值」那一遍的引用，避免引用与展示分数不匹配。"""
        best_quote = ""
        best_distance: float | None = None
        for response in responses:
            side_scores = response.pro if side == "pro" else response.con
            item = side_scores.by_key()[dimension]
            distance = abs(float(item.score) - target)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_quote = item.quote
        return best_quote

    def _merge_round(
        self,
        responses: Sequence[LLMRoundResponse],
        *,
        round_no: int,
        pro_text: str,
        con_text: str,
    ) -> RoundJudgement:
        notes: list[str] = []

        sides: dict[str, SideResult] = {}
        for side in ("pro", "con"):
            dimensions: list[DimensionResult] = []
            total = 0.0

            for dimension in DIMENSIONS:
                scores: list[float] = []
                missing_candidates: list[str] = []

                for response in responses:
                    side_scores = response.pro if side == "pro" else response.con
                    item = side_scores.by_key()[dimension.key]
                    scores.append(float(item.score))
                    if item.missing_evidence:
                        missing_candidates.append(item.missing_evidence)

                merged_score = round(_clamp(sum(scores) / len(scores)), 2)
                quote = self._pick_quote(
                    responses, side=side, dimension=dimension.key, target=merged_score  # type: ignore[arg-type]
                )

                missing: str | None = None
                if merged_score == SCORE_MIN:
                    missing = (
                        missing_candidates[0]
                        if missing_candidates
                        else "两次输出均未给出可引用的原文依据"
                    )
                    quote = ""
                    notes.append(f"{SIDE_TEXT[side]}的「{dimension.label}」为 0 分：{missing}")
                elif not quote:
                    # 理论上被 schema 挡住；这里只做兜底，避免 UI 显示空引用却不解释。
                    notes.append(f"{SIDE_TEXT[side]}的「{dimension.label}」缺少引用（模型输出异常）")

                total += merged_score * dimension.weight
                dimensions.append(
                    DimensionResult(
                        dimension=dimension.key,
                        label=dimension.label,
                        score=merged_score,
                        quote=quote,
                        missing_evidence=missing,
                    )
                )

            sides[side] = SideResult(
                side=side,  # type: ignore[arg-type]
                dimensions=tuple(dimensions),
                total=round(total, 2),
            )

        pro, con = sides["pro"], sides["con"]
        support = _support_pct(pro.total, con.total)
        if pro.total + con.total <= 0:
            notes.append("双方本轮总分均为 0，支持度按 50% 处理")

        return RoundJudgement(
            round_no=round_no,
            pro=pro,
            con=con,
            winner=_winner_from_scores(pro.average, con.average),
            support_pct=round(support, 1),
            notes=tuple(notes),
        )

    # -- 展示辅助 ---------------------------------------------------------

    @staticmethod
    def describe_winner(winner: Winner) -> str:
        return {"pro": "正方占优", "con": "反方占优", "tie": "平局"}[winner]


__all__ = [
    "DebateJudge",
    "DebateJudgement",
    "DimensionResult",
    "RoundJudgement",
    "SideResult",
    "WEIGHT_SUM",
    "Winner",
    "_clamp",
    "_support_pct",
    "_winner_from_scores",
]
