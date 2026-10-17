"""四维裁判 prompt：system 模板 + 单轮 user 模板 + 校准锚点。

来源与适配说明（对应第 1 / 2 步清单）
------------------------------------
- 结构提取自 Debatrix（源码实读，第 1 步已核对）：
  * ``preset/example/config/manager.yml`` -> ``dimensions[].prompt``
    的 ``judge_debate / analyze_speech / use_previous / judge_by_analysis`` 四段式；
  * ``preset/example/config/panel.yml`` -> ``judge_config.templates`` 的
    system/human 分段与 ``{{ }}`` 变量；``parser_config.schema_prompt_template``
    把 ``model_json_schema()`` 注入 prompt 的做法（本项目照此实现，见 ``SYSTEM_PROMPT``）。
- 适配方式：
  * Debatrix 是「每条 speech 一次 UPDATE 调用 + 全场一次 JUDGE 调用 + VerdictExtractor
    再调一次 LLM 把评语抽成分数」，本项目压成「**每轮一次调用**直接产出结构化四维分数」，
    因此保留了它的评分口径措辞，但不保留它的多段调用链；
  * Debatrix 的 ``use_previous``（把前序分析串进 prompt）在本项目中降级为
    「把前几轮原文截断后作为语境」——即保留时间顺序，但不做分析链式迭代；
  * Debatrix 分数是 1-10 整数、无引用约束、winner 由 LLM 直判；本项目改为 0-10、
    每维必须引用原文、胜方由数值重算。这些都是**新增**，不能声称复用。
- 校准锚点（0-2 / 3-4 / 5-6 / 7-8 / 9-10）与 ``missing_evidence`` 规则是本项目新增。
- 置信度：结构 = 高（Debatrix 源码实读）；措辞 = 本仓库原创。

设计约束
--------
- 本模块**不 import** ``parser`` / ``judge``，避免循环依赖：入参全部是字符串与元组。
- JSON schema 唯一来源是 ``schemas.public_schema()``，本文件不重复手写字段表。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from config import (
    DEFAULT_MODEL,
    DIMENSIONS,
    MAX_HISTORY_CHARS_PER_SIDE,
    PROMPT_VERSION,
    SCORE_MAX,
    SCORE_MIN,
)
from schemas import public_schema

Side = Literal["pro", "con"]

SIDE_LABEL: dict[str, str] = {"pro": "正方（Pro）", "con": "反方（Con）"}

# ---------------------------------------------------------------------------
# 校准锚点：每个维度 5 档行为描述。
# 目的：对抗 LLM 把分数堆在 7-8 的「安全分」倾向。档位描述必须是可观察的行为，
# 不能是「较好 / 一般」这类同义反复。
# ---------------------------------------------------------------------------

ANCHORS: dict[str, tuple[tuple[str, str], ...]] = {
    "logic": (
        ("0–2", "没有可辨认的论点，或论点与辩题无关；前提与结论之间找不到推理步骤；段落之间自相矛盾。"),
        ("3–4", "有论点但基本是断言；关键推理步骤缺失、偷换概念或循环论证；存在明显内部矛盾。"),
        ("5–6", "论点清楚、主干推理成立，但依赖未加说明的假设；存在少量逻辑跳跃。"),
        ("7–8", "论点—前提—结论链条完整且可追溯；能区分相关与因果；仅有细微瑕疵。"),
        ("9–10", "结构严密，主动限定前提与适用范围，并预判、封堵对方可能的反驳路径；无实质漏洞。"),
    ),
    "evidence": (
        ("0–2", "没有外部依据，或所谓依据与论点无关、无法核验、明显编造。"),
        ("3–4", "只有零散例子或个人感受，缺少来源、时间、口径，无法支撑论点强度。"),
        ("5–6", "至少有一条具体、相关、可核验的证据，但覆盖不足，或未说明其代表性。"),
        ("7–8", "多条相关证据，来源与机制说明较完整，足以支撑主要论点。"),
        ("9–10", "证据分层（数据 + 机制 + 反例排查），并说明证据的局限与适用边界。"),
    ),
    "persuasion": (
        ("0–2", "表达破碎、指代不清，读者无法复述其主张；或通篇情绪宣泄。"),
        ("3–4", "主张可以看懂，但结构松散、重点不明，难以被带动。"),
        ("5–6", "结构完整、语言清楚，有一定感染力，但缺少记忆点或价值升华。"),
        ("7–8", "取舍与节奏清晰，能抓住核心冲突，例子与价值陈述配合得当。"),
        ("9–10", "有强记忆点，把复杂问题讲成听众可感知的利害，且不靠夸张与人身攻击。"),
    ),
    "rebuttal": (
        ("0–2", "完全没有回应对方论点，或只是否定而不给理由。"),
        ("3–4", "提到对方论点但误述、稻草人化，攻击点不触及对方论证核心。"),
        ("5–6", "准确复述对方主要论点并给出针对性质疑，但没完成「为何因此不成立」。"),
        ("7–8", "直接攻击对方前提或证据，并说明该攻击如何导致其论点失效；能区分主次战场。"),
        ("9–10", "逐条拆解并按重要性排序攻击，迫使对方转入防守；同时修补自身被攻击处（攻防一体）。"),
    ),
}

_CALIBRATION_HINT = """\
## 校准方法（必须照做）

1. 先为该维度**选定档位**（0–2 / 3–4 / 5–6 / 7–8 / 9–10），再在档位内做最小幅度调整；
   档位判断优先于「感觉上还不错」。
2. **7–8 不是默认分**。给 7–8 的前提是：你能指出该方这一维具体做对了什么，
   同时也能指出一处仍然存在的、可被对手利用的瑕疵。
3. 表现中规中矩、无突出优点也无明显错误时，应落在 **5–6**。
4. **9–10 需要同时满足**：该维度在本轮明显优于一般竞技辩论发言，且你能在 quote 中
   引出对应原文作为证据。
5. 四个维度彼此独立打分，不要因为总分想给某一方领先而反向调整各维分数。"""


def _render_anchors() -> str:
    """把 ANCHORS 渲染成 prompt 中的文本块。"""
    blocks: list[str] = ["## 各维度校准锚点", ""]
    for dimension in DIMENSIONS:
        blocks.append(f"### {dimension.label}（{dimension.key}）")
        blocks.append(f"定义：{dimension.definition}。")
        for band, behavior in ANCHORS[dimension.key]:
            blocks.append(f"- {band} 分：{behavior}")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def _render_schema() -> str:
    """把 Pydantic 生成的 JSON Schema 渲染进 prompt（Debatrix schema_prompt 的做法）。"""
    schema_text = json.dumps(public_schema(), ensure_ascii=False, indent=2)
    return (
        "## 输出格式（严格 JSON）\n\n"
        "只输出一个 JSON 对象，不要 markdown 代码围栏、不要前后解释。"
        "字段名、类型、取值范围必须与下面的 schema 完全一致：\n\n"
        f"```json\n{schema_text}\n```\n\n"
        "字段含义补充：\n"
        "- `round_no`：轮次编号，必须等于用户消息中给出的当前轮次。\n"
        "- `pro` / `con`：两方的评分容器，各自 `side` 固定为 `\"pro\"` / `\"con\"`。\n"
        "- `dimensions`：恰好 4 项，`dimension` 取值只能是 "
        "`logic` / `evidence` / `persuasion` / `rebuttal`，每个维度只能出现一次。\n"
        f"- `score`：{SCORE_MIN}–{SCORE_MAX} 的整数。\n"
        "- `quote`：从**该方本轮原文**中逐字复制的片段；`score > 0` 时必填，不得改写。\n"
        "- `missing_evidence`：`score = 0` 时必填，说明找不到可引用依据的原因；其余情况为 `null`。"
    )


def _build_system_prompt() -> str:
    dimension_list = "\n".join(
        f"- **{d.label}（{d.key}）**：{d.definition}。" for d in DIMENSIONS
    )
    return f"""\
你是一名资深竞技辩论裁判，熟悉 Policy / BP 赛制的评判标准，现在只负责评判「一轮」交锋。

## 任务

针对用户给出的单轮材料，分别给正方（Pro）与反方（Con）按四个维度打分：
每个维度 {SCORE_MIN}–{SCORE_MAX} 分，且每一维都必须给出该方本轮原文的逐字引用作为**唯一依据**。

## 四个维度

{dimension_list}

## 硬性规则（违反任意一条，本次输出即作废）

1. 每一维分数必须附带 `quote`：该方在**本轮**原文中的逐字片段。不得改写、不得概括、不得引用对方或历史轮次的文字。
2. 若某一维在该方本轮原文中确实找不到可引用的依据，该维必须给 {SCORE_MIN} 分，`quote` 留空，
   并在 `missing_evidence` 中写明原因（例如「本轮未使用任何外部证据」）。
3. `score` 必须是 {SCORE_MIN}–{SCORE_MAX} 的整数，禁止小数。
4. 只依据本轮材料评分；历史轮次仅用于判断「本轮反驳是否命中对方此前论点」，不得重复计分。
5. 评分与材料呈现顺序无关：不要因为某一方先出现而给更高分。
6. 只输出 JSON 对象本身，不要任何解释文字。

{_CALIBRATION_HINT}

{_render_anchors()}

{_render_schema()}
"""


SYSTEM_PROMPT: str = _build_system_prompt()


# ---------------------------------------------------------------------------
# user prompt
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"……（已截断，原文共 {len(text)} 字）"


def _render_history(history: Sequence[tuple[int, str, str]]) -> str:
    if not history:
        return ""
    lines: list[str] = ["## 历史轮次（仅作语境参考，不重复计分）", ""]
    for round_no, pro_text, con_text in history:
        lines.append(f"### 第 {round_no} 轮")
        lines.append(f"正方：{_truncate(pro_text, MAX_HISTORY_CHARS_PER_SIDE)}")
        lines.append(f"反方：{_truncate(con_text, MAX_HISTORY_CHARS_PER_SIDE)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_user_prompt(
    *,
    motion: str | None,
    round_no: int,
    total_rounds: int,
    pro_text: str,
    con_text: str,
    order: tuple[Side, Side] = ("pro", "con"),
    history: Sequence[tuple[int, str, str]] = (),
) -> str:
    """构造单轮 user prompt。

    Args:
        motion: 辩题，可为 None。
        round_no: 当前轮次，从 1 开始。
        total_rounds: 总轮数，仅用于让模型知道所处阶段。
        pro_text: 正方本轮发言原文。
        con_text: 反方本轮发言原文。
        order: 呈现顺序；``("con", "pro")`` 即 A/B 对调的第二遍，用于消除位置偏见。
        history: 之前轮次的 ``(round_no, pro_text, con_text)``，仅作语境参考。
    """
    if set(order) != {"pro", "con"}:
        raise ValueError(f"order 必须恰好包含 pro 与 con，实际为 {order!r}")

    texts: dict[str, str] = {"pro": pro_text, "con": con_text}

    parts: list[str] = []
    if motion:
        parts.append(f"## 辩题\n\n{motion}")
    parts.append(f"## 当前轮次\n\n第 {round_no} 轮 / 共 {total_rounds} 轮")

    rendered_history = _render_history(history)
    if rendered_history:
        parts.append(rendered_history)

    parts.append("## 本轮材料（按呈现顺序编号）")
    for index, side in enumerate(order, start=1):
        parts.append(f"### 材料 {index}｜{SIDE_LABEL[side]}\n\n{_truncate(texts[side], 10000)}")

    parts.append(
        "## 要求\n\n"
        f"请对**正方**与**反方**分别按四个维度给出 {SCORE_MIN}–{SCORE_MAX} 整数分数，"
        "每一维附上该方本轮原文的逐字引用；找不到依据的维度给 0 分并在 "
        "`missing_evidence` 中说明。"
        f"`round_no` 填 {round_no}。只输出 JSON。"
    )
    return "\n\n".join(parts)


def build_messages(
    *,
    motion: str | None,
    round_no: int,
    total_rounds: int,
    pro_text: str,
    con_text: str,
    order: tuple[Side, Side] = ("pro", "con"),
    history: Sequence[tuple[int, str, str]] = (),
    model: str = DEFAULT_MODEL,
) -> list[dict[str, str]]:
    """返回 chat/completions 所需的 messages 列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                motion=motion,
                round_no=round_no,
                total_rounds=total_rounds,
                pro_text=pro_text,
                con_text=con_text,
                order=order,
                history=history,
            ),
        },
    ]


def build_repair_message(error_detail: str) -> str:
    """输出不合契约时的「修复」指令（Debatrix ``fix_prompt_template`` 的适配版）。"""
    return (
        "你上一次的输出未通过结构校验，不能使用。\n\n"
        f"错误信息：\n{error_detail}\n\n"
        "请严格按 system 中给出的 JSON Schema 重新输出**完整**的 JSON 对象，并注意：\n"
        "1. `quote` 必须从该方本轮原文中**逐字复制**（不要改写、不要合并两句）；\n"
        "2. `score = 0` 的维度必须填写 `missing_evidence`，且 `quote` 留空；\n"
        "3. `dimensions` 必须恰好包含 4 项，每个维度只出现一次；\n"
        "4. 只输出 JSON，不要任何解释。"
    )


def prompt_meta() -> dict[str, str]:
    """给 UI 展示用的 prompt 元信息（便于复现：同一文本 + 同一版本才可对比）。"""
    return {"prompt_version": PROMPT_VERSION, "model_default": DEFAULT_MODEL}


__all__ = [
    "ANCHORS",
    "SIDE_LABEL",
    "SYSTEM_PROMPT",
    "build_messages",
    "build_repair_message",
    "build_user_prompt",
    "prompt_meta",
]
