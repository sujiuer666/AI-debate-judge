"""项目常量：维度定义、评分尺度、平局阈值、环境变量名。

来源与适配说明（对应第 1 / 2 步清单）
------------------------------------
- 维度元数据结构（key / label / weight）参考 Debatrix
  ``debatrix/core/common.py::DimensionInfo(name, weight, allow_tie, prompt)``，
  但**维度集合不同**：Debatrix 为 argument / source / language / general
  （``preset/example/config/manager.yml`` 中权重 3 / 2 / 2 / 0），本项目为
  logic / evidence / persuasion / rebuttal（默认等权 1.0）。
  因此这是「按同一思路重新定义」，不是复用 Debatrix 的四维。
- 平局阈值 0.5 的语义参考 syed-waleed-ahmed/LLM-as-Judge
  ``src/llm_judge/judge.py::_TIE_THRESHOLD``；但本项目作用在 0-10 的
  四维加权均值上，而非该仓库的 criterion 分数上。
- 本项目**不 import** 上述任何仓库的代码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# ---------------------------------------------------------------------------
# 四维定义
# ---------------------------------------------------------------------------

# 维度 key 必须在类型层面固定，才能让 Pydantic 校验「LLM 是否返回了未知维度」。
DimensionKey = Literal["logic", "evidence", "persuasion", "rebuttal"]


@dataclass(frozen=True, slots=True)
class Dimension:
    """单个评分维度的元数据。"""

    key: DimensionKey
    label: str
    weight: float
    definition: str


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="logic",
        label="逻辑性",
        weight=1.0,
        definition="论点是否成立：前提是否成立、推理链是否完整、内部是否自洽",
    ),
    Dimension(
        key="evidence",
        label="证据",
        weight=1.0,
        definition="是否提供可核验、与论点相关、且足以支撑论点强度的外部依据",
    ),
    Dimension(
        key="persuasion",
        label="说服力",
        weight=1.0,
        definition="表达是否清楚有力，能否让中立听众理解并接受其立场",
    ),
    Dimension(
        key="rebuttal",
        label="反驳质量",
        weight=1.0,
        definition="是否准确命中对方论证并说明其为何失效（本项目新增维度）",
    ),
)

DIMENSION_KEYS: tuple[DimensionKey, ...] = tuple(d.key for d in DIMENSIONS)
DIMENSION_BY_KEY: dict[DimensionKey, Dimension] = {d.key: d for d in DIMENSIONS}

# 运行时自检：Literal 声明的取值必须与 DIMENSIONS 完全一致，防止两处漂移。
assert tuple(get_args(DimensionKey)) == DIMENSION_KEYS, "DimensionKey 与 DIMENSIONS 不一致"
assert all(d.weight > 0 for d in DIMENSIONS), "权重必须为正数"

# ---------------------------------------------------------------------------
# 评分尺度与判决规则
# ---------------------------------------------------------------------------

SCORE_MIN: int = 0
SCORE_MAX: int = 10

#: |pro_total - con_total| <= 该值判平局（作用在 0-10 的四维加权均值上）。
TIE_THRESHOLD: float = 0.5

#: 是否每轮做「先正后反 / 先反后正」两次调用并取平均（缓解位置偏见）。
#: 3 轮 = 6 次 LLM 调用；设为 False 可省一半成本，但会失去对调平均。
AB_SWAP: bool = True

# ---------------------------------------------------------------------------
# 输入限制（用于截断，避免超长文本撑爆上下文）
# ---------------------------------------------------------------------------

MAX_INPUT_CHARS: int = 20_000
MAX_HISTORY_CHARS_PER_SIDE: int = 400

# ---------------------------------------------------------------------------
# LLM 调用参数
# ---------------------------------------------------------------------------

LLM_TEMPERATURE: float = 0.0
LLM_TIMEOUT_SECONDS: float = 60.0
#: tenacity 的总尝试次数（含首次调用）；仅对可重试的瞬时错误生效。
LLM_MAX_RETRIES: int = 3
LLM_JSON_MODE: bool = True  # 请求 response_format={"type": "json_object"}
LLM_MAX_TOKENS: int = 4000
#: LLM 输出违反契约时，把「错误 + 原文」喂回去要求重写的次数（Debatrix fix_prompt 思路）。
MAX_REPAIR_ATTEMPTS: int = 1

# ---------------------------------------------------------------------------
# 环境变量名（API Key 只允许从这里读取，全仓库唯一入口见 llm_client.py）
# ---------------------------------------------------------------------------

ENV_API_KEY: str = "DEBATE_JUDGE_API_KEY"
ENV_BASE_URL: str = "DEBATE_JUDGE_BASE_URL"
ENV_MODEL: str = "DEBATE_JUDGE_MODEL"

DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
DEFAULT_MODEL: str = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Prompt 版本号：写入 UI 与日志之外的展示位，便于排查「换了 prompt 后分数变了」。
# ---------------------------------------------------------------------------

PROMPT_VERSION: str = "0.1.0"
