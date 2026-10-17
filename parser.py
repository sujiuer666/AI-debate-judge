"""统一内部格式：把各种来源的辩论材料规范成「轮次 + 正方 + 反方」。

来源与适配说明（对应第 1 / 2 步清单）
------------------------------------
- 内部数据结构参考 Debatrix ``debatrix/core/common.py`` 的
  ``Speech`` / ``DebateInfo`` 抽象（``speech_order`` 决定发言顺序），
  但本项目只需要「每轮正/反两段」这一个最小结构，因此重构为
  ``Utterance`` / ``Round`` / ``Debate``，不引入其多辩手与 panel 概念。
- 文本解析逻辑（``正方：/反方：``）为**新建**，Debatrix 的输入是 YAML 数据集，
  不解析自由文本，因此这里没有可复用实现。
- 适配器只负责「HTTP / 文件 -> Utterance[]」；轮次配对逻辑**只在本文件**，
  两个适配器共用 ``group_utterances``，避免出现两套配对规则。

命名注意
--------
模块名 ``parser.py`` 是需求指定的。Python 3.10+ 已移除同名标准库模块
（旧 ``parser`` 模块），因此在本项目内使用 ``from parser import ...`` 不会与
标准库冲突；但不要在同一环境里安装仍会 ``import parser`` 的老旧依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from config import MAX_INPUT_CHARS

Side = Literal["pro", "con"]

SIDE_TEXT: dict[Side, str] = {"pro": "正方", "con": "反方"}


class ParseError(ValueError):
    """文本/消息无法规范成辩论结构。

    携带「失败环节 + 位置（行号或消息序号）+ 原文片段（截断）」，
    供 app.py 直接 st.error 展示；不含任何隐私以外的附加信息。
    """

    def __init__(
        self,
        stage: str,
        detail: str,
        *,
        where: str | None = None,
        snippet: str = "",
    ) -> None:
        self.stage = stage
        self.detail = detail
        self.where = where
        self.snippet = (snippet or "").strip()[:200]

        message = f"[{stage}] {detail}"
        if where:
            message = f"{message}（{where}）"
        if self.snippet:
            message = f"{message} | 原文片段：{self.snippet}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Utterance:
    """一条原始发言。适配器只产出这个，不产出 Round。"""

    sender: str | None
    side: Side | None
    text: str
    ts: str | None = None


@dataclass(frozen=True, slots=True)
class Round:
    round_no: int
    pro_text: str
    con_text: str


@dataclass(frozen=True, slots=True)
class Debate:
    motion: str | None
    rounds: tuple[Round, ...]
    warnings: tuple[str, ...] = ()

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)


# ---------------------------------------------------------------------------
# 文本解析
# ---------------------------------------------------------------------------

_SIDE_RE = re.compile(r"^\s*(?P<side>正方|反方|pro|con)\s*[：:]\s*(?P<rest>.*)$", re.IGNORECASE)
_MOTION_RE = re.compile(r"^\s*(?:辩题|题目|话题|motion)\s*[：:]\s*(?P<motion>.+)$", re.IGNORECASE)


def _normalize_side(raw: str) -> Side:
    return "pro" if raw.strip().lower() in ("正方", "pro") else "con"


def _pair_segments(segments: list[tuple[Side, str, str]]) -> tuple[Round, ...]:
    """把「正方/反方」段落按顺序配成轮次。

    ``segments`` 的元素为 ``(side, where, text)``，其中 ``where`` 是给用户看的
    位置描述（如「第 12 行」或「消息 #7」）。
    """
    if not segments:
        raise ParseError("pairing", "没有解析出任何发言段落")

    rounds: list[Round] = []
    index = 0
    while index < len(segments):
        side, where, text = segments[index]
        round_no = len(rounds) + 1

        if side != "pro":
            raise ParseError(
                "pairing",
                f"第 {round_no} 轮应当是「正方」发言在先，实际遇到「反方」",
                where=where,
                snippet=text,
            )

        if index + 1 >= len(segments):
            raise ParseError(
                "pairing",
                f"第 {round_no} 轮缺少「反方」发言（正方段落比反方多一段）",
                where=where,
                snippet=text,
            )

        next_side, next_where, next_text = segments[index + 1]
        if next_side != "con":
            raise ParseError(
                "pairing",
                f"第 {round_no} 轮的正方发言之后应紧跟「反方」，却再次遇到「正方」",
                where=next_where,
                snippet=next_text,
            )

        rounds.append(Round(round_no=round_no, pro_text=text, con_text=next_text))
        index += 2

    return tuple(rounds)


def parse_text(raw: str) -> Debate:
    """解析「正方：... / 反方：...」格式的自由文本。

    支持：
    - 可选首行 ``辩题：xxx``（也接受 ``题目/话题/motion``）；
    - ``正方:`` / ``反方:``（中英文冒号均可，``pro/con`` 亦可）；
    - 标记行之后的多行内容归入同一段，直到下一个标记行；
    - 空行被保留为段落内换行。

    不支持的写法会抛 ``ParseError``，而不是静默猜测。
    """
    text = (raw or "").replace("\ufeff", "").strip()
    if not text:
        raise ParseError("input", "输入为空，请粘贴包含「正方：」「反方：」的辩论文本")
    if len(text) > MAX_INPUT_CHARS:
        raise ParseError(
            "input",
            f"文本长度 {len(text)} 字，超过上限 {MAX_INPUT_CHARS} 字，请分段评判",
        )

    motion: str | None = None
    segments: list[tuple[Side, str, str]] = []

    current_side: Side | None = None
    current_where: str = ""
    buffer: list[str] = []

    def flush() -> None:
        if current_side is None:
            return
        body = "\n".join(buffer).strip()
        if not body:
            raise ParseError(
                "format",
                f"「{SIDE_TEXT[current_side]}：」标记后面没有内容",
                where=current_where,
            )
        segments.append((current_side, current_where, body))

    for line_no, line in enumerate(text.splitlines(), start=1):
        where = f"第 {line_no} 行"

        if current_side is None and motion is None:
            motion_match = _MOTION_RE.match(line)
            if motion_match:
                motion = motion_match.group("motion").strip()
                continue

        side_match = _SIDE_RE.match(line)
        if side_match:
            flush()
            current_side = _normalize_side(side_match.group("side"))
            current_where = where
            buffer = []
            rest = side_match.group("rest").strip()
            if rest:
                buffer.append(rest)
            continue

        if current_side is None:
            if line.strip():
                raise ParseError(
                    "format",
                    "在第一个「正方：」之前出现了无法归属的文本；"
                    "如需设置辩题请写成「辩题：...」，否则请把发言放在「正方：」之后",
                    where=where,
                    snippet=line,
                )
            continue

        buffer.append(line.rstrip())

    flush()

    if not segments:
        raise ParseError(
            "format",
            "没有找到任何「正方：」/「反方：」发言标记，请检查格式",
        )

    return Debate(motion=motion, rounds=_pair_segments(segments))


# ---------------------------------------------------------------------------
# 消息解析（供 adapters 复用）
# ---------------------------------------------------------------------------


def group_utterances(utterances: list[Utterance], *, pro_sender: str) -> Debate:
    """把按时间排序的消息列表配对成轮次。

    Args:
        utterances: 适配器产出的消息；``sender`` 为 ``None`` 的消息会被忽略
            （无法归属到任何一方，例如系统提示）。
        pro_sender: 用户指定的「正方」发送者标识；另一方即视为反方。
            绝不自动猜测哪一方是正方。

    Raises:
        ParseError: 发送者数量不为 2、未知发送者、或消息无法严格交替配对。
    """
    usable = [item for item in utterances if item.sender]
    ignored = len(utterances) - len(usable)
    if not usable:
        raise ParseError("sender", "没有任何可归属发言人的消息")

    senders: list[str] = []
    for item in usable:
        assert item.sender is not None
        if item.sender not in senders:
            senders.append(item.sender)

    if pro_sender not in senders:
        raise ParseError(
            "sender",
            f"指定的正方发言人 {pro_sender!r} 不在消息中出现，实际发言人：{senders}",
        )

    opponents = [sender for sender in senders if sender != pro_sender]
    if len(opponents) != 1:
        raise ParseError(
            "sender",
            f"检测到 {len(senders)} 个发言人 {senders}，无法确定反方；"
            "请先在导出/接口层过滤为双方对话",
        )

    # 若所有消息都带时间戳，按时间排序；否则保持原始顺序（接口/文件本身的顺序）。
    if all(item.ts for item in usable):
        usable = sorted(usable, key=lambda item: item.ts or "")

    segments: list[tuple[Side, str, str]] = []
    for position, item in enumerate(usable, start=1):
        side: Side = "pro" if item.sender == pro_sender else "con"
        body = (item.text or "").strip()
        if not body:
            continue
        where = f"消息 #{position}"
        if segments and segments[-1][0] == side:
            previous_side, previous_where, previous_text = segments[-1]
            segments[-1] = (previous_side, previous_where, f"{previous_text}\n{body}")
        else:
            segments.append((side, where, body))

    if not segments:
        raise ParseError("content", "所有消息内容均为空")

    warnings: list[str] = []
    if ignored:
        warnings.append(f"有 {ignored} 条消息因缺少发送者被忽略（多为系统提示）")

    return Debate(motion=None, rounds=_pair_segments(segments), warnings=tuple(warnings))
