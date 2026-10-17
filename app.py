"""Streamlit 界面：AI 辩论裁判 MVP。

来源与适配说明（第 1 / 2 步清单）
--------------------------------
- 布局节奏（每轮 → 双列 → 色块 → 末尾汇总）参考 rd-serendipity/ai-debate-arena
  的 ``app.py``；但该仓库实际使用的是 ``st.columns`` + 内联 HTML 色块，
  **没有** ``st.container`` / ``st.metric`` / ``st.progress``。本项目这三个组件
  属新建（第 1 步已纠正该事实）。
- 错误提示：三类失败（解析 / 适配器 / LLM 输出校验）分别捕获并 ``st.error``，
  文案包含失败环节与原始片段（片段截断逻辑在各自的异常里）。
- 硬约束 3：两个适配器默认关闭（``st.session_state["enable_chatlog"] = False``、
  ``["enable_qq_upload"] = False``），只有用户勾选且填好必要条件后才调用。
- 硬约束 1：界面**不收集** API Key，只展示「环境变量是否就绪」与模型名。
- 硬约束 2：不写日志、不落盘；评判结果只存在 ``st.session_state``。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from adapters.base import AdapterError
from adapters.chatlog_adapter import ChatlogQuery, fetch_chatlog
from adapters.qq_export_adapter import load_qq_export
from config import AB_SWAP, DIMENSIONS, PROMPT_VERSION, SCORE_MAX
from judge import WEIGHT_SUM, DebateJudge, DebateJudgement, RoundJudgement
from llm_client import LLMConfig, LLMError, MissingAPIKeyError
from parser import SIDE_TEXT, Debate, ParseError, group_utterances, parse_text
from schemas import JudgeValidationError

SAMPLE_PATH = Path(__file__).parent / "samples" / "sample_debate.txt"
MAX_ROUND_TOTAL = int(SCORE_MAX * WEIGHT_SUM)

WINNER_TEXT: dict[str, str] = {
    "pro": ":orange[**正方占优**]",
    "con": ":blue[**反方占优**]",
    "tie": ":gray[**平局**]",
}
WINNER_SHORT: dict[str, str] = {"pro": "正方", "con": "反方", "tie": "平局"}
SIDE_COLOR: dict[str, str] = {"pro": "orange", "con": "blue"}


# ---------------------------------------------------------------------------
# 状态与输入
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """硬约束 3：两个适配器默认关闭，其余状态同样只放内存。"""
    st.session_state.setdefault("enable_chatlog", False)
    st.session_state.setdefault("enable_qq_upload", False)
    st.session_state.setdefault("loaded_debate", None)
    st.session_state.setdefault("judgement", None)
    st.session_state.setdefault("debate_text", "")


def _env_model() -> str | None:
    """只检查环境变量是否就绪；缺 Key 时给出可执行提示，绝不弹输入框。"""
    try:
        return LLMConfig.from_env().model
    except MissingAPIKeyError as exc:
        st.error(f"环境未就绪：{exc}")
        return None


def _load_sample_text() -> str | None:
    if not SAMPLE_PATH.exists():
        return None
    return SAMPLE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _render_side(round_judgement: RoundJudgement, side: str) -> None:
    side_result = round_judgement.pro if side == "pro" else round_judgement.con
    crown = " 🏆" if round_judgement.winner == side else ""
    st.markdown(
        f":{SIDE_COLOR[side]}[**{SIDE_TEXT[side]}**]　本轮总分 "
        f"`{side_result.total:g}/{MAX_ROUND_TOTAL}`{crown}"
    )
    for dimension in side_result.dimensions:
        st.markdown(f"**{dimension.label}**　`{dimension.score:g}/10`")
        if dimension.missing_evidence:
            st.caption(f"无引用依据（0 分）：{dimension.missing_evidence}")
        elif dimension.quote:
            st.caption(f"「{dimension.quote}」")


def _render_round(round_judgement: RoundJudgement) -> None:
    with st.container(border=True):
        st.markdown(f"#### 第 {round_judgement.round_no} 轮")
        col_pro, col_con = st.columns(2)
        with col_pro:
            _render_side(round_judgement, "pro")
        with col_con:
            _render_side(round_judgement, "con")

        metric_pro, metric_con, metric_support = st.columns(3)
        metric_pro.metric("本轮正方（加权）", f"{round_judgement.pro.total:g}")
        metric_con.metric("本轮反方（加权）", f"{round_judgement.con.total:g}")
        metric_support.metric("本轮回合支持度（正方）", f"{round_judgement.support_pct:.1f}%")

        st.markdown(f"本轮优势方：{WINNER_TEXT[round_judgement.winner]}")
        for note in round_judgement.notes:
            st.caption(f"· {note}")


def _render_summary(judgement: DebateJudgement) -> None:
    st.divider()
    st.subheader("总体结果")
    if judgement.motion:
        st.caption(f"辩题：{judgement.motion}")

    col_pro, col_con, col_support, col_winner = st.columns(4)
    col_pro.metric("正方累计（加权）", f"{judgement.pro_cum:g}")
    col_con.metric("反方累计（加权）", f"{judgement.con_cum:g}")
    col_support.metric("正方支持度", f"{judgement.support_pct:.1f}%")
    col_winner.metric("最终优势方", WINNER_SHORT[judgement.winner])

    st.progress(
        min(max(judgement.support_pct / 100.0, 0.0), 1.0),
        text=f"正方 {judgement.support_pct:.1f}% ｜ 反方 {100 - judgement.support_pct:.1f}%",
    )
    st.caption(
        f"模型：{judgement.model} ｜ prompt v{PROMPT_VERSION} ｜ "
        f"A/B 对调平均：{'开启' if judgement.ab_swapped else '关闭'} ｜ "
        f"四维各 0–{SCORE_MAX}，单方单轮加权满分 {MAX_ROUND_TOTAL}"
    )

    with st.expander("查看全部轮次明细（分数 + 引用原文）"):
        for round_judgement in judgement.rounds:
            st.markdown(
                f"**第 {round_judgement.round_no} 轮** ｜ "
                f"正方 {round_judgement.pro.total:g} vs 反方 {round_judgement.con.total:g} ｜ "
                f"{WINNER_TEXT[round_judgement.winner]}"
            )
            for side in ("pro", "con"):
                side_result = round_judgement.pro if side == "pro" else round_judgement.con
                st.caption(
                    f"{SIDE_TEXT[side]}：" + "；".join(
                        f"{item.label} {item.score:g}" + ("（无依据）" if item.missing_evidence else "")
                        for item in side_result.dimensions
                    )
                )


# ---------------------------------------------------------------------------
# 适配器面板（默认关闭）
# ---------------------------------------------------------------------------


def _render_chatlog_panel() -> None:
    st.caption(
        "需要本机已运行 chatlog server（默认 http://127.0.0.1:5030）。"
        "能力受限于其公开 HTTP API；字段名未在文档中公开，缺失时会给出带原始片段的错误。"
    )
    url = st.text_input("chatlog API 地址", value="http://127.0.0.1:5030", key="chatlog_url")
    talker = st.text_input("talker（wxid / 群 ID / 备注名）", key="chatlog_talker")
    col_start, col_end, col_limit = st.columns(3)
    start_date = col_start.text_input("开始日期（YYYY-MM-DD）", key="chatlog_start")
    end_date = col_end.text_input("结束日期（可空）", key="chatlog_end")
    limit = col_limit.number_input("limit", min_value=1, max_value=5000, value=200, key="chatlog_limit")
    pro_sender = st.text_input("哪一方是正方？（填发送者标识）", key="chatlog_pro_sender")

    if not st.button("拉取并识别轮次", key="chatlog_fetch"):
        return

    try:
        utterances = fetch_chatlog(
            ChatlogQuery(
                base_url=url,
                talker=talker,
                start_date=start_date,
                end_date=end_date or None,
                limit=int(limit),
            )
        )
        debate = group_utterances(utterances, pro_sender=pro_sender)
    except (AdapterError, ParseError) as exc:
        st.error(f"chatlog 导入失败：{exc}")
        return

    st.session_state["loaded_debate"] = debate
    st.success(f"已获取 {len(utterances)} 条消息，识别出 {debate.total_rounds} 轮")
    for warning in debate.warnings:
        st.warning(warning)
    for round_item in debate.rounds[:2]:
        st.caption(f"第 {round_item.round_no} 轮预览：{round_item.pro_text[:40]}… / {round_item.con_text[:40]}…")


def _render_qq_panel() -> None:
    st.caption(
        "上传 QQ Chat Exporter 导出的 JSON 文件（只解析你本地的文件，不上传、不落盘）。"
        "该导出格式的 schema 未公开，适配器按 type 分发并跳过/标记非文本消息。"
    )
    uploaded = st.file_uploader("选择导出的 JSON 文件", type=["json"], key="qq_file")
    pro_sender = st.text_input("哪一方是正方？（填发送者标识/昵称）", key="qq_pro_sender")

    if not st.button("解析文件", key="qq_parse"):
        return
    if uploaded is None:
        st.warning("请先选择文件")
        return

    try:
        utterances = load_qq_export(uploaded.read())
        debate = group_utterances(utterances, pro_sender=pro_sender)
    except (AdapterError, ParseError) as exc:
        st.error(f"QQ 导出文件解析失败：{exc}")
        return

    st.session_state["loaded_debate"] = debate
    st.success(f"已解析 {len(utterances)} 条消息，识别出 {debate.total_rounds} 轮")
    for warning in debate.warnings:
        st.warning(warning)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AI 辩论裁判 MVP", page_icon="⚖️", layout="wide")
    _init_state()

    st.title("⚖️ AI 辩论裁判 MVP")
    st.caption(
        "四维评分：逻辑性 / 证据 / 说服力 / 反驳质量 ｜ 每维 0–10 ｜ "
        "每维必须附一句原文引用 ｜ 胜方由数值重算，不由模型自称"
    )

    model = _env_model()
    if model is None:
        st.info(
            "设置方式（PowerShell）：`$env:DEBATE_JUDGE_API_KEY=\"你的Key\"` 后重启 `streamlit run app.py`。"
            "程序不会保存密钥，也不会在界面收集。"
        )
        st.stop()

    with st.sidebar:
        st.header("运行配置")
        st.caption(f"模型：{model}（来自环境变量）")
        st.caption("API Key：仅从环境变量读取，界面不收集")
        st.checkbox("A/B 顺序对调取平均（每轮调用 2 次）", value=AB_SWAP, key="ab_swap")
        st.caption(f"四维权重：{', '.join(f'{d.label} {d.weight:g}' for d in DIMENSIONS)}")

    tab_text, tab_adapters = st.tabs(["① 粘贴文本", "② 适配器导入（默认关闭）"])

    with tab_text:
        if st.button("载入示例（3 轮）", key="load_sample"):
            sample = _load_sample_text()
            if sample:
                st.session_state["debate_text"] = sample
                st.success("已载入 samples/sample_debate.txt")
            else:
                st.warning("未找到 samples/sample_debate.txt")
        st.text_area(
            "辩论内容（格式：辩题：… / 正方：… / 反方：…）",
            key="debate_text",
            height=280,
            placeholder="辩题：是否应全面禁止一次性塑料制品\n\n正方：……\n反方：……\n\n正方：……\n反方：……",
        )

    with tab_adapters:
        st.checkbox("启用 chatlog 适配器（读取本机 HTTP 服务）", key="enable_chatlog")
        if st.session_state.get("enable_chatlog"):
            with st.container(border=True):
                _render_chatlog_panel()
        else:
            st.caption("未启用：不会发起任何 HTTP 请求。")

        st.divider()
        st.checkbox("启用 QQ 导出文件上传", key="enable_qq_upload")
        if st.session_state.get("enable_qq_upload"):
            with st.container(border=True):
                _render_qq_panel()
        else:
            st.caption("未启用：不会读取任何上传文件。")

    st.divider()
    loaded_debate: Debate | None = st.session_state.get("loaded_debate")

    if loaded_debate is not None:
        source = st.radio(
            "评判对象",
            ["粘贴的文本", f"已导入的会话（{loaded_debate.total_rounds} 轮）"],
            horizontal=True,
            key="judge_source",
        )
    else:
        source = "粘贴的文本"

    if st.button("开始评判", type="primary", key="run_judge"):
        debate: Debate | None = None
        try:
            if source != "粘贴的文本" and loaded_debate is not None:
                debate = loaded_debate
            else:
                debate = parse_text(st.session_state.get("debate_text", ""))

            progress_bar = st.progress(0.0, text="准备中…")

            def _progress(done: int, total: int) -> None:
                progress_bar.progress(
                    done / total, text=f"已完成 {done}/{total} 轮（每轮含 A/B 两次调用时耗时更长）"
                )

            judge = DebateJudge(ab_swap=bool(st.session_state.get("ab_swap", AB_SWAP)))
            judgement = judge.judge_debate(debate, progress=_progress)
            progress_bar.empty()
            st.session_state["judgement"] = judgement
        except ParseError as exc:
            st.error(f"解析失败：{exc}")
        except AdapterError as exc:
            st.error(f"适配器失败：{exc}")
        except MissingAPIKeyError as exc:
            st.error(f"环境未就绪：{exc}")
        except JudgeValidationError as exc:
            st.error(f"LLM 输出校验失败（含重写重试后仍失败）：{exc}")
        except LLMError as exc:
            st.error(f"LLM 调用失败：{exc}")
        except ValueError as exc:
            st.error(f"输入或参数错误：{exc}")

    judgement = st.session_state.get("judgement")
    if isinstance(judgement, DebateJudgement):
        _render_summary(judgement)
        st.divider()
        for round_judgement in judgement.rounds:
            _render_round(round_judgement)


if __name__ == "__main__":
    main()
