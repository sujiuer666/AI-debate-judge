"""界面冒烟测试：用 Streamlit 官方 AppTest 真正执行 app.py。

要点
----
- 不需要 API Key 联网：只验证「环境未就绪」路径与「已有评判结果」的渲染路径；
- 注入一个手工构造的 ``DebateJudgement`` 来驱动渲染，从而校验每轮双列、
  四维分数、累计/支持度 metric 是否真的出现在组件树里；
- 顺带断言硬约束 3：两个适配器开关默认必须是 False。

未安装 streamlit 时整个模块跳过（requirements.txt 已包含 streamlit）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from config import ENV_API_KEY  # noqa: E402
from judge import (  # noqa: E402
    DebateJudgement,
    DimensionResult,
    RoundJudgement,
    SideResult,
)

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _side(side: str, scores: dict[str, float]) -> SideResult:
    labels = {"logic": "逻辑性", "evidence": "证据", "persuasion": "说服力", "rebuttal": "反驳质量"}
    dimensions = tuple(
        DimensionResult(
            dimension=key,
            label=labels[key],
            score=score,
            quote="示例引用" if score > 0 else "",
            missing_evidence=None if score > 0 else "本轮该维度找不到可引用依据",
        )
        for key, score in scores.items()
    )
    return SideResult(side=side, dimensions=dimensions, total=float(sum(scores.values())))


def _judgement() -> DebateJudgement:
    pro = _side("pro", {"logic": 8, "evidence": 8, "persuasion": 7, "rebuttal": 7})
    con = _side("con", {"logic": 6, "evidence": 7, "persuasion": 6, "rebuttal": 6})
    round_item = RoundJudgement(
        round_no=1,
        pro=pro,
        con=con,
        winner="pro",
        support_pct=54.1,
        notes=("反方的「反驳质量」为 0 分：示例",),
    )
    return DebateJudgement(
        motion="（示例）是否应全面禁止一次性塑料制品",
        rounds=(round_item,),
        pro_cum=30.0,
        con_cum=25.0,
        support_pct=54.1,
        winner="pro",
        model="fake-model",
        ab_swapped=True,
    )


def test_app_without_api_key_shows_env_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not at.exception
    assert any("环境未就绪" in error.value for error in at.error)


def test_app_defaults_and_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_KEY, "dummy-key-for-smoke-test")

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception

    # 硬约束 3：适配器默认关闭
    assert at.session_state["enable_chatlog"] is False
    assert at.session_state["enable_qq_upload"] is False

    # 注入一个已完成的评判结果，驱动渲染路径
    at.session_state["judgement"] = _judgement()
    at.run()
    assert not at.exception

    metric_labels = [metric.label for metric in at.metric]
    assert "正方累计（加权）" in metric_labels
    assert "反方累计（加权）" in metric_labels
    assert "正方支持度" in metric_labels
    assert "最终优势方" in metric_labels
    assert "本轮正方（加权）" in metric_labels
    assert "本轮回合支持度（正方）" in metric_labels

    markdown_values = "\n".join(markdown.value for markdown in at.markdown)
    assert "第 1 轮" in markdown_values
    assert "正方" in markdown_values and "反方" in markdown_values
    assert "逻辑性" in markdown_values
    assert "本轮优势方" in markdown_values

    # 结果只存在于 session_state（内存），app 不提供保存/下载入口
    assert at.session_state["judgement"] is not None
