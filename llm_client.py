"""OpenAI 兼容的 chat/completions 客户端 —— 本项目**唯一**读取 API Key 的地方。

来源与适配说明
--------------
- 角色与错误分类思路参考 syed-waleed-ahmed/LLM-as-Judge 的
  ``src/llm_judge/providers/groq_provider.py``（JSON mode + 把 429/5xx 标为可重试、
  把 4xx 认证错误标为不可重试）。
- 适配方式：不引入其 Groq SDK / provider 协议，改为通用 OpenAI 兼容 HTTP；
  ``tenacity`` 重试放在 ``judge.py``（本文件只负责「单次调用 + 错误归类」）。
- 硬约束 1：API Key 只从环境变量 ``DEBATE_JUDGE_API_KEY`` 读取，代码中无任何字面量；
  错误信息里**不回显** Key（异常只带 URL 与阶段）。
- 硬约束 2：不写日志、不落盘、不缓存请求内容；返回值只在内存中流转。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    LLM_JSON_MODE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)

FAILED_STATUS_RETRYABLE: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """LLM 调用相关错误基类：带失败环节与（可选）截断片段。"""

    def __init__(self, stage: str, detail: str, snippet: str = "") -> None:
        self.stage = stage
        self.detail = detail
        self.snippet = (snippet or "")[:200]
        message = f"[{stage}] {detail}"
        if self.snippet:
            message = f"{message} | 原始片段：{self.snippet}"
        super().__init__(message)


class MissingAPIKeyError(LLMError):
    """环境变量中没有 API Key（不提示、不收集、不落盘）。"""


class LLMTransientError(LLMError):
    """可重试的瞬时错误：429、5xx、超时、连接失败。"""


class LLMHTTPError(LLMError):
    """不可重试的 HTTP 错误（如 401/403/404）。"""


class LLMResponseError(LLMError):
    """HTTP 200 但响应结构不符合 OpenAI 兼容格式。"""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """运行时配置；``api_key`` 只能来自环境变量。"""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = LLM_TEMPERATURE
    timeout: float = LLM_TIMEOUT_SECONDS
    json_mode: bool = LLM_JSON_MODE
    max_tokens: int = LLM_MAX_TOKENS

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量构造配置；缺 Key 时抛 ``MissingAPIKeyError``。

        只读取三个变量：``DEBATE_JUDGE_API_KEY`` / ``DEBATE_JUDGE_BASE_URL`` /
        ``DEBATE_JUDGE_MODEL``。不读 ``.env``、不接受界面输入。
        """
        api_key = (os.environ.get(ENV_API_KEY) or "").strip()
        if not api_key:
            raise MissingAPIKeyError(
                "env",
                f"未检测到环境变量 {ENV_API_KEY}；请在启动 Streamlit 前设置（程序不会保存密钥，也不会在界面收集）",
            )
        base_url = (os.environ.get(ENV_BASE_URL) or "").strip() or DEFAULT_BASE_URL
        model = (os.environ.get(ENV_MODEL) or "").strip() or DEFAULT_MODEL
        return cls(api_key=api_key, base_url=base_url, model=model)


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    model: str
    usage: dict[str, Any] | None = None


def _build_payload(config: LLMConfig, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def chat_completion(
    config: LLMConfig,
    messages: Sequence[dict[str, str]],
    *,
    http_post: Any | None = None,
) -> ChatResult:
    """调用一次 ``POST {base_url}/chat/completions``。

    Args:
        config: 运行时配置（含 Key）。
        messages: ``[{"role": "system"|"user"|"assistant", "content": ...}]``。
        http_post: 可注入的 HTTP 函数，签名 ``(url, headers, payload, timeout) -> response``；
            仅用于离线测试，生产路径走 ``requests``。

    Raises:
        LLMTransientError: 429 / 5xx / 超时 / 连接失败（可被 tenacity 重试）。
        LLMHTTPError: 其它非 200（不可重试）。
        LLMResponseError: 响应结构不符合 OpenAI 兼容格式。
    """
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = _build_payload(config, messages)

    if http_post is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确提示
            raise LLMError("dependency", "缺少 requests，请先 pip install -r requirements.txt") from exc

        def http_post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> Any:  # type: ignore[misc]
            return requests.post(url, headers=headers, json=payload, timeout=timeout)

    try:
        response = http_post(url, headers, payload, config.timeout)
    except Exception as exc:  # noqa: BLE001 - 连接/超时统一归为可重试
        raise LLMTransientError(
            "llm-request",
            f"请求 {url} 失败：{type(exc).__name__}: {exc}",
        ) from exc

    status = getattr(response, "status_code", None)
    body_text = getattr(response, "text", "") or ""

    if status in FAILED_STATUS_RETRYABLE:
        raise LLMTransientError(
            "llm-http",
            f"HTTP {status}（瞬时错误，可重试）；请检查限流/服务端状态",
            body_text,
        )
    if status != 200:
        raise LLMHTTPError(
            "llm-http",
            f"HTTP {status}；请检查 API 地址、模型名与 Key 权限（Key 不会显示在错误里）",
            body_text,
        )

    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - 200 但非 JSON
        raise LLMResponseError("llm-json", "响应不是合法 JSON", body_text) from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(
            "llm-shape",
            "响应缺少 choices[0].message.content；请确认使用 OpenAI 兼容接口",
            body_text,
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("llm-content", "模型返回了空内容", body_text)

    return ChatResult(
        content=content,
        model=str(data.get("model") or config.model),
        usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
    )


__all__ = [
    "ChatResult",
    "LLMConfig",
    "LLMError",
    "LLMHTTPError",
    "LLMResponseError",
    "LLMTransientError",
    "MissingAPIKeyError",
    "chat_completion",
]
