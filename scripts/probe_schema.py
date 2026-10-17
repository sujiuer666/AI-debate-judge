"""未知 JSON 结构探测脚本（chatlog 响应 / QQ Chat Exporter 导出文件通用）。

为什么需要它
------------
第 1 步确认：chatlog 的响应字段名、QCE 导出的 JSON schema **均未在公开文档中暴露**。
本脚本不假设任何字段名，只把结构打印出来，用于一次性补齐
``adapters/base.py`` 中的候选字段表。

用法
----
::

    # 探测 QCE 导出文件
    python scripts/probe_schema.py --file sample.json

    # 探测 chatlog 响应（先启动 chatlog server）
    python scripts/probe_schema.py --url "http://127.0.0.1:5030/api/v1/chatlog?time=2026-01-01&talker=wxid_xxx&limit=2&format=json"

可选参数：``--max-depth``（默认 4）、``--sample-chars``（默认 800）。

隐私
----
本脚本**不写任何文件**、不发日志，只打印到 stdout；但输出可能包含聊天原文，
分享给他人前请自行删改。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _load_file(path: str) -> Any:
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return json.loads(raw.decode("utf-8"))


def _load_url(url: str, timeout: float = 20.0) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 仅本地调试
        body = response.read()
        status = getattr(response, "status", None)
    print(f"[http] status={status} bytes={len(body)}", file=sys.stderr)
    return json.loads(body.decode("utf-8-sig"))


def _summarize(node: Any, path: str, depth: int, max_depth: int, lines: list[str]) -> None:
    label = path or "$"
    if isinstance(node, dict):
        lines.append(f"{label}  object({len(node)} keys)")
        if depth < max_depth:
            for key, value in node.items():
                _summarize(value, f"{path}.{key}" if path else key, depth + 1, max_depth, lines)
    elif isinstance(node, list):
        lines.append(f"{label}  list({len(node)})")
        if node and depth < max_depth:
            _summarize(node[0], f"{path}[0]", depth + 1, max_depth, lines)
    else:
        preview = json.dumps(node, ensure_ascii=False)
        lines.append(f"{label}  {type(node).__name__} = {preview[:80]}")


def _find_object_arrays(node: Any, path: str = "", found: list[tuple[str, int, list[str]]] | None = None) -> list[tuple[str, int, list[str]]]:
    found = [] if found is None else found
    if isinstance(node, list) and node and all(isinstance(item, dict) for item in node[:5]):
        keys = sorted({key for item in node[:5] for key in item})
        found.append((path or "$", len(node), keys))
    if isinstance(node, dict):
        for key, value in node.items():
            _find_object_arrays(value, f"{path}.{key}" if path else key, found)
    elif isinstance(node, list):
        for index, value in enumerate(node[:3]):
            _find_object_arrays(value, f"{path}[{index}]", found)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测未知 JSON 结构（不假设字段名）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="本地 JSON 文件路径")
    source.add_argument("--url", help="HTTP GET URL（chatlog API）")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--sample-chars", type=int, default=800)
    args = parser.parse_args(argv)

    try:
        payload = _load_file(args.file) if args.file else _load_url(args.url)
    except (OSError, ValueError) as exc:
        print(f"[error] 读取或解析失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"[error] 请求失败：{exc}", file=sys.stderr)
        return 1

    lines: list[str] = []
    _summarize(payload, "", 0, args.max_depth, lines)
    print("=" * 70)
    print("结构概览（深度上限 %d）" % args.max_depth)
    print("=" * 70)
    print("\n".join(lines))

    arrays = _find_object_arrays(payload)
    print()
    print("=" * 70)
    print("疑似「消息数组」候选（元素为对象的数组）")
    print("=" * 70)
    if not arrays:
        print("未找到对象数组；这个 JSON 可能不是消息列表，或消息被包在字符串里。")
    for path, size, keys in arrays:
        print(f"- 路径: {path}  长度: {size}")
        print(f"  合并键名: {keys}")

    if arrays:
        path, _, _ = arrays[0]
        sample = _sample_at(payload, path)
        print()
        print("=" * 70)
        print(f"第一条消息原样（截断 {args.sample_chars} 字符）")
        print("=" * 70)
        print(json.dumps(sample, ensure_ascii=False, indent=2)[: args.sample_chars])

    print()
    print("提示：把上面看到的真实字段名补到 adapters/base.py 的候选表最前面，")
    print("      再跑 python -m pytest tests/test_offline.py 验证适配器。")
    return 0


def _sample_at(payload: Any, path: str) -> Any:
    """按 ``_find_object_arrays`` 输出的路径取回第一条样本。"""
    if path == "$":
        return payload[0] if isinstance(payload, list) and payload else payload
    current = payload
    for token in path.replace("[", ".").replace("]", "").split("."):
        if not token:
            continue
        if token.isdigit():
            current = current[int(token)]
        else:
            current = current[token]
    return current[0] if isinstance(current, list) and current else current


if __name__ == "__main__":
    raise SystemExit(main())
