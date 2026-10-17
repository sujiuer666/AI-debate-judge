# 本地自测（Node 18+，无需 npm install）

本目录是「Cloudflare Worker 单文件版」：`worker.js` 同时包含前端页面、API 与裁判逻辑，可直接粘贴到 Cloudflare 后台部署。

自测做的事：把 `fetch` 替换成假 LLM，验证以下路径（**不联网、不需要 API Key**）：

1. `parseText`：3 轮示例解析、轮次配对、错误定位；
2. 正常路径：6 次模型调用（3 轮 × A/B 两遍）→ 累计/支持度/胜方；
3. 非 JSON 输出 → 触发 1 次重写后成功；
4. 引用不是原文子串 → 触发重写；
5. 0 分没填 `missing_evidence` → 触发重写；
6. 缺 `OPENAI_API_KEY` → HTTP 500 + `stage=env`；
7. 设置 `ACCESS_CODE` 时：无口令 401、带对口令 200；
8. 文本格式错误（只有正方）→ HTTP 400 + `stage=pairing`。

```bash
node selftest.mjs
```

预期最后一行：`ALL_TESTS_PASSED`。

部署见仓库根目录 README 的「Cloudflare 部署」一节。
