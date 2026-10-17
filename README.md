# AI 辩论裁判 MVP（Python + Streamlit）

粘贴或导入一段辩论材料（`正方：... / 反方：...`），调用 LLM 从**逻辑性 / 证据 / 说服力 / 反驳质量**四个维度评分，展示每轮分数、累计分数、支持度百分比与本轮优势方。

## 1. 设计要点（先读，避免误解为“复刻”）

- **四维评分 + 引用锚定**：每维 0–10 整数，且每维分数必须附一句**该方本轮原文的逐字引用**；某维找不到可引用依据时必须给 0 分并在 `missing_evidence` 中说明。该规则由 prompt 约束 + Pydantic 校验 + 「引用必须是原文子串」断言三层保证。
- **胜方由数值重算**：LLM 输出 schema 里**没有** winner 字段；本轮/最终优势方都由四维加权分数算出（差值 ≤ 0.5 判平局）。
- **A/B 对调平均**：每轮分别以「先正后反」和「先反后正」各判一次，逐维平均，缓解位置偏见。默认开启，3 轮 = 6 次 LLM 调用。
- **与 Debatrix 的关系（必须明示）**：prompt 结构与评分口径参考 ljcleo/debatrix，但 **Debatrix 原生四维是 argument / source / language / general（权重 3/2/2/0），没有独立的“反驳质量”维度**（clash 在 argument 内部处理）；本项目拆出一个 `rebuttal` 维度，并把其「每条 speech 一次 UPDATE + 全场 JUDGE + 二次抽取分数」的多段调用压成「每轮一次结构化 JSON 调用」。因此这里是**结构借鉴 + 规则新建**，不是复用其评分实现。
- **不引入被参考仓库的依赖**：不使用 NiceGUI / LangChain / Go 二进制 / Groq SDK。

## 2. 目录结构

```text
debate_judge/
├── app.py                      # Streamlit 界面（每轮双列、四维、累计、支持度、优势方）
├── judge.py                    # 校验 / A-B 对调平均 / 裁剪 / 胜方重算 / tenacity 重试 / 修复重试
├── llm_client.py               # OpenAI 兼容客户端（唯一读取 API Key 的地方）
├── prompt.py                   # 四维 prompt + 校准锚点 + 注入式 JSON Schema
├── schemas.py                  # Pydantic 契约 + JSON 抽取 + 引用逐字校验
├── config.py                   # 维度/权重/阈值/env 变量名/LLM 参数
├── parser.py                   # 「正方：/反方：」文本 → 统一内部格式
├── adapters/
│   ├── base.py                 # 异常族 + 候选字段表 + .get() 链式取值工具
│   ├── chatlog_adapter.py      # chatlog HTTP API → Utterance[]（默认关闭）
│   └── qq_export_adapter.py    # QQ Chat Exporter JSON → Utterance[]（默认关闭）
├── scripts/probe_schema.py     # 未知 JSON 字段探测（本地确认用）
├── samples/sample_debate.txt   # 3 轮示例（可直接点“载入示例”）
├── tests/                      # 离线测试（不联网、不需要 Key）
├── requirements.txt
├── .env.example                # 仅键名参考；程序不读取 .env
└── README.md
```

## 3. 环境要求

- Python **3.10+**（代码使用 `X | None`、`dataclass(slots=True)`；实测环境为 3.12）
- 可访问的 OpenAI 兼容 LLM 接口

## 4. 安装与运行

PowerShell：

```powershell
cd debate_judge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:DEBATE_JUDGE_API_KEY="你的Key"
# 可选：$env:DEBATE_JUDGE_BASE_URL="https://api.openai.com/v1"
# 可选：$env:DEBATE_JUDGE_MODEL="gpt-4o-mini"

streamlit run app.py
```

bash / zsh：

```bash
cd debate_judge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 国内网络直连 PyPI 可能很慢（实测约 50KB/s），可加镜像：
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export DEBATE_JUDGE_API_KEY="你的Key"
streamlit run app.py
```

跑测试（可选，需要 `pip install pytest`；全部离线、不消耗额度）：

```bash
python -m pytest tests -q
```

`tests/test_app_smoke.py` 需要 `streamlit`（已在 requirements 中），未安装时会自动 skip。

## 5. 环境变量

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `DEBATE_JUDGE_API_KEY` | 是 | — | 只从环境变量读取；缺失时页面报错并停止，**不会**弹输入框 |
| `DEBATE_JUDGE_BASE_URL` | 否 | `https://api.openai.com/v1` | OpenAI 兼容地址 |
| `DEBATE_JUDGE_MODEL` | 否 | `gpt-4o-mini` | 模型名 |

`.env.example` 只是键名参考，**程序不会自动加载 `.env`**（避免“以为已隐藏、实际被提交”的误用）。

## 6. 使用流程

1. 打开页面 →「① 粘贴文本」→ 点「载入示例（3 轮）」或粘贴自己的材料（`辩题：` 可选，`正方：`/`反方：` 必须交替）。
2. 左侧栏可关闭「A/B 顺序对调取平均」以省一半调用。
3. 点「开始评判」→ 逐轮渲染：正/反双列、四维分数 + 引用原文、本轮加权分、本轮支持度、本轮优势方。
4. 页面底部汇总：正方/反方累计加权分、正方支持度、最终优势方，以及「全部轮次明细」。

适配器（**默认关闭**，不勾选不会发起任何请求）：

- 「② 适配器导入」→ 勾选「启用 chatlog 适配器」并填 API 地址 / talker / 日期范围；
- 或勾选「启用 QQ 导出文件上传」并选择本地 JSON；
- 两者都需要指定「哪一方是正方」（发送者标识），程序**不会自动猜**。

## 7. 本地确认适配器字段（首次使用必做）

chatlog 响应与 QCE 导出的 JSON **字段名均未在公开文档中完整暴露**，因此适配器是「候选字段名 + `.get()` 链式取值 + 带原始片段的异常」。用下面的命令一次拿到真实结构：

```powershell
# chatlog（先启动 chatlog server，默认 127.0.0.1:5030）
python scripts\probe_schema.py --url "http://127.0.0.1:5030/api/v1/chatlog?time=2026-01-01&talker=wxid_xxx&limit=2&format=json"

# QQ Chat Exporter（先导出一份很小的 JSON）
python scripts\probe_schema.py --file 你的导出.json
```

把输出中的「合并键名」和首条消息字段补到 `adapters/base.py` 的 `SENDER_KEYS` / `CONTENT_KEYS` / `TIME_KEYS` / `LIST_KEYS` 最前面即可，无需改适配器逻辑。

## 8. 验收清单

逐条勾选（每条都能在界面上观察到）：

- [ ] 粘贴 3 轮示例，点击「开始评判」后正常出结果，无异常栈
- [ ] 每轮以**正/反两列**并排展示（橙色正方、蓝色反方）
- [ ] 每方四个维度都有 0–10 分数，维度名为 逻辑性 / 证据 / 说服力 / 反驳质量
- [ ] 每个非 0 维度下方显示 `「原文引用」`，且该引用确实是该方本轮原文的子串
- [ ] 若某维度 0 分，显示「无引用依据（0 分）：<原因>」，且不显示引用
- [ ] 每轮显示双方加权总分与本轮回合支持度百分比
- [ ] 每轮显示「本轮优势方」，且与两侧分数一致（不会出现分数低却判占优）
- [ ] 底部显示正方/反方**累计分数**与**正方支持度**，并有进度条
- [ ] 底部显示「最终优势方」；平局时显示「平局」
- [ ] 侧栏默认勾选「A/B 顺序对调取平均」；关闭后每轮只调用一次 LLM
- [ ] 「chatlog 适配器」与「QQ 导出上传」默认都是**未勾选**；未勾选时页面明确提示不会发起请求/读取文件
- [ ] 未设置 `DEBATE_JUDGE_API_KEY` 时，页面显示「环境未就绪」错误并停止，**没有**任何密钥输入框
- [ ] 输入格式错误（如只有「正方：」、正方连续出现）时，`st.error` 给出行号与原文片段
- [ ] chatlog 地址错误 / 服务未启动时，`st.error` 显示 HTTP 状态与环节
- [ ] QQ 导出文件字段不匹配时，`st.error` 显示「第 N 条消息」与原始 JSON 片段（截断）
- [ ] LLM 返回非 JSON 或引用非原文时，界面显示「LLM 输出校验失败」，且程序内部先尝试过 1 次重写
- [ ] 全程没有生成日志文件、没有把聊天原文写入磁盘（结果只在 session 内存中）
- [ ] `.streamlit/` 与 `secrets.toml` 已被 `.gitignore` 覆盖（`git check-ignore -v .streamlit/secrets.toml` 应命中）
- [ ] `python -m pytest tests -q` 全绿：37 项离线用例通过；另 2 项界面冒烟需要已安装 streamlit（未安装时显示 skipped）

## 9. 本次交付的验证状态（可复现）

| 验证项 | 命令 | 结果 |
|---|---|---|
| 离线测试（parser / 适配器 / 评判引擎） | `python -m pytest tests -q`（不含 smoke） | 37 passed |
| 界面冒烟（官方 `AppTest` 真跑 `app.py`） | `python -m pytest tests/test_app_smoke.py -q` | 2 passed（streamlit 1.66.0） |
| 全量测试 | `python -m pytest tests -q` | **39 passed** |
| 干净环境安装 | 新建 venv + `pip install -r requirements.txt`（Python 3.12.14） | 成功 |
| 真实启动服务 | `streamlit run app.py --server.headless true --server.port 8599` | HTTP 200，无异常栈 |
| 3 轮示例端到端 dry-run（假 LLM） | 见测试用例 | 正方 90 / 反方 75、支持度 54.5%、胜方 pro，引用逐字校验通过 |

验证环境版本：streamlit 1.66.0 / pydantic 2.14.1 / tenacity 9.2.1 / pytest 9.1.1 / Python 3.12.14。

**仍未验证**：真实 LLM 调用（全部测试用假 client）、chatlog 响应与 QCE 导出的真实字段名（需按第 7 节本地 probe 后回填候选表）。

## 10. 网站部署（Streamlit Community Cloud，可选）

本项目是标准 Streamlit 单页应用：入口 `app.py`、依赖 `requirements.txt` 均在**仓库根**，满足 Community Cloud 要求（官方规则：入口文件可在任意位置，但应用从仓库根初始化；依赖文件放仓库根或入口同级，见 [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)）。

### 10.1 部署步骤

1. 打开 <https://share.streamlit.io> → **Create app**；
2. Repository 选 `sujiuer666/AI-debate-judge`，Branch `main`，**Main file path 填 `app.py`**；
3. **Advanced settings** → Python version 选 **3.12**（本项目需要 ≥3.10）；
4. Secrets 里粘贴（TOML）：

```toml
DEBATE_JUDGE_API_KEY = "sk-..."
DEBATE_JUDGE_MODEL = "gpt-4o-mini"
# 可选：自定义 OpenAI 兼容地址
# DEBATE_JUDGE_BASE_URL = "https://api.openai.com/v1"
```

5. **Deploy**（首次构建约 2–5 分钟）；
6. 打开应用 → 点「载入示例」确认界面 → 点「开始评判」确认 Key 生效。

> **关键点**：Secrets 里的键必须是**根级**（顶格、不在 `[section]` 内）。Streamlit 官方文档明确：根级键会「同时作为环境变量可用」；本项目 `llm_client.py` 只读 `os.environ`，因此这样才能生效。若写成 `[llm]` 再放 key，则**不会**变成环境变量，应用会报「环境未就绪」。

### 10.2 云端与本地运行的差异（重要）

| 项 | 本地运行 | 网站部署（Community Cloud） |
|---|---|---|
| API Key | 环境变量 | 平台 Secrets 根级键（自动注入环境变量） |
| chatlog 适配器 | 可用（连本机 `127.0.0.1:5030`） | **不可用**：云端没有你的 chatlog 服务，勾选后只会得到连接失败提示（默认关闭，不影响其它功能） |
| QQ 导出上传 | 文件只在本机内存中解析 | 文件会**先上传到托管服务器（美国机房）**再在内存解析；仍不落盘、不写日志，但「数据没离开你的机器」不再成立 |
| 进程生命周期 | 你关闭才停止 | 无流量 **12 小时休眠**，任何人访问可唤醒；重启后 session 内容丢失 |
| 资源 | 你的机器 | CPU 0.078–2 核、内存 690MB–2.7GB，超限会显示 "Oh no" |
| 错误详情 | 可看 traceback | 平台强制隐藏 traceback，只能看 `st.error` 文案与 Cloud logs |

### 10.3 成本与滥用（部署前必读）

Community Cloud 应用默认**公开可访问、无登录**，而 API Key 是你一个人的：任何拿到 URL 的人点「开始评判」都会消耗你的额度。至少做一项防护：

- 只把 URL 发给自己人（最弱）；
- 在 `app.py` 里加访问口令（读 Secrets 中的 `APP_ACCESS_CODE`，未设置则不启用）；
- 或改成「用户自带 Key」模式（会与本项目「Key 仅来自环境变量」的约束冲突，需明确同意后再改）。

> 当前仓库只提供**最小可部署形态**，未包含口令/限流/Dockerfile；需要这些可另行添加。

### 10.4 版本固定

`requirements.txt` 已将 `streamlit` 固定为 **1.66.0**（本项目实测版本），其余依赖仍为下限。这样可以避免托管平台自动升级 Streamlit 导致行为漂移（Streamlit 官方同样建议 pin 版本）。

## 11. 已知限制

- **适配器字段未最终确认**：chatlog / QCE 的真实 schema 需在第 7 节本地跑一次后回填候选表；在此之前适配器能保证「报错清晰」，不保证「一次解析成功」。
- **QCE 流式 JSONL 未支持**：适配器只接受单个 JSON 文件/对象；如导出选了 `STREAMING_JSONL`，请改用普通 JSON。
- **调用成本**：默认 A/B 对调 = 每轮 2 次调用；3 轮 6 次，另加失败时最多 1 次/轮的重写。
- **引用依赖模型遵循度**：模型可能改写引用，此时会触发「重写重试」，仍失败则整轮报错而不是降级给分（宁可失败也不给无依据的分）。
- **群聊/多人会话**：超过 2 个发言人会直接报错，需先过滤成双方对话。
- **平局阈值**为四维加权均值差 ≤ 0.5（0–10 尺度），可在 `config.py` 调整。

## 12. 隐私与安全

- API Key 只从环境变量读取，代码中无字面量，错误信息不回显 Key。
- 不写日志文件、不落盘聊天原文；上传文件在内存中解析；结果只存在 Streamlit session。
- **网站部署时**：QQ 导出文件会先上传到托管服务器（见第 10.2 节），且应用默认公开——需要「数据完全不离开本机」请本地运行。
- `.streamlit/` 与 `secrets.toml` 已加入 `.gitignore`，密钥不会被提交。
- `scripts/probe_schema.py` 只打印到 stdout，但输出可能含聊天原文，分享前请自行脱敏。

## 13. 参考与许可证

| 来源 | 用途 | 许可证 |
|---|---|---|
| [ljcleo/debatrix](https://github.com/ljcleo/debatrix) | 多维度 prompt 结构与迭代分析思路、Pydantic/JSON 解析兜底思路 | MIT |
| [xlight/chatlog](https://github.com/xlight/chatlog)（镜像 [sjzar/chatlog](https://github.com/sjzar/chatlog)） | 仅参考其公开 HTTP API 契约 | Apache-2.0 |
| [shuakami/qq-chat-exporter](https://github.com/shuakami/qq-chat-exporter) | 仅解析其导出的 JSON 文件，未复制代码 | GPL-3.0 |
| [syed-waleed-ahmed/LLM-as-Judge](https://github.com/syed-waleed-ahmed/LLM-as-Judge) | 严格校验 / 对调平均 / 裁剪与胜方重算 / tenacity 重试的算法模式 | MIT |
| [rd-serendipity/ai-debate-arena](https://github.com/rd-serendipity/ai-debate-arena) | 每轮双列 + 色块 + 末尾汇总的展示节奏 | MIT |

本项目未复制上述仓库代码；仅提取思路、数据结构与 prompt 组织方式，并按第 1 步清单逐项标注了「复用 / 适配 / 新建」。
