/**
 * AI 辩论裁判 —— Cloudflare Worker 单文件版（HTML + CSS + JS + API 全在一个文件里）
 *
 * 部署方式（无需本地工具链）：Cloudflare Dashboard → Workers & Pages → Create Worker
 * → 粘贴本文件 → Deploy → Settings → Variables and Secrets 添加：
 *   OPENAI_API_KEY   = 你的 key（必填，Secret）
 *   OPENAI_MODEL     = gpt-4o-mini（可选，Variable）
 *   OPENAI_BASE_URL  = https://api.openai.com/v1（可选，Variable）
 *   ACCESS_CODE      = 访问口令（可选；设置后页面必须输入同样口令才能评分）
 *
 * 与 Python 版（Streamlit）的关系：这是**同一套评分规则的独立实现**（JS 重写，不是移植库）：
 * - 四维：logic / evidence / persuasion / rebuttal，各 0-10 整数；
 * - 每维必须给该方本轮原文的逐字引用（quote），score>0 时缺失即判输出无效；
 * - score=0 必须填 missing_evidence；
 * - A/B 顺序对调两次取平均；分数 clamp；胜方由数值重算（平局阈值 0.5，按 0-10 加权均值）；
 * - 输出不合契约时把错误喂回模型重写 1 次；429/5xx 指数退避重试；
 * - 泄漏防护：不落盘、不打日志，答案只在内存中返回给浏览器。
 *
 * 注意：Cloudflare Worker 无法运行 Python/Streamlit，所以「部署到 Cloudflare」= 使用本文件，
 * Python 版仍保留用于本地运行与 Streamlit Community Cloud。
 */

/* ============================== 配置 ============================== */

const DIMENSIONS = [
  {
    key: "logic",
    label: "逻辑性",
    weight: 1,
    definition: "论点是否成立：前提是否成立、推理链是否完整、内部是否自洽",
  },
  {
    key: "evidence",
    label: "证据",
    weight: 1,
    definition: "是否提供可核验、与论点相关、且足以支撑论点强度的外部依据",
  },
  {
    key: "persuasion",
    label: "说服力",
    weight: 1,
    definition: "表达是否清楚有力，能否让中立听众理解并接受其立场",
  },
  {
    key: "rebuttal",
    label: "反驳质量",
    weight: 1,
    definition: "是否准确命中对方论证并说明其为何失效（本项目新增维度）",
  },
];

const DIM_KEYS = DIMENSIONS.map((d) => d.key);
const DIM_LABEL = Object.fromEntries(DIMENSIONS.map((d) => [d.key, d.label]));
const WEIGHT_SUM = DIMENSIONS.reduce((sum, d) => sum + d.weight, 0);

const SCORE_MIN = 0;
const SCORE_MAX = 10;
const TIE_THRESHOLD = 0.5;
const AB_SWAP = true;
const MAX_INPUT_CHARS = 20000;
const MAX_HISTORY_SIDE = 400;
const MAX_REPAIR_ATTEMPTS = 1;
const LLM_MAX_ATTEMPTS = 3;
const PROMPT_VERSION = "0.1.0-web";

const SIDE_TEXT = { pro: "正方", con: "反方" };
const SIDE_LABEL = { pro: "正方（Pro）", con: "反方（Con）" };
const WINNER_TEXT = { pro: "正方占优", con: "反方占优", tie: "平局" };

/* ============================== 错误 ============================== */

class AppError extends Error {
  constructor(stage, message, snippet) {
    super(message);
    this.name = "AppError";
    this.stage = stage;
    this.snippet = String(snippet || "").slice(0, 200);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ============================== 校准锚点 ============================== */

const ANCHORS = {
  logic: [
    ["0–2", "没有可辨认的论点，或论点与辩题无关；前提与结论之间找不到推理步骤；段落之间自相矛盾。"],
    ["3–4", "有论点但基本是断言；关键推理步骤缺失、偷换概念或循环论证；存在明显内部矛盾。"],
    ["5–6", "论点清楚、主干推理成立，但依赖未加说明的假设；存在少量逻辑跳跃。"],
    ["7–8", "论点—前提—结论链条完整且可追溯；能区分相关与因果；仅有细微瑕疵。"],
    ["9–10", "结构严密，主动限定前提与适用范围，并预判、封堵对方可能的反驳路径；无实质漏洞。"],
  ],
  evidence: [
    ["0–2", "没有外部依据，或所谓依据与论点无关、无法核验、明显编造。"],
    ["3–4", "只有零散例子或个人感受，缺少来源、时间、口径，无法支撑论点强度。"],
    ["5–6", "至少有一条具体、相关、可核验的证据，但覆盖不足，或未说明其代表性。"],
    ["7–8", "多条相关证据，来源与机制说明较完整，足以支撑主要论点。"],
    ["9–10", "证据分层（数据 + 机制 + 反例排查），并说明证据的局限与适用边界。"],
  ],
  persuasion: [
    ["0–2", "表达破碎、指代不清，读者无法复述其主张；或通篇情绪宣泄。"],
    ["3–4", "主张可以看懂，但结构松散、重点不明，难以被带动。"],
    ["5–6", "结构完整、语言清楚，有一定感染力，但缺少记忆点或价值升华。"],
    ["7–8", "取舍与节奏清晰，能抓住核心冲突，例子与价值陈述配合得当。"],
    ["9–10", "有强记忆点，把复杂问题讲成听众可感知的利害，且不靠夸张与人身攻击。"],
  ],
  rebuttal: [
    ["0–2", "完全没有回应对方论点，或只是否定而不给理由。"],
    ["3–4", "提到对方论点但误述、稻草人化，攻击点不触及对方论证核心。"],
    ["5–6", "准确复述对方主要论点并给出针对性质疑，但没完成「为何因此不成立」。"],
    ["7–8", "直接攻击对方前提或证据，并说明该攻击如何导致其论点失效；能区分主次战场。"],
    ["9–10", "逐条拆解并按重要性排序攻击，迫使对方转入防守；同时修补自身被攻击处（攻防一体）。"],
  ],
};

/* ============================== JSON Schema ============================== */

function dimensionSchema() {
  return {
    type: "object",
    additionalProperties: false,
    required: ["dimension", "score", "quote", "missing_evidence"],
    properties: {
      dimension: { type: "string", enum: DIM_KEYS },
      score: { type: "integer", minimum: SCORE_MIN, maximum: SCORE_MAX },
      quote: { type: "string", description: "该方本轮原文的逐字引用；score>0 时必填" },
      missing_evidence: {
        type: ["string", "null"],
        description: "score=0 时必填，说明找不到可引用依据的原因",
      },
    },
  };
}

function sideSchema(side) {
  return {
    type: "object",
    additionalProperties: false,
    required: ["side", "dimensions"],
    properties: {
      side: { const: side },
      dimensions: {
        type: "array",
        minItems: DIMENSIONS.length,
        maxItems: DIMENSIONS.length,
        items: dimensionSchema(),
      },
    },
  };
}

function roundSchema() {
  return {
    type: "object",
    additionalProperties: false,
    required: ["round_no", "pro", "con"],
    properties: {
      round_no: { type: "integer", minimum: 1 },
      pro: sideSchema("pro"),
      con: sideSchema("con"),
    },
  };
}

/* ============================== Prompt ============================== */

function buildSystemPrompt() {
  const dimensionList = DIMENSIONS.map(
    (d) => "- **" + d.label + "（" + d.key + "）**：" + d.definition + "。"
  ).join("\n");

  const anchorBlocks = DIMENSIONS.map((d) => {
    const lines = ANCHORS[d.key].map((pair) => "- " + pair[0] + " 分：" + pair[1]).join("\n");
    return "### " + d.label + "（" + d.key + "）\n定义：" + d.definition + "。\n" + lines;
  }).join("\n\n");

  const schemaText = JSON.stringify(roundSchema(), null, 2);

  return (
    "你是一名资深竞技辩论裁判，熟悉 Policy / BP 赛制的评判标准，现在只负责评判「一轮」交锋。\n\n" +
    "## 任务\n\n" +
    "针对用户给出的单轮材料，分别给正方（Pro）与反方（Con）按四个维度打分：\n" +
    "每个维度 " + SCORE_MIN + "–" + SCORE_MAX + " 分，且每一维都必须给出该方本轮原文的逐字引用作为**唯一依据**。\n\n" +
    "## 四个维度\n\n" +
    dimensionList +
    "\n\n## 硬性规则（违反任意一条，本次输出即作废）\n\n" +
    "1. 每一维分数必须附带 `quote`：该方在**本轮**原文中的逐字片段。不得改写、不得概括、不得引用对方或历史轮次的文字。\n" +
    "2. 若某一维在该方本轮原文中确实找不到可引用的依据，该维必须给 " + SCORE_MIN + " 分，`quote` 留空，并在 `missing_evidence` 中写明原因。\n" +
    "3. `score` 必须是 " + SCORE_MIN + "–" + SCORE_MAX + " 的整数，禁止小数。\n" +
    "4. 只依据本轮材料评分；历史轮次仅用于判断「本轮反驳是否命中对方此前论点」，不得重复计分。\n" +
    "5. 评分与材料呈现顺序无关：不要因为某一方先出现而给更高分。\n" +
    "6. 只输出 JSON 对象本身，不要任何解释文字、不要 markdown 围栏。\n\n" +
    "## 校准方法（必须照做）\n\n" +
    "1. 先为该维度**选定档位**（0–2 / 3–4 / 5–6 / 7–8 / 9–10），再在档位内做最小幅度调整。\n" +
    "2. **7–8 不是默认分**：给 7–8 的前提是你能指出具体做对了什么，也能指出一处仍可被对手利用的瑕疵。\n" +
    "3. 表现中规中矩、无突出优点也无明显错误时，应落在 **5–6**。\n" +
    "4. **9–10 需要同时满足**：该维度明显优于一般竞技辩论发言，且 quote 能引出对应原文。\n" +
    "5. 四个维度彼此独立打分，不要为了总分领先而反向调整各维分数。\n\n" +
    "## 各维度校准锚点\n\n" +
    anchorBlocks +
    "\n\n## 输出格式（严格 JSON）\n\n" +
    "只输出一个 JSON 对象，字段名、类型、取值范围必须与下面的 schema 完全一致：\n\n" +
    "```json\n" + schemaText + "\n```\n\n" +
    "字段含义：`round_no` 等于当前轮次；`pro`/`con` 的 `side` 固定为对应字符串；" +
    "`dimensions` 恰好 4 项且每个维度出现一次；`score` 为整数；" +
    "`quote` 为逐字引用（score>0 时必填）；`missing_evidence` 在 score=0 时必填、其余为 null。"
  );
}

function truncate(text, limit) {
  const value = String(text || "");
  if (value.length <= limit) return value;
  return value.slice(0, limit) + "……（已截断，原文共 " + value.length + " 字）";
}

function renderHistory(history) {
  if (!history || history.length === 0) return "";
  const lines = ["## 历史轮次（仅作语境参考，不重复计分）", ""];
  for (const item of history) {
    lines.push("### 第 " + item.roundNo + " 轮");
    lines.push("正方：" + truncate(item.proText, MAX_HISTORY_SIDE));
    lines.push("反方：" + truncate(item.conText, MAX_HISTORY_SIDE));
    lines.push("");
  }
  return lines.join("\n").trim();
}

function buildUserPrompt(ctx) {
  const texts = { pro: ctx.proText, con: ctx.conText };
  const parts = [];

  if (ctx.motion) parts.push("## 辩题\n\n" + ctx.motion);
  parts.push("## 当前轮次\n\n第 " + ctx.roundNo + " 轮 / 共 " + ctx.totalRounds + " 轮");

  const history = renderHistory(ctx.history);
  if (history) parts.push(history);

  parts.push("## 本轮材料（按呈现顺序编号）");
  ctx.order.forEach((side, index) => {
    parts.push("### 材料 " + (index + 1) + "｜" + SIDE_LABEL[side] + "\n\n" + truncate(texts[side], 10000));
  });

  parts.push(
    "## 要求\n\n" +
      "请对**正方**与**反方**分别按四个维度给出 " + SCORE_MIN + "–" + SCORE_MAX + " 整数分数，" +
      "每一维附上该方本轮原文的逐字引用；找不到依据的维度给 0 分并在 `missing_evidence` 中说明。" +
      "`round_no` 填 " + ctx.roundNo + "。只输出 JSON。"
  );
  return parts.join("\n\n");
}

function buildMessages(ctx) {
  return [
    { role: "system", content: buildSystemPrompt() },
    { role: "user", content: buildUserPrompt(ctx) },
  ];
}

function buildRepairMessage(err) {
  return (
    "你上一次的输出未通过结构校验，不能使用。\n\n" +
    "错误信息：\n" + err.stage + "：" + err.message + "\n\n" +
    "请严格按 system 中给出的 JSON Schema 重新输出**完整**的 JSON 对象，并注意：\n" +
    "1. `quote` 必须从该方本轮原文中**逐字复制**（不要改写、不要合并两句）；\n" +
    "2. `score = 0` 的维度必须填写 `missing_evidence`，且 `quote` 留空；\n" +
    "3. `dimensions` 必须恰好包含 4 项，每个维度只出现一次；\n" +
    "4. 只输出 JSON，不要任何解释。"
  );
}

/* ============================== LLM 调用 ============================== */

async function safeText(response) {
  try {
    return await response.text();
  } catch (err) {
    return "";
  }
}

async function callLLM(env, messages) {
  const baseUrl = String(env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/+$/, "");
  const model = String(env.OPENAI_MODEL || "gpt-4o-mini");
  // 部分 OpenAI 兼容服务不支持 response_format，可用 JSON_MODE=0 关闭。
  const rawJsonMode = env.JSON_MODE === undefined ? "1" : String(env.JSON_MODE);
  const jsonMode = !(rawJsonMode === "0" || rawJsonMode.toLowerCase() === "false");
  let lastError = null;

  for (let attempt = 1; attempt <= LLM_MAX_ATTEMPTS; attempt += 1) {
    let response;
    try {
      const payload = {
        model: model,
        messages: messages,
        temperature: 0,
        max_tokens: 4000,
      };
      if (jsonMode) payload.response_format = { type: "json_object" };

      response = await fetch(baseUrl + "/chat/completions", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: "Bearer " + env.OPENAI_API_KEY,
        },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      lastError = new AppError("llm-request", "调用 LLM 失败：" + err.message);
      if (attempt < LLM_MAX_ATTEMPTS) {
        await sleep(500 * Math.pow(2, attempt - 1));
        continue;
      }
      throw lastError;
    }

    if (response.status === 429 || response.status >= 500) {
      lastError = new AppError("llm-http", "HTTP " + response.status + "（瞬时错误，可重试）", await safeText(response));
      if (attempt < LLM_MAX_ATTEMPTS) {
        await sleep(500 * Math.pow(2, attempt - 1));
        continue;
      }
      throw lastError;
    }

    if (!response.ok) {
      throw new AppError(
        "llm-http",
        "HTTP " + response.status + "；请检查 API 地址、模型名与 Key 权限（Key 不会显示在错误里）",
        await safeText(response)
      );
    }

    let data;
    try {
      data = await response.json();
    } catch (err) {
      throw new AppError("llm-json", "响应不是合法 JSON", await safeText(response));
    }

    const choice = data && data.choices && data.choices[0];
    const content = choice && choice.message && choice.message.content;
    if (typeof content !== "string" || content.trim() === "") {
      throw new AppError("llm-content", "模型返回了空内容", JSON.stringify(data).slice(0, 200));
    }
    return { content: content, model: (data && data.model) || model };
  }

  throw lastError || new AppError("llm", "调用失败");
}

/* ============================== 输出校验 ============================== */

const isPlainObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

function rejectUnknownKeys(obj, allowed, where) {
  for (const key of Object.keys(obj)) {
    if (!allowed.includes(key)) {
      throw new AppError("schema", where + " 出现未知字段：" + key, JSON.stringify(obj).slice(0, 200));
    }
  }
}

function validateDimension(dim, where) {
  if (!isPlainObject(dim)) throw new AppError("schema", where + " 必须是对象", JSON.stringify(dim).slice(0, 200));
  rejectUnknownKeys(dim, ["dimension", "score", "quote", "missing_evidence"], where);

  if (!DIM_KEYS.includes(dim.dimension)) {
    throw new AppError("schema", where + " 的 dimension 非法：" + String(dim.dimension), JSON.stringify(dim).slice(0, 200));
  }
  if (!Number.isInteger(dim.score) || dim.score < SCORE_MIN || dim.score > SCORE_MAX) {
    throw new AppError("schema", where + " 的 score 必须是 " + SCORE_MIN + "–" + SCORE_MAX + " 的整数", JSON.stringify(dim).slice(0, 200));
  }
  const quote = typeof dim.quote === "string" ? dim.quote.trim() : "";
  const missing = dim.missing_evidence == null ? "" : String(dim.missing_evidence).trim();

  if (dim.score === SCORE_MIN) {
    if (!missing) throw new AppError("schema", where + " 得 0 分时必须填写 missing_evidence 说明原因", JSON.stringify(dim).slice(0, 200));
    if (quote) throw new AppError("schema", where + " 得 0 分时 quote 必须留空", JSON.stringify(dim).slice(0, 200));
  } else {
    if (!quote) throw new AppError("schema", where + " 得 " + dim.score + " 分时必须在 quote 中给出原文引用", JSON.stringify(dim).slice(0, 200));
    if (missing) throw new AppError("schema", where + " 得 " + dim.score + " 分时 missing_evidence 必须为 null", JSON.stringify(dim).slice(0, 200));
  }
}

function validateSide(side, expectedSide) {
  if (!isPlainObject(side)) throw new AppError("schema", expectedSide + " 必须是对象", JSON.stringify(side).slice(0, 200));
  rejectUnknownKeys(side, ["side", "dimensions"], expectedSide);
  if (side.side !== expectedSide) throw new AppError("schema", expectedSide + ".side 必须为 \"" + expectedSide + "\"", JSON.stringify(side.side));
  if (!Array.isArray(side.dimensions) || side.dimensions.length !== DIMENSIONS.length) {
    throw new AppError("schema", expectedSide + ".dimensions 必须恰好 " + DIMENSIONS.length + " 项", JSON.stringify(side.dimensions).slice(0, 200));
  }
  const seen = [];
  side.dimensions.forEach((dim, index) => {
    validateDimension(dim, expectedSide + ".dimensions[" + index + "]");
    if (seen.includes(dim.dimension)) {
      throw new AppError("schema", expectedSide + " 的维度重复：" + dim.dimension, JSON.stringify(side.dimensions).slice(0, 200));
    }
    seen.push(dim.dimension);
  });
  for (const key of DIM_KEYS) {
    if (!seen.includes(key)) {
      throw new AppError("schema", expectedSide + " 缺少维度：" + key, JSON.stringify(seen));
    }
  }
}

function parseRound(raw, expectedRoundNo) {
  if (!raw || raw.trim() === "") throw new AppError("json-parse", "LLM 返回了空内容");
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) throw new AppError("json-parse", "未在返回内容中找到 JSON 对象", raw);
  const jsonText = match[0];

  let data;
  try {
    data = JSON.parse(jsonText);
  } catch (err) {
    throw new AppError("json-parse", "JSON 语法错误：" + err.message, jsonText);
  }

  if (!isPlainObject(data)) throw new AppError("schema", "顶层必须是对象", jsonText);
  rejectUnknownKeys(data, ["round_no", "pro", "con"], "顶层");
  if (data.round_no !== expectedRoundNo) {
    throw new AppError("round-check", "LLM 回报的轮次为 " + data.round_no + "，与请求的第 " + expectedRoundNo + " 轮不一致", jsonText);
  }
  validateSide(data.pro, "pro");
  validateSide(data.con, "con");
  return data;
}

function assertQuotesVerbatim(round, proText, conText) {
  const texts = { pro: proText, con: conText };
  for (const side of ["pro", "con"]) {
    for (const dim of round[side].dimensions) {
      if (!dim.quote) continue;
      if (!texts[side].includes(dim.quote)) {
        throw new AppError(
          "quote-check",
          "第 " + round.round_no + " 轮 " + side + " 的 " + dim.dimension + " 维度引用并非原文逐字子串：" + JSON.stringify(dim.quote.slice(0, 80)),
          texts[side]
        );
      }
    }
  }
}

/* ============================== 评判引擎 ============================== */

const clamp = (value, low = SCORE_MIN, high = SCORE_MAX) => Math.max(low, Math.min(high, value));
const round2 = (value) => Math.round(value * 100) / 100;

function winnerFromScores(proAvg, conAvg, threshold = TIE_THRESHOLD) {
  if (Math.abs(proAvg - conAvg) <= threshold) return "tie";
  return proAvg > conAvg ? "pro" : "con";
}

function supportPct(pro, con) {
  const total = pro + con;
  if (total <= 0) return 50.0;
  return (pro / total) * 100;
}

async function requestRound(env, messages, roundNo, proText, conText) {
  let conversation = messages;
  let lastError = null;
  let model = "";

  for (let attempt = 0; attempt <= MAX_REPAIR_ATTEMPTS; attempt += 1) {
    const result = await callLLM(env, conversation);
    model = result.model || model;
    try {
      const round = parseRound(result.content, roundNo);
      assertQuotesVerbatim(round, proText, conText);
      return { round: round, model: model };
    } catch (err) {
      if (!(err instanceof AppError)) throw err;
      lastError = err;
      if (attempt >= MAX_REPAIR_ATTEMPTS) break;
      conversation = messages.concat([
        { role: "assistant", content: result.content },
        { role: "user", content: buildRepairMessage(err) },
      ]);
    }
  }

  throw new AppError(
    "repair-exhausted",
    "连续 " + (MAX_REPAIR_ATTEMPTS + 1) + " 次输出不符合数据契约；最后一次错误：" + lastError.message,
    lastError.snippet
  );
}

function pickQuote(rounds, side, dimensionKey, target) {
  let bestQuote = "";
  let bestDistance = null;
  for (const round of rounds) {
    const dim = round[side].dimensions.find((item) => item.dimension === dimensionKey);
    const distance = Math.abs(dim.score - target);
    if (bestDistance === null || distance < bestDistance) {
      bestDistance = distance;
      bestQuote = dim.quote || "";
    }
  }
  return bestQuote;
}

function mergeRound(responses, ctx) {
  const notes = [];
  const sides = {};

  for (const side of ["pro", "con"]) {
    const dimensions = [];
    let total = 0;

    for (const dimension of DIMENSIONS) {
      const scores = [];
      const missingList = [];
      for (const round of responses) {
        const dim = round[side].dimensions.find((item) => item.dimension === dimension.key);
        scores.push(dim.score);
        if (dim.missing_evidence) missingList.push(String(dim.missing_evidence));
      }

      const merged = round2(clamp(scores.reduce((sum, value) => sum + value, 0) / scores.length));
      let quote = pickQuote(responses, side, dimension.key, merged);
      let missing = null;

      if (merged === SCORE_MIN) {
        missing = missingList[0] || "两次输出均未给出可引用的原文依据";
        quote = "";
        notes.push(SIDE_TEXT[side] + "的「" + dimension.label + "」为 0 分：" + missing);
      } else if (!quote) {
        notes.push(SIDE_TEXT[side] + "的「" + dimension.label + "」缺少引用（模型输出异常）");
      }

      total += merged * dimension.weight;
      dimensions.push({
        dimension: dimension.key,
        label: dimension.label,
        score: merged,
        quote: quote,
        missing_evidence: missing,
      });
    }

    sides[side] = { side: side, dimensions: dimensions, total: round2(total) };
  }

  const pro = sides.pro;
  const con = sides.con;
  const proAvg = pro.total / WEIGHT_SUM;
  const conAvg = con.total / WEIGHT_SUM;
  const support = supportPct(pro.total, con.total);
  if (pro.total + con.total <= 0) notes.push("双方本轮总分均为 0，支持度按 50% 处理");

  return {
    round_no: ctx.roundNo,
    pro: pro,
    con: con,
    winner: winnerFromScores(proAvg, conAvg),
    support_pct: Math.round(support * 10) / 10,
    notes: notes,
  };
}

async function judgeRound(env, ctx) {
  const orders = AB_SWAP
    ? [["pro", "con"], ["con", "pro"]]
    : [["pro", "con"]];
  const responses = [];
  let model = "";

  for (const order of orders) {
    const messages = buildMessages({
      motion: ctx.motion,
      roundNo: ctx.roundNo,
      totalRounds: ctx.totalRounds,
      proText: ctx.proText,
      conText: ctx.conText,
      order: order,
      history: ctx.history,
    });
    const result = await requestRound(env, messages, ctx.roundNo, ctx.proText, ctx.conText);
    responses.push(result.round);
    model = result.model || model;
  }

  const merged = mergeRound(responses, ctx);
  return { round: merged, model: model };
}

async function judgeDebate(env, debate) {
  const rounds = [];
  const history = [];
  let model = "";

  for (const item of debate.rounds) {
    const result = await judgeRound(env, {
      motion: debate.motion,
      roundNo: item.roundNo,
      totalRounds: debate.rounds.length,
      proText: item.proText,
      conText: item.conText,
      history: history,
    });
    rounds.push(result.round);
    model = result.model || model;
    history.push({ roundNo: item.roundNo, proText: item.proText, conText: item.conText });
  }

  const proCum = round2(rounds.reduce((sum, item) => sum + item.pro.total, 0));
  const conCum = round2(rounds.reduce((sum, item) => sum + item.con.total, 0));
  const proAvg = proCum / (WEIGHT_SUM * rounds.length);
  const conAvg = conCum / (WEIGHT_SUM * rounds.length);

  return {
    motion: debate.motion,
    rounds: rounds,
    pro_cum: proCum,
    con_cum: conCum,
    support_pct: Math.round(supportPct(proCum, conCum) * 10) / 10,
    winner: winnerFromScores(proAvg, conAvg),
    model: model,
    ab_swapped: AB_SWAP,
    prompt_version: PROMPT_VERSION,
  };
}

/* ============================== 文本解析 ============================== */

const SIDE_RE = /^\s*(正方|反方|pro|con)\s*[：:]\s*(.*)$/i;
const MOTION_RE = /^\s*(辩题|题目|话题|motion)\s*[：:]\s*(.+)$/i;

function normalizeSide(raw) {
  const value = raw.trim().toLowerCase();
  return value === "正方" || value === "pro" ? "pro" : "con";
}

function pairSegments(segments) {
  if (segments.length === 0) throw new AppError("pairing", "没有解析出任何发言段落");
  const rounds = [];
  let index = 0;

  while (index < segments.length) {
    const current = segments[index];
    const roundNo = rounds.length + 1;

    if (current.side !== "pro") {
      throw new AppError("pairing", "第 " + roundNo + " 轮应当是「正方」发言在先，实际遇到「反方」（" + current.where + "）", current.text);
    }
    if (index + 1 >= segments.length) {
      throw new AppError("pairing", "第 " + roundNo + " 轮缺少「反方」发言（" + current.where + "）", current.text);
    }
    const next = segments[index + 1];
    if (next.side !== "con") {
      throw new AppError("pairing", "第 " + roundNo + " 轮正方之后应紧跟「反方」，却再次遇到「正方」（" + next.where + "）", next.text);
    }

    rounds.push({ roundNo: roundNo, proText: current.text, conText: next.text });
    index += 2;
  }
  return rounds;
}

function parseText(raw) {
  const text = String(raw || "").replace(/\ufeff/g, "").trim();
  if (text === "") throw new AppError("input", "输入为空，请粘贴包含「正方：」「反方：」的辩论文本");
  if (text.length > MAX_INPUT_CHARS) {
    throw new AppError("input", "文本长度 " + text.length + " 字，超过上限 " + MAX_INPUT_CHARS + " 字，请分段评判");
  }

  let motion = null;
  const segments = [];
  let currentSide = null;
  let currentWhere = "";
  let buffer = [];

  const flush = () => {
    if (!currentSide) return;
    const body = buffer.join("\n").trim();
    if (body === "") throw new AppError("format", "「" + SIDE_TEXT[currentSide] + "：」标记后面没有内容（" + currentWhere + "）");
    segments.push({ side: currentSide, where: currentWhere, text: body });
  };

  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const where = "第 " + (i + 1) + " 行";

    if (!currentSide && motion === null) {
      const motionMatch = line.match(MOTION_RE);
      if (motionMatch) {
        motion = motionMatch[2].trim();
        continue;
      }
    }

    const sideMatch = line.match(SIDE_RE);
    if (sideMatch) {
      flush();
      currentSide = normalizeSide(sideMatch[1]);
      currentWhere = where;
      buffer = [];
      const rest = sideMatch[2].trim();
      if (rest) buffer.push(rest);
      continue;
    }

    if (!currentSide) {
      if (line.trim() !== "") {
        throw new AppError(
          "format",
          "在第一个「正方：」之前出现了无法归属的文本；如需设置辩题请写成「辩题：...」（" + where + "）",
          line
        );
      }
      continue;
    }

    buffer.push(line.replace(/\s+$/, ""));
  }

  flush();
  if (segments.length === 0) {
    throw new AppError("format", "没有找到任何「正方：」/「反方：」发言标记，请检查格式");
  }
  return { motion: motion, rounds: pairSegments(segments) };
}

/* ============================== HTTP 处理 ============================== */

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status: status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function handleJudge(request, env) {
  if (!env.OPENAI_API_KEY) {
    return jsonResponse(
      {
        error:
          "服务器未配置 OPENAI_API_KEY。请在 Cloudflare 后台 Settings → Variables and Secrets 添加 Secret 后重试。",
        stage: "env",
      },
      500
    );
  }

  let body;
  try {
    body = await request.json();
  } catch (err) {
    return jsonResponse({ error: "请求体不是合法 JSON", stage: "request" }, 400);
  }

  const accessCode = String(env.ACCESS_CODE || "").trim();
  if (accessCode) {
    const given = String((body && body.accessCode) || request.headers.get("x-access-code") || "").trim();
    if (given !== accessCode) {
      return jsonResponse({ error: "访问口令不正确（未授权）", stage: "auth" }, 401);
    }
  }

  let debate;
  try {
    debate = parseText(body && body.text);
  } catch (err) {
    return jsonResponse({ error: err.message, stage: err.stage || "parse", snippet: err.snippet || "" }, 400);
  }

  try {
    const result = await judgeDebate(env, debate);
    return jsonResponse({ ok: true, result: result });
  } catch (err) {
    return jsonResponse(
      { error: err.message, stage: err.stage || "judge", snippet: err.snippet || "" },
      502
    );
  }
}

/* ============================== 前端页面 ============================== */

const PAGE_HTML = String.raw`<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI 辩论裁判</title>
<style>
  :root {
    --bg: #0b0f16;
    --panel: #121a26;
    --panel-2: #0f1620;
    --line: #22304a;
    --text: #e6edf7;
    --muted: #8b9bb4;
    --pro: #f5a524;
    --con: #4b9cf5;
    --ok: #2ecc71;
    --bad: #ff5c5c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    line-height: 1.65;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 28px 18px 80px; }
  header h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: .5px; }
  header p { margin: 0; color: var(--muted); font-size: 14px; }
  .tag { display: inline-block; font-size: 12px; color: var(--muted); border: 1px solid var(--line);
         border-radius: 999px; padding: 2px 10px; margin-left: 8px; vertical-align: middle; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin-top: 18px; }
  textarea {
    width: 100%; min-height: 220px; resize: vertical; padding: 14px; border-radius: 10px;
    border: 1px solid var(--line); background: var(--panel-2); color: var(--text);
    font-size: 14px; line-height: 1.7; font-family: inherit;
  }
  textarea:focus { outline: 1px solid #3d5a8a; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
  button {
    border: 1px solid var(--line); background: #1b2637; color: var(--text);
    padding: 9px 16px; border-radius: 9px; font-size: 14px; cursor: pointer;
  }
  button:hover { background: #22314a; }
  button.primary { background: linear-gradient(135deg, #2f6fed, #6a4bf0); border-color: transparent; font-weight: 600; }
  button.primary:disabled { opacity: .55; cursor: default; }
  input[type=password] {
    padding: 9px 12px; border-radius: 9px; border: 1px solid var(--line);
    background: var(--panel-2); color: var(--text); font-size: 14px;
  }
  .hint { color: var(--muted); font-size: 12px; }
  .err { border: 1px solid #7a2b2b; background: #2a1214; color: #ffb3b3; border-radius: 10px;
         padding: 12px 14px; margin-top: 14px; white-space: pre-wrap; font-size: 13px; }
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; }
  .metric { background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .metric .k { color: var(--muted); font-size: 12px; }
  .metric .v { font-size: 22px; font-weight: 700; margin-top: 2px; }
  .bar { height: 10px; border-radius: 999px; background: #1a2434; overflow: hidden; margin-top: 12px; }
  .bar > i { display: block; height: 100%; background: linear-gradient(90deg, var(--pro), var(--con)); }
  .round { border: 1px solid var(--line); border-radius: 12px; margin-top: 18px; overflow: hidden; background: var(--panel); }
  .round-head { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px;
                background: #16202f; border-bottom: 1px solid var(--line); font-weight: 600; }
  .badge { font-size: 12px; padding: 2px 10px; border-radius: 999px; border: 1px solid var(--line); }
  .badge.pro { color: var(--pro); border-color: #5a4416; background: #241c0c; }
  .badge.con { color: var(--con); border-color: #1c3a5e; background: #0d1a2a; }
  .badge.tie { color: var(--muted); }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  .col { padding: 14px 16px; }
  .col + .col { border-left: 1px solid var(--line); }
  .side-title { font-weight: 700; margin-bottom: 10px; }
  .side-title.pro { color: var(--pro); }
  .side-title.con { color: var(--con); }
  .dim { margin: 10px 0; }
  .dim-top { display: flex; justify-content: space-between; font-size: 13px; }
  .dim-score { color: var(--muted); font-variant-numeric: tabular-nums; }
  .dim-bar { height: 5px; border-radius: 999px; background: #1a2434; margin-top: 4px; overflow: hidden; }
  .dim-bar > i { display: block; height: 100%; background: #3d6fd8; }
  .quote { color: var(--muted); font-size: 12px; margin-top: 4px; border-left: 2px solid #2c3d59; padding-left: 8px; }
  .quote.missing { color: #c9924a; }
  .round-foot { display: flex; justify-content: space-between; padding: 10px 16px; border-top: 1px solid var(--line);
                color: var(--muted); font-size: 12px; flex-wrap: wrap; gap: 8px; }
  .loading { color: var(--muted); font-size: 13px; margin-top: 12px; }
  footer { color: var(--muted); font-size: 12px; margin-top: 26px; }
  @media (max-width: 760px) {
    .summary { grid-template-columns: repeat(2, 1fr); }
    .cols { grid-template-columns: 1fr; }
    .col + .col { border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>⚖️ AI 辩论裁判 <span class="tag" id="ver">web</span></h1>
    <p>粘贴辩论记录（<code>辩题：… / 正方：… / 反方：…</code>）→ 四维评分 · 每维必须附原文引用 · 胜方由数值重算</p>
  </header>

  <section class="card">
    <textarea id="input" placeholder="辩题：是否应全面禁止一次性塑料制品&#10;&#10;正方：……&#10;反方：……&#10;&#10;正方：……&#10;反方：……"></textarea>
    <div class="row">
      <button class="primary" id="run">开始评判</button>
      <button id="sample">载入示例（3 轮）</button>
      <button id="clear">清空</button>
      <span class="hint" id="hint">每轮 2 次模型调用（A/B 对调）</span>
    </div>
    <div class="row">
      <input type="password" id="code" placeholder="访问口令（若站长设置）" style="min-width: 220px;" />
      <span class="hint">口令仅用于本次请求，不保存</span>
    </div>
    <div id="error" class="err" style="display:none;"></div>
  </section>

  <section id="result"></section>
  <footer>四维：逻辑性 / 证据 / 说服力 / 反驳质量 ｜ 每维 0–10 · 引用必须逐字来自原文 · 结果不落盘</footer>
</div>

<script>
  var state = { loading: false };

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function showError(stage, message, snippet) {
    var box = document.getElementById("error");
    var text = "[" + (stage || "error") + "] " + (message || "未知错误");
    if (snippet) text += "\n原始片段：" + snippet;
    box.textContent = text;
    box.style.display = "block";
  }

  function clearError() {
    var box = document.getElementById("error");
    box.textContent = "";
    box.style.display = "none";
  }

  function dimRow(sideClass, dim) {
    var quoteClass = dim.missing_evidence ? "quote missing" : "quote";
    var quoteText = dim.missing_evidence
      ? "无引用依据（0 分）：" + dim.missing_evidence
      : "「" + dim.quote + "」";
    var pct = Math.max(0, Math.min(100, dim.score * 10));
    return '<div class="dim">' +
      '<div class="dim-top"><span>' + esc(dim.label) + '</span>' +
      '<span class="dim-score">' + esc(dim.score) + ' / 10</span></div>' +
      '<div class="dim-bar"><i style="width:' + pct + '%"></i></div>' +
      '<div class="' + quoteClass + '">' + esc(quoteText) + '</div>' +
      '</div>';
  }

  function renderRound(item) {
    var winBadge = item.winner === "tie"
      ? '<span class="badge tie">平局</span>'
      : (item.winner === "pro"
          ? '<span class="badge pro">正方占优</span>'
          : '<span class="badge con">反方占优</span>');
    var proDims = item.pro.dimensions.map(function (d) { return dimRow("pro", d); }).join("");
    var conDims = item.con.dimensions.map(function (d) { return dimRow("con", d); }).join("");
    var notes = (item.notes || []).map(function (n) { return "· " + esc(n); }).join("　");
    return '<div class="round">' +
      '<div class="round-head"><span>第 ' + esc(item.round_no) + ' 轮</span>' + winBadge + '</div>' +
      '<div class="cols">' +
        '<div class="col"><div class="side-title pro">正方 · 本轮 ' + esc(item.pro.total) + ' 分</div>' + proDims + '</div>' +
        '<div class="col"><div class="side-title con">反方 · 本轮 ' + esc(item.con.total) + ' 分</div>' + conDims + '</div>' +
      '</div>' +
      '<div class="round-foot"><span>本轮回合支持度（正方）：' + esc(item.support_pct) + '%</span><span>' + notes + '</span></div>' +
      '</div>';
  }

  function render(result) {
    var winnerText = result.winner === "tie" ? "平局" : (result.winner === "pro" ? "正方" : "反方");
    var html = '<div class="card">' +
      '<div style="font-weight:700;font-size:18px;">总体结果</div>' +
      (result.motion ? '<div class="hint" style="margin-top:4px;">辩题：' + esc(result.motion) + '</div>' : '') +
      '<div class="summary">' +
        '<div class="metric"><div class="k">正方累计</div><div class="v">' + esc(result.pro_cum) + '</div></div>' +
        '<div class="metric"><div class="k">反方累计</div><div class="v">' + esc(result.con_cum) + '</div></div>' +
        '<div class="metric"><div class="k">正方支持度</div><div class="v">' + esc(result.support_pct) + '%</div></div>' +
        '<div class="metric"><div class="k">最终优势方</div><div class="v">' + esc(winnerText) + '</div></div>' +
      '</div>' +
      '<div class="bar"><i style="width:' + esc(result.support_pct) + '%"></i></div>' +
      '<div class="hint" style="margin-top:8px;">模型：' + esc(result.model) + ' ｜ A/B 对调：' + (result.ab_swapped ? "开启" : "关闭") + ' ｜ prompt ' + esc(result.prompt_version) + '</div>' +
      '</div>';

    html += result.rounds.map(renderRound).join("");
    document.getElementById("result").innerHTML = html;
  }

  function loadSample() {
    document.getElementById("input").value = SAMPLE_TEXT;
  }

  async function runJudge() {
    if (state.loading) return;
    clearError();
    var text = document.getElementById("input").value;
    if (!text.trim()) { showError("input", "请先粘贴辩论内容"); return; }
    var code = document.getElementById("code").value;
    var btn = document.getElementById("run");
    btn.disabled = true;
    state.loading = true;
    document.getElementById("hint").textContent = "评判中…每轮 2 次模型调用，请稍候";
    document.getElementById("result").innerHTML = '<div class="loading">正在调用模型评判，可能需要 10–60 秒…</div>';

    try {
      var resp = await fetch("/api/judge", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: text, accessCode: code })
      });
      var data = await resp.json();
      if (!resp.ok || !data.ok) {
        document.getElementById("result").innerHTML = "";
        showError(data.stage || ("http-" + resp.status), data.error || "请求失败", data.snippet || "");
        return;
      }
      render(data.result);
    } catch (err) {
      document.getElementById("result").innerHTML = "";
      showError("network", "请求失败：" + err.message);
    } finally {
      btn.disabled = false;
      state.loading = false;
      document.getElementById("hint").textContent = "每轮 2 次模型调用（A/B 对调）";
    }
  }

  document.getElementById("run").addEventListener("click", runJudge);
  document.getElementById("sample").addEventListener("click", loadSample);
  document.getElementById("clear").addEventListener("click", function () {
    document.getElementById("input").value = "";
    document.getElementById("result").innerHTML = "";
    clearError();
  });
</script>
</body>
</html>`;

/* 3 轮示例（与 samples/sample_debate.txt 一致） */
const SAMPLE_TEXT = [
  "辩题：（示例）是否应全面禁止一次性塑料制品",
  "",
  "正方：我方主张全面禁止。理由有三：第一，替代品已经成熟，纸浆模塑与可降解聚乳酸餐具在2023年已覆盖多数连锁餐饮场景；第二，禁令有先例可循，欧盟一次性塑料指令自2021年起禁止吸管、餐具等十类制品；第三，禁令降低治理成本，回收体系对混杂塑料的分拣成本远高于源头禁用。",
  "反方：对方把禁止当成唯一解，却回避了替代品的环境代价。纸浆模塑需要大量原木浆，可降解材料需要工业堆肥条件，而多数城市并不具备；欧盟指令也并未禁止全部一次性塑料，而是限制、回收与标识的组合。我方主张押金回收与生产者责任延伸，德国PET瓶回收率长期维持在90%以上，这说明制度设计比一刀切更有效。",
  "",
  "正方：对方的成本论恰好证明了禁令的必要性。回收体系的边际成本随品类扩张递增，而禁用的治理成本递减；并且对方引用的德国数据只覆盖PET瓶，不覆盖餐具。我方要求对方说明：生产者责任延伸如何解决外卖场景中沾染油污、无法回收的餐盒？",
  "反方：我方正面回应。沾染油污的餐盒确实不可回收，但可要求平台强制使用可重复使用容器并建立清洗闭环，上海部分写字楼已在试点；同时对方所谓替代品成熟缺乏成本数据，纸浆模塑单价约为塑料的两到三倍。对方若不回应成本由谁承担，其禁令主张就只是道德姿态。",
  "",
  "正方：成本由谁承担，我方有明确答案：外部性内部化。塑料餐盒的环境治理成本目前由全体纳税人承担，禁令把成本前移到生产者与平台，反而更公平。至于价格，2024年国内纸浆模塑产能扩张后单价已下降到塑料的1.3倍左右。请对方比较溢价三成与每年数百万吨塑料泄漏的成本。",
  "反方：对方终于给出数字，但1.3倍这个数字没有出处，且回避了替代品的碳足迹。我方最后重申：辩论的目标不是比谁更环保，而是比谁的政策更可行。全面禁止在执法成本、替代品供给和地方差异上都不可行；分品类、分阶段的限制加生产者责任延伸才是可执行路径。",
].join("\n");

/* ============================== Worker 入口 ============================== */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      const html = PAGE_HTML.replace("</body>", '<script>var SAMPLE_TEXT = ' + JSON.stringify(SAMPLE_TEXT) + ';</script></body>');
      return new Response(html, {
        headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" },
      });
    }

    if (url.pathname === "/api/health") {
      return jsonResponse({ ok: true, prompt_version: PROMPT_VERSION });
    }

    if (request.method === "POST" && url.pathname === "/api/judge") {
      return handleJudge(request, env);
    }

    return new Response("Not Found", { status: 404 });
  },
};

/* 供 Node 本地自测使用（Cloudflare 运行时忽略这些导出） */
export { DIMENSIONS, PROMPT_VERSION, buildMessages, judgeDebate, mergeRound, parseRound, parseText };
