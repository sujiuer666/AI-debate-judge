/**
 * Cloudflare Worker 版本地自测：用假 LLM 替换 fetch，验证评分链路与错误分支。
 * 运行：node selftest.mjs   （需要 Node 18+，无第三方依赖）
 */

import worker, {
  DIMENSIONS,
  parseText,
  judgeDebate,
  parseRound,
} from "./worker.js";

const SAMPLE = [
  "辩题：（示例）是否应全面禁止一次性塑料制品",
  "",
  "正方：我方主张全面禁止。理由有三：第一，替代品已经成熟，纸浆模塑与可降解聚乳酸餐具在2023年已覆盖多数连锁餐饮场景。",
  "反方：对方把禁止当成唯一解，却回避了替代品的环境代价。纸浆模塑需要大量原木浆，而多数城市并不具备工业堆肥条件。",
  "",
  "正方：对方的成本论恰好证明了禁令的必要性。回收体系的边际成本随品类扩张递增，而禁用的治理成本递减。",
  "反方：我方正面回应。沾染油污的餐盒确实不可回收，但可要求平台强制使用可重复使用容器并建立清洗闭环。",
  "",
  "正方：成本由谁承担，我方有明确答案：外部性内部化。塑料餐盒的治理成本目前由全体纳税人承担。",
  "反方：对方终于给出数字，但回避了替代品的碳足迹。辩论的目标不是比谁更环保，而是比谁的政策更可行。",
].join("\n");

let passed = 0;
const failures = [];

function check(name, condition, extra) {
  if (condition) {
    passed += 1;
    console.log("  ok  " + name);
  } else {
    failures.push(name + (extra ? " -> " + extra : ""));
    console.log("  FAIL " + name + (extra ? " -> " + extra : ""));
  }
}

/* ---------------------------- 假 LLM ---------------------------- */

const MATERIAL_RE =
  /### 材料 \d｜(正方（Pro）|反方（Con）)\n\n([\s\S]*?)(?=\n\n### 材料 |\n\n## 要求)/g;

function buildRoundJson({ roundNo, proText, conText, proScores, conScores, breakQuotes, omitMissing }) {
  const make = (side, text, scores) => ({
    side: side,
    dimensions: DIMENSIONS.map((dim) => {
      const score = scores[dim.key];
      const base = { dimension: dim.key, score: score, quote: "", missing_evidence: null };
      if (score === 0) {
        base.missing_evidence = omitMissing ? null : "本轮该维度找不到可引用的原文依据";
        return base;
      }
      base.quote = breakQuotes ? "这句不是原文，纯属编造" : text.slice(0, 12);
      return base;
    }),
  });
  return JSON.stringify({
    round_no: roundNo,
    pro: make("pro", proText, proScores),
    con: make("con", conText, conScores),
  });
}

function extractMaterials(userPrompt) {
  const result = {};
  MATERIAL_RE.lastIndex = 0;
  let match;
  while ((match = MATERIAL_RE.exec(userPrompt)) !== null) {
    result[match[1]] = match[2];
  }
  return { pro: result["正方（Pro）"] || "", con: result["反方（Con）"] || "" };
}

function roundNoFrom(messages) {
  const all = messages.map((m) => m.content).join("\n");
  const match = all.match(/`round_no` 填 (\d+)/);
  return match ? Number(match[1]) : 1;
}

/** 重写重试时最后一条 user 是修复指令，材料在更早的 user 里，需要回溯查找。 */
function lastPromptWithMaterials(messages) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const item = messages[i];
    if (item.role === "user" && String(item.content).includes("### 材料")) return item.content;
  }
  return messages[messages.length - 1].content;
}

/**
 * 安装假 fetch。behavior(callIndex, messages) 可返回：
 *  - 字符串：作为 model 输出的 content
 *  - { status, body }：模拟 HTTP 错误
 */
function installFakeFetch(behavior) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const body = JSON.parse(options.body);
    calls.push({ url, body });
    const verdict = behavior(calls.length - 1, body.messages);
    if (verdict && typeof verdict === "object" && verdict.status) {
      return new Response(verdict.body || "", { status: verdict.status });
    }
    return new Response(
      JSON.stringify({
        model: "fake-model",
        choices: [{ message: { content: String(verdict) } }],
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  };
  return calls;
}

const DEFAULT_PRO = { logic: 8, evidence: 8, persuasion: 7, rebuttal: 7 };
const DEFAULT_CON = { logic: 6, evidence: 7, persuasion: 6, rebuttal: 6 };

function validResponder(overrides) {
  return (index, messages) => {
    const roundNo = roundNoFrom(messages);
    const materials = extractMaterials(lastPromptWithMaterials(messages));
    return buildRoundJson({
      roundNo,
      proText: materials.pro,
      conText: materials.con,
      proScores: (overrides && overrides.pro) || DEFAULT_PRO,
      conScores: (overrides && overrides.con) || DEFAULT_CON,
    });
  };
}

async function postJson(path, body, env) {
  const request = new Request("https://example.com" + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const response = await worker.fetch(request, env, {});
  return { status: response.status, data: await response.json() };
}

async function getPage() {
  const response = await worker.fetch(new Request("https://example.com/"), {}, {});
  return { status: response.status, text: await response.text() };
}

/* ---------------------------- 测试 ---------------------------- */

async function main() {
  console.log("== 1. 页面与健康检查 ==");
  const page = await getPage();
  check("GET / 返回 200", page.status === 200);
  check("页面包含标题", page.text.includes("AI 辩论裁判"));
  check("页面注入了示例文本", page.text.includes("var SAMPLE_TEXT"));
  check("页面没有残留模板占位", !page.text.includes("${"));
  const health = await worker.fetch(new Request("https://example.com/api/health"), {}, {});
  check("GET /api/health 返回 ok", (await health.json()).ok === true);

  console.log("== 2. 文本解析 ==");
  const debate = parseText(SAMPLE);
  check("解析出 3 轮", debate.rounds.length === 3, "实际 " + debate.rounds.length);
  check("解析出辩题", debate.motion === "（示例）是否应全面禁止一次性塑料制品");
  let parseError = null;
  try {
    parseText("正方：只有正方");
  } catch (err) {
    parseError = err;
  }
  check("缺反方时抛错", parseError && parseError.stage === "pairing");

  console.log("== 3. 正常评判（3 轮 × A/B = 6 次调用） ==");
  const calls = installFakeFetch(validResponder());
  const env = { OPENAI_API_KEY: "test-key-not-real", OPENAI_MODEL: "fake-model" };
  const okResult = await postJson("/api/judge", { text: SAMPLE }, env);
  check("HTTP 200", okResult.status === 200, JSON.stringify(okResult.data).slice(0, 200));
  check("ok=true", okResult.data.ok === true);
  const result = okResult.data.result;
  check("3 轮", result.rounds.length === 3);
  check("正方累计 90", result.pro_cum === 90, String(result.pro_cum));
  check("反方累计 75", result.con_cum === 75, String(result.con_cum));
  check("支持度 54.5", result.support_pct === 54.5, String(result.support_pct));
  check("胜方 pro", result.winner === "pro", result.winner);
  check("LLM 调用 6 次", calls.length === 6, String(calls.length));
  check("第一轮优势方 pro", result.rounds[0].winner === "pro");
  check(
    "引用来自原文",
    result.rounds.every((round) =>
      round.pro.dimensions.every((dim) => dim.score === 0 || debate.rounds[round.round_no - 1].proText.includes(dim.quote))
    )
  );

  console.log("== 4. 非 JSON 输出 → 重写后成功 ==");
  installFakeFetch((index, messages) => (index === 0 ? "我先解释一下评分理由……（无 JSON）" : validResponder()(index, messages)));
  const repaired = await postJson("/api/judge", { text: SAMPLE }, env);
  check("重写后 200", repaired.status === 200, JSON.stringify(repaired.data).slice(0, 200));
  check("重写后正方累计 90", repaired.data.result.pro_cum === 90);

  console.log("== 5. 引用非原文 → 重写后成功 ==");
  installFakeFetch((index, messages) => {
    const roundNo = roundNoFrom(messages);
    const materials = extractMaterials(lastPromptWithMaterials(messages));
    if (index === 0) {
      return buildRoundJson({
        roundNo,
        proText: materials.pro,
        conText: materials.con,
        proScores: DEFAULT_PRO,
        conScores: DEFAULT_CON,
        breakQuotes: true,
      });
    }
    return validResponder()(index, messages);
  });
  const quoteRepair = await postJson("/api/judge", { text: SAMPLE }, env);
  check("引用重写后 200", quoteRepair.status === 200);

  console.log("== 6. 0 分缺 missing_evidence → 重写 ==");
  installFakeFetch((index, messages) => {
    const roundNo = roundNoFrom(messages);
    const materials = extractMaterials(lastPromptWithMaterials(messages));
    const proScores = index === 0 ? { logic: 8, evidence: 0, persuasion: 7, rebuttal: 7 } : DEFAULT_PRO;
    if (index === 0) {
      return buildRoundJson({
        roundNo,
        proText: materials.pro,
        conText: materials.con,
        proScores,
        conScores: DEFAULT_CON,
        omitMissing: true,
      });
    }
    return validResponder()(index, messages);
  });
  const missingRepair = await postJson("/api/judge", { text: SAMPLE }, env);
  check("0 分规则修正后 200", missingRepair.status === 200);

  console.log("== 7. 环境与鉴权 ==");
  const noKey = await postJson("/api/judge", { text: SAMPLE }, {});
  check("缺 Key 返回 500", noKey.status === 500);
  check("缺 Key stage=env", noKey.data.stage === "env");

  const guardedEnv = { ...env, ACCESS_CODE: "s3cret" };
  const denied = await postJson("/api/judge", { text: SAMPLE }, guardedEnv);
  check("口令缺失返回 401", denied.status === 401);
  check("口令缺失 stage=auth", denied.data.stage === "auth");
  installFakeFetch(validResponder());
  const allowed = await postJson("/api/judge", { text: SAMPLE, accessCode: "s3cret" }, guardedEnv);
  check("口令正确返回 200", allowed.status === 200);

  console.log("== 8. 输入错误 ==");
  const badText = await postJson("/api/judge", { text: "正方：只有正方" }, env);
  check("格式错误返回 400", badText.status === 400);
  check("格式错误 stage=pairing", badText.data.stage === "pairing", String(badText.data.stage));
  const emptyText = await postJson("/api/judge", { text: "" }, env);
  check("空输入返回 400", emptyText.status === 400);

  console.log("== 9. 单轮解析与并发安全（纯函数） ==");
  const single = parseText("正方：A\n反方：B");
  check("单轮解析", single.rounds.length === 1);
  let schemaError = null;
  try {
    parseRound(JSON.stringify({ round_no: 1, pro: { side: "pro", dimensions: [] }, con: { side: "con", dimensions: [] } }), 1);
  } catch (err) {
    schemaError = err;
  }
  check("4 维缺失被拒", schemaError && schemaError.stage === "schema");

  console.log("");
  if (failures.length > 0) {
    console.log("FAILED: " + failures.length + " 项");
    failures.forEach((item) => console.log(" - " + item));
    process.exit(1);
  }
  console.log("passed " + passed + " checks");
  console.log("ALL_TESTS_PASSED");
}

main().catch((err) => {
  console.error("SELFTEST_CRASH", err);
  process.exit(1);
});
