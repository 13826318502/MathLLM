const STORAGE = {
  favorites: "mathllm_favorites",
  history: "mathllm_recent_history",
  apiBase: "mathllm_api_base",
  theme: "mathllm_theme",
};

const PAGE_META = {
  solve: ["学习工作区 / 开始解题", "今天想解决哪道题？"],
  favorites: ["学习工作区 / 我的收藏", "把值得复习的题目留下来"],
  "favorite-detail": ["学习工作区 / 我的收藏 / 题目详情", "收藏题目详情"],
  history: ["学习工作区 / 学习记录", "看看自己最近解决了什么"],
  toolkit: ["学习工作区 / 学习工具", "用适合自己的方式理解数学"],
  knowledge: ["学习工作区 / 知识库文档", "看看知识库里到底写了什么"],
  observability: ["系统 / 运行观测", "看看 Agent 每次运行到底发生了什么"],
  "rag-attribution": ["系统 / 知识库归因", "看看每次回答有没有依据知识库"],
  settings: ["系统 / 设置", "调整你的本地学习空间"],
};

const EXAMPLES = [
  ["一元二次方程", "解方程 $x^2-5x+6=0$。"],
  ["函数极值", "求函数 $f(x)=x^3-3x+1$ 的极值点。"],
  ["矩阵行列式", "计算矩阵 $\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}$ 的行列式。"],
  ["概率计算", "盒中有 3 个红球和 2 个白球，随机取出 2 个，求恰好取到 1 个红球的概率。"],
];

const FORMULA_TEMPLATES = [
  ["分数", "□⁄□", "上下两个占位框"],
  ["上标", "□^□", "底数和上标都可以填写"],
  ["下标", "□_□", "变量和下标都可以填写"],
  ["平方根", "√(□)", "根号内占位框"],
  ["n 次根", "□√(□)", "根指数和根式内容都可以填写"],
  ["绝对值", "|□|", "绝对值内占位框"],
  ["求和", "∑□", "求和表达式"],
  ["积分", "∫□ d□", "被积表达式和变量都可以填写"],
  ["极限", "lim_{□→□}", "趋近变量和趋近值都可以填写"],
  ["二阶矩阵", "⎡□ □⎤\n⎣□ □⎦", "按 Tab 依次填写"],
  ["分段函数", "⎧ □\n⎩ □", "两行表达式"],
  ["向量", "⟨□, □⟩", "向量分量"],
];

const FORMULA_SYMBOL_GROUPS = [
  ["基础运算", ["＋", "－", "×", "÷", "＝", "（", "）", "，", "．"]],
  ["关系符号", ["≤", "≥", "≠", "≈", "∝", "∞", "∈", "∉", "∴", "∵"]],
  ["集合与逻辑", ["∪", "∩", "⊂", "⊆", "∀", "∃", "¬", "⇒", "⇔"]],
  ["箭头与几何", ["→", "←", "↔", "⊥", "∥", "∠", "°", "△", "□"]],
];

const FORMULA_GREEK_GROUPS = [
  ["小写", ["α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "π", "ρ", "σ", "φ", "ω"]],
  ["大写", ["Α", "Β", "Γ", "Δ", "Θ", "Λ", "Π", "Σ", "Φ", "Ω"]],
];

const MEMORY_CONFIG = {
  triggerTokens: 1200,
  maxRecentMessages: 6,
  maxContextTokens: 2048,
  maxOutputTokens: 512,
};

const initialFavoriteDetail = location.hash.slice(1).match(/^favorite-detail\/(\d+)$/);
const state = {
  page: initialFavoriteDetail ? "favorite-detail" : (PAGE_META[location.hash.slice(1)] ? location.hash.slice(1) : "solve"),
  messages: [],
  loading: false,
  health: "checking",
  selectedFavoriteIndex: 0,
  favoriteDetailIndex: initialFavoriteDetail ? Number(initialFavoriteDetail[1]) : 0,
  favoriteDeleteMode: false,
  favoriteDeleteSelection: new Set(),
  chatPinnedToBottom: true,
  answerMode: "solve",
  memorySummary: "",
  memoryCursor: 0,
  activeRequestController: null,
  observability: { loading: false, days: 7, metrics: null, traces: [], error: "" },
  ragAttribution: { loading: false, days: 7, summary: null, runs: [], error: "" },
  knowledge: { loading: false, docs: [], current: "", content: "", error: "" },
  solveDocs: { loading: false, loaded: false, docs: [] },
};

const $ = (selector) => document.querySelector(selector);
const pageContainer = $("#page-container");

function readJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function writeJson(key, value) { localStorage.setItem(key, JSON.stringify(value)); }
function getFavorites() {
  return readJson(STORAGE.favorites, [])
    .filter(Boolean)
    .map((item) => typeof item === "string"
      ? { question: item, answer: "", createdAt: 0 }
      : { question: item.question || "", answer: item.answer || "", createdAt: item.createdAt || 0 })
    .filter((item) => item.question.trim());
}
function getHistory() { return readJson(STORAGE.history, []).filter((item) => item && item.question); }
function apiBase() {
  return (localStorage.getItem(STORAGE.apiBase) || "http://127.0.0.1:8080/api").replace(/\/$/, "");
}
function estimateTokens(text) {
  const value = String(text || "");
  let cjk = 0;
  for (const char of value) {
    if ((char >= "\u3400" && char <= "\u4dbf") || (char >= "\u4e00" && char <= "\u9fff")) cjk += 1;
  }
  return Math.max(1, Math.ceil(cjk + (value.length - cjk) / 2.5));
}
function estimateMessages(messages) {
  return messages.reduce((total, message) => total + estimateTokens(message.content) + 4, 0);
}
function splitHistory(messages, maxRecentMessages, maxRecentTokens) {
  if (!messages.length) return [[], []];
  const recent = [];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (recent.length >= maxRecentMessages) break;
    const candidate = [messages[index], ...recent];
    if (recent.length && estimateMessages(candidate) > maxRecentTokens) break;
    recent.unshift(messages[index]);
  }
  if (!recent.length) recent.push(messages[messages.length - 1]);
  if (recent[0]?.role === "assistant") recent.shift();
  const oldCount = Math.max(0, messages.length - recent.length);
  return [messages.slice(0, oldCount), recent.length ? recent : [messages[messages.length - 1]]];
}
function shouldSummarize(summary, messages) {
  if (messages.length <= MEMORY_CONFIG.maxRecentMessages) return false;
  return (summary ? estimateTokens(summary) : 0) + estimateMessages(messages) > MEMORY_CONFIG.triggerTokens;
}
async function requestConversationSummary(messages, existingSummary, signal) {
  const response = await fetch(`${apiBase()}/memory/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      messages,
      existing_summary: existingSummary || null,
    }),
  });
  if (!response.ok) throw new Error(`摘要请求失败（HTTP ${response.status}）`);
  const payload = await response.json();
  if (!payload.summary || typeof payload.summary !== "string") throw new Error("摘要接口没有返回有效内容");
  return payload.summary.trim();
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}
function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function inlineMarkdown(value) {
  let text = escapeHtml(value);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  text = text.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
  return text;
}

function markdownToHtml(source) {
  let text = String(source || "").replace(/\r\n/g, "\n");
  const blocks = [];
  const stash = (html) => { const token = `MATHLLMTOKEN${blocks.length}END`; blocks.push([token, html]); return token; };
  text = text.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => stash(`<pre><code>${escapeHtml(code.trim())}</code></pre>`));
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => { const value = formula.trim(); return stash(`<div class="math-block" data-katex="${escapeHtml(value)}">${escapeHtml(value)}</div>`); });
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => { const value = formula.trim(); return stash(`<div class="math-block" data-katex="${escapeHtml(value)}">${escapeHtml(value)}</div>`); });
  text = text.replace(/`([^`\n]+)`/g, (_, code) => stash(`<code>${escapeHtml(code)}</code>`));
  text = text.replace(/\$(?!\$)([^$\n]+?)\$(?!\$)/g, (_, formula) => stash(`<span class="math-inline" data-katex="${escapeHtml(formula)}">${escapeHtml(formula)}</span>`));
  text = text.replace(/\\\(([^\n]*?)\\\)/g, (_, formula) => stash(`<span class="math-inline" data-katex="${escapeHtml(formula)}">${escapeHtml(formula)}</span>`));

  text = text.replace(/\\begin\{(matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases|aligned|array)\}([\s\S]*?)\\end\{\1\}/g, (_, env, body) => {
    const value = "\\begin{" + env + "}" + body + "\\end{" + env + "}";
    const safeValue = escapeHtml(value);
    return stash('<span class="math-inline" data-katex="' + safeValue + '">' + safeValue + "</span>");
  });
  const lines = text.split("\n");
  const output = [];
  let paragraph = [];
  let listType = null;
  const flushParagraph = () => {
    if (paragraph.length) { output.push(`<p>${inlineMarkdown(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`); paragraph = []; }
  };
  const closeList = () => { if (listType) { output.push(`</${listType}>`); listType = null; } };
  for (const line of lines) {
    if (/^\s*$/.test(line)) { flushParagraph(); closeList(); continue; }
    const heading = line.match(/^\s*(#{1,6})\s+(.+)$/);
    if (heading) { flushParagraph(); closeList(); output.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`); continue; }
    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) { flushParagraph(); closeList(); output.push("<hr>"); continue; }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const nextType = ordered ? "ol" : "ul";
      if (listType !== nextType) { closeList(); output.push(`<${nextType}>`); listType = nextType; }
      output.push(`<li>${inlineMarkdown((ordered || unordered)[1])}</li>`);
      continue;
    }
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) { flushParagraph(); closeList(); output.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`); continue; }
    if (listType) closeList();
    paragraph.push(line);
  }
  flushParagraph(); closeList();
  let html = output.join("");
  for (const [token, value] of blocks) html = html.replace(new RegExp(escapeRegExp(token), "g"), value);
  return html || "<span class=\"muted\">正在思考…</span>";
}

function toast(message) {
  const node = $("#toast"); node.textContent = message; node.classList.add("show");
  clearTimeout(window.__mathllmToast); window.__mathllmToast = setTimeout(() => node.classList.remove("show"), 2200);
}
function setPage(page) {
  state.page = PAGE_META[page] ? page : "solve";
  location.hash = state.page;
  render();
  $("#sidebar").classList.remove("open");
  if (state.page === "observability") loadObservability();
  if (state.page === "rag-attribution") loadRagAttribution();
  if (state.page === "knowledge") loadKnowledge();
}
function showFavoriteDetail(index) {
  state.favoriteDetailIndex = Number(index);
  state.selectedFavoriteIndex = state.favoriteDetailIndex;
  state.page = "favorite-detail";
  location.hash = `favorite-detail/${state.favoriteDetailIndex}`;
  render();
}
function setActiveNav() {
  const activePage = state.page === "favorite-detail" ? "favorites" : state.page;
  document.querySelectorAll(".nav-item[data-page]").forEach((button) => button.classList.toggle("active", button.dataset.page === activePage));
  const meta = PAGE_META[state.page]; $("#page-kicker").textContent = meta[0]; $("#page-title").textContent = meta[1];
}
function updateCounts() { $("#favorite-count").textContent = getFavorites().length; }

function renderMessages() {
  if (!state.messages.length) return `<div class="empty-chat"><div><div class="empty-icon">∑</div><h3>从一道题开始吧</h3><p>输入题目后，我会尽量把思路、公式和每一步原因讲清楚。</p></div></div>`;
  return state.messages.map((message, index) => {
    const assistant = message.role === "assistant";
    const hasTrace = Boolean(assistant && message.trace);
    const content = markdownToHtml(message.content);
    const trace = hasTrace ? `<div class="trace-card">${renderTraceMarkup(message.trace, message.content)}</div>` : "";
    const bubble = hasTrace ? "" : `<div class="message-bubble">${content}</div>`;
    const actions = assistant && message.content ? `<div class="message-actions"><button data-copy-index="${index}">复制答案</button><button data-favorite-index="${index}">收藏上一道题</button></div>` : "";
    return `<div class="message-row ${assistant ? "assistant" : "user"}"><div class="message-avatar">${assistant ? "∑" : "我"}</div><div class="message-content">${trace}${bubble}${actions}</div></div>`;
  }).join("");
}

function formulaEditorMarkup() {
  const templateButtons = FORMULA_TEMPLATES.map(([label, value, hint]) => `<button type="button" class="formula-template" data-formula-template="${escapeHtml(value)}" title="${escapeHtml(hint)}"><span>${label}</span><strong>${escapeHtml(value)}</strong></button>`).join("");
  const symbolPanes = FORMULA_SYMBOL_GROUPS.map(([label, symbols]) => `<div class="formula-symbol-group"><span>${label}</span><div class="formula-symbol-grid">${symbols.map((symbol) => `<button type="button" data-formula-symbol="${escapeHtml(symbol)}">${escapeHtml(symbol)}</button>`).join("")}</div></div>`).join("");
  const greekPanes = FORMULA_GREEK_GROUPS.map(([label, symbols]) => `<div class="formula-symbol-group"><span>${label}</span><div class="formula-symbol-grid">${symbols.map((symbol) => `<button type="button" data-formula-symbol="${escapeHtml(symbol)}">${escapeHtml(symbol)}</button>`).join("")}</div></div>`).join("");
  return `<div class="formula-editor-panel" id="formula-editor-panel" hidden><div class="formula-editor-header"><div><strong>公式工具</strong><span>像 Word 一样选择结构和符号，不需要写 LaTeX</span></div><button class="formula-editor-clear" id="clear-formula-editor" type="button">清空</button></div><div class="formula-editor-input" id="formula-editor-input" contenteditable="true" role="textbox" aria-label="可视化公式输入" data-placeholder="先选择一个结构，例如分数、根式或矩阵"></div><div class="formula-editor-controls"><button type="button" class="formula-control" id="formula-next-placeholder">下一个占位框 Tab</button><span>每个 □ 都可以替换，结构符号也可以直接修改</span></div><div class="formula-tabbar" role="tablist"><button type="button" class="formula-tab active" data-formula-tab="structures">结构</button><button type="button" class="formula-tab" data-formula-tab="symbols">符号</button><button type="button" class="formula-tab" data-formula-tab="greek">希腊字母</button></div><div class="formula-tab-pane" data-formula-pane="structures"><div class="formula-template-grid">${templateButtons}</div></div><div class="formula-tab-pane" data-formula-pane="symbols" hidden>${symbolPanes}</div><div class="formula-tab-pane" data-formula-pane="greek" hidden>${greekPanes}</div><button class="btn primary formula-insert" id="insert-formula" type="button">插入到题目</button></div>`;
}

function solvePage() {
  const mode = state.answerMode;
  const memoryStatus = mode === "agent"
    ? `<div class="memory-status ${state.loading ? "active" : ""}"><span class="memory-status-dot"></span>${state.memorySummary ? "已启用压缩摘要，较早对话已整理" : "带上下文追问，过长时自动压缩旧内容"}</div>`
    : `<div class="memory-status"><span class="memory-status-dot"></span>单题模式不会发送之前的对话，响应更快</div>`;
  const generationButton = state.loading
    ? `<button class="btn danger" id="stop-generation" type="button">暂停输出</button>`
    : `<button class="btn primary" id="send-question">开始解题  →</button>`;
  return `<div class="hero-card"><span class="hero-chip">学习模式 · 逐步讲解</span><h2>把不会的题，变成会做的题。</h2><p>不用担心问得不完整。你可以直接粘贴题目，也可以继续追问“为什么”，我们一起把思路理清楚。</p></div>
    <div class="page-grid"><div><div class="section-heading"><h3>数学对话</h3><p>支持 Markdown 与可视化公式输入</p></div><div class="card chat-card"><div class="chat-toolbar"><strong>解题空间</strong><span>${state.loading ? "正在生成答案…" : "准备好了"}</span></div><div class="chat-messages" id="chat-messages">${renderMessages()}</div><div class="composer"><div class="answer-mode-switch" role="group" aria-label="答题模式"><button class="mode-button ${mode === "solve" ? "active" : ""}" data-answer-mode="solve" type="button"><strong>单题解答</strong><span>不带历史，速度更快</span></button><button class="mode-button ${mode === "agent" ? "active" : ""}" data-answer-mode="agent" type="button"><strong>Agent 模式</strong><span>带上下文，自动选工具并展示轨迹</span></button></div>${memoryStatus}<textarea id="question-input" placeholder="例如：求解方程 x² - 5x + 6 = 0，最好解释每一步…"></textarea>${formulaEditorMarkup()}<div class="composer-actions"><span class="composer-hint">Enter 提交 · Shift + Enter 换行</span><div class="composer-buttons"><button class="btn ghost" id="toggle-formula-editor" type="button">公式工具 ∑</button><button class="btn ghost" id="save-current-favorite">收藏题目</button><button class="btn ghost" id="clear-chat">清空对话</button>${generationButton}</div></div></div></div></div>
      <div class="side-stack"><div class="card side-card"><h3>知识库文档</h3><div class="knowledge-mini-list">${knowledgeMiniList()}</div></div><div class="card side-card"><h3>试试这些题</h3><div class="example-list">${EXAMPLES.map(([label, question]) => `<button class="example-btn" data-example="${escapeHtml(question)}"><strong>${label}</strong><br>${escapeHtml(question)}</button>`).join("")}</div></div></div>`;
}

function knowledgeMiniList() {
  const view = state.solveDocs;
  if (!view.docs.length) {
    return `<div class="obs-empty">${view.loaded ? "知识库暂无文档" : "正在读取文档…"}</div>`;
  }
  return view.docs
    .map((doc) => `<button class="knowledge-mini-item" data-open-doc="${escapeHtml(doc.name)}"><span class="knowledge-doc-name">${escapeHtml(doc.name)}</span><span class="knowledge-doc-meta">${Math.max(1, Math.round(doc.size / 1024))} KB</span></button>`)
    .join("");
}

async function loadSolveDocs() {
  const view = state.solveDocs;
  if (view.loaded || view.loading) return;
  view.loading = true;
  try {
    const response = await fetch(`${apiBase()}/knowledge/documents`);
    if (response.ok) view.docs = await response.json();
  } catch {
    // Ignore: the sidebar list is best-effort.
  }
  view.loading = false;
  view.loaded = true;
  if (state.page === "solve") render();
}

function openKnowledgeDoc(name) {
  state.knowledge.current = name;
  setPage("knowledge");
}

function favoritesPage() {
  const favorites = getFavorites();
  const list = favorites.length ? `<div class="item-list">${favorites.map((favorite, index) => `<div class="card list-item favorite-card" data-favorite-card="${index}"><div class="list-main"><strong title="${escapeHtml(favorite.question)}">${escapeHtml(favorite.question)}</strong><p>收藏题目 · ${index + 1}${favorite.answer ? " · 已保存回答" : " · 只有题目"}</p></div><span class="favorite-check-slot"></span></div>`).join("")}</div>` : `<div class="card empty-state"><strong>还没有收藏题目</strong><p>在解题页面点击“收藏当前题目”，把值得复习的题目放到这里。</p></div>`;
  return `<div class="page-title-row favorite-page-heading"><div><h2>我的收藏</h2><p>把错过的、重要的和想复习的题目集中管理。</p></div><div class="page-actions">${favorites.length ? '<button class="btn danger" id="clear-favorites">清空收藏</button>' : ""}</div></div>${list}`;
}

function favoriteDetailPage() {
  const favorites = getFavorites();
  const index = Math.min(Math.max(state.favoriteDetailIndex, 0), Math.max(favorites.length - 1, 0));
  const favorite = favorites[index];
  if (!favorite) return `<div class="card empty-state"><strong>找不到这条收藏</strong><p>这道题可能已经被移除。</p><button class="btn primary" id="back-to-favorites">返回我的收藏</button></div>`;
  state.favoriteDetailIndex = index;
  return `<div class="page-title-row favorite-detail-heading"><div><h2>收藏题目详情</h2><p>第 ${index + 1} 道收藏题目</p></div><div class="page-actions"><button class="btn" id="back-to-favorites">返回我的收藏</button><button class="btn primary" id="practice-favorite-detail">开始练习</button></div></div><div class="card favorite-detail-view"><div class="detail-section"><strong>我的问题</strong><div class="detail-content">${markdownToHtml(favorite.question)}</div></div><div class="detail-section"><strong>模型回答</strong><div class="detail-content">${favorite.answer ? markdownToHtml(favorite.answer) : '<span class="muted">这条收藏是在旧版本中保存的，没有记录模型回答。</span>'}</div></div></div>`;
}

function historyPage() {
  const history = getHistory();
  const list = history.length ? `<div class="item-list">${history.map((item, index) => `<div class="card list-item"><div class="list-main"><strong title="${escapeHtml(item.question)}">${escapeHtml(item.question)}</strong><p>${new Date(item.createdAt).toLocaleString()} · ${item.answer ? "已完成" : "未完成"}</p></div><div class="list-actions"><button class="btn" data-use-history="${index}">再次练习</button></div></div>`).join("")}</div>` : `<div class="card empty-state"><strong>还没有学习记录</strong><p>完成一道题后，这里会自动保留最近的练习记录。</p></div>`;
  return `<div class="page-title-row"><div><h2>学习记录</h2><p>回顾最近的问题，找到自己最需要巩固的知识点。</p></div><div class="page-actions">${history.length ? '<button class="btn danger" id="clear-history">清空记录</button>' : ""}</div></div><div class="stat-grid"><div class="card stat-card"><div class="stat-label">最近完成</div><div class="stat-value">${history.filter((item) => item.answer).length}</div><div class="stat-note">道练习题</div></div><div class="card stat-card"><div class="stat-label">收藏题目</div><div class="stat-value">${getFavorites().length}</div><div class="stat-note">可随时复习</div></div><div class="card stat-card"><div class="stat-label">学习建议</div><div class="stat-value">${history.length ? "继续" : "开始"}</div><div class="stat-note">保持练习节奏</div></div></div>${list}`;
}

function toolkitPage() {
  const tools = [
    ["分步讲解", "让助手把每一步的依据写出来，适合第一次接触新题型。", "请分步骤讲解这道题，每一步都说明依据。"],
    ["方法对比", "同一道题尝试两种方法，比较它们什么时候更方便。", "请用两种不同的方法解这道题，并比较它们的优缺点。"],
    ["错因分析", "不要只看答案，先定位自己是概念、计算还是审题出了问题。", "请分析我这道题错在哪里，并告诉我应该复习什么。"],
    ["举一反三", "解完一道题后生成相似但数字不同的新题，帮助真正掌握。", "请根据这道题生成一道相似的新题，先不要给答案。"],
  ];
  return `<div class="page-title-row"><div><h2>学习工具</h2><p>把同一个模型变成适合不同学习阶段的助手。</p></div></div><div class="tool-grid">${tools.map(([title, desc, instruction]) => `<div class="card tool-panel"><h3>${title}</h3><div class="tool-description">${markdownToHtml(desc)}</div><button class="btn primary" data-tool-instruction="${escapeHtml(instruction)}">使用这个模式</button></div>`).join("")}</div><div class="section-heading"><h3>常见公式速查</h3><p>先理解，再记忆</p></div><div class="tool-grid"><div class="card tool-panel"><h3>一元二次方程</h3><div class="tool-description">${markdownToHtml("当 $a \\\\ne 0$ 时，方程 $ax^2+bx+c=0$ 的根为：")}</div><div class="formula">${markdownToHtml("$x = \\\\frac{-b \\\\pm \\\\sqrt{b^2 - 4ac}}{2a}$")}</div></div><div class="card tool-panel"><h3>等差数列</h3><div class="tool-description">${markdownToHtml("首项为 $a_1$，公差为 $d$ 的数列：")}</div><div class="formula">${markdownToHtml("$a_n = a_1 + (n - 1)d$")}</div></div></div>`;
}

function formatPercent(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatSeconds(milliseconds) {
  if (!milliseconds) return "—";
  return milliseconds >= 10000
    ? `${(milliseconds / 1000).toFixed(1)}s`
    : `${(milliseconds / 1000).toFixed(2)}s`;
}

function distributionBars(data) {
  const entries = Object.entries(data || {});
  if (!entries.length) return `<div class="obs-empty">暂无数据</div>`;
  const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
  return entries.map(([name, count]) => `
    <div class="obs-bar-row">
      <span class="obs-bar-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
      <span class="obs-bar-track"><span class="obs-bar-fill" style="width:${Math.round((count / total) * 100)}%"></span></span>
      <span class="obs-bar-count">${count}</span>
    </div>`).join("");
}

function metricCard(label, value, note) {
  return `<div class="card stat-card"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value">${escapeHtml(value)}</div><div class="stat-note">${escapeHtml(note || "")}</div></div>`;
}

function observabilityMetrics(metrics) {
  const latency = metrics.latency_ms || {};
  const tokens = metrics.tokens || {};
  const stops = Object.entries(metrics.stopped_reasons || {}).map(([name, count]) => `${name} ${count}`).join(" · ");
  return `
    <div class="stat-grid">
      ${metricCard("运行总数", String(metrics.runs ?? 0), "当前窗口")}
      ${metricCard("失败率", formatPercent(metrics.failure_rate), stops || "无失败")}
      ${metricCard("完成率", formatPercent(metrics.completion_rate), "有非空答案")}
      ${metricCard("回退率", formatPercent(metrics.fallback_rate), `${metrics.runs_with_fallback ?? 0}/${metrics.runs ?? 0} 次运行发生过回退`)}
      ${metricCard("延迟 P50", formatSeconds(latency.p50), `P95 ${formatSeconds(latency.p95)} · 最大 ${formatSeconds(latency.max)}`)}
      ${metricCard("Token 总量", String(tokens.total ?? 0), `每次均 ${tokens.avg_per_run ?? 0}`)}
    </div>
    <div class="obs-grid">
      <div class="card card-pad">
        <h3 class="obs-title">路由分布（decision.tool）</h3>
        ${distributionBars(metrics.routes)}
        <h3 class="obs-title">意图分布</h3>
        ${distributionBars(metrics.intents)}
        <h3 class="obs-title">工具实际调用</h3>
        ${distributionBars(metrics.tool_calls)}
      </div>
      <div class="card card-pad">
        <h3 class="obs-title">验证结论</h3>
        ${distributionBars(metrics.verification)}
        <h3 class="obs-title">答案模型</h3>
        ${distributionBars(metrics.models)}
        <div class="obs-note">幻觉率代理 ${formatPercent(metrics.hallucination_rate)}（证伪 ${metrics.refuted ?? 0} · 来源未支持 ${metrics.ungrounded ?? 0}）</div>
      </div>
    </div>`;
}

function observabilityRun(trace) {
  const ok = !trace.error && trace.stopped_reason === "final";
  const verify = trace.verification;
  const usage = trace.usage || {};
  const time = String(trace.started_at || "").replace("T", " ").slice(5, 16);
  const tools = (trace.observations || []).map((observation) => `
    <div class="trace-step ${observation.success ? "ok" : "fail"}">
      <span class="trace-step-dot"></span>
      <span class="trace-step-name">${escapeHtml(observation.tool)}</span>
      <span class="trace-step-detail">${observation.success ? "成功" : "失败"} · ${(observation.duration_ms / 1000).toFixed(1)}s</span>
    </div>`).join("");
  return `<details class="card run-card">
    <summary class="run-summary">
      <span class="run-time">${escapeHtml(time)}</span>
      <span class="run-question" title="${escapeHtml(trace.question)}">${escapeHtml(trace.question)}</span>
      <span class="run-badge ${ok ? "ok" : "fail"}">${escapeHtml(trace.stopped_reason || "")}</span>
      <span class="run-meta">${formatSeconds(trace.duration_ms)} · ${usage.total_tokens ?? 0} tok</span>
    </summary>
    <div class="run-body">
      <div class="run-row"><span>run_id</span><code>${escapeHtml(trace.run_id)}</code></div>
      <div class="run-row"><span>路由</span><b>${trace.decision ? `${escapeHtml(trace.decision.intent)} → ${escapeHtml(trace.decision.tool)}` : "—"}</b></div>
      <div class="run-row"><span>验证</span><b>${verify ? `${escapeHtml(VERIFY_LABEL[verify.status] || verify.status)} · ${escapeHtml(VERIFY_METHOD[verify.method] || verify.method)}` : "未验证"}</b></div>
      <div class="run-row"><span>答案模型</span><b>${escapeHtml(trace.answer_model || "—")}</b></div>
      <div class="run-row"><span>模型调用</span><b>${usage.calls ?? 0} 次 · 回退 ${trace.fallbacks ?? 0} 次</b></div>
      ${trace.error ? `<div class="run-row fail"><span>错误</span><b>${escapeHtml(trace.error)}</b></div>` : ""}
      ${tools ? `<div class="trace-steps">${tools}</div>` : ""}
      ${verify && verify.detail ? `<div class="obs-note">${escapeHtml(verify.detail)}</div>` : ""}
      <div class="run-answer">${escapeHtml((trace.answer || "").slice(0, 400)) || "（无答案）"}</div>
    </div>
  </details>`;
}

function observabilityPage() {
  const view = state.observability;
  const options = [[1, "最近 1 天"], [7, "最近 7 天"], [30, "最近 30 天"], [0, "全部"]];
  const header = `<div class="page-title-row"><div><h2>运行观测</h2><p>每次 Agent 运行的调用链与整体指标，数据来自后端 <code>/api/metrics</code> 与 <code>/api/traces</code>。</p></div><div class="page-actions"><select id="obs-days" class="obs-select">${options.map(([value, label]) => `<option value="${value}" ${view.days === value ? "selected" : ""}>${label}</option>`).join("")}</select><button class="btn" id="obs-refresh">刷新</button></div></div>`;
  if (view.loading) return `${header}<div class="card empty-state"><strong>正在读取运行记录…</strong><p>需要后端已启动。</p></div>`;
  if (view.error) return `${header}<div class="card empty-state"><strong>读取失败</strong><p>${escapeHtml(view.error)}</p><button class="btn primary" id="obs-refresh">重试</button></div>`;
  if (!view.metrics) return `${header}<div class="card empty-state"><strong>还没有数据</strong><p>点右上角「刷新」读取运行记录。</p></div>`;
  const runs = view.traces.length
    ? view.traces.slice().reverse().map(observabilityRun).join("")
    : `<div class="card empty-state"><strong>窗口内没有运行记录</strong><p>在「开始解题」里用 Agent 模式提一个问题，这里就会出现一次运行。</p></div>`;
  return `${header}${observabilityMetrics(view.metrics)}<div class="section-heading"><h3>逐次运行</h3><p>按任务展开查看调用链，最新的在最上面</p></div>${runs}`;
}

async function loadObservability() {
  const view = state.observability;
  view.loading = true;
  view.error = "";
  render();
  try {
    const [metricsResponse, tracesResponse] = await Promise.all([
      fetch(`${apiBase()}/metrics?days=${view.days}`),
      fetch(`${apiBase()}/traces?limit=50`),
    ]);
    if (!metricsResponse.ok) throw new Error(`指标接口失败（HTTP ${metricsResponse.status}）`);
    if (!tracesResponse.ok) throw new Error(`明细接口失败（HTTP ${tracesResponse.status}）`);
    view.metrics = await metricsResponse.json();
    view.traces = await tracesResponse.json();
  } catch (error) {
    view.error = error.message || String(error);
  } finally {
    view.loading = false;
    if (state.page === "observability") render();
  }
}

const RAG_GROUNDING_LABEL = { verified: "已接地", unknown: "未接地", none: "未检查" };
const RAG_RETRIEVAL_LABEL = { ok: "检索成功", empty: "无命中", failed: "检索失败", none: "未检索" };

function ragRetrievalBadge(item) {
  const status = item.retrieval || "none";
  if (status === "none") return "";
  const cls = status === "ok" ? "ok" : status === "failed" ? "fail" : "";
  const title = item.retrieval_error ? ` title="${escapeHtml(item.retrieval_error)}"` : "";
  return `<span class="run-badge ${cls}"${title}>${escapeHtml(RAG_RETRIEVAL_LABEL[status] || status)}</span>`;
}

function ragGroundingBadge(status) {
  const cls = status === "verified" ? "ok" : status === "unknown" ? "fail" : "";
  return `<span class="run-badge ${cls}">${escapeHtml(RAG_GROUNDING_LABEL[status] || status || "未检查")}</span>`;
}

function ragAttributionRun(item) {
  const time = String(item.started_at || "").replace("T", " ").slice(5, 16);
  const sources = (item.sources || []).length ? item.sources.join("、") : "（未检索到片段）";
  const citations = item.citations || [];
  const cited = citations.length
    ? citations.map((citation) => `${citation.source}${citation.start_line ? ` 第 ${citation.start_line}-${citation.end_line} 行` : ""}`).join("；")
    : "（未判定引用片段）";
  const unsupported = item.unsupported || [];
  return `<details class="card run-card" data-rag-card="${escapeHtml(item.run_id)}">
    <summary class="run-summary">
      <span class="run-time">${escapeHtml(time)}</span>
      <span class="run-question" title="${escapeHtml(item.question)}">${escapeHtml(item.question)}</span>
      <span class="run-badge ok">知识库</span>
      ${ragRetrievalBadge(item)}
      ${ragGroundingBadge(item.grounding)}
      <span class="run-meta">${item.retrieved_count ?? 0} 片段 · 引用 ${citations.length} · ${formatSeconds(item.duration_ms)}</span>
    </summary>
    <div class="run-body">
      <div class="run-row"><span>run_id</span><code>${escapeHtml(item.run_id)}</code></div>
      <div class="run-row"><span>路由</span><b>${escapeHtml(item.intent || "—")} → ${escapeHtml(item.tool || "—")}</b></div>
      <div class="run-row"><span>检索查询</span><b>${escapeHtml(item.query || "—")}</b></div>
      <div class="run-row"><span>命中来源</span><b>${escapeHtml(sources)}</b></div>
      <div class="run-row"><span>引用位置</span><b>${escapeHtml(cited)}</b></div>
      ${unsupported.length ? `<div class="run-row fail"><span>未支持的结论</span><b>${escapeHtml(unsupported.join("；"))}</b></div>` : ""}
      <div class="rag-detail" id="rag-detail-${escapeHtml(item.run_id)}"></div>
      <div class="page-actions">
        <button class="btn" data-rag-detail="${escapeHtml(item.run_id)}">加载详情</button>
        <button class="btn primary" data-rag-check="${escapeHtml(item.run_id)}">重新核对</button>
      </div>
    </div>
  </details>`;
}

function ragAttributionPage() {
  const view = state.ragAttribution;
  const options = [[1, "最近 1 天"], [7, "最近 7 天"], [30, "最近 30 天"], [0, "全部"]];
  const header = `<div class="page-title-row"><div><h2>知识库归因</h2><p>只统计走过知识库的回答：检索到了什么、引用了哪些、是否被检索资料支持。数据来自 <code>/api/rag/attribution</code>。</p></div><div class="page-actions"><select id="rag-days" class="obs-select">${options.map(([value, label]) => `<option value="${value}" ${view.days === value ? "selected" : ""}>${label}</option>`).join("")}</select><button class="btn" id="rag-refresh">刷新</button></div></div>`;
  if (view.loading) return `${header}<div class="card empty-state"><strong>正在读取归因数据…</strong><p>需要后端已启动。</p></div>`;
  if (view.error) return `${header}<div class="card empty-state"><strong>读取失败</strong><p>${escapeHtml(view.error)}</p><button class="btn primary" id="rag-refresh">重试</button></div>`;
  if (!view.summary) return `${header}<div class="card empty-state"><strong>还没有数据</strong><p>点右上角「刷新」读取运行记录。</p></div>`;
  const summary = view.summary;
  const cards = `<div class="stat-grid">
    ${metricCard("走知识库", formatPercent(summary.knowledge_rate), `全部 ${summary.runs ?? 0} 次运行中 ${summary.knowledge_runs ?? 0} 次`)}
    ${metricCard("平均检索片段", String(summary.avg_retrieved ?? 0), "每次知识库回答")}
    ${metricCard("检索失败", String(summary.retrieval_failures ?? 0), `无命中 ${summary.retrieval?.empty ?? 0} 次`)}
    ${metricCard("接地准确率", formatPercent(summary.grounded_rate), `已接地 ${summary.grounding?.verified ?? 0} · 未接地 ${summary.grounding?.unknown ?? 0}`)}
    ${metricCard("未接地回答", String(summary.ungrounded_runs ?? 0), "资料未支持的结论")}
    ${metricCard("未检查", String(summary.grounding?.none ?? 0), "未做接地判定")}
  </div>`;
  const knowledgeRuns = view.runs.filter((item) => item.used_knowledge);
  const runs = knowledgeRuns.length
    ? knowledgeRuns.map(ragAttributionRun).join("")
    : `<div class="card empty-state"><strong>窗口内没有走知识库的回答</strong><p>在「开始解题」里用 Agent 模式提一个知识类问题，这里就会出现一次归因。</p></div>`;
  return `${header}${cards}<div class="section-heading"><h3>走知识库的回答</h3><p>展开查看检索片段、引用与接地结论，最新的在最上面</p></div>${runs}`;
}

async function loadRagAttribution() {
  const view = state.ragAttribution;
  view.loading = true;
  view.error = "";
  render();
  try {
    const response = await fetch(`${apiBase()}/rag/attribution?days=${view.days}&limit=50`);
    if (!response.ok) throw new Error(`归因接口失败（HTTP ${response.status}）`);
    const payload = await response.json();
    view.summary = payload.summary;
    view.runs = payload.runs || [];
  } catch (error) {
    view.error = error.message || String(error);
  } finally {
    view.loading = false;
    if (state.page === "rag-attribution") render();
  }
}

function ragDetailMarkup(trace) {
  const rag = trace.rag || {};
  const retrieved = rag.retrieved || [];
  const grounding = rag.grounding;
  const citations = grounding ? (grounding.citations || []) : [];
  const summary = state.ragAttribution.summary;
  const runRate = grounding ? (grounding.grounded ? "100%" : "0%") : "—";
  const windowRate = summary ? formatPercent(summary.grounded_rate) : "—";
  const searchObservations = (trace.observations || []).filter((item) => item.tool === "search_knowledge");
  const failedObservation = searchObservations.find((item) => !item.success);
  const retrievalStatus = failedObservation ? "failed" : retrieved.length ? "ok" : (rag.used_knowledge ? "empty" : "none");
  const retrievalText = RAG_RETRIEVAL_LABEL[retrievalStatus] || "—";
  const stats = `<div class="rag-stat-row">
    <div class="rag-stat"><span>相关片段</span><strong>${retrieved.length} 个</strong></div>
    <div class="rag-stat"><span>引用来源</span><strong>${citations.length} 个</strong></div>
    <div class="rag-stat"><span>检索状态</span><strong>${escapeHtml(retrievalText)}</strong>${failedObservation ? `<em>${escapeHtml(failedObservation.error || "")}</em>` : ""}</div>
    <div class="rag-stat"><span>接地准确率</span><strong>${runRate}</strong><em>窗口 ${windowRate}</em></div>
  </div>`;
  const chunks = retrieved.map((item) => `
    <div class="rag-chunk">
      <div class="rag-chunk-head"><span class="rag-rank">#${item.rank}</span><code>${escapeHtml(item.source)}</code>${item.start_line ? `<span class="rag-lines">第 ${item.start_line}-${item.end_line} 行</span>` : ""}<span class="rag-distance">距离 ${item.distance === null || item.distance === undefined ? "—" : Number(item.distance).toFixed(3)}</span></div>
      <div class="rag-chunk-body">${escapeHtml(item.snippet || "")}</div>
    </div>`).join("") || `<div class="obs-empty">未检索到知识库片段</div>`;
  const citationText = citations.length
    ? citations.map((item) => `${item.source}${item.start_line ? ` 第 ${item.start_line}-${item.end_line} 行` : ""}`).join("；")
    : "（未判定引用了哪些片段）";
  const groundingLine = grounding
    ? `${grounding.grounded ? "已接地" : "未接地"}：${escapeHtml(grounding.reason || "")}`
    : "未做接地检查";
  return `
    ${stats}
    <div class="rag-detail-block"><strong>检索片段</strong>${chunks}</div>
    <div class="rag-detail-block"><strong>引用位置</strong><div class="obs-note">${escapeHtml(citationText)}</div></div>
    <div class="rag-detail-block"><strong>接地结论</strong><div class="obs-note">${groundingLine}</div></div>`;
}

async function loadRagDetail(runId) {
  const target = document.getElementById(`rag-detail-${runId}`);
  if (!target) return;
  target.innerHTML = `<div class="obs-empty">正在加载详情…</div>`;
  try {
    const response = await fetch(`${apiBase()}/rag/attribution/${encodeURIComponent(runId)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    target.innerHTML = ragDetailMarkup(await response.json());
    renderMath();
  } catch (error) {
    target.innerHTML = `<div class="obs-note">加载失败：${escapeHtml(error.message || String(error))}</div>`;
  }
}

async function checkRagGrounding(runId) {
  try {
    const response = await fetch(`${apiBase()}/rag/attribution/${encodeURIComponent(runId)}/check`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    toast(payload.grounded ? "已接地：回答有检索资料支持。" : "未接地：存在资料未支持的结论。");
    await loadRagAttribution();
  } catch (error) {
    toast(`核对失败：${error.message || error}`);
  }
}

async function loadKnowledge() {
  const view = state.knowledge;
  view.loading = true;
  view.error = "";
  render();
  try {
    const response = await fetch(`${apiBase()}/knowledge/documents`);
    if (!response.ok) throw new Error(`文档接口失败（HTTP ${response.status}）`);
    view.docs = await response.json();
    if (!view.current || !view.docs.some((doc) => doc.name === view.current)) {
      view.current = view.docs.length ? view.docs[0].name : "";
    }
  } catch (error) {
    view.error = error.message || String(error);
  } finally {
    view.loading = false;
  }
  if (view.current) {
    await loadKnowledgeDoc(view.current, false);
  }
  if (state.page === "knowledge") render();
}

async function loadKnowledgeDoc(name, doRender = true) {
  const view = state.knowledge;
  view.current = name;
  try {
    const response = await fetch(`${apiBase()}/knowledge/documents/${encodeURIComponent(name)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    view.content = payload.content;
  } catch (error) {
    view.content = `加载失败：${error.message || error}`;
  }
  if (doRender && state.page === "knowledge") render();
}

function knowledgePage() {
  const view = state.knowledge;
  const header = `<div class="page-title-row"><div><h2>知识库文档</h2><p>RAG 检索用的 Markdown 文档，可对照「知识库归因」里的引用行范围阅读。</p></div><div class="page-actions"><button class="btn" id="knowledge-refresh">刷新</button></div></div>`;
  if (view.loading && !view.docs.length) return `${header}<div class="card empty-state"><strong>正在读取文档列表…</strong><p>需要后端已启动。</p></div>`;
  if (view.error && !view.docs.length) return `${header}<div class="card empty-state"><strong>读取失败</strong><p>${escapeHtml(view.error)}</p><button class="btn primary" id="knowledge-refresh">重试</button></div>`;
  const list = view.docs.length
    ? view.docs.map((doc) => `<button class="knowledge-doc-item ${doc.name === view.current ? "active" : ""}" data-knowledge-doc="${escapeHtml(doc.name)}"><span class="knowledge-doc-name">${escapeHtml(doc.name)}</span><span class="knowledge-doc-meta">${Math.max(1, Math.round(doc.size / 1024))} KB</span></button>`).join("")
    : `<div class="obs-empty">knowledge/ 目录下没有文档</div>`;
  const body = view.content ? markdownToHtml(view.content) : `<div class="obs-empty">选择左侧文档查看内容</div>`;
  return `${header}<div class="knowledge-layout"><div class="card knowledge-list">${list}</div><div class="card knowledge-view" id="knowledge-view">${body}</div></div>`;
}

function settingsPage() {
  const dark = document.body.classList.contains("dark");
  return `<div class="page-title-row"><div><h2>设置</h2><p>这些设置只保存在当前浏览器，不会上传到服务器。</p></div></div><div class="card card-pad settings-form"><div class="field"><label for="api-base-input">FastAPI 服务地址</label><input id="api-base-input" value="${escapeHtml(apiBase())}" placeholder="http://127.0.0.1:8080/api"/><small>默认由本地启动器提供。不要在这里填写 API Key，密钥只应放在后端环境变量中。</small></div><div class="setting-row"><div><strong>深色模式</strong><p>适合晚上学习，切换后会立即生效。</p></div><button class="switch ${dark ? "on" : ""}" id="settings-theme-toggle" aria-label="切换深色模式"></button></div><div class="setting-row"><div><strong>本地数据</strong><p>收藏和学习记录只保存在浏览器 localStorage。</p></div><button class="btn danger" id="clear-local-data">清除本地数据</button></div><div class="page-actions"><button class="btn primary" id="save-settings">保存设置</button></div></div>`;
}

function render() {
  setActiveNav(); updateCounts();
  pageContainer.innerHTML = state.page === "solve" ? solvePage() : state.page === "favorites" ? favoritesPage() : state.page === "favorite-detail" ? favoriteDetailPage() : state.page === "history" ? historyPage() : state.page === "toolkit" ? toolkitPage() : state.page === "observability" ? observabilityPage() : state.page === "rag-attribution" ? ragAttributionPage() : state.page === "knowledge" ? knowledgePage() : settingsPage();
  renderMarkdownSurfaces();
  renderFavoriteQuestionPreviews();
  renderFavoriteToolbar();
  renderMath();
  bindPageEvents();
  if (state.page === "solve") loadSolveDocs();
  scrollChatToLatest();
}

function scrollChatToLatest() {
  if (!state.loading || !state.chatPinnedToBottom) return;
  requestAnimationFrame(() => {
    const chat = $("#chat-messages");
    if (chat) chat.scrollTop = chat.scrollHeight;
  });
}

let streamingAnswerUpdatePending = false;
function updateStreamingAnswer() {
  if (streamingAnswerUpdatePending) return;
  streamingAnswerUpdatePending = true;
  requestAnimationFrame(() => {
    streamingAnswerUpdatePending = false;
    const answer = [...state.messages].reverse().find((message) => message.role === "assistant");
    const bubbles = document.querySelectorAll("#chat-messages .message-row.assistant .message-bubble");
    const bubble = bubbles[bubbles.length - 1];
    if (!answer || !bubble) return;
    bubble.innerHTML = markdownToHtml(answer.content);
    renderMath(bubble);
    scrollChatToLatest();
  });
}

function currentAssistantMessage() {
  return state.messages.length ? state.messages[state.messages.length - 1] : null;
}

const TRACE_THINKING = {
  classify: "正在分析问题类型…",
  decide: "正在判断下一步…",
  answer: "正在生成答案…",
  verify: "正在独立验证答案…",
};

const VERIFY_LABEL = { verified: "已验证", refuted: "已证伪", unknown: "无法验证" };
const SWITCH_STAGE = { classify: "路由", decide: "规划", answer: "回答", verify: "验证" };
const VERIFY_METHOD = {
  substitution: "代入检验",
  expression: "独立求值",
  grounding: "来源核对",
  llm_judge: "模型评审",
  none: "",
};

function modelTag(model) {
  return model ? `<span class="trace-model">${escapeHtml(model)}</span>` : "";
}

const VERIFY_SECTION_WORDS = ["检查与验证", "验证与检验", "检验", "验证"];
const FINAL_SECTION_WORDS = ["最终答案", "结论", "答案"];

function isSectionLine(line, keywords) {
  const trimmed = String(line || "").trim();
  if (!trimmed) return false;
  if (/^#{1,6}\s+/.test(trimmed)) {
    const plain = trimmed.replace(/^#{1,6}\s+/, "").replace(/[*_`]/g, "").trim();
    return keywords.some((word) => plain.startsWith(word));
  }
  const plain = trimmed.replace(/[*_`]/g, "").trim().replace(/[：:]$/, "").trim();
  return keywords.some((word) => plain === word);
}

function splitAnswer(content) {
  const lines = String(content || "").replace(/\r\n/g, "\n").split("\n");
  const marks = [];
  const verifyIndex = lines.findIndex((line) => isSectionLine(line, VERIFY_SECTION_WORDS));
  const finalIndex = lines.findIndex((line) => isSectionLine(line, FINAL_SECTION_WORDS));
  if (verifyIndex >= 0) marks.push({ kind: "verify", index: verifyIndex });
  if (finalIndex >= 0) marks.push({ kind: "final", index: finalIndex });
  if (!marks.length) return { solution: lines.join("\n").trim(), verifySection: "", finalSection: "" };
  marks.sort((left, right) => left.index - right.index);
  const solution = lines.slice(0, marks[0].index).join("\n").trim();
  let verifySection = "";
  let finalSection = "";
  marks.forEach((mark, position) => {
    const end = position + 1 < marks.length ? marks[position + 1].index : lines.length;
    const chunk = lines.slice(mark.index, end).join("\n").trim();
    if (!chunk) return;
    if (mark.kind === "verify") verifySection = verifySection ? `${verifySection}\n${chunk}` : chunk;
    else finalSection = finalSection ? `${finalSection}\n${chunk}` : chunk;
  });
  return { solution, verifySection, finalSection };
}

function renderTraceMarkup(trace, content = "") {
  if (!trace) return "";
  const { solution, verifySection, finalSection } = splitAnswer(content);
  const route = trace.decision
    ? `<div class="trace-route"><span class="trace-label">路由</span><span class="trace-badge">${escapeHtml(trace.decision.intent)}</span><span class="trace-arrow">→</span><span class="trace-badge tool">${escapeHtml(trace.decision.tool)}</span>${modelTag(trace.routeModel)}</div>`
    : `<div class="trace-route"><span class="trace-label">路由</span><span class="trace-muted">正在分析问题类型…</span></div>`;
  const steps = (trace.steps || []).map((step) => {
    const stateClass = step.status === "running" ? "running" : step.success ? "ok" : "fail";
    const detail = step.status === "running"
      ? "调用中…"
      : `${step.success ? "成功" : "失败"} · ${(step.duration_ms / 1000).toFixed(1)}s`;
    return `<div class="trace-step ${stateClass}"><span class="trace-step-dot"></span><span class="trace-step-name">${escapeHtml(step.tool)}</span>${modelTag(step.model)}<span class="trace-step-detail">${escapeHtml(detail)}</span></div>`;
  }).join("");
  const thinkingText = trace.status === "running" && trace.thinking ? (TRACE_THINKING[trace.thinking] || "处理中…") : "";
  const solveThinking = thinkingText && trace.thinking !== "verify" ? `<div class="trace-thinking">${escapeHtml(thinkingText)}</div>` : "";
  const solveAnswer = solution
    ? `<div class="trace-block-label">解题回答</div><div class="trace-answer">${markdownToHtml(solution)}</div>`
    : "";
  const solveBlock = `<div class="trace-block solve"><div class="trace-block-head"><span class="trace-block-step">1</span><span class="trace-block-title">解题路径</span><span class="trace-block-note">路由到解题工具，执行求解过程</span></div>${route}${steps ? `<div class="trace-steps">${steps}</div>` : ""}${solveThinking}${solveAnswer}</div>`;

  const switches = (trace.switches || []).map((item) => {
    const stage = SWITCH_STAGE[item.stage] || item.stage || "编排";
    const reason = item.reason ? ` · ${escapeHtml(item.reason)}` : "";
    return `<div class="trace-switch"><span class="trace-switch-icon">⇄</span><span>${escapeHtml(stage)}阶段切换：<b>${escapeHtml(item.from || "云端模型")}</b> → <b>${escapeHtml(item.to || "本地模型")}</b>${reason}</span></div>`;
  }).join("");
  const switchBlock = switches
    ? `<div class="trace-block switches"><div class="trace-block-head"><span class="trace-block-step">⇄</span><span class="trace-block-title">模型切换</span><span class="trace-block-note">编排模型回退到本地模型</span></div>${switches}</div>`
    : "";

  const retries = (trace.retries || []).map((item) => `<div class="trace-retry">第 ${item.attempt} 次修正：${escapeHtml(item.reason || "答案被证伪")}</div>`).join("");
  const verifying = trace.status === "running" && trace.thinking === "verify";
  const verifySkipped = trace.verificationSkipped || "";
  const verification = trace.verification
    ? `<div class="trace-verify ${escapeHtml(trace.verification.status)}"><span class="trace-verify-dot"></span><span class="trace-verify-label">验证 · ${escapeHtml(VERIFY_LABEL[trace.verification.status] || trace.verification.status)}</span><span class="trace-step-detail">${escapeHtml(VERIFY_METHOD[trace.verification.method] || "")}</span>${modelTag(trace.verifyModel)}</div>`
    : "";
  const verifyDetail = trace.verification && trace.verification.detail
    ? `<div class="trace-verify-detail">${escapeHtml(trace.verification.detail)}</div>`
    : "";
  const verifyNotice = verifySkipped
    ? `<div class="trace-notice skipped"><span class="trace-notice-icon">⚑</span><span>${escapeHtml(verifySkipped)}</span></div>`
    : `<div class="trace-notice"><span class="trace-notice-icon">⚑</span><span>已路由到验证路径：由独立模型重新检查答案是否成立，避免自证。</span></div>`;
  const verifyAnswer = (verifySection || finalSection)
    ? `<div class="trace-block-label">验证回答</div>${verifySection ? `<div class="trace-verify-answer">${markdownToHtml(verifySection)}</div>` : ""}${finalSection ? `<div class="trace-final-answer">${markdownToHtml(finalSection)}</div>` : ""}`
    : "";
  const verifyBlock = (verification || retries || verifying || verifyAnswer || verifySkipped)
    ? `<div class="trace-block verify${verifySkipped ? " skipped" : ""}"><div class="trace-block-head"><span class="trace-block-step">2</span><span class="trace-block-title">验证路径</span><span class="trace-block-note">${verifySkipped ? "已跳过" : "独立模型复核"}</span></div>${verifyNotice}${verifying ? `<div class="trace-thinking">正在独立验证答案…</div>` : verification}${verifyDetail}${retries}${verifyAnswer}</div>`
    : "";

  const answerNote = trace.answerModel ? ` · 答案由 ${escapeHtml(trace.answerModel)} 生成` : "";
  const footer = trace.stoppedReason ? `<div class="trace-footer">结束原因：${escapeHtml(trace.stoppedReason)}${answerNote}</div>` : "";
  const status = trace.status === "running" ? "运行中" : "已完成";
  return `<div class="trace-head"><span class="trace-title">Agent 执行轨迹</span><span class="trace-status">${status}</span></div>${solveBlock}${switchBlock}${verifyBlock}${footer}`;
}

let traceCardUpdatePending = false;
function updateTraceCard() {
  // Only the solve page renders a trace card. Calling render() from here while
  // another page is open would rebuild it and collapse any expanded run in
  // 「运行观测」, so a run in progress must not touch other pages.
  if (state.page !== "solve") return;
  if (traceCardUpdatePending) return;
  traceCardUpdatePending = true;
  requestAnimationFrame(() => {
    traceCardUpdatePending = false;
    const message = currentAssistantMessage();
    if (!message || !message.trace) return;
    const cards = document.querySelectorAll("#chat-messages .trace-card");
    const card = cards[cards.length - 1];
    if (!card) { render(); return; }
    card.innerHTML = renderTraceMarkup(message.trace, message.content);
    renderMath(card);
    scrollChatToLatest();
  });
}

function handleAgentEvent(event) {
  const message = currentAssistantMessage();
  if (!message) return;
  if (!message.trace) message.trace = { status: "running", decision: null, steps: [], switches: [], thinking: "", retries: [], verification: null, verificationSkipped: "", routeModel: "", verifyModel: "", answerModel: "", stoppedReason: "" };
  const trace = message.trace;
  if (event.type === "thinking") {
    trace.thinking = event.stage;
  } else if (event.type === "route") {
    trace.decision = event.decision;
    trace.routeModel = event.model || "";
    trace.thinking = "";
  } else if (event.type === "tool_start") {
    trace.thinking = "";
    trace.steps.push({ step: event.step, tool: event.tool, model: event.model || "", status: "running", success: null, duration_ms: 0 });
  } else if (event.type === "tool_end") {
    const step = trace.steps.find((item) => item.step === event.step && item.tool === event.tool);
    if (step) {
      step.status = "done"; step.success = event.success; step.duration_ms = event.duration_ms;
      if (event.model) step.model = event.model;
    } else {
      trace.steps.push({ step: event.step, tool: event.tool, model: event.model || "", status: "done", success: event.success, duration_ms: event.duration_ms });
    }
  } else if (event.type === "answer_delta") {
    trace.thinking = "";
    message.content += event.content || "";
  } else if (event.type === "model_switch") {
    trace.switches.push({ stage: event.stage, from: event.from_model, to: event.to_model, reason: event.reason || "" });
  } else if (event.type === "verify_start") {
    trace.thinking = "verify";
  } else if (event.type === "verify_skipped") {
    trace.thinking = "";
    trace.verificationSkipped = event.reason || "已跳过独立验证";
  } else if (event.type === "verify") {
    trace.thinking = "";
    trace.verification = event.verification;
    trace.verifyModel = event.model || "";
  } else if (event.type === "retry") {
    trace.thinking = "";
    trace.retries.push({ attempt: event.attempt, reason: event.reason });
  } else if (event.type === "answer_reset") {
    trace.thinking = "answer";
    message.content = "";
  } else if (event.type === "done") {
    trace.status = "done";
    trace.thinking = "";
    trace.stoppedReason = event.stopped_reason || "";
    trace.answerModel = event.answer_model || "";
    if (event.verification) trace.verification = event.verification;
    if (event.run && typeof event.run.answer === "string" && event.run.answer) {
      message.content = event.run.answer;
    }
  } else if (event.type === "error") {
    throw new Error(event.message || "Agent 请求失败");
  }
  updateTraceCard();
}

function renderFavoriteToolbar() {
  if (state.page !== "favorites") return;
  const actions = document.querySelector(".favorite-page-heading .page-actions");
  const favorites = getFavorites();
  if (!actions || !favorites.length) return;

  state.selectedFavoriteIndex = Math.min(Math.max(state.selectedFavoriteIndex, 0), favorites.length - 1);
  const selectedIndex = state.selectedFavoriteIndex;
  const clearButton = $("#clear-favorites");
  const select = document.createElement("select");
  select.className = "btn favorite-toolbar-control favorite-selector";
  select.setAttribute("aria-label", "选择收藏题目");
  favorites.forEach((favorite, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    const preview = plainTextPreview(favorite.question);
    option.textContent = (index + 1) + ". " + preview.slice(0, 24) + (preview.length > 24 ? "…" : "");
    select.appendChild(option);
  });
  select.value = String(selectedIndex);
  select.onchange = () => { state.selectedFavoriteIndex = Number(select.value); render(); };

  const toggleButton = document.createElement("button");
  toggleButton.className = "btn favorite-toolbar-control";
  toggleButton.textContent = "查看详情";
  toggleButton.onclick = () => showFavoriteDetail(selectedIndex);

  const useButton = document.createElement("button");
  useButton.className = "btn favorite-toolbar-control";
  useButton.textContent = "开始练习";
  useButton.onclick = () => useQuestion(favorites[selectedIndex].question);

  const deleteButton = document.createElement("button");
  deleteButton.className = "btn danger favorite-toolbar-control";
  const selectedForDeletion = () => state.favoriteDeleteSelection.size;
  const updateDeleteButton = () => {
    deleteButton.textContent = state.favoriteDeleteMode
      ? `确认移除${selectedForDeletion() ? ` (${selectedForDeletion()})` : ""}`
      : "移除";
  };
  updateDeleteButton();
  deleteButton.onclick = () => {
    if (!state.favoriteDeleteMode) {
      state.favoriteDeleteMode = true;
      state.favoriteDeleteSelection = new Set();
      render();
      return;
    }
    if (!selectedForDeletion()) return toast("请先勾选要移除的题目。");
    const remaining = favorites.filter((_, index) => !state.favoriteDeleteSelection.has(index));
    writeJson(STORAGE.favorites, remaining);
    state.favoriteDeleteMode = false;
    state.favoriteDeleteSelection = new Set();
    state.selectedFavoriteIndex = Math.min(state.selectedFavoriteIndex, Math.max(remaining.length - 1, 0));
    render();
    toast("已移除选中的收藏题目。");
  };

  [select, toggleButton, useButton, deleteButton].forEach((control) => actions.insertBefore(control, clearButton || null));
  document.querySelectorAll(".favorite-card").forEach((card, index) => {
    card.classList.toggle("selected", index === selectedIndex);
    const checkSlot = card.querySelector(".favorite-check-slot");
    if (checkSlot) {
      checkSlot.replaceChildren();
      if (state.favoriteDeleteMode) {
        const label = document.createElement("label");
        label.className = "favorite-check-label";
        label.title = "选择要移除的题目";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = state.favoriteDeleteSelection.has(index);
        checkbox.setAttribute("aria-label", `选择收藏题目 ${index + 1}`);
        checkbox.onchange = () => {
          if (checkbox.checked) state.favoriteDeleteSelection.add(index);
          else state.favoriteDeleteSelection.delete(index);
          updateDeleteButton();
        };
        label.appendChild(checkbox);
        checkSlot.appendChild(label);
      }
    }
    card.onclick = (event) => {
      if (event.target.closest("a, button, select, input, label")) return;
      state.selectedFavoriteIndex = index;
      render();
    };
  });
}

function mathFallback(formula) {
  let text = escapeHtml(formula)
    .replace(/\\\\(?=(?:frac|pm|sqrt|ne|left|right|leq?|geq?|times|cdot|begin|end)\b)/g, "\\")
    .replace(/\\left|\\right/g, "")
    .replace(/\\pm/g, "±")
    .replace(/\\ne/g, "≠")
    .replace(/\\leq?/g, "≤")
    .replace(/\\geq?/g, "≥")
    .replace(/\\times/g, "×")
    .replace(/\\cdot/g, "·");
  text = text.replace(/\\begin\{(pmatrix|bmatrix|Bmatrix|vmatrix)\}([\s\S]*?)\\end\{\1\}/g, (_, env, body) => `<span class="matrix ${env}">${body.split(/\\\\/).map((row) => `<span class="matrix-row">${row.split("&").map((cell) => `<span class="matrix-cell">${cell.trim()}</span>`).join("")}</span>`).join("")}</span>`);
  text = text.replace(/\\frac\{(.+)\}\{([^{}]+)\}/g, '<span class="math-frac"><span class="math-frac-num">$1</span><span class="math-frac-den">$2</span></span>');
  text = text.replace(/\\sqrt\{([^{}]+)\}/g, '√<span class="sqrt-body">$1</span>');
  return text
    .replace(/\^\{([^}]+)\}/g, "<sup>$1</sup>")
    .replace(/\^([A-Za-z0-9])/g, "<sup>$1</sup>")
    .replace(/_\{([^}]+)\}/g, "<sub>$1</sub>")
    .replace(/_([A-Za-z0-9])/g, "<sub>$1</sub>");
}

function renderMarkdownSurfaces() {
  document.querySelectorAll(".example-btn").forEach((button) => {
    const label = button.querySelector("strong")?.textContent || "";
    button.innerHTML = `<strong>${escapeHtml(label)}</strong><br>${markdownToHtml(button.dataset.example || "")}`;
  });
}

function plainTextPreview(source) {
  return String(source || "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$([\s\S]*?)\$\$/g, " $1 ")
    .replace(/\\\[([\s\S]*?)\\\]/g, " $1 ")
    .replace(/\$([^$\n]+?)\$/g, " $1 ")
    .replace(/\\\(([^\n]*?)\\\)/g, " $1 ")
    .replace(/^\s*#{1,6}\s+/gm, "")
    .replace(/[*_`>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function renderFavoriteQuestionPreviews() {
  document.querySelectorAll(".favorite-card .list-main strong").forEach((node) => {
    const preview = document.createElement("div");
    preview.className = "favorite-question-preview";
    preview.textContent = plainTextPreview(node.getAttribute("title") || node.textContent);
    node.replaceWith(preview);
  });
}

function renderMath(root = document) {
  root.querySelectorAll("[data-katex]").forEach((element) => {
    if (element.dataset.katexRendered === "true") return;
    if (!window.katex) {
      element.innerHTML = mathFallback(element.dataset.katex);
      element.dataset.katexRendered = "fallback";
      return;
    }
    try {
      const formula = element.dataset.katex.replace(/\\\\(?=(?:frac|pm|sqrt|ne|left|right|leq?|geq?|times|cdot|begin|end)\b)/g, "\\");
      window.katex.render(formula, element, {
        displayMode: element.classList.contains("math-block"),
        throwOnError: false,
        strict: false,
      });
      element.dataset.katexRendered = "true";
    } catch { /* Keep the readable text fallback when a formula is malformed. */ }
  });
}

function addFavorite(question, answer = "") {
  const value = String(question || "").trim(); if (!value) return toast("请先输入一道题目。");
  const favorites = getFavorites();
  if (favorites.some((item) => item.question === value)) return toast("这道题已经在收藏中了。");
  writeJson(STORAGE.favorites, [{ question: value, answer: String(answer || ""), createdAt: Date.now() }, ...favorites].slice(0, 50)); updateCounts(); toast("已收藏当前题目和回答。");
}
function addHistory(question, answer) {
  const history = getHistory().filter((item) => item.question !== question);
  writeJson(STORAGE.history, [{ question, answer, createdAt: Date.now() }, ...history].slice(0, 30));
}
function useQuestion(question) { state.page = "solve"; state.messages = []; state.memorySummary = ""; state.memoryCursor = 0; location.hash = "solve"; render(); const input = $("#question-input"); input.value = question; input.focus(); }
function appendInstruction(instruction) { const input = $("#question-input"); if (!input) return; input.value = input.value.trim() ? `${input.value.trim()}\n\n${instruction}` : instruction; input.focus(); }

function insertAtTextareaCursor(input, value) {
  if (!input) return;
  const start = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? start;
  input.value = input.value.slice(0, start) + value + input.value.slice(end);
  input.focus();
  const cursor = start + value.length;
  input.setSelectionRange(cursor, cursor);
}

function insertIntoFormulaEditor(value, selectFirstPlaceholder = false) {
  const editor = $("#formula-editor-input");
  if (!editor) return;
  editor.focus();
  const selection = window.getSelection();
  let range;
  if (selection && selection.rangeCount && editor.contains(selection.getRangeAt(0).commonAncestorContainer)) {
    range = selection.getRangeAt(0);
  } else {
    range = document.createRange();
    range.selectNodeContents(editor);
    range.collapse(false);
  }
  range.deleteContents();
  const node = document.createTextNode(value);
  range.insertNode(node);
  const placeholderIndex = selectFirstPlaceholder ? value.indexOf("□") : -1;
  if (placeholderIndex >= 0) {
    range.setStart(node, placeholderIndex);
    range.setEnd(node, placeholderIndex + 1);
  } else {
    range.setStartAfter(node);
    range.collapse(true);
  }
  selection?.removeAllRanges();
  selection?.addRange(range);
}

function selectNextFormulaPlaceholder() {
  const editor = $("#formula-editor-input");
  if (!editor) return;
  const selection = window.getSelection();
  let offset = 0;
  if (selection && selection.rangeCount && editor.contains(selection.anchorNode)) {
    const range = document.createRange();
    range.selectNodeContents(editor);
    range.setEnd(selection.anchorNode, selection.anchorOffset);
    offset = range.toString().length + (selection.isCollapsed ? 0 : 1);
  }
  const text = editor.textContent || "";
  let index = text.indexOf("□", offset);
  if (index < 0) index = text.indexOf("□");
  if (index < 0) return;
  const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
  let node;
  let position = 0;
  while ((node = walker.nextNode())) {
    const end = position + node.nodeValue.length;
    if (index >= position && index < end) {
      const range = document.createRange();
      range.setStart(node, index - position);
      range.setEnd(node, index - position + 1);
      selection?.removeAllRanges();
      selection?.addRange(range);
      editor.focus();
      return;
    }
    position = end;
  }
}

async function sendQuestion() {
  const input = $("#question-input"); const question = input.value.trim(); if (!question || state.loading) return;
  const requestController = new AbortController();
  state.activeRequestController = requestController;
  const assistantMessage = { role: "assistant", content: "" };
  if (state.answerMode === "agent") {
    assistantMessage.trace = { status: "running", decision: null, steps: [], switches: [], thinking: "classify", retries: [], verification: null, verificationSkipped: "", routeModel: "", verifyModel: "", answerModel: "", stoppedReason: "" };
  }
  state.loading = true; state.chatPinnedToBottom = true; state.messages.push({ role: "user", content: question }, assistantMessage); render();
  try {
    let endpoint;
    let body;
    if (state.answerMode === "agent") {
      const allMessages = state.messages.slice(0, -1).map(({ role, content }) => ({ role, content }));
      const recentBudget = Math.max(600, MEMORY_CONFIG.maxContextTokens - MEMORY_CONFIG.maxOutputTokens - 450);
      const [oldMessages, recentMessages] = splitHistory(
        allMessages,
        MEMORY_CONFIG.maxRecentMessages,
        recentBudget,
      );
      state.memoryCursor = Math.min(state.memoryCursor, oldMessages.length);
      const newOldMessages = oldMessages.slice(state.memoryCursor);
      if (shouldSummarize(state.memorySummary, allMessages) && newOldMessages.length) {
        try {
          state.memorySummary = await requestConversationSummary(newOldMessages, state.memorySummary, requestController.signal);
          state.memoryCursor = oldMessages.length;
          toast("较早对话已自动压缩，后续响应会更稳定。");
        } catch (summaryError) {
          if (summaryError.name === "AbortError") throw summaryError;
          // The current question can still be answered from the recent window.
          toast("自动摘要暂时失败，将只使用最近对话继续回答。");
        }
      }
      endpoint = "/agent/stream";
      body = { question, max_steps: 4, messages: recentMessages, summary: state.memorySummary || undefined };
    } else {
      endpoint = "/solve";
      body = { question, stream: true };
      state.memorySummary = "";
      state.memoryCursor = 0;
    }
    const response = await fetch(`${apiBase()}${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: requestController.signal });
    if (!response.ok) throw new Error(`后端请求失败（HTTP ${response.status}）`);
    const reader = response.body.getReader(); const decoder = new TextDecoder("utf-8"); let buffer = "";
    while (true) {
      const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(/\r?\n\r?\n/); buffer = parts.pop() || "";
      for (const part of parts) {
        const line = part.split("\n").find((item) => item.startsWith("data:")); if (!line) continue;
        const payloadText = line.slice(5).trim(); if (!payloadText || payloadText === "[DONE]") continue;
        let payload; try { payload = JSON.parse(payloadText); } catch { continue; }
        if (payload.error) throw new Error(payload.error);
        if (payload.type) { handleAgentEvent(payload); continue; }
        if (payload.content && state.messages.length) { state.messages[state.messages.length - 1].content += payload.content; updateStreamingAnswer(); }
      }
    }
    if (state.messages.length) addHistory(question, state.messages[state.messages.length - 1].content);
  } catch (error) {
    const last = state.messages[state.messages.length - 1];
    if (last?.trace) {
      last.trace.status = "done";
      last.trace.steps.forEach((step) => { if (step.status === "running") { step.status = "done"; step.success = false; } });
    }
    if (error.name === "AbortError") {
      if (last) {
        last.content = last.content.trim()
          ? `${last.content}\n\n*（输出已暂停）*`
          : "*（输出已暂停，尚未生成回答）*";
        addHistory(question, last.content);
      }
      toast("已暂停模型输出。");
    } else {
      if (last) last.content = `**暂时无法完成请求**\n\n${error.message}`;
      toast("请求失败，请检查后端和模型服务状态。");
    }
  } finally {
    if (state.activeRequestController === requestController) state.activeRequestController = null;
    state.loading = false; render(); checkHealth();
  }
}

function stopGeneration() {
  if (!state.loading || !state.activeRequestController) return;
  state.activeRequestController.abort();
}

async function checkHealth() {
  const set = (dot, title, detail, status) => { dot.className = `status-dot ${status}`; title.textContent = detail; };
  try {
    const response = await fetch(`${apiBase()}/health`, { signal: AbortSignal.timeout(5000) }); const payload = await response.json();
    state.health = "online"; set($("#header-status-dot"), $("#header-status"), "服务正常 · " + (payload.model || "模型已连接"), "online"); set($("#sidebar-status-dot"), $("#sidebar-status-title"), "服务正常", "online"); $("#sidebar-status-detail").textContent = payload.model || "模型已连接";
  } catch { state.health = "offline"; set($("#header-status-dot"), $("#header-status"), "服务未连接", "offline"); set($("#sidebar-status-dot"), $("#sidebar-status-title"), "服务未连接", "offline"); $("#sidebar-status-detail").textContent = "请检查 FastAPI"; }
}

function bindPageEvents() {
  document.querySelectorAll("[data-page]").forEach((button) => button.onclick = () => setPage(button.dataset.page));
  document.querySelectorAll("[data-answer-mode]").forEach((button) => button.onclick = () => {
    const mode = button.dataset.answerMode;
    state.answerMode = mode === "agent" ? "agent" : "solve";
    if (state.answerMode !== "agent") {
      state.memorySummary = "";
      state.memoryCursor = 0;
    }
    render();
  });
  document.querySelectorAll("#obs-refresh").forEach((button) => button.addEventListener("click", loadObservability));
  $("#obs-days")?.addEventListener("change", (event) => {
    state.observability.days = Number(event.target.value) || 0;
    loadObservability();
  });
  document.querySelectorAll("#rag-refresh").forEach((button) => button.addEventListener("click", loadRagAttribution));
  $("#rag-days")?.addEventListener("change", (event) => {
    state.ragAttribution.days = Number(event.target.value) || 0;
    loadRagAttribution();
  });
  document.querySelectorAll("[data-rag-detail]").forEach((button) => button.addEventListener("click", () => loadRagDetail(button.dataset.ragDetail)));
  document.querySelectorAll("[data-rag-check]").forEach((button) => button.addEventListener("click", () => checkRagGrounding(button.dataset.ragCheck)));
  document.querySelectorAll("#knowledge-refresh").forEach((button) => button.addEventListener("click", loadKnowledge));
  document.querySelectorAll("[data-knowledge-doc]").forEach((button) => button.addEventListener("click", () => loadKnowledgeDoc(button.dataset.knowledgeDoc)));
  document.querySelectorAll("[data-open-doc]").forEach((button) => button.onclick = () => openKnowledgeDoc(button.dataset.openDoc));
  $("#stop-generation")?.addEventListener("click", stopGeneration);
  $("#chat-messages")?.addEventListener("scroll", (event) => {
    const chat = event.currentTarget;
    state.chatPinnedToBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 48;
  });
  $("#send-question")?.addEventListener("click", sendQuestion); $("#save-current-favorite")?.addEventListener("click", () => { const question = $("#question-input").value; const answer = [...state.messages].reverse().find((message) => message.role === "assistant")?.content || ""; addFavorite(question, answer); }); $("#clear-chat")?.addEventListener("click", () => { stopGeneration(); state.messages = []; state.memorySummary = ""; state.memoryCursor = 0; render(); });
  $("#question-input")?.addEventListener("keydown", (event) => { if (event.isComposing) return; if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(); } });
  document.querySelectorAll("[data-example]").forEach((button) => button.onclick = () => { $("#question-input").value = button.dataset.example; $("#question-input").focus(); });
  document.querySelectorAll("[data-instruction], [data-tool-instruction]").forEach((button) => button.onclick = () => appendInstruction(button.dataset.instruction || button.dataset.toolInstruction));
  $("#toggle-formula-editor")?.addEventListener("click", () => {
    const panel = $("#formula-editor-panel");
    if (!panel) return;
    panel.hidden = !panel.hidden;
    if (!panel.hidden) $("#formula-editor-input")?.focus();
  });
  document.querySelectorAll("[data-formula-symbol]").forEach((button) => {
    button.onmousedown = (event) => event.preventDefault();
    button.onclick = () => insertIntoFormulaEditor(button.dataset.formulaSymbol || "");
  });
  document.querySelectorAll("[data-formula-template]").forEach((button) => {
    button.onmousedown = (event) => event.preventDefault();
    button.onclick = () => insertIntoFormulaEditor(button.dataset.formulaTemplate || "", true);
  });
  document.querySelectorAll("[data-formula-tab]").forEach((button) => {
    button.onclick = () => {
      const tab = button.dataset.formulaTab;
      document.querySelectorAll("[data-formula-tab]").forEach((item) => item.classList.toggle("active", item === button));
      document.querySelectorAll("[data-formula-pane]").forEach((pane) => { pane.hidden = pane.dataset.formulaPane !== tab; });
    };
  });
  $("#formula-next-placeholder")?.addEventListener("click", selectNextFormulaPlaceholder);
  $("#formula-editor-input")?.addEventListener("keydown", (event) => {
    if (event.key === "Tab") {
      event.preventDefault();
      selectNextFormulaPlaceholder();
    }
  });
  $("#clear-formula-editor")?.addEventListener("click", () => { $("#formula-editor-input").textContent = ""; $("#formula-editor-input").focus(); });
  $("#insert-formula")?.addEventListener("click", () => {
    const formula = $("#formula-editor-input")?.textContent.trim();
    if (!formula) return toast("请先输入或选择公式符号。");
    insertAtTextareaCursor($("#question-input"), formula);
    $("#formula-editor-panel").hidden = true;
  });
  document.querySelectorAll("[data-copy-index]").forEach((button) => button.onclick = async () => { await navigator.clipboard.writeText(state.messages[Number(button.dataset.copyIndex)].content); toast("答案已复制。"); });
  document.querySelectorAll("[data-favorite-index]").forEach((button) => button.onclick = () => { const answerIndex = Number(button.dataset.favoriteIndex); const user = [...state.messages.slice(0, answerIndex)].reverse().find((message) => message.role === "user"); addFavorite(user?.content, state.messages[answerIndex]?.content); });
  $("#clear-favorites")?.addEventListener("click", () => { localStorage.removeItem(STORAGE.favorites); state.selectedFavoriteIndex = 0; state.favoriteDetailIndex = 0; state.favoriteDeleteMode = false; state.favoriteDeleteSelection = new Set(); render(); toast("收藏已清空。"); });
  $("#back-to-favorites")?.addEventListener("click", () => setPage("favorites"));
  $("#practice-favorite-detail")?.addEventListener("click", () => useQuestion(getFavorites()[state.favoriteDetailIndex]?.question));
  document.querySelectorAll("[data-use-history]").forEach((button) => button.onclick = () => useQuestion(getHistory()[Number(button.dataset.useHistory)].question));
  $("#clear-history")?.addEventListener("click", () => { localStorage.removeItem(STORAGE.history); render(); toast("学习记录已清空。"); });
  $("#save-settings")?.addEventListener("click", () => { localStorage.setItem(STORAGE.apiBase, $("#api-base-input").value.trim().replace(/\/$/, "")); toast("设置已保存，正在检查服务。"); checkHealth(); });
  $("#settings-theme-toggle")?.addEventListener("click", toggleTheme);
  $("#clear-local-data")?.addEventListener("click", () => { localStorage.removeItem(STORAGE.favorites); localStorage.removeItem(STORAGE.history); state.selectedFavoriteIndex = 0; state.favoriteDetailIndex = 0; state.favoriteDeleteMode = false; state.favoriteDeleteSelection = new Set(); toast("本地收藏和学习记录已清除。"); render(); });
}
function toggleTheme() { const dark = document.body.classList.toggle("dark"); localStorage.setItem(STORAGE.theme, dark ? "dark" : "light"); render(); }

document.querySelectorAll("[data-page]").forEach((button) => button.onclick = () => setPage(button.dataset.page));
$("#mobile-menu").onclick = () => $("#sidebar").classList.toggle("open"); $("#sidebar-refresh").onclick = checkHealth; $("#theme-toggle").onclick = toggleTheme;
window.addEventListener("hashchange", () => {
  const hash = location.hash.slice(1);
  const detail = hash.match(/^favorite-detail\/(\d+)$/);
  state.page = detail ? "favorite-detail" : (PAGE_META[hash] ? hash : "solve");
  if (detail) state.favoriteDetailIndex = Number(detail[1]);
  render();
  if (state.page === "observability") loadObservability();
  if (state.page === "rag-attribution") loadRagAttribution();
  if (state.page === "knowledge") loadKnowledge();
});
if (localStorage.getItem(STORAGE.theme) === "dark") document.body.classList.add("dark");
render(); checkHealth();
if (state.page === "observability") loadObservability();
if (state.page === "rag-attribution") loadRagAttribution();
if (state.page === "knowledge") loadKnowledge();
