const STORAGE = {
  favorites: "mathllm_favorites",
  history: "mathllm_recent_history",
  apiBase: "mathllm_api_base",
  theme: "mathllm_theme",
};

const PAGE_META = {
  solve: ["学习工作区 / 开始解题", "今天想解决哪道题？"],
  favorites: ["学习工作区 / 我的收藏", "把值得复习的题目留下来"],
  history: ["学习工作区 / 学习记录", "看看自己最近解决了什么"],
  toolkit: ["学习工作区 / 学习工具", "用适合自己的方式理解数学"],
  settings: ["系统 / 设置", "调整你的本地学习空间"],
};

const EXAMPLES = [
  ["一元二次方程", "解方程 $x^2-5x+6=0$。"],
  ["函数极值", "求函数 $f(x)=x^3-3x+1$ 的极值点。"],
  ["矩阵行列式", "计算矩阵 \\begin{pmatrix}1&2\\3&4\\end{pmatrix} 的行列式。"],
  ["概率计算", "盒中有 3 个红球和 2 个白球，随机取出 2 个，求恰好取到 1 个红球的概率。"],
];

const state = {
  page: location.hash.slice(1) || "solve",
  messages: [],
  loading: false,
  health: "checking",
};

const $ = (selector) => document.querySelector(selector);
const pageContainer = $("#page-container");

function readJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function writeJson(key, value) { localStorage.setItem(key, JSON.stringify(value)); }
function getFavorites() { return readJson(STORAGE.favorites, []).filter(Boolean); }
function getHistory() { return readJson(STORAGE.history, []).filter((item) => item && item.question); }
function apiBase() {
  return (localStorage.getItem(STORAGE.apiBase) || "http://127.0.0.1:8080/api").replace(/\/$/, "");
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
  text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  text = text.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
  text = text.replace(/\$([^$\n]+)\$/g, (_, formula) => `<span class="math-inline" data-katex="${escapeHtml(formula)}">${escapeHtml(formula)}</span>`);
  text = text.replace(/\\\(([^\n]*?)\\\)/g, (_, formula) => `<span class="math-inline" data-katex="${escapeHtml(formula)}">${escapeHtml(formula)}</span>`);
  return text;
}

function markdownToHtml(source) {
  let text = String(source || "").replace(/\r\n/g, "\n");
  const blocks = [];
  const stash = (html) => { const token = `MATHLLM_BLOCK_${blocks.length}`; blocks.push([token, html]); return token; };
  text = text.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => stash(`<pre><code>${escapeHtml(code.trim())}</code></pre>`));
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => { const value = formula.trim(); return stash(`<div class="math-block" data-katex="${escapeHtml(value)}">${escapeHtml(value)}</div>`); });
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => { const value = formula.trim(); return stash(`<div class="math-block" data-katex="${escapeHtml(value)}">${escapeHtml(value)}</div>`); });

  const lines = text.split("\n");
  const output = [];
  let paragraph = [];
  let inList = false;
  const flushParagraph = () => {
    if (paragraph.length) { output.push(`<p>${inlineMarkdown(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`); paragraph = []; }
  };
  const closeList = () => { if (inList) { output.push("</ul>"); inList = false; } };
  for (const line of lines) {
    if (/^\s*$/.test(line)) { flushParagraph(); closeList(); continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) { flushParagraph(); closeList(); output.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`); continue; }
    const list = line.match(/^\s*[-*]\s+(.+)$/);
    if (list) { flushParagraph(); if (!inList) { output.push("<ul>"); inList = true; } output.push(`<li>${inlineMarkdown(list[1])}</li>`); continue; }
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) { flushParagraph(); closeList(); output.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`); continue; }
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
}
function setActiveNav() {
  document.querySelectorAll(".nav-item[data-page]").forEach((button) => button.classList.toggle("active", button.dataset.page === state.page));
  const meta = PAGE_META[state.page]; $("#page-kicker").textContent = meta[0]; $("#page-title").textContent = meta[1];
}
function updateCounts() { $("#favorite-count").textContent = getFavorites().length; }

function renderMessages() {
  if (!state.messages.length) return `<div class="empty-chat"><div><div class="empty-icon">∑</div><h3>从一道题开始吧</h3><p>输入题目后，我会尽量把思路、公式和每一步原因讲清楚。</p></div></div>`;
  return state.messages.map((message, index) => {
    const assistant = message.role === "assistant";
    const content = assistant ? markdownToHtml(message.content) : escapeHtml(message.content).replace(/\n/g, "<br>");
    const actions = assistant && message.content ? `<div class="message-actions"><button data-copy-index="${index}">复制答案</button><button data-favorite-index="${index}">收藏上一道题</button></div>` : "";
    return `<div class="message-row ${assistant ? "assistant" : "user"}"><div class="message-avatar">${assistant ? "∑" : "我"}</div><div><div class="message-bubble">${content}</div>${actions}</div></div>`;
  }).join("");
}

function solvePage() {
  const quick = [
    ["💡", "只给我提示", "先自己想一想", "请先给我解题思路和关键提示，不要直接给出最终答案。"],
    ["🧩", "讲简单一点", "适合初学者", "请用适合初学者的简单语言解释，并说明每一步为什么这样做。"],
    ["✅", "检查我的答案", "找出思路漏洞", "请检查我写出的答案或思路，指出错误并给出改进建议。"],
  ];
  return `<div class="hero-card"><span class="hero-chip">学习模式 · 逐步讲解</span><h2>把不会的题，变成会做的题。</h2><p>不用担心问得不完整。你可以直接粘贴题目，也可以继续追问“为什么”，我们一起把思路理清楚。</p></div>
    <div class="page-grid"><div><div class="section-heading"><h3>数学对话</h3><p>支持 Markdown 与 LaTeX 公式</p></div><div class="card chat-card"><div class="chat-toolbar"><strong>解题空间</strong><span>${state.loading ? "正在生成答案…" : "准备好了"}</span></div><div class="chat-messages" id="chat-messages">${renderMessages()}</div><div class="composer"><textarea id="question-input" placeholder="例如：求解方程 x² - 5x + 6 = 0，最好解释每一步…"></textarea><div class="composer-actions"><span class="composer-hint">Ctrl + Enter 快速提交</span><div class="composer-buttons"><button class="btn ghost" id="save-current-favorite">收藏题目</button><button class="btn ghost" id="clear-chat">清空对话</button><button class="btn primary" id="send-question">${state.loading ? "生成中…" : "开始解题  →"}</button></div></div></div></div></div>
      <div class="side-stack"><div class="card side-card"><h3>试试这些题</h3><div class="example-list">${EXAMPLES.map(([label, question]) => `<button class="example-btn" data-example="${escapeHtml(question)}"><strong>${label}</strong><br>${escapeHtml(question)}</button>`).join("")}</div></div><div class="card side-card"><h3>快捷学习工具</h3>${quick.map(([icon, title, desc, instruction]) => `<button class="quick-tool" data-instruction="${escapeHtml(instruction)}"><span class="tool-icon">${icon}</span><span><strong>${title}</strong><span>${desc}</span></span></button>`).join("")}</div></div></div>`;
}

function favoritesPage() {
  const favorites = getFavorites();
  const list = favorites.length ? `<div class="item-list">${favorites.map((question, index) => `<div class="card list-item"><div class="list-main"><strong title="${escapeHtml(question)}">${escapeHtml(question)}</strong><p>收藏题目 · ${index + 1}</p></div><div class="list-actions"><button class="btn" data-use-favorite="${index}">开始练习</button><button class="btn danger" data-delete-favorite="${index}">移除</button></div></div>`).join("")}</div>` : `<div class="card empty-state"><strong>还没有收藏题目</strong><p>在解题页面点击“收藏当前题目”，把值得复习的题目放到这里。</p></div>`;
  return `<div class="page-title-row"><div><h2>我的收藏</h2><p>把错过的、重要的和想复习的题目集中管理。</p></div><div class="page-actions">${favorites.length ? '<button class="btn danger" id="clear-favorites">清空收藏</button>' : ""}</div></div>${list}`;
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
  return `<div class="page-title-row"><div><h2>学习工具</h2><p>把同一个模型变成适合不同学习阶段的助手。</p></div></div><div class="tool-grid">${tools.map(([title, desc, instruction]) => `<div class="card tool-panel"><h3>${title}</h3><p>${desc}</p><button class="btn primary" data-tool-instruction="${escapeHtml(instruction)}">使用这个模式</button></div>`).join("")}</div><div class="section-heading"><h3>常见公式速查</h3><p>先理解，再记忆</p></div><div class="tool-grid"><div class="card tool-panel"><h3>一元二次方程</h3><p>当 $a \\ne 0$ 时，方程 $ax^2+bx+c=0$ 的根为：</p><div class="formula">x = (-b ± √(b² - 4ac)) / 2a</div></div><div class="card tool-panel"><h3>等差数列</h3><p>首项为 $a_1$，公差为 $d$ 的数列：</p><div class="formula">aₙ = a₁ + (n - 1)d</div></div></div>`;
}

function settingsPage() {
  const dark = document.body.classList.contains("dark");
  return `<div class="page-title-row"><div><h2>设置</h2><p>这些设置只保存在当前浏览器，不会上传到服务器。</p></div></div><div class="card card-pad settings-form"><div class="field"><label for="api-base-input">FastAPI 服务地址</label><input id="api-base-input" value="${escapeHtml(apiBase())}" placeholder="http://127.0.0.1:8080/api"/><small>默认由本地启动器提供。不要在这里填写 API Key，密钥只应放在后端环境变量中。</small></div><div class="setting-row"><div><strong>深色模式</strong><p>适合晚上学习，切换后会立即生效。</p></div><button class="switch ${dark ? "on" : ""}" id="settings-theme-toggle" aria-label="切换深色模式"></button></div><div class="setting-row"><div><strong>本地数据</strong><p>收藏和学习记录只保存在浏览器 localStorage。</p></div><button class="btn danger" id="clear-local-data">清除本地数据</button></div><div class="page-actions"><button class="btn primary" id="save-settings">保存设置</button></div></div>`;
}

function render() {
  setActiveNav(); updateCounts();
  pageContainer.innerHTML = state.page === "solve" ? solvePage() : state.page === "favorites" ? favoritesPage() : state.page === "history" ? historyPage() : state.page === "toolkit" ? toolkitPage() : settingsPage();
  renderMath();
  bindPageEvents();
}

function mathFallback(formula) {
  return escapeHtml(formula)
    .replace(/\^\{([^}]+)\}/g, "<sup>$1</sup>")
    .replace(/\^([A-Za-z0-9])/g, "<sup>$1</sup>")
    .replace(/_\{([^}]+)\}/g, "<sub>$1</sub>")
    .replace(/_([A-Za-z0-9])/g, "<sub>$1</sub>");
}

function renderMath() {
  document.querySelectorAll("[data-katex]").forEach((element) => {
    if (element.dataset.katexRendered === "true") return;
    if (!window.katex) {
      element.innerHTML = mathFallback(element.dataset.katex);
      element.dataset.katexRendered = "fallback";
      return;
    }
    try {
      window.katex.render(element.dataset.katex, element, {
        displayMode: element.classList.contains("math-block"),
        throwOnError: false,
        strict: false,
      });
      element.dataset.katexRendered = "true";
    } catch { /* Keep the readable text fallback when a formula is malformed. */ }
  });
}

function addFavorite(question) {
  const value = String(question || "").trim(); if (!value) return toast("请先输入一道题目。");
  const favorites = getFavorites(); if (favorites.includes(value)) return toast("这道题已经在收藏中了。");
  writeJson(STORAGE.favorites, [value, ...favorites].slice(0, 50)); updateCounts(); toast("已收藏当前题目。");
}
function addHistory(question, answer) {
  const history = getHistory().filter((item) => item.question !== question);
  writeJson(STORAGE.history, [{ question, answer, createdAt: Date.now() }, ...history].slice(0, 30));
}
function useQuestion(question) { state.page = "solve"; state.messages = []; location.hash = "solve"; render(); const input = $("#question-input"); input.value = question; input.focus(); }
function appendInstruction(instruction) { const input = $("#question-input"); if (!input) return; input.value = input.value.trim() ? `${input.value.trim()}\n\n${instruction}` : instruction; input.focus(); }

async function sendQuestion() {
  const input = $("#question-input"); const question = input.value.trim(); if (!question || state.loading) return;
  state.loading = true; state.messages.push({ role: "user", content: question }, { role: "assistant", content: "" }); render();
  const messages = state.messages.slice(0, -1).map(({ role, content }) => ({ role, content }));
  try {
    const response = await fetch(`${apiBase()}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages, stream: true }) });
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
        if (payload.content) { state.messages[state.messages.length - 1].content += payload.content; render(); }
      }
    }
    addHistory(question, state.messages[state.messages.length - 1].content);
  } catch (error) {
    state.messages[state.messages.length - 1].content = `**暂时无法完成请求**\n\n${error.message}`;
    toast("请求失败，请检查后端和模型服务状态。");
  } finally { state.loading = false; render(); checkHealth(); }
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
  $("#send-question")?.addEventListener("click", sendQuestion); $("#save-current-favorite")?.addEventListener("click", () => addFavorite($("#question-input").value)); $("#clear-chat")?.addEventListener("click", () => { state.messages = []; render(); });
  $("#question-input")?.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); sendQuestion(); } });
  document.querySelectorAll("[data-example]").forEach((button) => button.onclick = () => { $("#question-input").value = button.dataset.example; $("#question-input").focus(); });
  document.querySelectorAll("[data-instruction], [data-tool-instruction]").forEach((button) => button.onclick = () => appendInstruction(button.dataset.instruction || button.dataset.toolInstruction));
  document.querySelectorAll("[data-copy-index]").forEach((button) => button.onclick = async () => { await navigator.clipboard.writeText(state.messages[Number(button.dataset.copyIndex)].content); toast("答案已复制。"); });
  document.querySelectorAll("[data-favorite-index]").forEach((button) => button.onclick = () => { const user = [...state.messages].reverse().find((message) => message.role === "user"); addFavorite(user?.content); });
  document.querySelectorAll("[data-use-favorite]").forEach((button) => button.onclick = () => useQuestion(getFavorites()[Number(button.dataset.useFavorite)]));
  document.querySelectorAll("[data-delete-favorite]").forEach((button) => button.onclick = () => { const favorites = getFavorites(); favorites.splice(Number(button.dataset.deleteFavorite), 1); writeJson(STORAGE.favorites, favorites); render(); toast("已移除收藏。"); });
  $("#clear-favorites")?.addEventListener("click", () => { localStorage.removeItem(STORAGE.favorites); render(); toast("收藏已清空。"); });
  document.querySelectorAll("[data-use-history]").forEach((button) => button.onclick = () => useQuestion(getHistory()[Number(button.dataset.useHistory)].question));
  $("#clear-history")?.addEventListener("click", () => { localStorage.removeItem(STORAGE.history); render(); toast("学习记录已清空。"); });
  $("#save-settings")?.addEventListener("click", () => { localStorage.setItem(STORAGE.apiBase, $("#api-base-input").value.trim().replace(/\/$/, "")); toast("设置已保存，正在检查服务。"); checkHealth(); });
  $("#settings-theme-toggle")?.addEventListener("click", toggleTheme);
  $("#clear-local-data")?.addEventListener("click", () => { localStorage.removeItem(STORAGE.favorites); localStorage.removeItem(STORAGE.history); toast("本地收藏和学习记录已清除。"); render(); });
}
function toggleTheme() { const dark = document.body.classList.toggle("dark"); localStorage.setItem(STORAGE.theme, dark ? "dark" : "light"); render(); }

document.querySelectorAll("[data-page]").forEach((button) => button.onclick = () => setPage(button.dataset.page));
$("#mobile-menu").onclick = () => $("#sidebar").classList.toggle("open"); $("#sidebar-refresh").onclick = checkHealth; $("#theme-toggle").onclick = toggleTheme;
window.addEventListener("hashchange", () => { state.page = location.hash.slice(1) || "solve"; render(); });
if (localStorage.getItem(STORAGE.theme) === "dark") document.body.classList.add("dark");
render(); checkHealth();
