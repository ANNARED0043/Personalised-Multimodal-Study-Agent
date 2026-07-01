const state = {
  profile: null,
  busy: false,
  loadingTimer: null,
  loadingStartedAt: 0,
  activeBranch: null,
  branchCount: 0,
  gallery: [],
  galleryIndex: 0,
  currentSessionId: null,
  uploadedFiles: [],
  branchRecords: [],
  activeBranchDetailId: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

async function apiStream(path, payload, handlers = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const data = await response.json();
      message = data.error || message;
    } catch (_) {}
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      handlers[event.type]?.(event);
    }
  }
  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    handlers[event.type]?.(event);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pagesFromLine(line) {
  const pages = new Set();
  const rangePattern = /p\.(\d+)\s*[-–]\s*p?\.?(\d+)/gi;
  let match;
  while ((match = rangePattern.exec(line))) {
    const start = Number(match[1]);
    const end = Number(match[2]);
    for (let page = start; page <= Math.min(end, start + 8); page += 1) pages.add(page);
  }
  const singlePattern = /p\.(\d+)/gi;
  while ((match = singlePattern.exec(line))) pages.add(Number(match[1]));
  return Array.from(pages);
}

function inlineEvidenceHtml(line, evidence = []) {
  const pages = pagesFromLine(line);
  if (!pages.length) return "";
  const matched = evidence
    .filter((item) => item.image_path && pages.includes(Number(item.page)))
    .sort((a, b) => Number(a.page || 0) - Number(b.page || 0));
  if (!matched.length) return "";
  return `
    <details class="inline-slide-group" open>
      <summary>
        <span>${matched.length} reference slide${matched.length > 1 ? "s" : ""}</span>
        <strong>${matched.map((item) => `p.${item.page || "-"}`).join(", ")}</strong>
      </summary>
      <div class="ig-slide-stack">
        ${matched.slice(0, 6).map((item, index) => `
          <button class="ig-slide-card image-open" style="--i:${index}" data-image="/${escapeHtml(item.image_path)}" data-caption="${escapeHtml(item.source)} p.${escapeHtml(item.page || "-")} · ${escapeHtml(item.title)}" data-detail="${escapeHtml(item.visual_summary || item.excerpt || item.title)}">
            <img src="/${escapeHtml(item.image_path)}" alt="${escapeHtml(item.title)}" />
            <span>p.${escapeHtml(item.page || "-")}</span>
          </button>
        `).join("")}
      </div>
      <p class="inline-slide-note">${escapeHtml(summariseSlideGroup(matched))}</p>
    </details>
  `;
}

function summariseSlideGroup(items) {
  const pages = items.map((item) => `p.${item.page || "-"}`).join(", ");
  const topics = [...new Set(items.flatMap((item) => item.topics || item.metadata?.topics || []).filter(Boolean))].slice(0, 4);
  const first = items[0]?.visual_summary || items[0]?.excerpt || items[0]?.title || "";
  return `这一部分主要参考 ${pages}${topics.length ? `，重点是 ${topics.join("、")}` : ""}。${first.slice(0, 160)}`;
}

function pagesFromLineStable(line) {
  const pages = new Set();
  let match;
  const rangePattern = /p\.(\d+)\s*(?:-|\u2013|\u2014|to)\s*p?\.?(\d+)/gi;
  while ((match = rangePattern.exec(line))) {
    const start = Number(match[1]);
    const end = Number(match[2]);
    for (let page = start; page <= Math.min(end, start + 8); page += 1) pages.add(page);
  }
  const singlePattern = /p\.(\d+)/gi;
  while ((match = singlePattern.exec(line))) pages.add(Number(match[1]));
  return Array.from(pages);
}

pagesFromLine = pagesFromLineStable;

function renderMarkdownLite(text, evidence = [], options = {}) {
  const inlineEvidence = options.inlineEvidence !== false;
  const safe = escapeHtml(text).replace(/\r\n/g, "\n");
  const lines = safe.split("\n");
  const html = [];
  let listOpen = false;
  let orderedOpen = false;
  let tableOpen = false;
  let tableHeaderDone = false;
  const inlineFormat = (value) => value.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const linkFormat = (value) => inlineFormat(value).replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const closeList = () => {
    if (listOpen) html.push("</ul>");
    if (orderedOpen) html.push("</ol>");
    listOpen = false;
    orderedOpen = false;
  };
  const closeTable = () => {
    if (tableOpen) html.push("</tbody></table>");
    tableOpen = false;
    tableHeaderDone = false;
  };
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      closeTable();
      continue;
    }
    if (line.startsWith("# ")) {
      closeList(); closeTable(); html.push(`<h3>${linkFormat(line.slice(2))}</h3>`);
    } else if (line.startsWith("## ")) {
      closeList(); closeTable(); html.push(`<h3>${linkFormat(line.slice(3))}</h3>`);
    } else if (line.startsWith("### ")) {
      closeList(); closeTable(); html.push(`<h3>${linkFormat(line.slice(4))}</h3>`);
    } else if (/^(总览|知识点地图|易混淆点|考试\/作业可用答题框架|下一步学习行动|Lecture overview|Key ideas|Exam focus|Common traps|Next study action):$/i.test(line)) {
      closeList(); closeTable(); html.push(`<h3>${linkFormat(line.replace(/:$/, ""))}</h3>`);
    } else if (/^\|.+\|$/.test(line)) {
      closeList();
      if (/^\|\s*-+/.test(line)) continue;
      const cells = line.slice(1, -1).split("|").map((cell) => linkFormat(cell.trim()));
      if (!tableOpen) {
        html.push('<table class="answer-table"><tbody>');
        tableOpen = true;
      }
      html.push(`<tr>${cells.map((cell) => tableHeaderDone ? `<td>${cell}</td>` : `<th>${cell}</th>`).join("")}</tr>`);
      tableHeaderDone = true;
    } else if (/^[-*]\s+/.test(line)) {
      closeTable();
      if (orderedOpen) { html.push("</ol>"); orderedOpen = false; }
      if (!listOpen) { html.push("<ul>"); listOpen = true; }
      html.push(`<li>${linkFormat(line.replace(/^[-*]\s+/, ""))}${inlineEvidence ? inlineEvidenceHtml(line, evidence) : ""}</li>`);
    } else if (/^\d+\.\s+/.test(line)) {
      closeTable();
      if (listOpen) { html.push("</ul>"); listOpen = false; }
      if (!orderedOpen) { html.push("<ol>"); orderedOpen = true; }
      html.push(`<li>${linkFormat(line.replace(/^\d+\.\s+/, ""))}${inlineEvidence ? inlineEvidenceHtml(line, evidence) : ""}</li>`);
    } else {
      closeList(); closeTable(); html.push(`<p>${linkFormat(line)}${inlineEvidence ? inlineEvidenceHtml(line, evidence) : ""}</p>`);
    }
  }
  closeList();
  closeTable();
  return html.join("");
}

function renderAnswerHtml(result, text = null) {
  const answer = text ?? result.answer ?? "";
  const inlineEvidence = !["document_overview", "quiz_generation"].includes(result.intent || "");
  return renderMarkdownLite(answer, result.evidence || [], { inlineEvidence });
}

function setBusy(isBusy) {
  state.busy = isBusy;
  $$("button").forEach((button) => {
    if (!button.classList.contains("nav-button") && !button.classList.contains("tab-button")) {
      button.disabled = isBusy;
    }
  });
}

async function refreshStatus() {
  const status = await api("/api/status");
  $("#docCount").textContent = status.documents;
  $("#chunkCount").textContent = status.evidence_chunks;
  $("#frameworkName").textContent = status.agent_framework;
  $("#modelName").textContent = status.ollama_model;
  if (status.text_model_status?.startsWith("fallback")) {
    $("#modelName").title = `Text model fallback active. Install ${status.text_model || "qwen2.5:7b"} for better Chinese text answers.`;
  }
}

async function loadProfile() {
  const profile = await api("/api/profile");
  state.profile = profile;
  $("#userName").value = profile.user_name;
  $("#course").value = profile.course;
  $("#language").value = profile.preferred_language;
  $("#answerStyle").value = profile.answer_style;
  $("#goal").value = profile.current_goal;
  $("#dailyMinutes").value = profile.daily_minutes;
  $("#weakTopics").value = profile.weak_topics.map((item) => item.topic).join("\n");
  renderMemory(profile);
}

function renderMemory(profile) {
  $("#memoryPanel").innerHTML = `
    <div class="memory-card">
      <h3>${escapeHtml(profile.user_name)}</h3>
      <p>${escapeHtml(profile.current_goal)}</p>
      <div class="chip-row">${profile.weak_topics.map((item) => `<span>${escapeHtml(item.topic)}</span>`).join("")}</div>
    </div>
  `;
}

async function saveProfile() {
  const payload = {
    user_name: $("#userName").value.trim(),
    course: $("#course").value.trim(),
    preferred_language: $("#language").value,
    answer_style: $("#answerStyle").value.trim(),
    current_goal: $("#goal").value.trim(),
    daily_minutes: Number($("#dailyMinutes").value || 90),
    weak_topics: $("#weakTopics").value.split(/\n|,/).map((item) => item.trim()).filter(Boolean),
  };
  const profile = await api("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.profile = profile;
  renderMemory(profile);
  toast("Profile saved.");
}

function toast(text) {
  const box = $("#uploadResult");
  box.textContent = text;
  box.classList.add("show");
  setTimeout(() => box.classList.remove("show"), 4000);
}

function renderAttachmentTray() {
  const tray = $("#attachmentTray");
  if (!tray) return;
  if (!state.uploadedFiles.length) {
    tray.hidden = true;
    tray.innerHTML = "";
    return;
  }
  tray.hidden = false;
  tray.innerHTML = state.uploadedFiles.map((file, index) => `
    <span class="attachment-chip" title="${escapeHtml(file.name)}">
      <strong>${escapeHtml(file.name)}</strong>
      <small>${escapeHtml(file.status)}</small>
      <button type="button" data-remove-upload="${index}" aria-label="Remove uploaded file">×</button>
    </span>
  `).join("");
}

function queryMentionsCurrentAttachment(query = "") {
  return /(这张图|这个图|这幅图|这张图片|这个图片|这张截图|这个截图|刚上传|上传的图|上传的图片|attached image|uploaded image|this image|this screenshot|the attached|the uploaded)/i.test(query);
}

function activeAttachmentContext(query = "") {
  const indexed = state.uploadedFiles.filter((file) => file.status === "indexed");
  if (!indexed.length) return "";
  const imageNames = indexed
    .map((file) => file.name)
    .filter((name) => /\.(png|jpe?g|webp)$/i.test(name));
  const otherNames = indexed
    .map((file) => file.name)
    .filter((name) => !imageNames.includes(name));
  const lines = [];
  if (imageNames.length && queryMentionsCurrentAttachment(query)) {
    lines.push(`Attached image files for this turn: ${imageNames.join(", ")}. If the user says this, it, image, picture, screenshot, or 这/这个/这张图, answer using the most recent uploaded image evidence.`);
  }
  if (otherNames.length) {
    lines.push(`Attached indexed files for this turn: ${otherNames.join(", ")}.`);
  }
  return lines.join("\n");
}

function renderBranchRecords() {
  const panel = $("#branchRecords");
  if (!panel) return;
  if (state.activeBranchDetailId) {
    const record = state.branchRecords.find((item) => item.id === state.activeBranchDetailId);
    if (record) {
      const status = record.status === "pending" ? `<p class="branch-detail-status">Thinking...</p>` : "";
      const error = record.error ? `<p class="branch-detail-error">${escapeHtml(record.error)}</p>` : "";
      panel.innerHTML = `
        <section class="branch-detail">
          <button class="branch-back" type="button" data-branch-back>&lt; Back</button>
          <div class="branch-detail-meta">
            <strong>${escapeHtml(record.question || "Branch question")}</strong>
            <span>${escapeHtml(record.context || "Selected branch context")}</span>
          </div>
          ${status}
          ${error}
          <div class="branch-detail-answer">${record.answer ? renderMarkdownLite(record.answer, record.evidence || []) : ""}</div>
        </section>
      `;
      return;
    }
    state.activeBranchDetailId = null;
  }
  if (!state.branchRecords.length) {
    panel.innerHTML = `<div class="empty-state compact">No branch questions yet.</div>`;
    return;
  }
  panel.innerHTML = state.branchRecords.map((record, index) => `
    <div class="branch-record" data-branch-record="${escapeHtml(record.id)}" role="button" tabindex="0">
      <div>
        <strong>${escapeHtml(record.question || `Branch ${index + 1}`)}</strong>
        <span>${escapeHtml(record.context)}</span>
      </div>
      <button class="branch-expand" type="button" data-branch-expand="${escapeHtml(record.id)}" title="Open branch answer" aria-label="Open branch answer">
        ${record.status === "pending" ? "..." : "Expand"}
      </button>
    </div>
  `).join("");
}

function addBranchRecord(record) {
  state.activeBranchDetailId = null;
  state.branchRecords.unshift(record);
  renderBranchRecords();
  switchTab("branch");
  return record;
}

function updateBranchRecord(id, patch) {
  const record = state.branchRecords.find((item) => item.id === id);
  if (!record) return;
  Object.assign(record, patch);
  renderBranchRecords();
}

function openBranchDetail(id) {
  state.activeBranchDetailId = id;
  switchTab("branch");
  renderBranchRecords();
}

function closeBranchDetail() {
  state.activeBranchDetailId = null;
  renderBranchRecords();
}

function jumpToBranchRecord(id) {
  const record = state.branchRecords.find((item) => item.id === id);
  if (!record) return;
  const target = document.querySelector(`[data-branch-anchor="${CSS.escape(record.anchorId)}"]`);
  switchView("chat", true);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("branch-highlight");
    setTimeout(() => target.classList.remove("branch-highlight"), 1600);
  }
}

async function askBranchToPanel(record) {
  if (!record || state.busy) return;
  updateBranchRecord(record.id, { status: "pending", error: "", answer: "" });
  setBusy(true);
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: `Branch question without changing the original answer. Selected paragraph:\n${record.fullContext || record.context || ""}\n\nQuestion: ${record.question}`,
        session_id: state.currentSessionId,
        is_branch: true,
        answer_mode: "",
      }),
    });
    state.currentSessionId = result.session_id || state.currentSessionId;
    updateBranchRecord(record.id, {
      status: "done",
      answer: result.answer || "",
      evidence: result.evidence || [],
      trace: result.trace || [],
    });
    renderTrace(result.trace || []);
    renderEvidence(result.evidence || []);
    if (result.memory_updates?.length) loadProfile();
    await Promise.all([refreshStatus(), loadHistory()]);
  } catch (error) {
    updateBranchRecord(record.id, { status: "error", error: error.message });
  } finally {
    setBusy(false);
  }
}

function addUserMessage(text) {
  const node = document.createElement("div");
  node.className = "message user";
  node.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
  $("#chatLog").appendChild(node);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
}

function addSystemMessage(text) {
  const node = document.createElement("div");
  node.className = "message agent compact";
  node.innerHTML = `<div class="answer-body"><p>${escapeHtml(text)}</p></div>`;
  $("#chatLog").appendChild(node);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
}

function addLoadingMessage() {
  const node = document.createElement("div");
  node.className = "message agent loading-message";
  node.id = "loadingMessage";
  state.loadingStartedAt = Date.now();
  node.innerHTML = `
    <div class="loading-row">
      <span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <span id="loadingText">Processing knowledge base · 0s</span>
    </div>`;
  $("#chatLog").appendChild(node);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
  state.loadingTimer = setInterval(() => {
    const text = $("#loadingText");
    if (text) text.textContent = `Processing knowledge base · ${Math.floor((Date.now() - state.loadingStartedAt) / 1000)}s`;
  }, 500);
}

function removeLoadingMessage() {
  if (state.loadingTimer) clearInterval(state.loadingTimer);
  state.loadingTimer = null;
  $("#loadingMessage")?.remove();
}

function addAgentMessage(result) {
  const node = document.createElement("div");
  node.className = "message agent";
  if (result.intent === "revision_planning") {
    node.classList.add("plan-message");
  }
  const citations = result.citations?.length
    ? `<details class="source-details"><summary>Sources</summary>${result.citations.map((item) => `<p>${escapeHtml(item)}</p>`).join("")}</details>`
    : "";
  node.innerHTML = `
    <div class="assistant-row"><span class="assistant-dot"></span><strong>Study Agent</strong></div>
    <div class="answer-body">${renderAnswerHtml(result)}</div>
    ${result.intent === "revision_planning" ? renderPlanBranches(result.answer) : ""}
    ${citations}`;
  $("#chatLog").appendChild(node);
  decorateAnswerAnchors(node);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
}

function addStreamingAgentMessage() {
  const node = document.createElement("div");
  node.className = "message agent";
  node.innerHTML = `
    <div class="assistant-row"><span class="assistant-dot"></span><strong>Study Agent</strong></div>
    <div class="answer-body"></div>`;
  $("#chatLog").appendChild(node);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
  return {
    node,
    answer: "",
    result: null,
    append(text) {
      this.answer += text;
      node.querySelector(".answer-body").innerHTML = this.result
        ? renderAnswerHtml(this.result, this.answer)
        : renderMarkdownLite(this.answer, []);
      $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
    },
    finish(result) {
      this.result = result;
      const citations = result.citations?.length
        ? `<details class="source-details"><summary>Sources</summary>${result.citations.map((item) => `<p>${escapeHtml(item)}</p>`).join("")}</details>`
        : "";
      node.querySelector(".answer-body").innerHTML = renderAnswerHtml(result);
      if (result.intent === "revision_planning" && !node.querySelector(".plan-branch-panel")) {
        node.insertAdjacentHTML("beforeend", renderPlanBranches(result.answer));
      }
      if (citations) node.insertAdjacentHTML("beforeend", citations);
      decorateAnswerAnchors(node);
      $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
    },
  };
}

function decorateAnswerAnchors(messageNode) {
  messageNode.querySelectorAll(".answer-body > h3, .answer-body > p, .answer-body > ul > li, .answer-body > ol > li").forEach((block, index) => {
    if (block.querySelector(".anchor-plus")) return;
    if (!block.dataset.branchAnchor) {
      block.dataset.branchAnchor = `branch-anchor-${Date.now()}-${Math.random().toString(36).slice(2)}-${index}`;
    }
    const button = document.createElement("button");
    button.className = "anchor-plus";
    button.textContent = "+";
    button.title = "Ask a branch question about this paragraph";
    button.setAttribute("aria-label", "Ask a branch question about this paragraph");
    button.dataset.anchorText = block.textContent.trim().slice(0, 500);
    block.appendChild(button);
  });
}

function openInlineBranchQuestion(anchor) {
  const block = anchor.closest("h3, p, li");
  if (!block) return;
  block.parentElement?.querySelectorAll(".inline-branch-form").forEach((node) => node.remove());
  const selected = anchor.dataset.anchorText || block.textContent.trim();
  const anchorId = block.dataset.branchAnchor || `branch-anchor-${Date.now()}`;
  block.dataset.branchAnchor = anchorId;
  const form = document.createElement("form");
  form.className = "inline-branch-form";
  form.dataset.branchContext = selected;
  form.dataset.anchorId = anchorId;
  form.innerHTML = `
    <textarea rows="2" placeholder="Ask a branch question about this paragraph..."></textarea>
    <div>
      <button type="submit">Ask branch</button>
      <button type="button" data-close-inline-branch>Cancel</button>
    </div>
  `;
  block.insertAdjacentElement("afterend", form);
  form.querySelector("textarea").focus();
}

function parsePlanNodes(answer) {
  const lines = answer.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const nodes = [];
  for (const line of lines) {
    const clean = line.replace(/^[-*]\s+/, "").replace(/\*\*/g, "");
    if (/^(Day\s*\d+|第\s*\d+\s*天|阶段\s*\d+|Step\s*\d+)/i.test(clean)) {
      nodes.push(clean.slice(0, 120));
    }
  }
  if (!nodes.length) {
    const headings = lines.filter((line) => /^#{2,3}\s+/.test(line)).map((line) => line.replace(/^#{2,3}\s+/, ""));
    nodes.push(...headings.filter((line) => /day|天|阶段|复习|learn|review/i.test(line)).slice(0, 6));
  }
  return nodes.slice(0, 10);
}

function renderPlanBranches(answer) {
  const nodes = parsePlanNodes(answer);
  if (!nodes.length) return "";
  return `
    <div class="plan-branch-panel">
      <div class="plan-branch-title">Plan branches</div>
      ${nodes.map((title, index) => `
        <div class="plan-node" data-node-index="${index}" data-node-title="${escapeHtml(title)}">
          <div class="plan-node-main">
            <button class="plan-plus" data-plan-plus title="Ask inside this stage">+</button>
            <span>${escapeHtml(title)}</span>
          </div>
          <div class="subthread" hidden>
            <form class="subthread-form">
              <input placeholder="在这个阶段插入一个问题..." />
              <button type="submit">Ask</button>
            </form>
            <div class="subthread-log"></div>
          </div>
        </div>
      `).join("")}
      <p class="plan-hint">点击 + 在节点下提问；Ctrl + 点击 + 会创建一个新的分支入口。</p>
    </div>
  `;
}

function renderSlideStack(evidence) {
  const slides = (evidence || []).filter((item) => item.image_path).slice(0, 8);
  if (!slides.length) return "";
  return `
    <div class="slide-stack" data-slide-stack>
      ${slides.map((item, index) => `
        <button class="stack-card" style="--i:${index}" data-stack-index="${index}" data-image="/${escapeHtml(item.image_path)}" data-caption="${escapeHtml(item.source)} p.${escapeHtml(item.page || "-")} · ${escapeHtml(item.title)}">
          <img src="/${escapeHtml(item.image_path)}" alt="${escapeHtml(item.title)}" />
        </button>
      `).join("")}
      <span>${slides.length} related slides</span>
    </div>
  `;
}

function renderTrace(trace) {
  $("#traceList").innerHTML = trace.map((item) => `
    <li><strong>${escapeHtml(item.node.replaceAll("_", " "))}</strong><span>${escapeHtml(item.detail)}</span></li>
  `).join("");
}

function renderEvidence(evidence) {
  if (!evidence.length) {
    state.gallery = [];
    $("#evidenceList").innerHTML = `<div class="empty-state">No evidence yet. Upload lecture PDFs or ask a question.</div>`;
    return;
  }
  state.gallery = evidence.filter((item) => item.image_path).map((item) => ({
    image: `/${item.image_path}`,
    detail: item.knowledge_summary || item.visual_summary || item.excerpt || "",
    caption: `${item.source} p.${item.page || "-"} · ${item.title}`,
  }));
  $("#evidenceList").innerHTML = evidence.slice(0, 6).map((item) => {
    const image = item.image_path ? `<button class="image-open" data-image="/${escapeHtml(item.image_path)}" data-caption="${escapeHtml(item.source)} p.${escapeHtml(item.page || "-")} · ${escapeHtml(item.title)}"><img src="/${escapeHtml(item.image_path)}" alt="Evidence from ${escapeHtml(item.source)}" loading="lazy" /></button>` : "";
    const topics = (item.topics || item.metadata?.topics || []).slice(0, 4).map((topic) => `<span>${escapeHtml(topic)}</span>`).join("");
    return `
      <article class="evidence-card">
        ${image}
        <div class="evidence-body">
          <div class="evidence-meta">
            <span>${escapeHtml(item.modality)}</span>
            <span>${item.page ? `p.${escapeHtml(item.page)}` : "no page"}</span>
            <span>${escapeHtml(item.metadata?.visual_type || "evidence")}</span>
          </div>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.source)} · score ${escapeHtml(item.score)}</p>
          ${item.image_path ? `<a class="open-link" href="/${escapeHtml(item.image_path)}" target="_blank" rel="noopener">Open slide</a>` : ""}
          ${item.visual_summary ? `<div class="visual-summary">${renderMarkdownLite(item.visual_summary)}</div>` : ""}
          <div class="chip-row">${topics}</div>
        </div>
      </article>`;
  }).join("");
}

function openSlideModal(image, caption, detail = "") {
  const existing = state.gallery.findIndex((item) => item.image === image);
  if (existing >= 0) state.galleryIndex = existing;
  const galleryDetail = existing >= 0 ? state.gallery[existing].detail : "";
  $("#modalImage").src = image;
  $("#modalCaption").innerHTML = `
    <strong>${escapeHtml(caption)}</strong>
    ${(detail || galleryDetail) ? `<div class="modal-study-notes"><h3>Page Focus</h3>${renderMarkdownLite(detail || galleryDetail)}</div>` : ""}
  `;
  $("#slideModal").hidden = false;
}

function showGalleryOffset(offset) {
  if (!state.gallery.length) return;
  state.galleryIndex = (state.galleryIndex + offset + state.gallery.length) % state.gallery.length;
  const item = state.gallery[state.galleryIndex];
  $("#modalImage").src = item.image;
  $("#modalCaption").innerHTML = `<strong>${escapeHtml(item.caption)}</strong>${item.detail ? `<div class="modal-study-notes"><h3>Page Focus</h3>${renderMarkdownLite(item.detail)}</div>` : ""}`;
}

function closeSlideModal() {
  $("#slideModal").hidden = true;
  $("#modalImage").src = "";
}

async function askAgent(query) {
  if (!query || state.busy) return;
  const sendQuery = state.activeBranch
    ? `${state.activeBranch.context}。我的分支问题：${query}。请直接回答这个局部问题，并在最后说明如何回到原计划。`
    : query;
  addUserMessage(state.activeBranch ? `[${state.activeBranch.title}] ${query}` : query);
  $("#queryInput").value = "";
  autoResizeComposer();
  setBusy(true);
  addLoadingMessage();
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: sendQuery }),
    });
    removeLoadingMessage();
    addAgentMessage(result);
    renderTrace(result.trace);
    renderEvidence(result.evidence);
    if (result.memory_updates?.length) await loadProfile();
    await Promise.all([refreshStatus(), loadDocuments(), loadHistory()]);
  } catch (error) {
    removeLoadingMessage();
    addSystemMessage(error.message);
  } finally {
    setBusy(false);
  }
}

async function askSubQuestion(planNode, question) {
  if (!question || state.busy) return;
  const log = planNode.querySelector(".subthread-log");
  const nodeTitle = planNode.dataset.nodeTitle;
  const user = document.createElement("div");
  user.className = "sub-message sub-user";
  user.textContent = question;
  log.appendChild(user);
  setBusy(true);
  const loading = document.createElement("div");
  loading.className = "sub-message sub-agent";
  loading.textContent = "Processing...";
  log.appendChild(loading);
  try {
    const payload = {
      query: `在复习计划的「${nodeTitle}」阶段，我有一个分支问题：${question}。请不要重写整个计划，只回答这个局部问题，并说明如何回到原计划。`,
    };
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    loading.innerHTML = renderMarkdownLite(result.answer, result.evidence || []);
    renderTrace(result.trace);
    renderEvidence(result.evidence);
    if (result.memory_updates?.length) await loadProfile();
    await refreshStatus();
  } catch (error) {
    loading.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function createBranchContext(title) {
  state.branchCount += 1;
  const id = `branch-${state.branchCount}`;
  const branch = {
    id,
    title: `Branch ${state.branchCount}`,
    context: `在复习计划的「${title}」阶段`,
  };
  state.activeBranch = branch;
  const button = document.createElement("button");
  button.className = "nav-button branch-nav active";
  button.dataset.branchId = id;
  button.textContent = `B${state.branchCount}`;
  button.title = title;
  $$(".nav-button").forEach((item) => item.classList.remove("active"));
  $(".nav-rail").appendChild(button);
  switchView("chat", true);
  $$(".nav-button").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  addSystemMessage(`已创建分支：${title}。现在你的主输入框会在这个阶段内提问；点击 Chat 可回到普通主线。`);
  $("#queryInput").focus();
}

async function askAgentStream(query) {
  if (!query || state.busy) return;
  const sendQuery = state.activeBranch
    ? `${state.activeBranch.context}。我的分支问题：${query}。请直接回答这个局部问题，不要重写主对话。`
    : query;
  addUserMessage(state.activeBranch ? `[${state.activeBranch.title}] ${query}` : query);
  $("#queryInput").value = "";
  autoResizeComposer();
  setBusy(true);
  addLoadingMessage();
  try {
    const payload = {
      query: sendQuery,
      session_id: state.currentSessionId,
      is_branch: Boolean(state.activeBranch),
    };
    let streamMessage = null;
    await apiStream("/api/chat/stream", payload, {
      status(event) {
        const text = $("#loadingText");
        if (text) text.textContent = `${event.message} · ${Math.floor((Date.now() - state.loadingStartedAt) / 1000)}s`;
      },
      meta(event) {
        state.currentSessionId = event.session_id || state.currentSessionId;
        removeLoadingMessage();
        streamMessage = addStreamingAgentMessage();
        renderTrace(event.trace || []);
        renderEvidence(event.evidence || []);
      },
      token(event) {
        if (!streamMessage) {
          removeLoadingMessage();
          streamMessage = addStreamingAgentMessage();
        }
        streamMessage.append(event.text || "");
      },
      done(result) {
        state.currentSessionId = result.session_id || state.currentSessionId;
        if (streamMessage) streamMessage.finish(result);
        renderTrace(result.trace || []);
        renderEvidence(result.evidence || []);
        if (result.memory_updates?.length) loadProfile();
      },
    });
    await Promise.all([refreshStatus(), loadDocuments(), loadHistory()]);
  } catch (error) {
    removeLoadingMessage();
    addSystemMessage(error.message);
  } finally {
    setBusy(false);
  }
}

async function summariseSession() {
  if (state.busy) return;
  setBusy(true);
  addLoadingMessage();
  try {
    const result = await api("/api/session-summary", { method: "POST" });
    removeLoadingMessage();
    addAgentMessage({
      answer: result.answer,
      evidence: result.evidence,
      citations: [],
      selected_tools: ["session_notes", "slide_evidence", "weakness_highlight"],
    });
    renderEvidence(result.evidence);
    switchView("chat");
  } catch (error) {
    removeLoadingMessage();
    addSystemMessage(error.message);
  } finally {
    setBusy(false);
  }
}

async function uploadSelectedFiles() {
  const files = $("#fileInput").files;
  if (!files.length) return;
  const selected = Array.from(files).map((file) => ({ name: file.name, status: "indexing" }));
  state.uploadedFiles.unshift(...selected);
  renderAttachmentTray();
  setBusy(true);
  toast(`Indexing ${files.length} file(s)...`);
  try {
    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    const result = await api("/api/upload", { method: "POST", body: form });
    const names = result.uploaded.map((item) => item.error ? `${item.filename}: ${item.error}` : `${item.filename} indexed`).join(" · ");
    result.uploaded.forEach((item) => {
      const match = state.uploadedFiles.find((file) => file.name === item.filename && file.status === "indexing");
      if (match) match.status = item.error ? "failed" : "indexed";
    });
    renderAttachmentTray();
    toast(names);
    $("#fileInput").value = "";
    await Promise.all([refreshStatus(), loadDocuments()]);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

function setupResizer() {
  const handle = $("#resizeHandle");
  if (!handle) return;
  let dragging = false;
  handle.addEventListener("mousedown", () => {
    dragging = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    const width = Math.min(Math.max(window.innerWidth - event.clientX, 300), 720);
    document.documentElement.style.setProperty("--details-width", `${width}px`);
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  });
}

async function loadDocuments() {
  const [data, indexData] = await Promise.all([api("/api/documents"), api("/api/indexes")]);
  $("#indexesList").innerHTML = indexData.indexes.map((item) => `
    <article class="index-card">
      <strong>${escapeHtml(item.name)}</strong>
      <span>${escapeHtml(item.chunks)} chunks</span>
      <span>${escapeHtml(item.vector_backend)}</span>
      <span>${escapeHtml(item.fusion)}</span>
    </article>
  `).join("");
  if (!data.documents.length) {
    $("#documentsList").innerHTML = `<div class="empty-state">No documents indexed yet. Attach a lecture PDF to begin.</div>`;
    return;
  }
  $("#documentsList").innerHTML = data.documents.map((doc) => {
    const modalities = Object.entries(doc.modality_counts).map(([key, value]) => `<span>${escapeHtml(key)} ${escapeHtml(value)}</span>`).join("");
    const topics = doc.topics.map((topic) => `<span>${escapeHtml(topic)}</span>`).join("");
    return `
      <article class="document-card">
        <div>
          <h3>${escapeHtml(doc.filename)}</h3>
          <p>${escapeHtml(doc.doc_type)} · ${escapeHtml(doc.page_count)} pages/items · ${escapeHtml(doc.chunks)} evidence chunks</p>
          <div class="chip-row">${modalities}</div>
          <div class="chip-row topic-row">${topics}</div>
        </div>
        <div class="doc-actions">
          <button data-reindex="${doc.id}">Reindex</button>
          <button data-delete="${doc.id}">Remove</button>
        </div>
      </article>`;
  }).join("");
}

async function loadHistory() {
  const items = await api("/api/sessions");
  if (!items.length) {
    $("#historyList").innerHTML = `<div class="empty-state">No chat history yet.</div>`;
    return;
  }
  $("#historyList").innerHTML = items.map((item) => `
    <article class="history-card">
      <div>
        <h3>${escapeHtml(item.title)}</h3>
        <p>${escapeHtml(item.intent)} · ${escapeHtml(item.created_at)}</p>
      </div>
      <div class="history-actions">
        <button data-open-history="${item.id}">Open</button>
        <button data-delete-history="${item.id}">Delete</button>
      </div>
    </article>
  `).join("");
}

async function openHistory(id) {
  const data = await api(`/api/sessions/${id}`);
  state.currentSessionId = data.session.id;
  $("#chatLog").innerHTML = "";
  let lastTurn = null;
  data.turns.filter((turn) => !turn.is_branch).forEach((turn) => {
    addUserMessage(turn.query);
    addAgentMessage({
      answer: turn.answer,
      evidence: turn.evidence,
      trace: turn.trace,
      citations: [],
      selected_tools: ["history_resume"],
      intent: turn.intent,
    });
    lastTurn = turn;
  });
  if (lastTurn) {
    renderEvidence(lastTurn.evidence);
    renderTrace(lastTurn.trace);
  }
  switchView("chat");
}

async function deleteHistory(id) {
  await api(`/api/sessions/${id}`, { method: "DELETE" });
  await loadHistory();
  await refreshStatus();
}

async function newChat() {
  const session = await api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New study chat" }),
  });
  state.currentSessionId = session.id;
  state.activeBranch = null;
  $("#chatLog").innerHTML = "";
  renderEvidence([]);
  renderTrace([]);
  switchView("chat");
  addSystemMessage("New chat started. Attach lecture files or ask a study question.");
  await loadHistory();
}

async function runEvaluation() {
  const panel = $("#evaluationPanel");
  panel.innerHTML = `<div class="empty-state">Running benchmark and ablations...</div>`;
  setBusy(true);
  try {
    const data = await api("/api/evaluate", { method: "POST" });
    renderEvaluation(data);
    toast("Evaluation artifacts written to data/evaluation.");
  } catch (error) {
    panel.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  } finally {
    setBusy(false);
  }
}

function renderEvaluation(data) {
  const summaryRows = Object.entries(data.summary || {}).map(([mode, item]) => `
    <tr>
      <td>${escapeHtml(mode)}</td>
      <td>${escapeHtml(item.mean_recall_at_k)}</td>
      <td>${escapeHtml(item.mean_page_recall_at_k)}</td>
      <td>${escapeHtml(item.mean_mrr)}</td>
      <td>${escapeHtml(item.mean_answer_success_proxy)}</td>
      <td>${escapeHtml(item.mean_latency_ms)}ms</td>
    </tr>
  `).join("");
  const cases = (data.results || []).filter((item) => item.mode === "final_agent").map((item) => `
    <article class="eval-case">
      <h3>${escapeHtml(item.case_id)}</h3>
      <p>${escapeHtml(item.family)} · Recall@k ${escapeHtml(item.recall_at_k)} · MRR ${escapeHtml(item.mrr)} · Page ${escapeHtml(item.page_recall_at_k ?? "-")}</p>
      <p>${escapeHtml(item.query)}</p>
    </article>
  `).join("");
  $("#evaluationPanel").innerHTML = `
    <table class="answer-table">
      <tbody>
        <tr><th>Mode</th><th>Recall@k</th><th>Page Recall</th><th>MRR</th><th>Answer Proxy</th><th>Latency</th></tr>
        ${summaryRows}
      </tbody>
    </table>
    <div class="eval-grid">${cases}</div>
  `;
}

async function reindexDocument(id) {
  setBusy(true);
  try {
    await api(`/api/documents/${id}/reindex`, { method: "POST" });
    toast("Document reindexed.");
    await Promise.all([refreshStatus(), loadDocuments()]);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function deleteDocument(id) {
  setBusy(true);
  try {
    await api(`/api/documents/${id}`, { method: "DELETE" });
    toast("Document removed from the knowledge base.");
    await Promise.all([refreshStatus(), loadDocuments()]);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

function switchView(name, preserveBranch = false) {
  if (name === "chat" && !preserveBranch) state.activeBranch = null;
  $$(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  $$(".view-panel").forEach((panel) => panel.classList.remove("active"));
  $(`#${name}View`).classList.add("active");
}

function switchTab(name) {
  $$(".tab-button").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.remove("active"));
  $(`#${name}Tab`).classList.add("active");
}

function autoResizeComposer() {
  const input = $("#queryInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function bindEvents() {
  $("#newChatBtn").addEventListener("click", newChat);
  $$(".nav-button[data-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$(".tab-button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  $("#attachBtn").addEventListener("click", () => $("#fileInput").click());
  $("#libraryUploadBtn").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", uploadSelectedFiles);
  $("#attachmentTray")?.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-remove-upload]");
    if (!remove) return;
    state.uploadedFiles.splice(Number(remove.dataset.removeUpload), 1);
    renderAttachmentTray();
  });
  $("#refreshDocsBtn").addEventListener("click", loadDocuments);
  $("#refreshHistoryBtn").addEventListener("click", loadHistory);
  $("#runEvalBtn").addEventListener("click", runEvaluation);
  $("#saveProfileBtn").addEventListener("click", saveProfile);
  $("#sessionSummaryBtn").addEventListener("click", summariseSession);
  $("#closeModalBtn").addEventListener("click", closeSlideModal);
  $("#prevSlideBtn").addEventListener("click", () => showGalleryOffset(-1));
  $("#nextSlideBtn").addEventListener("click", () => showGalleryOffset(1));
  $("#slideModal").addEventListener("click", (event) => {
    if (event.target.id === "slideModal") closeSlideModal();
  });
  $("#evidenceList").addEventListener("click", (event) => {
    const opener = event.target.closest("[data-image]");
    if (opener) openSlideModal(opener.dataset.image, opener.dataset.caption, opener.dataset.detail || "");
  });
  $("#documentsList").addEventListener("click", (event) => {
    const reindex = event.target.closest("[data-reindex]");
    const remove = event.target.closest("[data-delete]");
    if (reindex) reindexDocument(reindex.dataset.reindex);
    if (remove) deleteDocument(remove.dataset.delete);
  });
  $("#chatForm").addEventListener("submit", (event) => {
    event.preventDefault();
    askAgentStream($("#queryInput").value.trim());
  });
  $("#chatLog").addEventListener("click", (event) => {
    const closeInline = event.target.closest("[data-close-inline-branch]");
    if (closeInline) {
      closeInline.closest(".inline-branch-form")?.remove();
      return;
    }
    const imageOpener = event.target.closest("[data-image]");
    if (imageOpener) {
      openSlideModal(imageOpener.dataset.image, imageOpener.dataset.caption, imageOpener.dataset.detail || "");
      return;
    }
    const anchor = event.target.closest(".anchor-plus");
    if (anchor) {
      openInlineBranchQuestion(anchor);
      return;
    }
    const stackCard = event.target.closest(".stack-card");
    if (stackCard) {
      openSlideModal(stackCard.dataset.image, stackCard.dataset.caption);
      return;
    }
    const plus = event.target.closest("[data-plan-plus]");
    if (!plus) return;
    const planNode = plus.closest(".plan-node");
    const subthread = planNode.querySelector(".subthread");
    if (event.ctrlKey || event.metaKey) {
      createBranchContext(planNode.dataset.nodeTitle);
      return;
    }
    subthread.hidden = !subthread.hidden;
    if (!subthread.hidden) subthread.querySelector("input").focus();
  });
  $("#chatLog").addEventListener("submit", (event) => {
    const inlineForm = event.target.closest(".inline-branch-form");
    if (inlineForm) {
      event.preventDefault();
      event.stopImmediatePropagation();
      const textarea = inlineForm.querySelector("textarea");
      const question = textarea.value.trim();
      if (!question) return;
      const context = inlineForm.dataset.branchContext || "";
      const anchorId = inlineForm.dataset.anchorId || "";
      const record = addBranchRecord({
        id: `record-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        anchorId,
        question,
        context: context.slice(0, 180),
        fullContext: context,
        status: "pending",
      });
      inlineForm.remove();
      askBranchToPanel(record);
      return;
    }
    const form = event.target.closest(".subthread-form");
    if (!form) return;
    event.preventDefault();
    const input = form.querySelector("input");
    const question = input.value.trim();
    input.value = "";
    askSubQuestion(form.closest(".plan-node"), question);
  });
  $("#historyList").addEventListener("click", (event) => {
    const open = event.target.closest("[data-open-history]");
    const remove = event.target.closest("[data-delete-history]");
    if (open) openHistory(open.dataset.openHistory);
    if (remove) deleteHistory(remove.dataset.deleteHistory);
  });
  $("#branchQuestionForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("#branchQuestionInput").value.trim();
    if (!question) return;
    const refs = $$(".branch-ref").map((node, index) => `${index + 1}. ${node.textContent.trim()}`).join("\n");
    $("#branchQuestionInput").value = "";
    switchView("chat", true);
    askAgentStream(`基于右侧分支参考内容回答我的问题。\n参考内容：\n${refs || "暂无显式参考，请结合最近上下文。"}\n\n问题：${question}`);
  });
  $("#branchRecords")?.addEventListener("click", (event) => {
    const record = event.target.closest("[data-branch-record]");
    if (record) jumpToBranchRecord(record.dataset.branchRecord);
  });
  $(".nav-rail").addEventListener("click", (event) => {
    const branchButton = event.target.closest("[data-branch-id]");
    if (!branchButton) return;
    $$(".nav-button").forEach((item) => item.classList.remove("active"));
    branchButton.classList.add("active");
    state.activeBranch = {
      id: branchButton.dataset.branchId,
      title: branchButton.textContent,
      context: `在复习计划的「${branchButton.title}」阶段`,
    };
    switchView("chat", true);
    $$(".nav-button").forEach((item) => item.classList.remove("active"));
    branchButton.classList.add("active");
  });
  $("#queryInput").addEventListener("input", autoResizeComposer);
  $("#queryInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askAgentStream($("#queryInput").value.trim());
    }
  });
  $$(".mode-button").forEach((button) => button.addEventListener("click", () => setAnswerMode(button.dataset.mode || "")));
}

function cleanSlideCaption(item) {
  return `${item.source} p.${item.page || "-"} - ${item.title}`;
}

summariseSlideGroup = function summariseSlideGroupClean(items) {
  const pages = items.map((item) => `p.${item.page || "-"}`).join(", ");
  const topics = [...new Set(items.flatMap((item) => item.topics || item.metadata?.topics || []).filter(Boolean))].slice(0, 4);
  const first = items[0]?.visual_summary || items[0]?.excerpt || items[0]?.title || "";
  return `Main reference pages: ${pages}${topics.length ? `; focus: ${topics.join(", ")}` : ""}. ${first.slice(0, 160)}`;
};

renderEvidence = function renderEvidenceClean(evidence) {
  if (!evidence.length) {
    state.gallery = [];
    $("#evidenceList").innerHTML = `<div class="empty-state">No evidence yet. Ask a question or upload lecture PDFs.</div>`;
    return;
  }
  state.gallery = evidence.filter((item) => item.image_path).map((item) => ({
    image: `/${item.image_path}`,
    detail: item.knowledge_summary || item.visual_summary || item.excerpt || "",
    caption: cleanSlideCaption(item),
  }));
  $("#evidenceList").innerHTML = evidence.slice(0, 8).map((item) => {
    const caption = cleanSlideCaption(item);
    const summary = item.knowledge_summary || item.visual_summary || item.excerpt || "";
    const image = item.image_path ? `<button class="image-open" data-image="/${escapeHtml(item.image_path)}" data-caption="${escapeHtml(caption)}" data-detail="${escapeHtml(summary)}"><img src="/${escapeHtml(item.image_path)}" alt="Evidence from ${escapeHtml(item.source)}" loading="lazy" /></button>` : "";
    const topics = (item.topics || item.metadata?.topics || []).slice(0, 4).map((topic) => `<span>${escapeHtml(topic)}</span>`).join("");
    return `
      <article class="evidence-card">
        ${image}
        <div class="evidence-body">
          <div class="evidence-meta">
            <span>${escapeHtml(item.modality)}</span>
            <span>${item.page ? `p.${escapeHtml(item.page)}` : "no page"}</span>
            <span>${escapeHtml(item.metadata?.visual_type || "evidence")}</span>
          </div>
          <h3>${escapeHtml(item.title)}</h3>
          <p>${escapeHtml(item.source)} - score ${escapeHtml(item.score)}</p>
          ${item.image_path ? `<a class="open-link" href="/${escapeHtml(item.image_path)}" target="_blank" rel="noopener">Open slide</a>` : ""}
          ${summary ? `<div class="visual-summary">${renderMarkdownLite(summary)}</div>` : ""}
          <div class="chip-row">${topics}</div>
        </div>
      </article>`;
  }).join("");
};

askAgentStream = async function askAgentStreamClean(query) {
  if (!query || state.busy) return;
  const branchMatch = query.match(/Branch question without changing the original answer\. Selected paragraph:\n([\s\S]*?)\n\nQuestion:\s*([\s\S]*)/i);
  if (branchMatch) {
    const context = branchMatch[1].trim();
    const question = branchMatch[2].trim();
    const record = addBranchRecord({
      id: `record-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      anchorId: "",
      question,
      context: context.slice(0, 180),
      fullContext: context,
      status: "pending",
    });
    askBranchToPanel(record);
    return;
  }
  if (state.activeBranch) {
    const record = addBranchRecord({
      id: `record-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      anchorId: "",
      question: query,
      context: state.activeBranch.context.slice(0, 180),
      fullContext: state.activeBranch.context,
      status: "pending",
    });
    $("#queryInput").value = "";
    autoResizeComposer();
    askBranchToPanel(record);
    return;
  }
  const attachmentContext = activeAttachmentContext(query);
  const sendQuery = attachmentContext ? `${query}\n\n${attachmentContext}` : query;
  addUserMessage(query);
  $("#queryInput").value = "";
  autoResizeComposer();
  setBusy(true);
  addLoadingMessage();
  try {
    const payload = {
      query: sendQuery,
      session_id: state.currentSessionId,
      is_branch: Boolean(state.activeBranch),
      answer_mode: state.activeBranch ? "" : (state.answerMode || ""),
    };
    let streamMessage = null;
    await apiStream("/api/chat/stream", payload, {
      status(event) {
        const text = $("#loadingText");
        if (text) text.textContent = `${event.message} - ${Math.floor((Date.now() - state.loadingStartedAt) / 1000)}s`;
      },
      meta(event) {
        state.currentSessionId = event.session_id || state.currentSessionId;
        removeLoadingMessage();
        streamMessage = addStreamingAgentMessage();
        renderTrace(event.trace || []);
        renderEvidence(event.evidence || []);
      },
      token(event) {
        if (!streamMessage) {
          removeLoadingMessage();
          streamMessage = addStreamingAgentMessage();
        }
        streamMessage.append(event.text || "");
      },
      done(result) {
        state.currentSessionId = result.session_id || state.currentSessionId;
        if (streamMessage) streamMessage.finish(result);
        renderTrace(result.trace || []);
        renderEvidence(result.evidence || []);
        if (result.memory_updates?.length) loadProfile();
      },
    });
    await Promise.all([refreshStatus(), loadDocuments(), loadHistory()]);
  } catch (error) {
    removeLoadingMessage();
    addSystemMessage(error.message);
  } finally {
    setBusy(false);
  }
};

askSubQuestion = async function askSubQuestionClean(planNode, question) {
  if (!question || state.busy) return;
  const nodeTitle = planNode.dataset.nodeTitle;
  const record = addBranchRecord({
    id: `record-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    anchorId: "",
    question,
    context: `Plan stage: ${nodeTitle}`.slice(0, 180),
    fullContext: `In the revision-plan stage "${nodeTitle}"`,
    status: "pending",
  });
  planNode.querySelector(".subthread")?.setAttribute("hidden", "");
  askBranchToPanel(record);
};

createBranchContext = function createBranchContextClean(title) {
  state.branchCount += 1;
  const id = `branch-${state.branchCount}`;
  const branch = {
    id,
    title: `Branch ${state.branchCount}`,
    context: `In the revision-plan stage "${title}"`,
  };
  state.activeBranch = branch;
  const button = document.createElement("button");
  button.className = "nav-button branch-nav active";
  button.dataset.branchId = id;
  button.textContent = `B${state.branchCount}`;
  button.title = title;
  $$(".nav-button").forEach((item) => item.classList.remove("active"));
  $(".nav-rail").appendChild(button);
  switchView("chat", true);
  button.classList.add("active");
  addSystemMessage(`Branch created: ${title}. Your next message will answer this local stage without rewriting the main plan.`);
  $("#queryInput").focus();
};

loadHistory = async function loadHistoryClean() {
  const items = await api("/api/sessions");
  if (!items.length) {
    $("#historyList").innerHTML = `<div class="empty-state">No chat history yet.</div>`;
    return;
  }
  $("#historyList").innerHTML = items.map((item) => `
    <article class="history-card">
      <div>
        <h3>${escapeHtml(item.last_query || item.title)}</h3>
        <p>${escapeHtml(item.turns || 0)} turns - ${escapeHtml(item.last_intent || "chat")} - ${escapeHtml(item.updated_at || item.created_at)}</p>
        ${item.title && item.last_query && item.title !== item.last_query ? `<p class="history-subtitle">${escapeHtml(item.title)}</p>` : ""}
      </div>
      <div class="history-actions">
        <button data-open-history="${item.id}">Open</button>
        <button data-delete-history="${item.id}">Delete</button>
      </div>
    </article>
  `).join("");
};

newChat = async function newChatClean() {
  const session = await api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New study chat" }),
  });
  state.currentSessionId = session.id;
  state.activeBranch = null;
  $("#chatLog").innerHTML = "";
  renderEvidence([]);
  renderTrace([]);
  switchView("chat");
  addSystemMessage("New chat started. Ask in Chinese or English; I will match the query language and cite course evidence.");
  await loadHistory();
};

addLoadingMessage = function addLoadingMessageClean() {
  const node = document.createElement("div");
  node.className = "message agent loading-message";
  node.id = "loadingMessage";
  state.loadingStartedAt = Date.now();
  node.innerHTML = `
    <div class="loading-row">
      <span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <span id="loadingText">Reading knowledge base - 0s</span>
    </div>`;
  $("#chatLog").appendChild(node);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
  state.loadingTimer = setInterval(() => {
    const text = $("#loadingText");
    if (text) text.textContent = `Reading knowledge base - ${Math.floor((Date.now() - state.loadingStartedAt) / 1000)}s`;
  }, 500);
};

function setAnswerMode(mode = "") {
  state.answerMode = mode;
  $$(".mode-button").forEach((button) => button.classList.toggle("active", (button.dataset.mode || "") === mode));
  const label = mode ? mode[0].toUpperCase() + mode.slice(1) : "General";
  $("#queryInput").placeholder = mode ? `${label} mode: type your own question...` : "Message Study Agent...";
}

parsePlanNodes = function parsePlanNodesClean(answer) {
  const lines = answer.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const nodes = [];
  for (const line of lines) {
    const clean = line.replace(/^[-*]\s+/, "").replace(/\*\*/g, "");
    if (/^(Day\s*\d+|第\s*\d+\s*天|Stage\s*\d+|Step\s*\d+)/i.test(clean)) nodes.push(clean.slice(0, 120));
  }
  if (!nodes.length) {
    nodes.push(...lines.filter((line) => /^#{2,3}\s+/.test(line)).map((line) => line.replace(/^#{2,3}\s+/, "")).filter((line) => /day|stage|review|复习|学习|计划/i.test(line)).slice(0, 6));
  }
  return nodes.slice(0, 10);
};

renderPlanBranches = function renderPlanBranchesClean(answer) {
  const nodes = parsePlanNodes(answer);
  if (!nodes.length) return "";
  return `
    <div class="plan-branch-panel">
      <div class="plan-branch-title">Plan branches</div>
      ${nodes.map((title, index) => `
        <div class="plan-node" data-node-index="${index}" data-node-title="${escapeHtml(title)}">
          <div class="plan-node-main">
            <button class="plan-plus" data-plan-plus title="Ask inside this stage">+</button>
            <span>${escapeHtml(title)}</span>
          </div>
          <div class="subthread" hidden>
            <form class="subthread-form">
              <input placeholder="Ask a local question for this stage..." />
              <button type="submit">Ask</button>
            </form>
            <div class="subthread-log"></div>
          </div>
        </div>
      `).join("")}
      <p class="plan-hint">Click + to open a question box under that stage. Ctrl/Cmd + click creates a branch in the left rail, so your next chat message becomes a local follow-up instead of a new plan.</p>
    </div>
  `;
};

bindEvents = function bindEventsClean() {
  $("#newChatBtn").addEventListener("click", newChat);
  $$(".nav-button[data-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$(".tab-button").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  $("#attachBtn").addEventListener("click", () => $("#fileInput").click());
  $("#libraryUploadBtn").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", uploadSelectedFiles);
  $("#attachmentTray")?.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-remove-upload]");
    if (!remove) return;
    state.uploadedFiles.splice(Number(remove.dataset.removeUpload), 1);
    renderAttachmentTray();
  });
  $("#refreshDocsBtn").addEventListener("click", loadDocuments);
  $("#refreshHistoryBtn").addEventListener("click", loadHistory);
  $("#runEvalBtn").addEventListener("click", runEvaluation);
  $("#saveProfileBtn").addEventListener("click", saveProfile);
  $("#sessionSummaryBtn").addEventListener("click", summariseSession);
  $("#closeModalBtn").addEventListener("click", closeSlideModal);
  $("#prevSlideBtn").addEventListener("click", () => showGalleryOffset(-1));
  $("#nextSlideBtn").addEventListener("click", () => showGalleryOffset(1));
  $("#slideModal").addEventListener("click", (event) => {
    if (event.target.id === "slideModal") closeSlideModal();
  });
  $("#evidenceList").addEventListener("click", (event) => {
    const opener = event.target.closest("[data-image]");
    if (opener) openSlideModal(opener.dataset.image, opener.dataset.caption, opener.dataset.detail || "");
  });
  $("#documentsList").addEventListener("click", (event) => {
    const reindex = event.target.closest("[data-reindex]");
    const remove = event.target.closest("[data-delete]");
    if (reindex) reindexDocument(reindex.dataset.reindex);
    if (remove) deleteDocument(remove.dataset.delete);
  });
  $("#chatForm").addEventListener("submit", (event) => {
    event.preventDefault();
    askAgentStream($("#queryInput").value.trim());
  });
  $("#chatLog").addEventListener("click", (event) => {
    const closeInline = event.target.closest("[data-close-inline-branch]");
    if (closeInline) {
      closeInline.closest(".inline-branch-form")?.remove();
      return;
    }
    const imageOpener = event.target.closest("[data-image]");
    if (imageOpener) {
      openSlideModal(imageOpener.dataset.image, imageOpener.dataset.caption, imageOpener.dataset.detail || "");
      return;
    }
    const anchor = event.target.closest(".anchor-plus");
    if (anchor) {
      openInlineBranchQuestion(anchor);
      return;
    }
    const plus = event.target.closest("[data-plan-plus]");
    if (!plus) return;
    const planNode = plus.closest(".plan-node");
    const subthread = planNode.querySelector(".subthread");
    if (event.ctrlKey || event.metaKey) {
      createBranchContext(planNode.dataset.nodeTitle);
      return;
    }
    subthread.hidden = !subthread.hidden;
    if (!subthread.hidden) subthread.querySelector("input").focus();
  });
  $("#chatLog").addEventListener("submit", (event) => {
    const inlineForm = event.target.closest(".inline-branch-form");
    if (inlineForm) {
      event.preventDefault();
      event.stopImmediatePropagation();
      const textarea = inlineForm.querySelector("textarea");
      const question = textarea.value.trim();
      if (!question) return;
      const context = inlineForm.dataset.branchContext || "";
      const anchorId = inlineForm.dataset.anchorId || "";
      const record = addBranchRecord({
        id: `record-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        anchorId,
        question,
        context: context.slice(0, 180),
        fullContext: context,
        status: "pending",
      });
      inlineForm.remove();
      askBranchToPanel(record);
      return;
    }
    const form = event.target.closest(".subthread-form");
    if (!form) return;
    event.preventDefault();
    const input = form.querySelector("input");
    const question = input.value.trim();
    input.value = "";
    askSubQuestion(form.closest(".plan-node"), question);
  });
  $("#historyList").addEventListener("click", (event) => {
    const open = event.target.closest("[data-open-history]");
    const remove = event.target.closest("[data-delete-history]");
    if (open) openHistory(open.dataset.openHistory);
    if (remove) deleteHistory(remove.dataset.deleteHistory);
  });
  $("#branchQuestionForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("#branchQuestionInput").value.trim();
    if (!question) return;
    const refs = $$(".branch-ref").map((node, index) => `${index + 1}. ${node.textContent.trim()}`).join("\n");
    $("#branchQuestionInput").value = "";
    const record = addBranchRecord({
      id: `record-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      anchorId: "",
      question,
      context: (refs || "Recent conversation context").slice(0, 180),
      fullContext: refs || "No explicit references; use recent context.",
      status: "pending",
    });
    askBranchToPanel(record);
  });
  $("#branchRecords")?.addEventListener("click", (event) => {
    const back = event.target.closest("[data-branch-back]");
    if (back) {
      closeBranchDetail();
      return;
    }
    const expand = event.target.closest("[data-branch-expand]");
    if (expand) {
      openBranchDetail(expand.dataset.branchExpand);
      return;
    }
    const record = event.target.closest("[data-branch-record]");
    if (record) jumpToBranchRecord(record.dataset.branchRecord);
  });
  $(".nav-rail").addEventListener("click", (event) => {
    const branchButton = event.target.closest("[data-branch-id]");
    if (!branchButton) return;
    $$(".nav-button").forEach((item) => item.classList.remove("active"));
    branchButton.classList.add("active");
    state.activeBranch = {
      id: branchButton.dataset.branchId,
      title: branchButton.textContent,
      context: `In the revision-plan stage "${branchButton.title}"`,
    };
    switchView("chat", true);
    branchButton.classList.add("active");
  });
  $("#queryInput").addEventListener("input", autoResizeComposer);
  $("#queryInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askAgentStream($("#queryInput").value.trim());
    }
  });
  $$(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      setAnswerMode(button.dataset.mode || "");
      $("#queryInput").focus();
    });
  });
};

async function boot() {
  bindEvents();
  setupResizer();
  await Promise.all([loadProfile(), refreshStatus(), loadDocuments(), loadHistory()]);
  renderEvidence([]);
  renderBranchRecords();
  addSystemMessage("Attach lecture PDFs, then ask for summaries, slide search, revision plans, or practice questions.");
}

boot().catch((error) => addSystemMessage(error.message));
