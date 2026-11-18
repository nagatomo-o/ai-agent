const els = {
  connectionStatus: document.getElementById("connection-status"),
  headerWorkspace: document.getElementById("header-workspace"),
  resetBtn: document.getElementById("reset-btn"),
  modelSelect: document.getElementById("model-select"),
  workspaceInput: document.getElementById("workspace-input"),
  workspaceError: document.getElementById("workspace-error"),
  workspaceApplyBtn: document.getElementById("workspace-apply-btn"),
  toolList: document.getElementById("tool-list"),
  helpBtn: document.getElementById("help-btn"),
  helpDialog: document.getElementById("help-dialog"),
  helpCloseBtn: document.getElementById("help-close-btn"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  promptInput: document.getElementById("prompt-input"),
  sendBtn: document.getElementById("send-btn"),
  stopBtn: document.getElementById("stop-btn"),
  agentStatus: document.getElementById("agent-status"),
  footerInfo: document.getElementById("footer-info"),
  toast: document.getElementById("toast"),
};

let sessionId = null;
let abortController = null;
let isBusy = false;
const chatOnlyModels = new Set();
const CHAT_ONLY_STATUS = "チャットのみモード";
const CHAT_ONLY_NOTICE = "このモデルはツールを使えません。ファイルの読み書きは行われません。";

function isChatOnlyModel(name) {
  return chatOnlyModels.has(name);
}

// 選択中のモデルに合わせてステータス表示を更新し、通知が必要なら警告を出す
function applyModelMode(notify = false) {
  const chatOnly = isChatOnlyModel(els.modelSelect.value);
  if (!isBusy) {
    els.agentStatus.textContent = chatOnly ? CHAT_ONLY_STATUS : "";
  }
  if (chatOnly && notify) {
    toastMsg(CHAT_ONLY_NOTICE, "error");
  }
}

function toastMsg(text, type = "success") {
  const p = document.createElement("p");
  p.textContent = text;
  p.dataset.type = type;
  els.toast.appendChild(p);
  setTimeout(() => p.remove(), 3000);
}

// 取得済みのモデルが選択されているときだけ送信できる
function hasUsableModel() {
  const selected = els.modelSelect.selectedOptions[0];
  return Boolean(selected && selected.value && !selected.disabled);
}

function updateSendAvailability() {
  els.sendBtn.disabled = isBusy || !hasUsableModel();
}

function setBusy(busy, statusText = "") {
  isBusy = busy;
  els.sendBtn.disabled = busy || !hasUsableModel();
  els.stopBtn.disabled = !busy;
  els.resetBtn.disabled = busy;
  els.workspaceApplyBtn.disabled = busy;
  els.modelSelect.disabled = busy;
  els.agentStatus.textContent =
    statusText || (!busy && isChatOnlyModel(els.modelSelect.value) ? CHAT_ONLY_STATUS : "");
}

function scrollToBottom() {
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function createMessageArticle(role, label) {
  const article = document.createElement("article");
  article.dataset.role = role;
  const header = document.createElement("header");
  const title = document.createElement("strong");
  title.textContent = label;
  header.appendChild(title);
  article.appendChild(header);
  els.chatLog.appendChild(article);
  scrollToBottom();
  return article;
}

function appendUserMessage(text) {
  const article = createMessageArticle("user", "あなた");
  const pre = document.createElement("pre");
  pre.textContent = text;
  article.appendChild(pre);
  scrollToBottom();
}

function createAssistantTurnView(modelName) {
  const article = createMessageArticle("assistant", modelName);

  const thinkingDetails = document.createElement("details");
  thinkingDetails.hidden = true;
  const thinkingSummary = document.createElement("summary");
  thinkingSummary.textContent = "思考過程";
  const thinkingPre = document.createElement("pre");
  thinkingDetails.append(thinkingSummary, thinkingPre);

  const toolsContainer = document.createElement("div");

  const bodyPre = document.createElement("pre");
  bodyPre.textContent = "";

  article.append(thinkingDetails, toolsContainer, bodyPre);
  return {
    article,
    appendThinking(delta) {
      thinkingDetails.hidden = false;
      thinkingPre.textContent += delta;
      scrollToBottom();
    },
    appendToken(delta) {
      bodyPre.textContent += delta;
      scrollToBottom();
    },
    addToolCall(payload) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `ツール実行: ${payload.name}`;
      const pre = document.createElement("pre");
      pre.textContent = `引数: ${JSON.stringify(payload.arguments, null, 2)}`;
      details.append(summary, pre);
      toolsContainer.appendChild(details);
      details._pre = pre;
      details._summary = summary;
      scrollToBottom();
      return details;
    },
    attachToolResult(detailsEl, payload) {
      if (!detailsEl) {
        const fallback = this.addToolCall({ name: payload.name, arguments: {} });
        fallback._summary.textContent = `ツール結果: ${payload.name} (${payload.ok ? "成功" : "エラー"})`;
        fallback._pre.textContent = payload.output;
        return;
      }
      detailsEl._summary.textContent = `ツール結果: ${payload.name} (${payload.ok ? "成功" : "エラー"})`;
      detailsEl._pre.textContent += `\n\n結果:\n${payload.output}`;
      scrollToBottom();
    },
  };
}

function renderApprovalCard(payload) {
  const article = document.createElement("article");
  article.dataset.kind = "approval";

  const header = document.createElement("header");
  const title = document.createElement("strong");
  const actionMap = {
    create: "ファイル新規作成の承認",
    overwrite: "ファイル上書きの承認",
    edit: "ファイル編集の承認",
  };
  title.textContent = actionMap[payload.action_type] || `操作承認: ${payload.name}`;
  const pathBadge = document.createElement("code");
  pathBadge.textContent = payload.path || "";
  header.append(title, pathBadge);

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = payload.diff || payload.content_preview || JSON.stringify(payload.arguments, null, 2);
  pre.appendChild(code);

  const actions = document.createElement("p");
  const approveBtn = document.createElement("button");
  approveBtn.textContent = "承認して実行";
  const rejectBtn = document.createElement("button");
  rejectBtn.type = "button";
  rejectBtn.textContent = "拒否";
  actions.append(approveBtn, rejectBtn);

  const handleDecision = async (decision) => {
    approveBtn.disabled = true;
    rejectBtn.disabled = true;
    await streamSse(
      `/api/sessions/${sessionId}/approvals/${payload.approval_id}`,
      { decision },
      els.modelSelect.value,
    );
  };

  approveBtn.addEventListener("click", () => handleDecision("approve"));
  rejectBtn.addEventListener("click", () => handleDecision("reject"));

  article.append(header, pre, actions);
  els.chatLog.appendChild(article);
  scrollToBottom();
}

async function streamSse(url, body, modelLabel) {
  abortController = new AbortController();
  setBusy(true, "処理中...");
  const turnView = createAssistantTurnView(modelLabel);
  let lastToolDetails = null;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortController.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "リクエストに失敗しました");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        if (!block.trim()) continue;
        let eventName = "message";
        const dataLines = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trim());
          }
        }
        const payload = JSON.parse(dataLines.join("\n"));

        if (eventName === "status") {
          els.agentStatus.textContent = payload.message;
        } else if (eventName === "thinking") {
          turnView.appendThinking(payload.delta);
        } else if (eventName === "token") {
          turnView.appendToken(payload.delta);
        } else if (eventName === "tool_call") {
          lastToolDetails = turnView.addToolCall(payload);
        } else if (eventName === "tool_result") {
          turnView.attachToolResult(lastToolDetails, payload);
          lastToolDetails = null;
        } else if (eventName === "approval_required") {
          els.agentStatus.textContent = "承認待ちです";
          renderApprovalCard(payload);
        } else if (eventName === "error") {
          toastMsg(payload.message, "error");
          turnView.appendToken(`\n[エラー] ${payload.message}`);
        } else if (eventName === "done") {
          els.agentStatus.textContent = "";
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      turnView.appendToken("\n（生成を停止しました）");
      toastMsg("生成を停止しました", "error");
    } else {
      turnView.appendToken(`\n[エラー] ${err.message}`);
      toastMsg(err.message, "error");
    }
  } finally {
    abortController = null;
    setBusy(false, "");
  }
}

async function initApp() {
  const healthRes = await fetch("/api/health");
  const health = await healthRes.json();

  if (health.ollama_connected) {
    els.connectionStatus.textContent = "Ollama 接続済み";
    els.connectionStatus.dataset.status = "ok";
  } else {
    els.connectionStatus.textContent = "Ollama 未接続";
    els.connectionStatus.dataset.status = "error";
    toastMsg(health.error || "Ollama に接続できません", "error");
  }

  els.modelSelect.innerHTML = "";
  chatOnlyModels.clear();
  const options =
    health.model_options ||
    (health.models || []).map((name) => ({ name, installed: true, tools: null }));
  if (options.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.disabled = true;
    opt.selected = true;
    opt.textContent = "利用できるモデルがありません";
    els.modelSelect.appendChild(opt);
  }
  for (const item of options) {
    const opt = document.createElement("option");
    opt.value = item.name;
    if (!item.installed) {
      opt.disabled = true;
      opt.textContent = `${item.name} (未取得。ollama run ${item.name} で取得してください)`;
    } else {
      const chatOnly = item.tools === false;
      if (chatOnly) chatOnlyModels.add(item.name);
      opt.textContent = chatOnly ? `${item.name} (チャットのみ)` : item.name;
      if (item.name === health.default_model_resolved) opt.selected = true;
    }
    els.modelSelect.appendChild(opt);
  }
  updateSendAvailability();

  els.toolList.innerHTML = "";
  for (const tool of health.tools) {
    const li = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = tool.name;
    const badge = document.createTextNode(tool.requires_approval ? " [要承認]" : " [自動]");
    const br = document.createElement("br");
    const desc = document.createElement("small");
    desc.textContent = tool.description;
    li.append(code, badge, br, desc);
    els.toolList.appendChild(li);
  }

  els.footerInfo.textContent = `接続先: ${health.ollama_host} / 1回あたりツール上限: ${health.max_tool_steps} ステップ`;
  await createNewSession();
  applyModelMode(true);
}

async function createNewSession() {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: els.modelSelect.value || undefined,
      workspace_root: els.workspaceInput.value.trim() || undefined,
    }),
  });
  const data = await res.json();
  sessionId = data.session_id;
  els.workspaceInput.value = data.workspace_root;
  els.headerWorkspace.textContent = data.workspace_root;
  els.workspaceInput.removeAttribute("aria-invalid");
  els.workspaceError.hidden = true;
  els.chatLog.innerHTML = "";
}

els.workspaceApplyBtn.addEventListener("click", async () => {
  if (!sessionId || isBusy) return;
  const newPath = els.workspaceInput.value.trim();
  const res = await fetch(`/api/sessions/${sessionId}/workspace`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workspace_root: newPath }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "フォルダの切替に失敗しました" }));
    els.workspaceInput.setAttribute("aria-invalid", "true");
    els.workspaceError.textContent = err.detail;
    els.workspaceError.hidden = false;
    toastMsg(err.detail, "error");
    return;
  }
  const data = await res.json();
  els.workspaceInput.removeAttribute("aria-invalid");
  els.workspaceError.hidden = true;
  els.headerWorkspace.textContent = data.workspace_root;
  toastMsg("作業フォルダを切り替えました", "success");
});

els.modelSelect.addEventListener("change", async () => {
  updateSendAvailability();
  if (!sessionId || isBusy) return;
  await fetch(`/api/sessions/${sessionId}/model`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: els.modelSelect.value }),
  });
  toastMsg(`モデルを ${els.modelSelect.value} に切り替えました`, "success");
  applyModelMode(true);
});

els.helpBtn.addEventListener("click", () => {
  els.helpDialog.showModal();
});

els.helpCloseBtn.addEventListener("click", () => {
  els.helpDialog.close();
});

els.resetBtn.addEventListener("click", async () => {
  if (isBusy) return;
  await createNewSession();
  toastMsg("会話をリセットしました", "success");
});

els.stopBtn.addEventListener("click", () => {
  if (abortController) {
    abortController.abort();
  }
});

els.chatForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!sessionId || isBusy || !hasUsableModel()) return;
  const prompt = els.promptInput.value.trim();
  if (!prompt) return;
  els.promptInput.value = "";
  appendUserMessage(prompt);
  await streamSse(
    `/api/sessions/${sessionId}/chat`,
    { prompt },
    els.modelSelect.value,
  );
});

initApp();
