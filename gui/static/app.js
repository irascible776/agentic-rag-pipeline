// ==========================================================================
// WORKBENCH CLIENT CONTROLLER: AUTO-ACTIVATION, TUNING & TELEMETRY
// ==========================================================================

// --- 1. Session ID Management ---
let sessionId = localStorage.getItem("rag_session_id");
if (!sessionId) {
  sessionId = "session_" + (crypto.randomUUID ? crypto.randomUUID().slice(0, 8) : Math.random().toString(36).substring(2, 10));
  localStorage.setItem("rag_session_id", sessionId);
}

// Conversation memory & document queue
let chatHistory = [];
let pendingFiles = [];

// --- 2. DOM Element Selectors ---
const statusDot = document.getElementById("status-dot");
const storageStatusText = document.getElementById("storage-status-text");
const sessionIdDisplay = document.getElementById("session-id-display");
const sessionPill = document.getElementById("session-pill");
const corpusStatusBadge = document.getElementById("corpus-status-badge");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const fileCount = document.getElementById("file-count");
const chkPermanent = document.getElementById("chk-permanent");

const progressBox = document.getElementById("progress-box");
const progressMsg = document.getElementById("progress-msg");
const stepExtract = document.getElementById("step-extract");
const stepChunk = document.getElementById("step-chunk");
const stepEmbed = document.getElementById("step-embed");

const statsCard = document.getElementById("stats-card");
const statFiles = document.getElementById("stat-files");
const statChunks = document.getElementById("stat-chunks");
const statEngine = document.getElementById("stat-engine");

const messagesContainer = document.getElementById("messages-container");
const guideContainer = document.getElementById("guide-container");
const btnToggleGuide = document.getElementById("btn-toggle-guide");
const promptSuggestions = document.getElementById("prompt-suggestions");
const chatStatusSubtitle = document.getElementById("chat-status-subtitle");
const btnClearChat = document.getElementById("btn-clear-chat");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");

// Tuning Drawer DOM
const btnOpenTuning = document.getElementById("btn-open-tuning");
const btnCloseDrawer = document.getElementById("btn-close-drawer");
const btnSaveDrawer = document.getElementById("btn-save-drawer");
const btnResetTuning = document.getElementById("btn-reset-tuning");
const tuningDrawer = document.getElementById("tuning-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");

const chunkSizeSlider = document.getElementById("chunk-size-slider");
const chunkOverlapSlider = document.getElementById("chunk-overlap-slider");
const retrieverKSlider = document.getElementById("retriever-k-slider");
const temperatureSlider = document.getElementById("temperature-slider");
const modelSelect = document.getElementById("model-select");

const valChunkSize = document.getElementById("val-chunk-size");
const valChunkOverlap = document.getElementById("val-chunk-overlap");
const valRetrieverK = document.getElementById("val-retriever-k");
const valTemperature = document.getElementById("val-temperature");

const telemetryStorage = document.getElementById("telemetry-storage");
const telemetryEmbedding = document.getElementById("telemetry-embedding");
const telemetryRetrievalTime = document.getElementById("telemetry-retrieval-time");
const telemetryGenerationTime = document.getElementById("telemetry-generation-time");
const telemetryTotalTime = document.getElementById("telemetry-total-time");


// --- 3. Tuning Preferences & Sliders State ---
const TuningConfig = {
  chunkSize: parseInt(localStorage.getItem("wb_chunk_size")) || 1500,
  chunkOverlap: parseInt(localStorage.getItem("wb_chunk_overlap")) || 150,
  retrieverK: parseInt(localStorage.getItem("wb_retriever_k")) || 5,
  temperature: parseFloat(localStorage.getItem("wb_temperature")) || 0.1,
  model: localStorage.getItem("wb_model") || "gemini-3.6-flash",

  save() {
    localStorage.setItem("wb_chunk_size", String(this.chunkSize));
    localStorage.setItem("wb_chunk_overlap", String(this.chunkOverlap));
    localStorage.setItem("wb_retriever_k", String(this.retrieverK));
    localStorage.setItem("wb_temperature", String(this.temperature));
    localStorage.setItem("wb_model", this.model);
  },

  syncUI() {
    chunkSizeSlider.value = this.chunkSize;
    valChunkSize.textContent = Number(this.chunkSize).toLocaleString();

    chunkOverlapSlider.value = this.chunkOverlap;
    valChunkOverlap.textContent = Number(this.chunkOverlap).toLocaleString();

    retrieverKSlider.value = this.retrieverK;
    valRetrieverK.textContent = this.retrieverK;

    temperatureSlider.value = this.temperature;
    valTemperature.textContent = this.temperature.toFixed(2);

    modelSelect.value = this.model;
  },

  reset() {
    this.chunkSize = 1500;
    this.chunkOverlap = 150;
    this.retrieverK = 5;
    this.temperature = 0.1;
    this.model = "gemini-3.6-flash";
    this.save();
    this.syncUI();
  }
};

// Bind Slider Input Listeners
chunkSizeSlider.addEventListener("input", (e) => {
  TuningConfig.chunkSize = parseInt(e.target.value);
  valChunkSize.textContent = Number(TuningConfig.chunkSize).toLocaleString();
  TuningConfig.save();
});

chunkOverlapSlider.addEventListener("input", (e) => {
  TuningConfig.chunkOverlap = parseInt(e.target.value);
  valChunkOverlap.textContent = Number(TuningConfig.chunkOverlap).toLocaleString();
  TuningConfig.save();
});

retrieverKSlider.addEventListener("input", (e) => {
  TuningConfig.retrieverK = parseInt(e.target.value);
  valRetrieverK.textContent = TuningConfig.retrieverK;
  TuningConfig.save();
});

temperatureSlider.addEventListener("input", (e) => {
  TuningConfig.temperature = parseFloat(e.target.value);
  valTemperature.textContent = TuningConfig.temperature.toFixed(2);
  TuningConfig.save();
});

modelSelect.addEventListener("change", (e) => {
  TuningConfig.model = e.target.value;
  TuningConfig.save();
});

btnResetTuning.addEventListener("click", () => TuningConfig.reset());


// --- 4. Drawer Toggle Logic ---
function openTuningDrawer() {
  tuningDrawer.classList.remove("hidden");
  drawerBackdrop.classList.remove("hidden");
}

function closeTuningDrawer() {
  tuningDrawer.classList.add("hidden");
  drawerBackdrop.classList.add("hidden");
}

btnOpenTuning.addEventListener("click", openTuningDrawer);
btnCloseDrawer.addEventListener("click", closeTuningDrawer);
btnSaveDrawer.addEventListener("click", closeTuningDrawer);
drawerBackdrop.addEventListener("click", closeTuningDrawer);

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !tuningDrawer.classList.contains("hidden")) {
    closeTuningDrawer();
  }
});


// --- 5. Guide Toggle Logic ---
btnToggleGuide.addEventListener("click", () => {
  if (guideContainer.classList.contains("hidden")) {
    guideContainer.classList.remove("hidden");
    messagesContainer.scrollTop = 0;
  } else {
    guideContainer.classList.add("hidden");
  }
});


// --- 6. Session Copy Toast ---
sessionIdDisplay.textContent = sessionId;
sessionPill.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(sessionId);
    const original = sessionIdDisplay.textContent;
    sessionIdDisplay.textContent = "Copied!";
    setTimeout(() => {
      sessionIdDisplay.textContent = original;
    }, 1500);
  } catch (err) {
    // clipboard fallback
  }
});


// --- 7. Initial System Config Check ---
async function checkSystemConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();

    statusDot.classList.add("ready");
    const isSupabase = data.mode === "supabase";
    storageStatusText.textContent = isSupabase ? "Supabase pgvector" : "Local ChromaDB";
    corpusStatusBadge.textContent = isSupabase ? "Cloud Ready" : "Local Ready";

    if (telemetryStorage) {
      telemetryStorage.textContent = isSupabase ? "🟢 Supabase Cloud" : "💻 Local ChromaDB";
    }
    if (telemetryEmbedding && data.embedding_model) {
      telemetryEmbedding.textContent = `${data.embedding_model} (768d)`;
    }
    if (statEngine) {
      statEngine.textContent = isSupabase ? "Supabase" : "Chroma";
    }
  } catch (err) {
    statusDot.classList.remove("ready");
    storageStatusText.textContent = "Server Offline";
    corpusStatusBadge.textContent = "Offline";
    chatStatusSubtitle.textContent = "Backend connection unreachable";
  }
}


// --- 8. Drag & Drop and Instant Auto-Ingestion ---
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleIncomingFiles(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length > 0) {
    handleIncomingFiles(fileInput.files);
  }
});

function handleIncomingFiles(files) {
  const allowed = [".pdf", ".docx", ".txt", ".md"];
  const newValidFiles = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const nameLower = file.name.toLowerCase();
    const isValid = allowed.some(ext => nameLower.endsWith(ext));
    if (isValid && !pendingFiles.some(f => f.name === file.name)) {
      pendingFiles.push(file);
      newValidFiles.push(file);
    }
  }

  if (newValidFiles.length === 0) return;

  renderFileList();
  // Automatically trigger processing! Zero redundant continue clicks
  triggerAutoIngest(newValidFiles);
}

function renderFileList() {
  fileList.innerHTML = "";
  fileCount.textContent = pendingFiles.length;

  if (pendingFiles.length === 0) {
    fileList.innerHTML = '<li class="empty-file-hint">No documents loaded yet. Drop files above to index automatically.</li>';
    return;
  }

  pendingFiles.forEach((file, idx) => {
    const li = document.createElement("li");
    li.className = "file-item";
    const ext = file.name.split(".").pop().toUpperCase();
    const sizeKB = (file.size / 1024).toFixed(1);

    li.innerHTML = `
      <span class="file-name" title="${file.name}">${file.name}</span>
      <span class="file-ext-badge">${ext}</span>
      <span class="file-size-tag">${sizeKB} KB</span>
    `;

    fileList.appendChild(li);
  });
}

// Ingestion Execution
async function triggerAutoIngest(filesToIngest) {
  const isPermanent = chkPermanent.checked;

  progressBox.classList.remove("hidden");
  stepExtract.className = "step-item active";
  stepChunk.className = "step-item";
  stepEmbed.className = "step-item";
  progressMsg.textContent = "Extracting document text...";

  const formData = new FormData();
  filesToIngest.forEach(f => formData.append("files", f));
  formData.append("session_id", sessionId);
  formData.append("is_permanent", isPermanent);
  formData.append("chunk_size", TuningConfig.chunkSize);
  formData.append("chunk_overlap", TuningConfig.chunkOverlap);

  // Animate stepper transition
  setTimeout(() => {
    stepChunk.className = "step-item active";
    progressMsg.textContent = `Chunking (${TuningConfig.chunkSize} chars)...`;
  }, 400);

  try {
    const res = await fetch("/api/upload-chunk", {
      method: "POST",
      body: formData
    });

    stepEmbed.className = "step-item active";
    progressMsg.textContent = "Embedding & Indexing vectors...";

    const data = await res.json();
    progressBox.classList.add("hidden");

    if (data.status === "success") {
      statFiles.textContent = data.files_count;
      statChunks.textContent = data.chunks_count;
      if (statEngine) statEngine.textContent = data.mode === "supabase" ? "Supabase" : "Chroma";
      statsCard.classList.remove("hidden");

      // Save state to sessionStorage
      sessionStorage.setItem("wb_indexed", "true");
      sessionStorage.setItem("wb_files_count", String(data.files_count));
      sessionStorage.setItem("wb_chunks_count", String(data.chunks_count));
      sessionStorage.setItem("wb_mode", data.mode);

      // Auto-unlock chat interface!
      enableChatInterface();
      chatStatusSubtitle.textContent = `Indexed ${data.chunks_count} chunks ready for questions`;

      // Collapse guide if open, show prompt suggestions
      if (guideContainer) guideContainer.classList.add("hidden");
      if (promptSuggestions) promptSuggestions.classList.remove("hidden");

    } else {
      alert("Ingestion error: " + (data.message || "Failed to process files."));
    }
  } catch (err) {
    progressBox.classList.add("hidden");
    alert("Network error during document ingestion.");
  }
}

function enableChatInterface() {
  chatInput.disabled = false;
  btnSend.disabled = false;
  chatInput.placeholder = "Ask a question about your documents... (Press Enter)";
  chatInput.focus();
}


// --- 9. Conversational Chat & Grounded Responses ---
async function handleSendMessage(overrideQuery = null) {
  const query = (overrideQuery || chatInput.value).trim();
  if (!query) return;

  chatInput.value = "";
  chatInput.disabled = true;
  btnSend.disabled = true;

  // Append user row
  appendMessage("user", query);

  // Append typing dots
  const typingIndicator = appendTypingIndicator();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: query,
        history: chatHistory,
        session_id: sessionId,
        k: TuningConfig.retrieverK,
        temperature: TuningConfig.temperature,
        model: TuningConfig.model
      })
    });

    const data = await res.json();
    if (typingIndicator && typingIndicator.parentNode) {
      typingIndicator.remove();
    }

    if (data.answer) {
      appendMessage("ai", data.answer, data.sources || []);
      chatHistory.push({ role: "user", content: query });
      chatHistory.push({ role: "assistant", content: data.answer });
      sessionStorage.setItem("wb_chat_history", JSON.stringify(chatHistory));

      // Update live telemetry in drawer!
      if (data.telemetry) {
        if (telemetryRetrievalTime) telemetryRetrievalTime.textContent = `${data.telemetry.retrieval_ms} ms`;
        if (telemetryGenerationTime) telemetryGenerationTime.textContent = `${data.telemetry.generation_ms} ms`;
        if (telemetryTotalTime) telemetryTotalTime.textContent = `${data.telemetry.total_ms} ms`;
      }
    } else {
      appendMessage("ai", "I was unable to retrieve a grounded answer from the indexed documents.");
    }
  } catch (err) {
    if (typingIndicator && typingIndicator.parentNode) {
      typingIndicator.remove();
    }
    appendMessage("ai", "Connection error: Could not reach the server.");
  } finally {
    chatInput.disabled = false;
    btnSend.disabled = false;
    chatInput.focus();
  }
}

// Enter sends without newline (Shift+Enter adds newline)
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    e.stopPropagation();
    handleSendMessage();
  }
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  handleSendMessage();
});

btnSend.addEventListener("click", () => handleSendMessage());


// --- 10. Suggested Prompt Chips ---
document.querySelectorAll(".suggestion-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const prompt = chip.getAttribute("data-prompt");
    if (prompt) {
      handleSendMessage(prompt);
    }
  });
});


// --- 11. Clear Chat History ---
btnClearChat.addEventListener("click", () => {
  chatHistory = [];
  sessionStorage.removeItem("wb_chat_history");
  
  // Clear messages but keep guide if no docs
  messagesContainer.innerHTML = "";
  const isIndexed = sessionStorage.getItem("wb_indexed") === "true";
  if (!isIndexed && guideContainer) {
    guideContainer.classList.remove("hidden");
    messagesContainer.appendChild(guideContainer);
  }
});


// --- 12. Message Rendering & Utilities ---
function appendMessage(role, text, sources = []) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const meta = document.createElement("div");
  meta.className = "message-sender-meta";
  meta.textContent = role === "user" ? "You" : "Workbench Assistant";
  row.appendChild(meta);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatMarkdown(text);

  // Citation pills
  if (sources && sources.length > 0) {
    const sourcesBox = document.createElement("div");
    sourcesBox.className = "sources-box";
    sourcesBox.innerHTML = '<span class="sources-label">Grounded Sources:</span>';
    sources.forEach(src => {
      const pill = document.createElement("span");
      pill.className = "source-pill";
      pill.textContent = src;
      pill.title = `Passage retrieved from: ${src}`;
      sourcesBox.appendChild(pill);
    });
    bubble.appendChild(sourcesBox);
  }

  row.appendChild(bubble);

  // Micro-actions for AI responses (Copy Markdown)
  if (role === "ai") {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    
    const copyBtn = document.createElement("button");
    copyBtn.className = "btn-msg-copy";
    copyBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy</span>
    `;
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.innerHTML = `<span>✓ Copied</span>`;
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>
          `;
        }, 1500);
      } catch (e) {}
    });

    actions.appendChild(copyBtn);
    row.appendChild(actions);
  }

  messagesContainer.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return row;
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row ai";

  const meta = document.createElement("div");
  meta.className = "message-sender-meta";
  meta.textContent = "Workbench Assistant";
  row.appendChild(meta);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `
    <div class="typing-dots">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>
  `;

  row.appendChild(bubble);
  messagesContainer.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return row;
}

// Markdown Formatter (Bold, code blocks, bullet lists, paragraphs)
function formatMarkdown(text) {
  if (!text) return "";

  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Inline code: `code`
  escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold: **text**
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Split into paragraphs & list items
  const lines = escaped.split("\n");
  let html = "";
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (line.startsWith("* ") || line.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${line.substring(2)}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      if (line.length > 0) {
        html += `<p>${line}</p>`;
      }
    }
  }

  if (inList) html += "</ul>";
  return html || "<p>" + escaped + "</p>";
}


// --- 13. Data Retention Lifecycle: Background Session Purge ---
// If "Retain in Knowledge Base" is unchecked, silently clear session on tab close
window.addEventListener("beforeunload", () => {
  if (chkPermanent && !chkPermanent.checked) {
    const payload = JSON.stringify({ session_id: sessionId });
    const blob = new Blob([payload], { type: "application/json" });
    navigator.sendBeacon("/api/session/clear", blob);
  }
});


// --- 14. Session Restoration on Page Reload ---
function restoreSession() {
  TuningConfig.syncUI();

  const isIndexed = sessionStorage.getItem("wb_indexed") === "true";
  if (isIndexed) {
    const filesCount = sessionStorage.getItem("wb_files_count") || "0";
    const chunksCount = sessionStorage.getItem("wb_chunks_count") || "0";
    const mode = sessionStorage.getItem("wb_mode") || "local";

    statFiles.textContent = filesCount;
    statChunks.textContent = chunksCount;
    if (statEngine) statEngine.textContent = mode === "supabase" ? "Supabase" : "Chroma";
    statsCard.classList.remove("hidden");

    enableChatInterface();
    chatStatusSubtitle.textContent = `Indexed ${chunksCount} chunks ready for questions`;

    if (guideContainer) guideContainer.classList.add("hidden");
    if (promptSuggestions) promptSuggestions.classList.remove("hidden");
  }

  const savedHistory = sessionStorage.getItem("wb_chat_history");
  if (savedHistory) {
    try {
      chatHistory = JSON.parse(savedHistory);
      if (chatHistory.length > 0) {
        if (guideContainer) guideContainer.classList.add("hidden");
        chatHistory.forEach(msg => {
          appendMessage(msg.role === "user" ? "user" : "ai", msg.content);
        });
      }
    } catch (e) {
      chatHistory = [];
    }
  }
}

// Initial Boot
checkSystemConfig();
restoreSession();
