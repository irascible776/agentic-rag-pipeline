// ==========================================================================
// NEXUS CLIENT CONTROLLER: VISIONARY 2.0 WORKSPACE
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
const appContainer = document.getElementById("app-container");
const sidebarPanel = document.getElementById("sidebar-panel");
const btnSidebarToggle = document.getElementById("btn-sidebar-toggle");
const btnSidebarOpen = document.getElementById("btn-sidebar-open");

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
const progressBarFill = document.getElementById("progress-bar-fill");
const stepExtract = document.getElementById("step-extract");
const stepChunk = document.getElementById("step-chunk");
const stepEmbed = document.getElementById("step-embed");

const statsCard = document.getElementById("stats-card");
const statFiles = document.getElementById("stat-files");
const statChunks = document.getElementById("stat-chunks");
const statEngine = document.getElementById("stat-engine");

const messagesContainer = document.getElementById("messages-container");
const guideContainer = document.getElementById("guide-container");
const guideModal = document.getElementById("guide-modal");
const guideModalClose = document.getElementById("guide-modal-close");
const btnToggleGuide = document.getElementById("btn-toggle-guide");
const promptSuggestions = document.getElementById("prompt-suggestions");
const chatStatusSubtitle = document.getElementById("chat-status-subtitle");
const btnClearChat = document.getElementById("btn-clear-chat");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");

// --- 3. Sidebar Collapse & Auto-Hide Management (Step 4) ---
function collapseSidebar() {
  if (appContainer) appContainer.classList.add("sidebar-collapsed");
  if (btnSidebarOpen) btnSidebarOpen.classList.remove("hidden");
}

function expandSidebar() {
  if (appContainer) appContainer.classList.remove("sidebar-collapsed");
  if (btnSidebarOpen) btnSidebarOpen.classList.add("hidden");
}

function toggleSidebarManual() {
  if (appContainer.classList.contains("sidebar-collapsed")) {
    expandSidebar();
    localStorage.setItem("wb_sidebar_manual", "open");
  } else {
    collapseSidebar();
    localStorage.setItem("wb_sidebar_manual", "collapsed");
  }
}

if (btnSidebarToggle) {
  btnSidebarToggle.addEventListener("click", () => {
    collapseSidebar();
    localStorage.setItem("wb_sidebar_manual", "collapsed");
  });
}

if (btnSidebarOpen) {
  btnSidebarOpen.addEventListener("click", () => {
    expandSidebar();
    localStorage.setItem("wb_sidebar_manual", "open");
  });
}

// Ctrl/Cmd+B Keyboard Shortcut to Toggle Sidebar (Step 4)
window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && (e.key === "b" || e.key === "B")) {
    e.preventDefault();
    toggleSidebarManual();
  }
  if (e.key === "Escape") {
    if (guideModal && !guideModal.classList.contains("hidden")) {
      guideModal.classList.add("hidden");
    }
  }
});

// Guide Modal Triggers (Step 3)
if (btnToggleGuide) {
  btnToggleGuide.addEventListener("click", () => {
    if (guideModal) guideModal.classList.remove("hidden");
  });
}

if (guideModalClose) {
  guideModalClose.addEventListener("click", () => {
    if (guideModal) guideModal.classList.add("hidden");
  });
}

if (guideModal) {
  guideModal.addEventListener("click", (e) => {
    if (e.target === guideModal) guideModal.classList.add("hidden");
  });
}

// --- 4. Tuning Preferences & Sliders State ---
const TuningConfig = {
  chunkSize: parseInt(localStorage.getItem("wb_chunk_size")) || 1500,
  chunkOverlap: parseInt(localStorage.getItem("wb_chunk_overlap")) || 150,
  retrieverK: parseInt(localStorage.getItem("wb_retriever_k")) || 5,
  temperature: parseFloat(localStorage.getItem("wb_temperature")) || 0.1,
  model: localStorage.getItem("wb_model") || "gemini-3.6-flash",
  enableHybrid: localStorage.getItem("wb_enable_hybrid") !== "false",
  enableReranker: localStorage.getItem("wb_enable_reranker") !== "false",

  save() {
    localStorage.setItem("wb_chunk_size", String(this.chunkSize));
    localStorage.setItem("wb_chunk_overlap", String(this.chunkOverlap));
    localStorage.setItem("wb_retriever_k", String(this.retrieverK));
    localStorage.setItem("wb_temperature", String(this.temperature));
    localStorage.setItem("wb_model", this.model);
    localStorage.setItem("wb_enable_hybrid", String(this.enableHybrid));
    localStorage.setItem("wb_enable_reranker", String(this.enableReranker));
  },

  syncUI() {
    if (chunkSizeSlider && valChunkSize) {
      chunkSizeSlider.value = this.chunkSize;
      valChunkSize.textContent = Number(this.chunkSize).toLocaleString();
    }
    if (chunkOverlapSlider && valChunkOverlap) {
      chunkOverlapSlider.value = this.chunkOverlap;
      valChunkOverlap.textContent = Number(this.chunkOverlap).toLocaleString();
    }
    if (retrieverKSlider && valRetrieverK) {
      retrieverKSlider.value = this.retrieverK;
      valRetrieverK.textContent = this.retrieverK;
    }
    if (temperatureSlider && valTemperature) {
      temperatureSlider.value = this.temperature;
      valTemperature.textContent = this.temperature.toFixed(2);
    }
    if (modelSelect) modelSelect.value = this.model;
    if (chkHybrid) chkHybrid.checked = this.enableHybrid;
    if (chkReranker) chkReranker.checked = this.enableReranker;
  },

  reset() {
    this.chunkSize = 1500;
    this.chunkOverlap = 150;
    this.retrieverK = 5;
    this.temperature = 0.1;
    this.model = "gemini-3.6-flash";
    this.enableHybrid = true;
    this.enableReranker = true;
    this.save();
    this.syncUI();
  }
};

// Bind Slider Input Listeners
if (chunkSizeSlider) {
  chunkSizeSlider.addEventListener("input", (e) => {
    TuningConfig.chunkSize = parseInt(e.target.value);
    if (valChunkSize) valChunkSize.textContent = Number(TuningConfig.chunkSize).toLocaleString();
    TuningConfig.save();
  });
}

if (chunkOverlapSlider) {
  chunkOverlapSlider.addEventListener("input", (e) => {
    TuningConfig.chunkOverlap = parseInt(e.target.value);
    if (valChunkOverlap) valChunkOverlap.textContent = Number(TuningConfig.chunkOverlap).toLocaleString();
    TuningConfig.save();
  });
}

if (retrieverKSlider) {
  retrieverKSlider.addEventListener("input", (e) => {
    TuningConfig.retrieverK = parseInt(e.target.value);
    if (valRetrieverK) valRetrieverK.textContent = TuningConfig.retrieverK;
    TuningConfig.save();
  });
}

if (temperatureSlider) {
  temperatureSlider.addEventListener("input", (e) => {
    TuningConfig.temperature = parseFloat(e.target.value);
    if (valTemperature) valTemperature.textContent = TuningConfig.temperature.toFixed(2);
    TuningConfig.save();
  });
}

if (modelSelect) {
  modelSelect.addEventListener("change", (e) => {
    TuningConfig.model = e.target.value;
    TuningConfig.save();
  });
}

if (chkHybrid) {
  chkHybrid.addEventListener("change", (e) => {
    TuningConfig.enableHybrid = e.target.checked;
    TuningConfig.save();
  });
}

if (chkReranker) {
  chkReranker.addEventListener("change", (e) => {
    TuningConfig.enableReranker = e.target.checked;
    TuningConfig.save();
  });
}

if (btnResetTuning) {
  btnResetTuning.addEventListener("click", () => TuningConfig.reset());
}

// --- 5. Drawer Toggle Logic ---
function openTuningDrawer() {
  if (tuningDrawer) tuningDrawer.classList.remove("hidden");
  if (drawerBackdrop) drawerBackdrop.classList.remove("hidden");
}

function closeTuningDrawer() {
  if (tuningDrawer) tuningDrawer.classList.add("hidden");
  if (drawerBackdrop) drawerBackdrop.classList.add("hidden");
}

if (btnOpenTuning) btnOpenTuning.addEventListener("click", openTuningDrawer);
if (btnCloseDrawer) btnCloseDrawer.addEventListener("click", closeTuningDrawer);
if (btnSaveDrawer) btnSaveDrawer.addEventListener("click", closeTuningDrawer);
if (drawerBackdrop) drawerBackdrop.addEventListener("click", closeTuningDrawer);

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && tuningDrawer && !tuningDrawer.classList.contains("hidden")) {
    closeTuningDrawer();
  }
});

// --- 6. Guide Toggle Logic ---
if (btnToggleGuide) {
  btnToggleGuide.addEventListener("click", () => {
    if (guideContainer) {
      if (guideContainer.classList.contains("hidden")) {
        guideContainer.classList.remove("hidden");
        messagesContainer.scrollTop = 0;
      } else {
        guideContainer.classList.add("hidden");
      }
    }
  });
}

// --- 7. Session Copy Toast ---
if (sessionIdDisplay) sessionIdDisplay.textContent = sessionId;
if (sessionPill) {
  sessionPill.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
      const original = sessionIdDisplay.textContent;
      sessionIdDisplay.textContent = "Copied!";
      setTimeout(() => {
        sessionIdDisplay.textContent = original;
      }, 1500);
    } catch (err) {}
  });
}

// --- 8. Initial System Config Check ---
async function checkSystemConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();

    if (statusDot) statusDot.classList.add("ready");
    const isSupabase = data.mode === "supabase";
    if (storageStatusText) storageStatusText.textContent = isSupabase ? "Supabase pgvector" : "Local ChromaDB";
    if (corpusStatusBadge) {
      corpusStatusBadge.textContent = isSupabase ? "Hybrid + Re-Rank" : "Local Chroma";
    }

    if (telemetryStorage) {
      telemetryStorage.textContent = isSupabase ? "🟢 Supabase pgvector" : "💻 Local ChromaDB";
    }
    if (telemetryEmbedding && data.embedding_model) {
      telemetryEmbedding.textContent = `${data.embedding_model}`;
    }
    if (statEngine) {
      statEngine.textContent = isSupabase ? "pgvector" : "Chroma";
    }
  } catch (err) {
    if (statusDot) statusDot.classList.remove("ready");
    if (storageStatusText) storageStatusText.textContent = "Server Offline";
    if (corpusStatusBadge) corpusStatusBadge.textContent = "Offline";
    if (chatStatusSubtitle) chatStatusSubtitle.textContent = "Backend connection unreachable";
  }
}

// --- 9. Drag & Drop and Instant Auto-Ingestion ---
if (dropZone && fileInput) {
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
}

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
  triggerAutoIngest(newValidFiles);
}

function renderFileList() {
  if (!fileList || !fileCount) return;
  fileList.innerHTML = "";
  fileCount.textContent = pendingFiles.length;

  if (pendingFiles.length === 0) {
    fileList.innerHTML = '<li class="empty-file-hint">No documents loaded yet. Drop files above to index automatically.</li>';
    return;
  }

  pendingFiles.forEach((file) => {
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

// Ingestion Execution with Single Thin Progress Bar & Smart Auto-Collapse (Step 4)
async function triggerAutoIngest(filesToIngest) {
  const isPermanent = chkPermanent ? chkPermanent.checked : true;

  // Reset manual override on new upload
  localStorage.removeItem("wb_sidebar_manual");

  if (progressBox) progressBox.classList.remove("hidden");
  if (progressBarFill) {
    progressBarFill.style.width = "25%";
    progressBarFill.style.background = "var(--accent)";
  }
  if (progressMsg) {
    progressMsg.textContent = "Extracting document text...";
    progressMsg.style.color = "";
  }

  if (chatStatusSubtitle) {
    chatStatusSubtitle.textContent = "Indexing documents in progress...";
    chatStatusSubtitle.style.color = "var(--accent)";
  }

  const formData = new FormData();
  filesToIngest.forEach(f => formData.append("files", f));
  formData.append("session_id", sessionId);
  formData.append("is_permanent", isPermanent);
  formData.append("chunk_size", TuningConfig.chunkSize);
  formData.append("chunk_overlap", TuningConfig.chunkOverlap);

  setTimeout(() => {
    if (progressBarFill) progressBarFill.style.width = "60%";
    if (progressMsg) progressMsg.textContent = `Chunking (${TuningConfig.chunkSize} chars)...`;
    if (chatStatusSubtitle) {
      chatStatusSubtitle.textContent = `Chunking text (${TuningConfig.chunkSize} chars)...`;
    }
  }, 400);

  try {
    const res = await fetch("/api/upload-chunk", {
      method: "POST",
      body: formData
    });

    if (progressBarFill) progressBarFill.style.width = "90%";
    if (progressMsg) progressMsg.textContent = "Embedding & indexing vectors...";
    if (chatStatusSubtitle) {
      chatStatusSubtitle.textContent = "Embedding & indexing vectors...";
    }

    const data = await res.json();

    if (data.status === "success") {
      if (progressBarFill) progressBarFill.style.width = "100%";
      if (progressMsg) progressMsg.textContent = "Indexing complete!";

      setTimeout(() => {
        if (progressBox) progressBox.classList.add("hidden");
      }, 500);

      if (statFiles) statFiles.textContent = data.files_count;
      if (statChunks) statChunks.textContent = data.chunks_count;
      if (statEngine) statEngine.textContent = data.mode === "supabase" ? "pgvector" : "Chroma";
      if (statsCard) statsCard.classList.remove("hidden");

      sessionStorage.setItem("wb_indexed", "true");
      sessionStorage.setItem("wb_files_count", String(data.files_count));
      sessionStorage.setItem("wb_chunks_count", String(data.chunks_count));
      sessionStorage.setItem("wb_mode", data.mode);

      // Auto-unlock chat interface
      enableChatInterface();
      if (chatStatusSubtitle) {
        chatStatusSubtitle.textContent = `Indexed ${data.chunks_count} chunks • Ready`;
        chatStatusSubtitle.style.color = "";
      }

      // Hide guide, show prompt suggestions
      if (guideContainer) guideContainer.classList.add("hidden");
      if (promptSuggestions) promptSuggestions.classList.remove("hidden");

      // Auto-collapse smoothly after 800ms unless manually reopened (Step 4)
      setTimeout(() => {
        if (localStorage.getItem("wb_sidebar_manual") !== "open") {
          collapseSidebar();
        }
        if (chatInput) chatInput.focus();
      }, 800);

    } else {
      // On failure: stay open with an inline error
      if (progressBarFill) {
        progressBarFill.style.width = "100%";
        progressBarFill.style.background = "#ef4444";
      }
      if (progressMsg) {
        progressMsg.textContent = "Indexing issue: " + (data.message || "Failed to process files.");
        progressMsg.style.color = "#ef4444";
      }
      if (chatStatusSubtitle) {
        chatStatusSubtitle.textContent = "Indexing encountered an issue";
        chatStatusSubtitle.style.color = "#ef4444";
      }
    }
  } catch (err) {
    // Stay open with inline error
    if (progressBarFill) {
      progressBarFill.style.width = "100%";
      progressBarFill.style.background = "#ef4444";
    }
    if (progressMsg) {
      progressMsg.textContent = "Network error during document ingestion.";
      progressMsg.style.color = "#ef4444";
    }
    if (chatStatusSubtitle) {
      chatStatusSubtitle.textContent = "Indexing failed. Please retry.";
      chatStatusSubtitle.style.color = "#ef4444";
    }
  }
}

function enableChatInterface() {
  if (chatInput) {
    chatInput.disabled = false;
    chatInput.placeholder = "Describe what you'd like to query or ask Nexus... (Enter to send)";
    chatInput.focus();
  }
  if (btnSend) {
    btnSend.disabled = false;
    btnSend.title = "Send Question (Enter)";
  }
}

// --- 10. Conversational Chat & Grounded Responses ---
async function handleSendMessage(overrideQuery = null) {
  const query = (overrideQuery || (chatInput ? chatInput.value : "")).trim();
  if (!query) return;

  if (chatInput) {
    chatInput.value = "";
    chatInput.disabled = true;
  }
  if (btnSend) btnSend.disabled = true;

  // Append user bubble
  appendMessage("user", query);

  // Append Nexus typing indicator
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
        model: TuningConfig.model,
        enable_hybrid: TuningConfig.enableHybrid,
        enable_reranker: TuningConfig.enableReranker
      })
    });

    const data = await res.json();

    if (typingIndicator && typingIndicator.parentNode) {
      typingIndicator.remove();
    }

    if (data.answer) {
      appendMessage("ai", data.answer, data.sources || [], data.media || []);
      chatHistory.push({ role: "user", content: query });
      chatHistory.push({
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        media: data.media || []
      });
      sessionStorage.setItem("wb_chat_history", JSON.stringify(chatHistory));

      // Update live telemetry in drawer
      if (data.telemetry) {
        if (telemetryRetrievalTime) telemetryRetrievalTime.textContent = `${data.telemetry.retrieval_ms} ms`;
        if (telemetryGenerationTime) telemetryGenerationTime.textContent = `${data.telemetry.generation_ms} ms`;
        if (telemetryTotalTime) telemetryTotalTime.textContent = `${data.telemetry.total_ms} ms`;
        if (telemetryDenseTime) telemetryDenseTime.textContent = `${data.telemetry.dense_search_ms ?? 0} ms`;
        if (telemetryBm25Time) telemetryBm25Time.textContent = `${data.telemetry.bm25_search_ms ?? 0} ms`;
        if (telemetryRerankTime) telemetryRerankTime.textContent = `${data.telemetry.rerank_ms ?? 0} ms`;
        if (telemetryCandidates) telemetryCandidates.textContent = `${data.telemetry.candidates_pooled ?? '--'} pooled → ${data.telemetry.chunks_retrieved ?? '--'} sent`;
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
    if (chatInput) {
      chatInput.disabled = false;
      chatInput.focus();
    }
    if (btnSend) btnSend.disabled = false;
  }
}

// Enter sends without newline (Shift+Enter adds newline)
if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();
      handleSendMessage();
    }
  });
}

if (chatForm) {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });
}

if (btnSend) {
  btnSend.addEventListener("click", () => handleSendMessage());
}

// --- 11. Suggested Prompt Chips ---
document.querySelectorAll(".suggestion-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const prompt = chip.getAttribute("data-prompt");
    if (prompt) {
      handleSendMessage(prompt);
    }
  });
});

// --- 12. Clear Chat History ---
if (btnClearChat) {
  btnClearChat.addEventListener("click", () => {
    chatHistory = [];
    sessionStorage.removeItem("wb_chat_history");
    
    if (messagesContainer) {
      messagesContainer.innerHTML = "";
      const isIndexed = sessionStorage.getItem("wb_indexed") === "true";
      if (!isIndexed && guideContainer) {
        guideContainer.classList.remove("hidden");
        messagesContainer.appendChild(guideContainer);
      }
    }
  });
}

// --- 13. Dynamic Instagram Recency Gradient Updater ---
function updateChatRecency() {
  const userRows = document.querySelectorAll(".message-row.user");
  const total = userRows.length;
  userRows.forEach((row, idx) => {
    row.classList.remove("recency-newest", "recency-recent", "recency-older");
    const distFromEnd = total - 1 - idx;
    if (distFromEnd === 0) {
      row.classList.add("recency-newest"); // 1st newest (bottom)
    } else if (distFromEnd <= 2) {
      row.classList.add("recency-recent"); // 2nd & 3rd newest
    } else {
      row.classList.add("recency-older");  // older messages
    }
  });
}

// --- 14. Message Rendering: Visionary 2.0 Avatars & Overlapping Citations ---
function appendMessage(role, text, sources = [], media = []) {
  if (!messagesContainer) return;

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  if (role === "ai") {
    // Left Avatar Badge for Nexus ("V" / "N")
    const avatar = document.createElement("div");
    avatar.className = "message-avatar ai-avatar";
    avatar.textContent = "N";
    row.appendChild(avatar);

    // Message Bubble
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatMarkdown(text);

    // Citation Pills
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

    // Media Evidence (Overlapping Card Fan-Out Stack Style)
    if (media && media.length > 0) {
      const mediaGallery = document.createElement("div");
      mediaGallery.className = "media-gallery";
      mediaGallery.innerHTML = '<span class="media-gallery-label">Retrieved Evidence & Figures:</span>';

      const mediaGrid = document.createElement("div");
      mediaGrid.className = "media-grid";

      media.forEach(item => {
        const card = document.createElement("div");
        card.className = "media-card";
        card.innerHTML = `
          <div class="media-thumb-wrap">
            <img src="${item.url}" alt="${item.caption || 'Figure'}" loading="lazy" class="media-thumb" />
            <div class="media-zoom-overlay">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                <line x1="11" y1="8" x2="11" y2="14"></line>
                <line x1="8" y1="11" x2="14" y2="11"></line>
              </svg>
              <span>Expand</span>
            </div>
          </div>
          <div class="media-meta">
            <span class="media-caption" title="${item.caption || ''}">${item.caption || 'Visual Asset'}</span>
            <span class="media-source-tag">${item.source}${item.page ? ` (p.${item.page})` : ''}</span>
          </div>
        `;

        card.addEventListener("click", () => {
          openLightbox(item.url, item.caption, item.source, item.page);
        });

        mediaGrid.appendChild(card);
      });

      mediaGallery.appendChild(mediaGrid);
      bubble.appendChild(mediaGallery);
    }

    row.appendChild(bubble);

    // Inline Micro-Actions: Thumbs Up, Thumbs Down, Copy
    const actions = document.createElement("div");
    actions.className = "message-actions";
    
    // Copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn btn-msg-copy";
    copyBtn.title = "Copy answer";
    copyBtn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.classList.add("copied");
        setTimeout(() => copyBtn.classList.remove("copied"), 1500);
      } catch (e) {}
    });

    // Thumbs up / down feedback
    const thumbUp = document.createElement("button");
    thumbUp.className = "msg-action-btn";
    thumbUp.title = "Helpful";
    thumbUp.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>`;
    thumbUp.addEventListener("click", () => thumbUp.classList.toggle("active"));

    const thumbDown = document.createElement("button");
    thumbDown.className = "msg-action-btn";
    thumbDown.title = "Not helpful";
    thumbDown.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg>`;
    thumbDown.addEventListener("click", () => thumbDown.classList.toggle("active"));

    actions.appendChild(thumbUp);
    actions.appendChild(thumbDown);
    actions.appendChild(copyBtn);
    row.appendChild(actions);

  } else {
    // User Message Bubble
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatMarkdown(text);
    row.appendChild(bubble);

    // User Avatar Badge on the right
    const avatar = document.createElement("div");
    avatar.className = "message-avatar user-avatar";
    avatar.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
        <circle cx="12" cy="7" r="4"></circle>
      </svg>
    `;
    row.appendChild(avatar);
  }

  messagesContainer.appendChild(row);
  updateChatRecency();
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return row;
}

// Lightbox Expansion Modal for Figures & Images
function openLightbox(url, caption, source, page) {
  let lightbox = document.getElementById("media-lightbox");
  if (!lightbox) {
    lightbox = document.createElement("div");
    lightbox.id = "media-lightbox";
    lightbox.className = "lightbox-backdrop";
    lightbox.innerHTML = `
      <div class="lightbox-modal">
        <button class="lightbox-close" id="lightbox-close-btn" aria-label="Close modal">&times;</button>
        <div class="lightbox-body">
          <img class="lightbox-img" id="lightbox-img" src="" alt="Enlarged figure" />
          <div class="lightbox-footer">
            <div class="lightbox-caption" id="lightbox-caption"></div>
            <div class="lightbox-tag" id="lightbox-tag"></div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(lightbox);

    const close = () => lightbox.classList.remove("active");
    lightbox.querySelector("#lightbox-close-btn").addEventListener("click", close);
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && lightbox.classList.contains("active")) close();
    });
  }

  document.getElementById("lightbox-img").src = url;
  document.getElementById("lightbox-caption").textContent = caption || "";
  document.getElementById("lightbox-tag").textContent = `${source}${page ? ` (Page ${page})` : ""}`;
  lightbox.classList.add("active");
}

function appendTypingIndicator() {
  if (!messagesContainer) return;

  const row = document.createElement("div");
  row.className = "message-row ai typing-row";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar ai-avatar";
  avatar.textContent = "N";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing-bubble";
  bubble.innerHTML = `
    <div class="typing-status-wrap">
      <span class="pulsing-spinner"></span>
      <span class="typing-text">Nexus is reasoning & retrieving sources, please wait...</span>
    </div>
  `;

  row.appendChild(avatar);
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

  escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

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

// --- 15. Data Retention Lifecycle: Background Session Purge ---
window.addEventListener("beforeunload", () => {
  if (chkPermanent && !chkPermanent.checked) {
    const payload = JSON.stringify({ session_id: sessionId });
    const blob = new Blob([payload], { type: "application/json" });
    navigator.sendBeacon("/api/session/clear", blob);
  }
});

// --- 16. Session Restoration on Page Reload ---
function restoreSession() {
  TuningConfig.syncUI();

  const isIndexed = sessionStorage.getItem("wb_indexed") === "true";
  if (isIndexed) {
    const filesCount = sessionStorage.getItem("wb_files_count") || "0";
    const chunksCount = sessionStorage.getItem("wb_chunks_count") || "0";
    const mode = sessionStorage.getItem("wb_mode") || "local";

    if (statFiles) statFiles.textContent = filesCount;
    if (statChunks) statChunks.textContent = chunksCount;
    if (statEngine) statEngine.textContent = mode === "supabase" ? "pgvector" : "Chroma";
    if (statsCard) statsCard.classList.remove("hidden");

    enableChatInterface();
    if (chatStatusSubtitle) chatStatusSubtitle.textContent = `Indexed ${chunksCount} chunks • Ready with Nexus`;

    if (guideContainer) guideContainer.classList.add("hidden");
    if (promptSuggestions) promptSuggestions.classList.remove("hidden");

    // Honor manual user sidebar preference on reload
    if (localStorage.getItem("wb_sidebar_manual") === "open") {
      expandSidebar();
    } else {
      collapseSidebar();
    }
  }

  const savedHistory = sessionStorage.getItem("wb_chat_history");
  if (savedHistory) {
    try {
      chatHistory = JSON.parse(savedHistory);
      if (chatHistory.length > 0) {
        if (guideContainer) guideContainer.classList.add("hidden");
        chatHistory.forEach(msg => {
          appendMessage(
            msg.role === "user" ? "user" : "ai",
            msg.content,
            msg.sources || [],
            msg.media || []
          );
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

// --- 17. Visual Theme Switcher ---
const btnToggleTheme = document.getElementById("btn-toggle-theme");
const themeIconSun = document.getElementById("theme-icon-sun");
const themeIconMoon = document.getElementById("theme-icon-moon");

function updateThemeUI(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("wb_theme", theme);
  if (theme === "light") {
    if (themeIconSun) themeIconSun.classList.remove("hidden");
    if (themeIconMoon) themeIconMoon.classList.add("hidden");
  } else {
    if (themeIconSun) themeIconSun.classList.add("hidden");
    if (themeIconMoon) themeIconMoon.classList.remove("hidden");
  }
}

if (btnToggleTheme) {
  btnToggleTheme.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    updateThemeUI(next);
  });
  const savedTheme = localStorage.getItem("wb_theme") || "dark";
  updateThemeUI(savedTheme);
}
