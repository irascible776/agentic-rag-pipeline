// gui/static/app.js - Clean Client Controller with Session Management & Dual-Mode UI

// 1. Session ID Management: Generates a persistent UUID per browser tab/session
let sessionId = localStorage.getItem("rag_session_id");
if (!sessionId) {
  sessionId = "session_" + (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 11));
  localStorage.setItem("rag_session_id", sessionId);
}

// Conversation memory
let chatHistory = [];
// In-memory files pending ingestion
let pendingFiles = [];

// DOM Element References
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const fileCount = document.getElementById("file-count");
const chkPermanent = document.getElementById("chk-permanent");

const btnIngest = document.getElementById("btn-ingest");
const progressBox = document.getElementById("progress-box");
const progressMsg = document.getElementById("progress-msg");
const statsCard = document.getElementById("stats-card");
const statFiles = document.getElementById("stat-files");
const statChunks = document.getElementById("stat-chunks");
const statEngine = document.getElementById("stat-engine");

const btnStartChat = document.getElementById("btn-start-chat");
const btnClear = document.getElementById("btn-clear");

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const chatStatusSubtitle = document.getElementById("chat-status-subtitle");

const messagesContainer = document.getElementById("messages-container");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const btnClearChat = document.getElementById("btn-clear-chat");


// --- 2. Check Engine & Server Configuration ---
async function checkSystemConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();

    statusDot.classList.add("ready");

    statusText.textContent = "System Active";
    chatStatusSubtitle.textContent = "Grounded in indexed documents";
    if (statEngine) statEngine.textContent = "Connected";
  } catch (err) {
    statusDot.classList.remove("ready");
    statusText.textContent = "Server Offline";
    chatStatusSubtitle.textContent = "Cannot connect to backend server";
  }
}


// --- 3. File Selection & Drag & Drop ---
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
    addFiles(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length > 0) {
    addFiles(fileInput.files);
  }
});

function addFiles(newFiles) {
  // list of accepted file extensions
  const validExtensions = [".pdf", ".docx", ".txt", ".md"];

  for (let i = 0; i < newFiles.length; i++) {
    const f = newFiles[i];
    const nameLower = f.name.toLowerCase();
    const isValid = validExtensions.some(ext => nameLower.endsWith(ext));

    if (isValid) {
      // avoid duplicates in the list
      if (!pendingFiles.some(existing => existing.name === f.name)) {
        pendingFiles.push(f);
      }
    }
  }
  renderFileList();
}

function renderFileList() {
  fileList.innerHTML = "";
  fileCount.textContent = pendingFiles.length;

  if (pendingFiles.length === 0) {
    fileList.innerHTML = '<li class="empty-hint">No documents uploaded yet.</li>';
    btnIngest.disabled = true;
    return;
  }

  btnIngest.disabled = false;

  pendingFiles.forEach((file) => {
    const li = document.createElement("li");
    li.className = "file-item";
    const ext = file.name.split(".").pop().toUpperCase();
    li.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
      </svg>
      <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${file.name}</span>
      <span style="background: rgba(99, 102, 241, 0.15); color: var(--primary); padding: 1px 5px; border-radius: 4px; font-size: 0.68rem; font-weight: 600;">${ext}</span>
      <span style="color: var(--text-dim); font-size: 0.72rem;">${(file.size / 1024).toFixed(1)} KB</span>
    `;
    fileList.appendChild(li);
  });
}


// --- 4. Step 2: Ingest & Chunk Documents ---
btnIngest.addEventListener("click", async () => {
  if (pendingFiles.length === 0) return;

  const isPermanent = chkPermanent ? chkPermanent.checked : false;

  progressBox.classList.remove("hidden");
  progressMsg.textContent = isPermanent 
    ? "Chunking & storing in Global Knowledge Base..." 
    : "Chunking & indexing for current session...";
  btnIngest.disabled = true;

  const formData = new FormData();
  pendingFiles.forEach(file => {
    formData.append("files", file);
  });
  formData.append("session_id", sessionId);
  formData.append("is_permanent", isPermanent);

  try {
    const res = await fetch("/api/upload-chunk", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    progressBox.classList.add("hidden");

    if (data.status === "success") {
      statFiles.textContent = data.files_count;
      statChunks.textContent = data.chunks_count;
      if (statEngine) {
        statEngine.textContent = data.mode === "supabase" ? "Supabase" : "Chroma";
      }
      statsCard.classList.remove("hidden");

      // Save state to sessionStorage
      sessionStorage.setItem("rag_indexed", "true");
      sessionStorage.setItem("rag_files_count", String(data.files_count));
      sessionStorage.setItem("rag_chunks_count", String(data.chunks_count));
      sessionStorage.setItem("rag_mode", data.mode);

      // Unlock Step 3: Start Chat button
      btnStartChat.disabled = false;
      chatStatusSubtitle.textContent = `Indexed ${data.chunks_count} chunks ready for questions`;
    } else {
      alert("Ingestion error: " + (data.message || "Failed to process files."));
      btnIngest.disabled = false;
    }
  } catch (err) {
    progressBox.classList.add("hidden");
    alert("Network error during ingestion. Please check server.");
    btnIngest.disabled = false;
  }
});


// --- 5. Step 3: Start Chat ---
btnStartChat.addEventListener("click", () => {
  enableChatInterface();
  chatInput.focus();

  // If empty state is still visible, show welcome message
  if (emptyState && !emptyState.classList.contains("hidden")) {
    emptyState.classList.add("hidden");
    appendMessage(
      "ai",
      "**Your documents are loaded and indexed!**\n\nAsk any question and I will synthesize grounded answers with citations."
    );
  }
});

function enableChatInterface() {
  chatInput.disabled = false;
  btnSend.disabled = false;
  if (emptyState) emptyState.classList.add("hidden");
}


// --- 6. Conversational Chat Submissions ---
async function handleSendMessage() {
  const query = chatInput.value.trim();
  if (!query) return;

  chatInput.value = "";
  chatInput.disabled = true;
  btnSend.disabled = true;

  // Append user bubble
  appendMessage("user", query);

  // Show typing animation dots
  const typingIndicator = appendTypingIndicator();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: query,
        history: chatHistory,
        session_id: sessionId
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
      sessionStorage.setItem("rag_chat_history", JSON.stringify(chatHistory));
    } else {
      appendMessage("ai", "I was unable to retrieve an answer. Please try rephrasing your question.");
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

// Enter key sends without reloading
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    e.stopPropagation();
    handleSendMessage();
  }
});

// Form submit event stopped completely
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  e.stopPropagation();
  handleSendMessage();
  return false;
});

// Send button click
btnSend.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  handleSendMessage();
});


// --- 7. Delete Session Uploads ---
btnClear.addEventListener("click", async () => {
  const confirmed = confirm("Are you sure you want to delete your uploaded documents from the vector database?");
  if (!confirmed) return;

  try {
    const res = await fetch("/api/session/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await res.json();

    // Clear sessionStorage
    sessionStorage.removeItem("rag_indexed");
    sessionStorage.removeItem("rag_files_count");
    sessionStorage.removeItem("rag_chunks_count");
    sessionStorage.removeItem("rag_mode");
    sessionStorage.removeItem("rag_chat_history");

    // Reset UI state
    pendingFiles = [];
    renderFileList();
    statsCard.classList.add("hidden");
    btnStartChat.disabled = true;

    // Reset chat
    chatHistory = [];
    messagesContainer.innerHTML = "";
    if (emptyState) {
      emptyState.classList.remove("hidden");
      messagesContainer.appendChild(emptyState);
    }
    chatInput.disabled = true;
    btnSend.disabled = true;

    alert("Session documents cleared successfully!");
  } catch (err) {
    alert("Error clearing session documents.");
  }
});

// Clear Chat History button (keeps vectors intact)
btnClearChat.addEventListener("click", () => {
  chatHistory = [];
  sessionStorage.removeItem("rag_chat_history");
  messagesContainer.innerHTML = "";
  if (emptyState) {
    emptyState.classList.remove("hidden");
    messagesContainer.appendChild(emptyState);
  }
});


// --- 8. UI Rendering Helpers & Rich Markdown Formatting ---
function appendMessage(role, text, sources = []) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user-avatar" : "ai-avatar"}`;
  avatar.textContent = role === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatMarkdown(text);

  // Add cyan source citations badges
  if (sources && sources.length > 0) {
    const sourcesBox = document.createElement("div");
    sourcesBox.className = "sources-box";
    sourcesBox.innerHTML = '<span class="sources-label">Sources:</span>';
    sources.forEach(src => {
      const pill = document.createElement("span");
      pill.className = "source-pill";
      pill.textContent = src;
      sourcesBox.appendChild(pill);
    });
    bubble.appendChild(sourcesBox);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);

  messagesContainer.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  return row;
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row ai";

  const avatar = document.createElement("div");
  avatar.className = "avatar ai-avatar";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `
    <div class="typing-dots">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>
  `;

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesContainer.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  return row;
}

// Simple, robust markdown formatter (bold, lists, code, paragraphs)
function formatMarkdown(text) {
  if (!text) return "";

  // 1. Escape HTML
  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // 2. Inline code: `code`
  escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

  // 3. Bold text: **text**
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // 4. Split into paragraphs and bullet lists
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

  if (inList) {
    html += "</ul>";
  }

  return html || "<p>" + escaped + "</p>";
}

// Restore UI and chat history if user refreshes browser
function restoreSession() {
  const isIndexed = sessionStorage.getItem("rag_indexed") === "true";
  if (isIndexed) {
    const filesCount = sessionStorage.getItem("rag_files_count") || "0";
    const chunksCount = sessionStorage.getItem("rag_chunks_count") || "0";
    const mode = sessionStorage.getItem("rag_mode") || "local";

    statFiles.textContent = filesCount;
    statChunks.textContent = chunksCount;
    if (statEngine) statEngine.textContent = mode === "supabase" ? "Supabase" : "Chroma";
    statsCard.classList.remove("hidden");

    btnStartChat.disabled = false;
    chatStatusSubtitle.textContent = `Indexed ${chunksCount} chunks ready for questions`;
  }

  const savedHistory = sessionStorage.getItem("rag_chat_history");
  if (savedHistory) {
    try {
      chatHistory = JSON.parse(savedHistory);
      if (chatHistory.length > 0) {
        enableChatInterface();
        if (emptyState) emptyState.classList.add("hidden");
        chatHistory.forEach(msg => {
          appendMessage(msg.role === "user" ? "user" : "ai", msg.content);
        });
      }
    } catch (e) {
      chatHistory = [];
    }
  }
}

// Initial status check & session restoration on boot
checkSystemConfig();
restoreSession();
