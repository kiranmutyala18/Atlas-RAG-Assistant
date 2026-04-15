const chatForm = document.getElementById("chat-form");
const uploadForm = document.getElementById("upload-form");
const chatLog = document.getElementById("chat-log");
const messageInput = document.getElementById("message");
const sendButton = document.getElementById("send-button");
const uploadButton = document.getElementById("upload-button");
const fileInput = document.getElementById("file-input");
const newSessionButton = document.getElementById("new-session-button");

const messageTemplate = document.getElementById("message-template");
const sessionTemplate = document.getElementById("session-template");
const documentTemplate = document.getElementById("document-template");

const serverStatus = document.getElementById("server-status");
const documentCount = document.getElementById("document-count");
const chunkCount = document.getElementById("chunk-count");
const generationMode = document.getElementById("generation-mode");
const sessionList = document.getElementById("session-list");
const documentList = document.getElementById("document-list");
const uploadStatus = document.getElementById("upload-status");
const activeSessionTitle = document.getElementById("active-session-title");

let activeSessionId = null;
let sessionCache = [];

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

async function loadHealth() {
  try {
    const data = await requestJson("/api/health");
    serverStatus.textContent = data.status;
    documentCount.textContent = data.documents;
    chunkCount.textContent = data.chunks;
    generationMode.textContent = data.generation_mode;
  } catch (error) {
    serverStatus.textContent = "offline";
  }
}

function formatDate(timestamp) {
  if (!timestamp) {
    return "just now";
  }
  return new Date(timestamp * 1000).toLocaleString();
}

function clearMessages() {
  chatLog.innerHTML = "";
}

function renderMessages(messages, latestSources = []) {
  clearMessages();

  if (!messages.length) {
    addMessage(
      "bot",
      "This chat is empty. Ask a question after uploading or selecting documents."
    );
    return;
  }

  messages.forEach((message, index) => {
    const isLastAssistant =
      index === messages.length - 1 && message.role === "assistant";
    addMessage(
      message.role === "assistant" ? "bot" : "user",
      message.content,
      isLastAssistant ? latestSources : []
    );
  });
}

function addMessage(role, text, sources = []) {
  const fragment = messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const roleNode = fragment.querySelector(".message-role");
  const bodyNode = fragment.querySelector(".message-body");
  const sourcesNode = fragment.querySelector(".sources");

  article.classList.add(role);
  roleNode.textContent = role === "user" ? "You" : "Assistant";
  bodyNode.textContent = text;

  sources.forEach((source) => {
    const pill = document.createElement("span");
    pill.className = "source-pill";
    pill.textContent = `${source.source} | score ${source.score}`;
    sourcesNode.appendChild(pill);
  });

  chatLog.appendChild(fragment);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderSessions(sessions) {
  sessionCache = sessions;
  sessionList.innerHTML = "";

  sessions.forEach((session) => {
    const fragment = sessionTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".session-item");
    button.textContent = session.title;
    button.title = `Updated ${formatDate(session.updated_at)}`;
    if (session.id === activeSessionId) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => loadSession(session.id));
    sessionList.appendChild(fragment);
  });
}

function renderDocuments(documents) {
  documentList.innerHTML = "";

  documents.forEach((document) => {
    const fragment = documentTemplate.content.cloneNode(true);
    fragment.querySelector(".document-name").textContent = document.name;
    fragment.querySelector(".document-meta").textContent =
      `${document.source_type} | added ${formatDate(document.created_at)}`;
    documentList.appendChild(fragment);
  });
}

async function loadDocuments() {
  const data = await requestJson("/api/documents");
  renderDocuments(data.documents);
}

async function loadSessions() {
  const data = await requestJson("/api/sessions");
  renderSessions(data.sessions);

  if (!activeSessionId && data.sessions.length) {
    await loadSession(data.sessions[0].id);
  }
}

async function loadSession(sessionId) {
  activeSessionId = sessionId;
  const session = sessionCache.find((item) => item.id === sessionId);
  activeSessionTitle.textContent = session ? session.title : `Chat ${sessionId}`;
  renderSessions(sessionCache);
  const data = await requestJson(`/api/sessions/${sessionId}/messages`);
  renderMessages(data.messages);
}

async function createSession() {
  const title = `Chat ${sessionCache.length + 1}`;
  const session = await requestJson("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  activeSessionId = session.id;
  await loadSessions();
  await loadSession(session.id);
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message || !activeSessionId) {
    return;
  }

  sendButton.disabled = true;
  sendButton.textContent = "Thinking...";

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ session_id: activeSessionId, message }),
    });
    messageInput.value = "";
    renderMessages(data.messages ?? [], data.sources ?? []);
    await loadSessions();
    await loadHealth();
  } catch (error) {
    addMessage(
      "bot",
      "The request failed. Please try again after the server is ready."
    );
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    return;
  }

  uploadButton.disabled = true;
  uploadButton.textContent = "Uploading...";
  uploadStatus.textContent = "Ingesting document into the knowledge base...";

  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    await requestJson("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    fileInput.value = "";
    uploadStatus.textContent = "Document uploaded and indexed successfully.";
    await loadDocuments();
    await loadHealth();
  } catch (error) {
    uploadStatus.textContent = "Upload failed. Please use a UTF-8 .txt or .md file.";
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Upload";
  }
});

newSessionButton.addEventListener("click", async () => {
  await createSession();
});

async function initializeApp() {
  await loadHealth();
  await loadDocuments();
  await loadSessions();
  if (!activeSessionId) {
    await createSession();
  }
}

initializeApp();
