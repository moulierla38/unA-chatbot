const CHAT_API_URL = "http://127.0.0.1:5000/api/chat";
const HEALTH_API_URL = "http://127.0.0.1:5000/api/health";

const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const datasetOnlyToggle = document.querySelector("#dataset-only-toggle");
const backendStatusIndicators = document.querySelectorAll(".backend-status-indicator");
const backendStatusLabels = document.querySelectorAll(".backend-status-label");

function updateBackendStatus(status, label) {
  for (const indicator of backendStatusIndicators) {
    indicator.classList.remove("is-checking", "is-online", "is-offline");
    indicator.classList.add(`is-${status}`);
  }

  for (const statusLabel of backendStatusLabels) {
    statusLabel.textContent = label;
  }
}

async function checkBackendHealth() {
  try {
    const response = await fetch(HEALTH_API_URL, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || data.status !== "success") {
      throw new Error(data.message || `Health check failed (HTTP ${response.status}).`);
    }

    updateBackendStatus("online", "Backend online");
  } catch {
    updateBackendStatus("offline", "Backend unavailable");
  }
}

function scrollToLatestMessage() {
  messages.scrollTop = messages.scrollHeight;
}

function appendInlineFormatting(container, text) {
  const formatting = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|https?:\/\/[^\s<]+)/gi;
  let lastIndex = 0;

  for (const match of text.matchAll(formatting)) {
    container.append(document.createTextNode(text.slice(lastIndex, match.index)));

    const token = match[0];
    if (/^https?:\/\//i.test(token)) {
      const url = token.replace(/[.,!?;:)\]}]+$/, "");
      const link = document.createElement("a");
      link.href = url;
      link.textContent = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      container.append(link, document.createTextNode(token.slice(url.length)));
      lastIndex = match.index + token.length;
      continue;
    }

    const element = token.startsWith("**")
      ? document.createElement("strong")
      : token.startsWith("`")
        ? document.createElement("code")
        : document.createElement("em");
    const wrapperLength = token.startsWith("**") ? 2 : 1;
    element.textContent = token.slice(wrapperLength, -wrapperLength);
    container.append(element);
    lastIndex = match.index + token.length;
  }

  container.append(document.createTextNode(text.slice(lastIndex)));
}

function parseMarkdownTableRow(line) {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) {
    return null;
  }

  const content = trimmed.replace(/^\|/, "").replace(/\|$/, "");
  return content.split("|").map((cell) => cell.trim());
}

function renderAnswer(container, text) {
  let list = null;
  let listType = null;
  const lines = text.split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const headerCells = parseMarkdownTableRow(line);
    const separatorCells = parseMarkdownTableRow(lines[index + 1] || "");
    if (
      headerCells
      && separatorCells
      && separatorCells.length === headerCells.length
      && separatorCells.every((cell) => /^:?-{3,}:?$/.test(cell))
    ) {
      const table = document.createElement("table");
      const tableHead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      for (const cellText of headerCells) {
        const cell = document.createElement("th");
        appendInlineFormatting(cell, cellText);
        headerRow.append(cell);
      }
      tableHead.append(headerRow);
      table.append(tableHead);

      const tableBody = document.createElement("tbody");
      let rowIndex = index + 2;
      while (rowIndex < lines.length) {
        const rowCells = parseMarkdownTableRow(lines[rowIndex]);
        if (!rowCells) {
          break;
        }

        const row = document.createElement("tr");
        for (let cellIndex = 0; cellIndex < headerCells.length; cellIndex += 1) {
          const cell = document.createElement("td");
          appendInlineFormatting(cell, rowCells[cellIndex] || "");
          row.append(cell);
        }
        tableBody.append(row);
        rowIndex += 1;
      }
      table.append(tableBody);
      container.append(table);
      index = rowIndex - 1;
      list = null;
      listType = null;
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    const heading = line.match(/^\s*#{1,4}\s+(.+)$/);
    const listMatch = bullet || numbered;

    if (!line.trim()) {
      list = null;
      listType = null;
      continue;
    }

    if (listMatch) {
      const nextListType = bullet ? "ul" : "ol";
      if (listType !== nextListType) {
        list = document.createElement(nextListType);
        container.append(list);
        listType = nextListType;
      }

      const item = document.createElement("li");
      appendInlineFormatting(item, listMatch[1]);
      list.append(item);
      continue;
    }

    list = null;
    listType = null;
    const block = document.createElement(heading ? "h4" : "p");
    appendInlineFormatting(block, heading ? heading[1] : line);
    container.append(block);
  }
}

checkBackendHealth();

function addMessage(text, sender, { error = false, loading = false } = {}) {
  const article = document.createElement("article");
  article.className = `message ${sender}-message${error ? " error-message" : ""}`;

  if (sender === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "uA";
    article.append(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";

  const senderLabel = document.createElement("span");
  senderLabel.className = "message-sender";
  senderLabel.textContent = sender === "user" ? "You" : "unA";

  const messageText = document.createElement(
    sender === "assistant" && !loading ? "div" : "p"
  );

  if (loading) {
    messageText.textContent = text;
    const dots = document.createElement("span");
    dots.className = "loading-dots";
    dots.setAttribute("aria-hidden", "true");
    for (let index = 0; index < 3; index += 1) {
      dots.append(document.createElement("span"));
    }
    messageText.append(dots);
    messageText.setAttribute("aria-label", text);
  } else if (sender === "assistant") {
    renderAnswer(messageText, text);
  } else {
    messageText.textContent = text;
  }

  content.append(senderLabel, messageText);
  article.append(content);
  messages.append(article);
  scrollToLatestMessage();
  return article;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }
  const datasetOnly = datasetOnlyToggle.checked;

  addMessage(question, "user");
  questionInput.value = "";
  questionInput.disabled = true;
  sendButton.disabled = true;
  sendButton.setAttribute("aria-busy", "true");
  const loadingMessage = addMessage("Thinking through your question", "assistant", {
    loading: true,
  });

  try {
    const response = await fetch(CHAT_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, dataset_only: datasetOnly }),
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The server returned an unreadable response. Please try again.");
    }

    if (!response.ok) {
      throw new Error(data.error || `The request failed (HTTP ${response.status}).`);
    }

    if (typeof data.answer !== "string" || !data.answer.trim()) {
      throw new Error("The server did not return an answer. Please try again.");
    }

    addMessage(data.answer, "assistant");
  } catch (error) {
    const message = error instanceof TypeError
      ? "Could not connect to unA. Check that the Flask backend is running at 127.0.0.1:5000, then try again."
      : error.message;
    addMessage(message, "assistant", { error: true });
  } finally {
    loadingMessage.remove();
    questionInput.disabled = false;
    sendButton.disabled = false;
    sendButton.removeAttribute("aria-busy");
    questionInput.focus();
    scrollToLatestMessage();
  }
});
