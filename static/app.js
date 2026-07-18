const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const chatArea = document.getElementById("chatArea");

// Auto-resize textarea as user types.
input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = input.scrollHeight + "px";
});

// Enter sends, Shift+Enter adds new line.
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

function sendQuestion() {
    const question = input.value.trim();
    if (!question) return;

    // Remove welcome screen on first message.
    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    // Append user bubble.
    appendMessage("user", question);

    // Clear and reset input.
    input.value = "";
    input.style.height = "auto";
    sendBtn.disabled = true;

    // Show typing indicator.
    const typingId = appendTyping();

    // Send to Flask backend.
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
    })
    .then((res) => res.json().then((data) => ({ status: res.status, data })))
    .then(({ status, data }) => {
        removeTyping(typingId);
        appendMessage("assistant", data.answer || "No answer returned.");
        if (status === 401) requestApiKey();
        sendBtn.disabled = false;
    })
    .catch(() => {
        removeTyping(typingId);
        appendMessage("assistant", "Something went wrong. Please try again.");
        sendBtn.disabled = false;
    });
}

function appendMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const label = document.createElement("p");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "Assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "assistant") {
        const rawHtml = marked.parse(text);
        bubble.innerHTML = DOMPurify.sanitize(rawHtml);

        // Long structured answers get a copy button
        if (text.length > 200) {
            const copyBtn = document.createElement("button");
            copyBtn.className = "copy-btn";
            copyBtn.textContent = "Copy answer";
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(text);
                copyBtn.textContent = "Copied!";
                setTimeout(() => (copyBtn.textContent = "Copy answer"), 1500);
            };
            bubble.appendChild(copyBtn);
        }
    } else {
        bubble.textContent = text;
    }

    msg.appendChild(label);
    msg.appendChild(bubble);
    chatArea.appendChild(msg);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function appendTyping() {
    const id = "typing-" + Date.now();
    const msg = document.createElement("div");
    msg.className = "message assistant typing";
    msg.id = id;

    const label = document.createElement("p");
    label.className = "message-label";
    label.textContent = "Assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;

    msg.appendChild(label);
    msg.appendChild(bubble);
    chatArea.appendChild(msg);
    chatArea.scrollTop = chatArea.scrollHeight;
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function requestApiKey() {
    const key = prompt("Paste your Groq (gsk_...) or Gemini (AIza...) API key:");
    if (!key) return;
    fetch("/set-api-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key.trim() }),
    })
    .then((res) => res.json())
    .then((data) => { if (!data.valid) { alert(data.message); requestApiKey(); } });
}

fetch("/has-api-key").then((res) => res.json()).then((data) => { if (!data.has_key) requestApiKey(); });