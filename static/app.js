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
        appendMessage("assistant", data.answer || "No answer returned.", data.sources || []);
        if (status === 401) requestApiKey();
        sendBtn.disabled = false;
    })
    .catch(() => {
        removeTyping(typingId);
        appendMessage("assistant", "Something went wrong. Please try again.");
        sendBtn.disabled = false;
    });
}

function appendMessage(role, text, sources = []) {
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

        if (sources && sources.length > 0) {
            const sourcesEl = document.createElement("div");
            sourcesEl.className = "sources";
            const seen = new Set();
            const items = sources
                .filter((s) => {
                    const key = `${s.filename}|${s.page}`;
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                })
                .map((s) => `${s.filename} (p.${s.page})`)
                .join(" · ");
            sourcesEl.textContent = `Source: ${items}`;
            bubble.appendChild(sourcesEl);
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

const keyModalOverlay = document.getElementById("keyModalOverlay");
const apiKeyInput = document.getElementById("apiKeyInput");
const keyError = document.getElementById("keyError");
const saveKeyBtn = document.getElementById("saveKeyBtn");
const apiKeyNavItem = document.getElementById("apiKeyNavItem");

function requestApiKey() {
    keyError.textContent = "";
    apiKeyInput.value = "";
    keyModalOverlay.classList.add("visible");
    apiKeyInput.focus();
}

function closeKeyModal() {
    keyModalOverlay.classList.remove("visible");
}

function submitApiKey() {
    const key = apiKeyInput.value.trim();
    if (!key) return;
    saveKeyBtn.disabled = true;
    saveKeyBtn.textContent = "Checking...";
    fetch("/set-api-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key }),
    })
    .then((res) => res.json())
    .then((data) => {
        saveKeyBtn.disabled = false;
        saveKeyBtn.textContent = "Save & Continue";
        if (data.valid) {
            closeKeyModal();
        } else {
            keyError.textContent = data.message;
        }
    })
    .catch(() => {
        saveKeyBtn.disabled = false;
        saveKeyBtn.textContent = "Save & Continue";
        keyError.textContent = "Something went wrong. Try again.";
    });
}

saveKeyBtn.addEventListener("click", submitApiKey);
apiKeyInput.addEventListener("keydown", (e) => { if (e.key === "Enter") submitApiKey(); });
apiKeyNavItem.addEventListener("click", (e) => { e.preventDefault(); requestApiKey(); });

fetch("/has-api-key").then((res) => res.json()).then((data) => { if (!data.has_key) requestApiKey(); });