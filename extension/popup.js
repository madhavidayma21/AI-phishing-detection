const API_BASE = "http://localhost:8000";

let currentUrl = null;

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentUrl = tab.url;

  const urlDisplay = document.getElementById("urlDisplay");
  const content = document.getElementById("content");

  if (!currentUrl || !(currentUrl.startsWith("http://") || currentUrl.startsWith("https://"))) {
    urlDisplay.textContent = "This page cannot be checked";
    content.innerHTML = "";
    return;
  }

  urlDisplay.textContent = currentUrl;

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentUrl }),
    });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const data = await response.json();
    renderResult(data);
  } catch (err) {
    content.innerHTML = `<div class="loading">Could not reach the detection API.<br>Is the FastAPI server running on localhost:8000?</div>`;
    console.error(err);
  }
}

function renderResult(data) {
  const content = document.getElementById("content");
  const isPhishing = data.prediction === "phishing";
  const confidencePct = (data.confidence * 100).toFixed(1);

  const signalRows = Object.entries(data.top_signals)
    .map(([key, value]) => `<li>${formatSignalName(key)}: <strong>${formatSignalValue(key, value)}</strong></li>`)
    .join("");

  content.innerHTML = `
    <div class="verdict-box ${isPhishing ? "phishing" : "legitimate"}">
      <div class="verdict-text">${isPhishing ? "⚠ Likely Phishing" : "✓ Looks Legitimate"}</div>
      <div class="confidence">${confidencePct}% confidence</div>
    </div>
    <div class="signals">
      <strong>Key signals:</strong>
      <ul>${signalRows}</ul>
    </div>
    <button id="explainBtn">Explain why (AI-generated)</button>
    <div id="explanationBox"></div>
  `;

  document.getElementById("explainBtn").addEventListener("click", fetchExplanation);
}

function formatSignalName(key) {
  const names = {
    has_valid_ssl: "Valid SSL",
    domain_age_days: "Domain age (days)",
    whois_available: "WHOIS available",
    num_digits: "Digits in URL",
    has_suspicious_word: "Suspicious keywords",
    domain_entropy: "Domain randomness score",
  };
  return names[key] || key;
}

function formatSignalValue(key, value) {
  if (key === "domain_age_days" && value === -1) return "unknown";
  if (typeof value === "number" && (value === 0 || value === 1) && key !== "domain_entropy" && key !== "domain_age_days" && key !== "num_digits") {
    return value === 1 ? "Yes" : "No";
  }
  return value;
}

async function fetchExplanation() {
  const btn = document.getElementById("explainBtn");
  const box = document.getElementById("explanationBox");
  btn.disabled = true;
  btn.textContent = "Generating explanation...";

  try {
    const response = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentUrl }),
    });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const data = await response.json();
    box.innerHTML = `<div class="explanation">${data.explanation}</div>`;
    btn.style.display = "none";
  } catch (err) {
    box.innerHTML = `<div class="explanation">Could not generate an explanation. Is Ollama running?</div>`;
    btn.disabled = false;
    btn.textContent = "Explain why (AI-generated)";
    console.error(err);
  }
}

init();
