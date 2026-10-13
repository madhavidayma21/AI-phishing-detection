// Background service worker.
// Runs on every tab navigation, calls the /analyze endpoint (fast, no LLM),
// and sets a colored badge on the extension icon so the user gets a
// signal without needing to open the popup. Full explanation is
// fetched on-demand only when the user opens the popup (saves calling
// the slower LLM-backed /explain endpoint on every single page load).

const API_BASE = "http://localhost:8000";

// Simple in-memory cache so re-visiting a tab doesn't always re-hit the API
const resultCache = new Map();

async function analyzeUrl(url) {
  if (resultCache.has(url)) {
    return resultCache.get(url);
  }
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const data = await response.json();
    resultCache.set(url, data);
    return data;
  } catch (err) {
    console.error("Phishing check failed:", err);
    return null;
  }
}

function setBadge(tabId, result) {
  if (!result) {
    chrome.action.setBadgeText({ tabId, text: "?" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#999999" });
    return;
  }
  if (result.prediction === "phishing") {
    chrome.action.setBadgeText({ tabId, text: "!" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#d93025" });
  } else {
    chrome.action.setBadgeText({ tabId, text: "OK" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#188038" });
  }
}

// Only check http(s) URLs — skip chrome://, chrome-extension://, about:blank, etc.
function isCheckableUrl(url) {
  return url && (url.startsWith("http://") || url.startsWith("https://"));
}

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && isCheckableUrl(tab.url)) {
    chrome.action.setBadgeText({ tabId, text: "..." });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#f9ab00" });
    const result = await analyzeUrl(tab.url);
    setBadge(tabId, result);
  }
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  const tab = await chrome.tabs.get(tabId);
  if (isCheckableUrl(tab.url)) {
    const result = await analyzeUrl(tab.url);
    setBadge(tabId, result);
  }
});
