# AI-Powered Phishing & URL Detection System

Final-year capstone project — an ML classifier with an LLM-based explanation layer, delivered as a REST API, Chrome extension, and web dashboard.

**Author:** Madhavi
**Program:** B.Tech, AI & Data Science (Cybersecurity minor) — MGM's Jawaharlal Nehru College of Engineering

---

## 1. Problem Statement

Phishing remains one of the most common initial-access vectors in real-world attacks. Most consumer-facing phishing detectors are either pure blacklist lookups (fast, but blind to brand-new attacks not yet indexed) or black-box ML classifiers that give a verdict with no explanation, leaving users unable to judge whether to trust the tool.

This project addresses both gaps:
- A **structural/behavioral ML classifier** that judges a URL on its own merits (SSL validity, domain age, entropy, etc.) rather than relying on a pre-compiled blacklist — meaning it can catch phishing sites within minutes of going live, before any blacklist has indexed them.
- An **LLM explanation layer** that translates the model's raw feature values into a plain-English justification, so the verdict is auditable rather than opaque.

---

## 2. System Architecture

```
                         ┌─────────────────────┐
                         │   Chrome Extension    │
                         │  (badge + popup UI)   │
                         └──────────┬───────────┘
                                    │
                         ┌─────────────────────┐
                         │   Web Dashboard        │
                         │  (URL Trust Console)   │
                         └──────────┬───────────┘
                                    │  HTTP (CORS-enabled)
                                    ▼
                    ┌───────────────────────────────┐
                    │        FastAPI Backend          │
                    │  ┌─────────────────────────┐   │
                    │  │  Feature Extraction Engine │   │
                    │  │  • URL structure (no net)  │   │
                    │  │  • WHOIS domain age        │   │
                    │  │  • SSL/TLS certificate     │   │
                    │  │  • Page content analysis   │   │
                    │  └───────────┬─────────────┘   │
                    │              ▼                   │
                    │  ┌─────────────────────────┐   │
                    │  │  XGBoost Classifier        │   │
                    │  │  (27 engineered features)  │   │
                    │  └───────────┬─────────────┘   │
                    │              ▼                   │
                    │  ┌─────────────────────────┐   │
                    │  │  LangGraph Explanation Agent│   │
                    │  │  1. select_signals (rules)  │   │
                    │  │  2. generate_explanation    │   │
                    │  │     (local LLM via Ollama)  │   │
                    │  └─────────────────────────┘   │
                    └───────────────────────────────┘
```

**Key design principle:** the ML classifier makes the security decision; the LLM only explains a decision that's already been made, and is only ever shown pre-verified signal values, never the raw feature vector. This separation means a bad LLM explanation can never flip the actual verdict, and the LLM cannot hallucinate facts about the URL that weren't independently confirmed.

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Feature extraction | Python, `python-whois`, `tldextract`, `requests`, `ssl` |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Backend API | FastAPI, Uvicorn |
| Explanation agent | LangGraph, Ollama (llama3.1, local) |
| Browser extension | Chrome Manifest V3, vanilla JS |
| Web dashboard | Vanilla HTML/CSS/JS (no framework/build step) |
| Datasets | PhiUSIIL (Kaggle), OpenPhish live feed |

---

## 4. Dataset & Methodology

### 4.1 Data Sourcing

Originally planned to use PhishTank, but **PhishTank has had new-registration closed since 2020** (confirmed still closed as of mid-2026, pending a Cisco Talos rework) — this is documented here because it's a real-world constraint any practitioner would hit, not a shortcut taken for convenience.

Final dataset combines:
- **PhiUSIIL** (Kaggle, 235,795 pre-labeled URLs) — sampled 5,000 phishing + 5,000 legitimate
- **OpenPhish** live feed — 300 freshly-reported phishing URLs, added for recency

Total: **10,298 URLs** (5,298 phishing / 5,000 legitimate) after deduplication.

### 4.2 Feature Engineering

27 features across four categories, each extracted independently with graceful failure handling (a WHOIS timeout never crashes the pipeline — it just yields an "unknown" signal, which the model can itself learn from):

- **URL structure** (no network call): length, entropy, subdomain count, suspicious keywords, IP-address usage, digit count
- **Domain metadata**: WHOIS-derived registration age, expiry
- **SSL/TLS**: certificate validity, issuer, age
- **Content** (optional pass): login form detection, brand-impersonation heuristics

### 4.3 Dataset Leakage — Discovery and Fix

This is the most important methodological finding in the project and is documented in detail because it materially changed the reported results.

**Initial baseline result:** 99.6% F1, which is suspiciously high for this task.

**Investigation:** feature importance analysis showed `path_length` and `has_https` alone accounted for ~95% of the XGBoost model's decision-making. A distribution check revealed the cause: **every single legitimate URL in the PhiUSIIL dataset was a bare homepage root** (`path_length` = 0, `std` = 0 for the entire legitimate class; `has_https` = 1 for 100% of the legitimate class). The model had learned "anything with a path is phishing" — a fact about how the dataset was collected, not a fact about phishing behavior. This would have caused the model to misclassify something as ordinary as `https://en.wikipedia.org/wiki/Phishing` as phishing.

**Fix:** built a systematic leakage diagnostic (`diagnose_leakage()` in `train_baseline.py`) that flags any feature with near-zero variance in one class combined with a large between-class mean gap. It surfaced **7 leaking features** in total: `path_length`, `num_underscores`, `num_slashes`, `num_at_symbols`, `num_question_marks`, `num_equal_signs`, `has_https`. All seven were excluded from the final model.

**Result after fix:** F1 dropped to a more defensible **97.2%** initially, then **98.2%** after hyperparameter tuning — with the top features now being genuinely explainable ones (SSL validity, digit count, subdomain count, suspicious keywords, domain age).

**Known remaining limitation:** the fix removes the leaking features rather than correcting the underlying data quality issue. PhiUSIIL's legitimate URLs are still homepage-only; a more thorough fix would enrich the legitimate set with real subpages (e.g., crawling a few links deep into Tranco top domains). Flagged here as future work rather than solved, in the interest of accurate reporting.

---

## 5. Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8971 | 0.9030 | 0.8962 | 0.8996 | 0.9548 |
| Random Forest | 0.9291 | 0.9369 | 0.9245 | 0.9307 | 0.9769 |
| **XGBoost (tuned)** | **0.9816** | **0.9830** | **0.9811** | **0.9821** | **0.9983** |

**5-fold cross-validation (XGBoost):** F1 = 0.9807 ± 0.0031 — tight spread across folds, confirming the result is not an artifact of a lucky train/test split.

**Hyperparameter tuning:** RandomizedSearchCV, 25 candidates × 3-fold CV. Best parameters: `n_estimators=300, max_depth=5, learning_rate=0.2, subsample=0.9, colsample_bytree=1.0, min_child_weight=1`.

**Top features (post-leakage-fix):** `has_valid_ssl`, `num_digits`, `hyphen_in_domain`, `num_subdomains`, `url_length`, `domain_age_days` — all independently defensible as real phishing indicators.

---

## 6. Benchmark: Comparison Against Google Safe Browsing

Ran 40 held-out URLs (30 fresh OpenPhish phishing URLs never used in training + 10 well-known legitimate sites) through both this model and the Google Safe Browsing v4 API.

| | Our Model | Google Safe Browsing |
|---|---|---|
| Accuracy | 97.4% | 50.0% |
| Recall (phishing only) | 96.4% | 32.1% |
| Phishing caught that the other missed | 18 | 0 |

**Interpretation (important — do not overstate this):** this is not a claim that the model is generally "better" than Google's system. Google Safe Browsing is blacklist-based — it flags URLs it has previously indexed, and there is an inherent propagation delay between a phishing site going live and it being crawled/reported/blacklisted. The test set was deliberately built from OpenPhish's *live* feed (URLs reported hours before testing), which structurally favors a real-time structural classifier over a blacklist. The honest, defensible claim: **this model detects newly-emerging phishing URLs that blacklist-based systems haven't yet indexed**, because it evaluates the URL's own structural and behavioral properties rather than checking it against a list of previously-seen threats. The two approaches are complementary, not competing — a production system would likely use both.

---

## 7. API Reference

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Classify a URL. Fast (~1-13s, dominated by WHOIS/SSL network lookups), no LLM call. |
| `/explain` | POST | Classify + generate a plain-English explanation via the LangGraph agent. Slower (adds an LLM call). |
| `/robustness-test` | POST | Score up to 30 controlled synthetic URL variations offline. |
| `/health` | GET | Confirms the model loaded and reports which model is in use. |
| `/docs` | GET | Interactive Swagger UI. |

Both `/analyze` and `/explain` accept `{"url": "..."}` and cache results in memory per URL for the life of the server process (repeat lookups skip the slow network calls).

### Adversarial Phishing Lab

The Adversarial Phishing Lab evaluates the consistency of the detector under controlled synthetic URL perturbations. It reuses the existing XGBoost model and URL feature pipeline, with URL-only features and fixed unknown values for network-derived signals. It does not contact the submitted or generated URLs.

Mutation categories include character substitution, hyphen manipulation, synthetic subdomains, path changes, query changes, and insertion of common URL tokens. The deterministic generator removes duplicates and labels each generated string as a synthetic test case. `POST /robustness-test` accepts `{"url":"https://example.com/login","num_mutations":20,"mutation_types":["character_substitution","token_manipulation"]}`; the count must be between 1 and 30. Omitting `mutation_types` selects all categories.

The response reports original and mutated model classes and confidence, probability-band labels (legitimate below 35%, suspicious from 35% to below 70%, phishing at 70% or above), prediction-change counts, consistency overall and by mutation category, and category totals. Probability bands are descriptive UI labels, not a second classifier. When a ground-truth `known_label` is explicitly supplied, detected/missed counts compare the model's binary class to that label; these metrics are meaningful only when the label is correct and applies to each mutation.

This testing is useful for observing sensitivity to small string changes and identifying cases for further evaluation. It does not establish actual detection performance for unlabeled synthetic URLs, semantic equivalence, or resistance to adversarial attacks. Generated URLs are never visited, crawled, resolved, registered, deployed, or sent to reputation services. The lab is an offline string-mutation experiment only.

---

## 8. Chrome Extension

Manifest V3 extension that runs a lightweight classification (`/analyze`) automatically on every page visited, shown as a colored badge (green = legitimate, red = phishing, gray = checking). Opening the popup shows the full signal breakdown and offers an on-demand "Explain why" button that calls `/explain`.

Deliberately does *not* call the slower `/explain` endpoint automatically on every page load — only on user request — to avoid taxing the local LLM unnecessarily.

## 9. Web Dashboard

Standalone HTML/CSS/JS page ("URL Trust Console") for manual URL lookups outside the browser extension context — useful for demos and for checking URLs shared via email/chat rather than visited directly. Displays an animated confidence dial, the same signal breakdown as the extension, and the AI explanation, styled as a security console rather than a generic web form.

---

## 10. Setup Instructions

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000
```

### Chrome Extension
1. `chrome://extensions` → enable Developer Mode → Load Unpacked → select `extension/` folder

### Web Dashboard
```bash
cd frontend
python -m http.server 5500
```
Visit `http://localhost:5500`

### Requirements
- Python 3.12 (not 3.14 — ML packages don't yet ship pre-built wheels for it)
- Ollama running locally with `llama3.1` pulled, for the `/explain` endpoint

---

## 11. Limitations

- **WHOIS coverage is incomplete.** Certain TLDs (notably `.de`, `.eu`) restrict WHOIS output for privacy reasons; the pipeline treats this as "unknown" rather than failing, but domain-age-based features have reduced signal for these domains.
- **Legitimate-URL training data is homepage-only** (see Section 4.3) — the model has not been trained on real subpages of legitimate sites, which is a residual risk even after removing the directly-leaking features.
- **Content-based features** (login form detection, brand impersonation) are implemented but were not included in the final model's feature set for this dataset pass, since bulk content-fetching against thousands of URLs (many already dead) was prohibitively slow; this is a natural next step for improving recall further.
- **Benchmark comparison** (Section 6) used a test set favorable to real-time classifiers over blacklists by construction; results should not be read as a general claim of superiority over Google Safe Browsing.

## 12. Future Work

- Enrich legitimate-URL training data with real subpages, not just homepages
- Enable content-based features in the production pipeline with async/batched fetching
- Feedback loop: let users flag misclassifications, periodically retrain
- Deploy the backend publicly (currently local-only for the project demo)


## 13. To run project 
1. Start Ollama (needed only if you want AI explanations via /explain):

powershell
ollama serve

(Often already running in the background on Windows — check with ollama list first; if it responds, it's running.)

2. Start the FastAPI backend — open a terminal:

powershell
cd "C:\Users\omprakash dayma\Downloads\phishing-detector\phishing-detector\backend"
venv\Scripts\activate
uvicorn app.api.main:app --reload --port 8000

3. Choose your frontend (you have two — use either or both):

Option A — Web Dashboard (open a second terminal):

powershell
cd "C:\Users\omprakash dayma\Downloads\phishing-detector\phishing-detector\frontend"
python -m http.server 5500

Then visit http://localhost:5500 in your browser.

visit : C:/Users/omprakash dayma/Downloads/phishing-detector/phishing-detector/frontend/index.html

file:///C:/Users/omprakash%20dayma/Downloads/phishing-detector/phishing-detector/frontend/index.html
