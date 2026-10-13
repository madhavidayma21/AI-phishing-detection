"""
Week 7-8 + Month 3: Real-time inference API + LLM explanation layer.

Run with (from backend/):
    uvicorn app.api.main:app --reload --port 8000

Endpoints:
    POST /analyze  — ML classification only (fast, no LLM)
    POST /explain  — classification + plain-English explanation (slower, calls Ollama)
    GET  /health   — status check
    GET  /docs     — interactive Swagger UI
"""
import time
import re
from email.utils import parseaddr
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.features.extractor import extract_all_features
from app.features.url_features import extract_url_features
from app.explain.explainer import explain_result
from app.robustness.mutations import MUTATION_TYPES, generate_mutations
from email_features import extract_email_features, generate_email_reasons
from text_features import extract_text_features, generate_text_reasons

MODEL_DIR = "data/processed"  # relative to backend/, since that's the working directory when running `uvicorn app.api.main:app` from backend/

app = FastAPI(title="Phishing Detector API", version="0.3.0")

# Chrome extensions run as chrome-extension://<id> origins — without CORS enabled,
# the browser blocks the extension's fetch() calls to this API entirely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local dev project; restrict to your extension's ID for production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
feature_columns = None
imputation_medians = None
dropped_features = None
best_model_name = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_columns, imputation_medians, dropped_features, best_model_name
    model = joblib.load(f"{MODEL_DIR}/best_model.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.joblib")
    imputation_medians = joblib.load(f"{MODEL_DIR}/imputation_medians.joblib")
    dropped_features = joblib.load(f"{MODEL_DIR}/dropped_features.joblib")
    best_model_name = joblib.load(f"{MODEL_DIR}/best_model_name.joblib")
    print(f"Loaded model: {best_model_name}")


class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="The URL to classify", examples=["https://www.google.com"])


class AnalyzeResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    phishing_probability: float
    top_signals: dict
    processing_time_ms: float


class ExplainResponse(AnalyzeResponse):
    explanation: str


class RobustnessRequest(BaseModel):
    url: str = Field(..., max_length=2000, description="URL to mutate and test")
    num_mutations: int = Field(default=20, ge=1, le=30, description="Maximum synthetic variants to score")
    mutation_types: list[Literal[
        "character_substitution", "subdomain_manipulation", "hyphen_manipulation",
        "path_manipulation", "query_manipulation", "token_manipulation",
    ]] = Field(default_factory=lambda: list(MUTATION_TYPES))
    known_label: Literal["phishing", "legitimate"] | None = None


class EmailRequest(BaseModel):
    email: str = Field(..., max_length=320, description="Sender email address")


class TextRequest(BaseModel):
    text: str = Field(..., max_length=10000, description="Message text to inspect")


def heuristic_response(risk: float, signals: dict, reasons: list, elapsed: float, subject: str):
    risk = max(0.0, min(1.0, risk))
    prediction = "phishing" if risk >= 0.7 else "suspicious" if risk >= 0.35 else "legitimate"
    confidence = risk if prediction == "phishing" else (1 - risk if prediction == "legitimate" else 0.5 + abs(risk - 0.5))
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "phishing_probability": round(risk, 4),
        "top_signals": signals,
        "reasons": reasons,
        "processing_time_ms": round(elapsed, 1),
        "method": "heuristic",
        "notice": "Heuristic risk estimate; it does not verify sender identity or guarantee a message is safe.",
        "subject": subject,
    }


@app.post("/analyze-email")
def analyze_email(request: EmailRequest):
    value = request.email.strip()
    _, address = parseaddr(value)
    if not address or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", address):
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    started = time.time()
    features = extract_email_features(address, "", "", timeout=1)
    domain = address.rsplit("@", 1)[-1].lower()
    local = address.split("@", 1)[0].lower()
    risk = 0.08
    if features["sender_domain_is_free_provider"]:
        risk += 0.18
    if features["sender_num_digits"] >= 4:
        risk += 0.18
    if len(local) > 24 or re.search(r"[^a-z0-9._+-]", local):
        risk += 0.15
    if any(word in domain for word in ("secure", "verify", "account", "login", "support", "update")):
        risk += 0.35
    reasons = []
    if features["sender_domain_is_free_provider"]:
        reasons.append("Address uses a public email provider; this is common and is not suspicious by itself")
    if features["sender_num_digits"] >= 4:
        reasons.append("Address contains several digits")
    if any(word in domain for word in ("secure", "verify", "account", "login", "support", "update")):
        reasons.append("Domain contains a word often used in impersonation addresses")
    if not reasons:
        reasons.append("No obvious address-pattern red flags found; this cannot verify who controls the address")
    return heuristic_response(risk, {"domain": domain, "public_provider": bool(features["sender_domain_is_free_provider"]), "digits_in_address": features["sender_num_digits"]}, reasons, (time.time()-started)*1000, address)


@app.post("/analyze-text")
def analyze_text(request: TextRequest):
    value = request.text.strip()
    if not value:
        raise HTTPException(status_code=400, detail="Message text cannot be empty")
    started = time.time()
    features = extract_text_features(value, timeout=1)
    weighted = (features["urgency_score"] * .12 + features["threat_score"] * .15 + features["credential_request_count"] * .2 + features["financial_keyword_count"] * .12 + features["reward_keyword_count"] * .12 + features["num_shortened_urls"] * .2 + features["any_url_is_ip_based"] * .3 + features["any_url_suspicious_word"] * .15 + (0.1 if features["has_url"] else 0))
    risk = min(.98, .05 + weighted)
    return heuristic_response(risk, {"urgency_terms": features["urgency_score"], "credential_requests": features["credential_request_count"], "financial_terms": features["financial_keyword_count"], "reward_terms": features["reward_keyword_count"], "urls_found": features["num_urls"], "shortened_urls": features["num_shortened_urls"]}, generate_text_reasons(features), (time.time()-started)*1000, value[:160])


def build_feature_vector(url: str) -> pd.DataFrame:
    raw_features = extract_all_features(url, include_content=False)
    row = dict(raw_features)

    for col, median_val in imputation_medians.items():
        if col in row:
            row[f"{col}_missing"] = 1 if row[col] == -1 else 0
            if row[col] == -1:
                row[col] = median_val

    for col in dropped_features:
        row.pop(col, None)

    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    return df, raw_features


def build_offline_feature_vector(url: str) -> tuple[pd.DataFrame, dict]:
    """Build a model-compatible vector without resolving, fetching, or querying a URL."""
    raw_features = extract_url_features(url)
    raw_features.update({
        "domain_age_days": -1, "whois_available": 0, "expiry_days_remaining": -1,
        "domain_age_under_30_days": 0, "domain_age_under_180_days": 0,
        "has_valid_ssl": 0, "ssl_cert_age_days": -1, "ssl_days_until_expiry": -1,
        "ssl_issuer_is_known_ca": 0,
    })
    row = dict(raw_features)
    for col, median_val in imputation_medians.items():
        if col in row:
            row[f"{col}_missing"] = 1 if row[col] == -1 else 0
            if row[col] == -1:
                row[col] = median_val
    for col in dropped_features:
        row.pop(col, None)
    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns], raw_features


def predict_vector(X: pd.DataFrame, raw_features: dict, start: float) -> dict:
    X_input = scaler.transform(X) if best_model_name == "Logistic Regression" else X
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    phishing_proba = float(proba[1])
    confidence = float(proba[pred])
    return {
        "prediction": "phishing" if pred == 1 else "legitimate",
        "confidence": round(confidence, 4),
        "phishing_probability": round(phishing_proba, 4),
        "top_signals": {
            "has_valid_ssl": raw_features.get("has_valid_ssl"), "domain_age_days": raw_features.get("domain_age_days"),
            "whois_available": raw_features.get("whois_available"), "num_digits": raw_features.get("num_digits"),
            "has_suspicious_word": raw_features.get("has_suspicious_word"), "domain_entropy": round(raw_features.get("domain_entropy", 0), 3),
        },
        "processing_time_ms": round((time.time() - start) * 1000, 1),
        "_raw_features": raw_features,
    }


@lru_cache(maxsize=2048)
def cached_predict(url: str):
    start = time.time()
    X, raw_features = build_feature_vector(url)
    return predict_vector(X, raw_features, start)


def offline_predict(url: str) -> dict:
    start = time.time()
    X, raw_features = build_offline_feature_vector(url)
    return predict_vector(X, raw_features, start)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.url or len(request.url.strip()) == 0:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    try:
        result = cached_predict(request.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    response_fields = {k: v for k, v in result.items() if not k.startswith("_")}
    return AnalyzeResponse(url=request.url, **response_fields)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: AnalyzeRequest):
    if not request.url or len(request.url.strip()) == 0:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    try:
        result = cached_predict(request.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    try:
        explanation_text = explain_result(
            url=request.url,
            prediction=result["prediction"],
            confidence=result["confidence"],
            raw_features=result["_raw_features"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Explanation generation failed (is Ollama running? try 'ollama serve'): {str(e)}"
        )

    response_fields = {k: v for k, v in result.items() if not k.startswith("_")}
    return ExplainResponse(url=request.url, explanation=explanation_text, **response_fields)


@app.post("/robustness-test")
def robustness_test(request: RobustnessRequest):
    """Score synthetic URL changes locally; generated URLs are never contacted."""
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    if not request.mutation_types:
        raise HTTPException(status_code=400, detail="Select at least one mutation type")
    try:
        original = offline_predict(request.url)
        mutations = generate_mutations(request.url, request.mutation_types, request.num_mutations)
        variants = []
        for mutation in mutations:
            result = offline_predict(mutation["url"])
            risk = result["phishing_probability"]
            variants.append({
                **mutation,
                "prediction": result["prediction"],
                "response_label": "phishing" if risk >= 0.7 else "suspicious" if risk >= 0.35 else "legitimate",
                "confidence": result["confidence"],
                "phishing_probability": result["phishing_probability"],
                "prediction_changed": result["prediction"] != original["prediction"],
            })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Offline robustness test failed: {str(exc)}")

    flagged = sum(item["response_label"] == "phishing" for item in variants)
    changed = sum(item["prediction_changed"] for item in variants)
    metrics = {
        "original_urls_tested": 1,
        "total_variants": len(variants),
        "total_mutations_generated": len(variants),
        "phishing": flagged,
        "suspicious": sum(item["response_label"] == "suspicious" for item in variants),
        "legitimate": sum(item["response_label"] == "legitimate" for item in variants),
        "flagged_as_phishing": flagged,
        "not_flagged": len(variants) - flagged,
        "prediction_changes": changed,
        "prediction_consistency": round((len(variants) - changed) / len(variants), 4) if variants else 1.0,
    }
    if request.known_label:
        detected = sum(item["prediction"] == request.known_label for item in variants)
        metrics.update({"detected": detected, "missed": len(variants) - detected, "detection_rate": round(detected / len(variants), 4) if variants else 0})

    category_metrics = []
    for mutation_type in request.mutation_types:
        group = [item for item in variants if item["mutation_type"] == mutation_type]
        category_metrics.append({
            "mutation_type": mutation_type,
            "label": MUTATION_TYPES[mutation_type],
            "total": len(group),
            "consistent": sum(not item["prediction_changed"] for item in group),
            "flagged_as_phishing": sum(item["response_label"] == "phishing" for item in group),
            "prediction_changes": sum(item["prediction_changed"] for item in group),
            "prediction_consistency": round(sum(not item["prediction_changed"] for item in group) / len(group), 4) if group else None,
        })
    return {
        "original": {
            "url": request.url, "prediction": original["prediction"],
            "response_label": "phishing" if original["phishing_probability"] >= 0.7 else "suspicious" if original["phishing_probability"] >= 0.35 else "legitimate",
            "confidence": original["confidence"], "phishing_probability": original["phishing_probability"],
        },
        "known_label": request.known_label,
        "metrics": metrics,
        "category_metrics": category_metrics,
        "variants": variants,
        "notice": "Synthetic variations are generated and scored offline only. They are not visited, registered, or deployed.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_name": best_model_name}


@app.get("/")
def root():
    return {
        "message": "Phishing Detector API",
        "endpoints": {
            "POST /analyze": "Classify a URL (fast, ML only)",
            "POST /analyze-email": "Heuristic check of an email address",
            "POST /analyze-text": "Heuristic check of a message body",
            "POST /explain": "Classify + plain-English explanation (slower, uses local LLM)",
            "POST /robustness-test": "Generate and score synthetic URL mutations offline",
            "GET /health": "Check API + model status",
            "GET /docs": "Interactive API documentation",
        }
    }
