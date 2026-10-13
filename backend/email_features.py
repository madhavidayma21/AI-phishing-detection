"""
Email phishing feature extraction.

Extracts sender, subject, and body features, and reuses the EXISTING
URL feature extractor (app.features.extractor.extract_all_features)
for any URLs found inside the email body — no duplicate URL-analysis
logic, per project requirement.

Returns a flat dict: numeric/boolean engineered features (for the
Logistic Regression side of the model) plus the raw subject/body text
(for the TF-IDF side) plus a human-readable `reasons` list generated
purely from the detected feature values (Explainable AI — no LLM
required for this fast path).
"""
import re
from email.utils import parseaddr

from url_finder import find_urls
from app.features.extractor import extract_all_features

FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "protonmail.com", "mail.com", "yandex.com"
}

URGENCY_WORDS = [
    "urgent", "immediately", "action required", "act now", "expire",
    "expiring", "suspended", "locked", "restricted", "final notice",
    "warning", "alert", "deadline"
]

SUSPICIOUS_KEYWORDS = [
    "verify", "urgent", "account", "suspended", "password", "security",
    "payment", "invoice", "winner", "congratulations", "click", "confirm",
    "update", "unusual activity", "limited time"
]

CREDENTIAL_REQUEST_WORDS = [
    "password", "username", "login", "ssn", "social security",
    "credit card", "cvv", "pin number", "verify your identity"
]

FINANCIAL_WORDS = [
    "payment", "invoice", "bank", "transfer", "refund", "billing",
    "transaction", "wire", "paypal", "card number"
]

REWARD_WORDS = [
    "winner", "congratulations", "prize", "reward", "free", "gift",
    "claim now", "you have won", "lottery"
]

THREAT_WORDS = [
    "suspended", "terminated", "legal action", "closed", "blocked",
    "unauthorized", "penalty", "fine"
]


def _keyword_count(text: str, keywords: list) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_email_features(sender: str, subject: str, body: str, timeout: int = 3) -> dict:
    sender = sender or ""
    subject = subject or ""
    body = body or ""

    display_name, sender_address = parseaddr(sender)
    sender_domain = sender_address.split("@")[-1].lower() if "@" in sender_address else ""

    # --- Sender features ---
    sender_length = len(sender_address)
    is_free_email_provider = 1 if sender_domain in FREE_EMAIL_PROVIDERS else 0
    sender_num_digits = sum(c.isdigit() for c in sender_address)
    # crude display-name / address mismatch check: does a brand word in the
    # display name fail to appear in the actual domain?
    name_domain_mismatch = 0
    if display_name and sender_domain:
        name_tokens = re.findall(r'[a-zA-Z]+', display_name.lower())
        if name_tokens and not any(tok in sender_domain for tok in name_tokens if len(tok) > 3):
            name_domain_mismatch = 1

    # --- Subject features ---
    subject_length = len(subject)
    subject_word_count = len(subject.split())
    subject_special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', subject))
    subject_uppercase_chars = sum(1 for c in subject if c.isupper())
    subject_urgency_count = _keyword_count(subject, URGENCY_WORDS)
    subject_suspicious_count = _keyword_count(subject, SUSPICIOUS_KEYWORDS)

    # --- Body features ---
    body_length = len(body)
    body_word_count = len(body.split())
    urls_in_body = find_urls(body)
    num_urls = len(urls_in_body)
    num_html_forms = body.lower().count("<form")
    num_password_fields = body.lower().count('type="password"') + body.lower().count("type='password'")
    num_buttons_links = body.lower().count("<button") + body.lower().count("<a href")

    body_suspicious_count = _keyword_count(body, SUSPICIOUS_KEYWORDS)
    body_urgency_count = _keyword_count(body, URGENCY_WORDS)
    body_credential_count = _keyword_count(body, CREDENTIAL_REQUEST_WORDS)
    body_financial_count = _keyword_count(body, FINANCIAL_WORDS)
    body_reward_count = _keyword_count(body, REWARD_WORDS)
    body_threat_count = _keyword_count(body, THREAT_WORDS)

    # --- URLs inside the email: reuse the EXISTING url extractor ---
    url_feature_list = []
    for u in urls_in_body[:5]:  # cap at 5 to keep prediction latency bounded
        try:
            url_feature_list.append(extract_all_features(u, include_content=False, timeout=timeout))
        except Exception:
            pass  # a single bad URL must never break email analysis

    any_url_is_ip_based = 1 if any(f.get("has_ip_address") == 1 for f in url_feature_list) else 0
    any_url_shortened = 1 if any(f.get("is_shortened") == 1 for f in url_feature_list) else 0
    any_url_suspicious_word = 1 if any(f.get("has_suspicious_word") == 1 for f in url_feature_list) else 0
    any_url_no_https = 1 if any(f.get("has_https") == 0 for f in url_feature_list) else 0
    any_url_no_valid_ssl = 1 if any(f.get("has_valid_ssl") == 0 for f in url_feature_list) else 0

    features = {
        "sender_length": sender_length,
        "sender_domain_is_free_provider": is_free_email_provider,
        "sender_num_digits": sender_num_digits,
        "sender_name_domain_mismatch": name_domain_mismatch,
        "subject_length": subject_length,
        "subject_word_count": subject_word_count,
        "subject_special_chars": subject_special_chars,
        "subject_uppercase_chars": subject_uppercase_chars,
        "subject_urgency_count": subject_urgency_count,
        "subject_suspicious_count": subject_suspicious_count,
        "body_length": body_length,
        "body_word_count": body_word_count,
        "num_urls": num_urls,
        "num_html_forms": num_html_forms,
        "num_password_fields": num_password_fields,
        "num_buttons_links": num_buttons_links,
        "body_suspicious_count": body_suspicious_count,
        "body_urgency_count": body_urgency_count,
        "body_credential_request_count": body_credential_count,
        "body_financial_count": body_financial_count,
        "body_reward_count": body_reward_count,
        "body_threat_count": body_threat_count,
        "any_url_is_ip_based": any_url_is_ip_based,
        "any_url_shortened": any_url_shortened,
        "any_url_suspicious_word": any_url_suspicious_word,
        "any_url_no_https": any_url_no_https,
        "any_url_no_valid_ssl": any_url_no_valid_ssl,
        "_subject_text": subject,
        "_body_text": body,
        "_detected_urls": urls_in_body,
    }
    return features


def generate_email_reasons(features: dict) -> list:
    """
    Rule-based, LLM-free explanation — mirrors the pattern already used by
    app/explain/explainer.py's select_signals() for URLs. Every reason here
    is derived directly from a real detected feature value, never invented.
    """
    reasons = []

    if features.get("sender_domain_is_free_provider"):
        reasons.append("Sender uses a free/public email provider rather than a corporate domain")
    if features.get("sender_name_domain_mismatch"):
        reasons.append("Sender display name does not match the sender's email domain")
    if features.get("subject_urgency_count", 0) > 0 or features.get("body_urgency_count", 0) > 0:
        reasons.append("Message uses urgent or time-pressure language")
    if features.get("body_credential_request_count", 0) > 0:
        reasons.append("Message requests credentials or sensitive personal information")
    if features.get("body_financial_count", 0) > 0:
        reasons.append("Message references payment, billing, or financial account details")
    if features.get("body_reward_count", 0) > 0:
        reasons.append("Message offers a prize, reward, or unexpected winnings")
    if features.get("body_threat_count", 0) > 0:
        reasons.append("Message contains threatening language (e.g. account suspension, legal action)")
    if features.get("num_password_fields", 0) > 0:
        reasons.append("Message embeds a password input field")
    if features.get("any_url_is_ip_based"):
        reasons.append("Message contains a link using a raw IP address instead of a domain name")
    if features.get("any_url_shortened"):
        reasons.append("Message contains a shortened URL, which can hide the true destination")
    if features.get("any_url_suspicious_word"):
        reasons.append("Message contains a link with phishing-associated keywords (e.g. 'login', 'verify')")
    if features.get("any_url_no_valid_ssl"):
        reasons.append("Message contains a link without a valid SSL certificate")
    if features.get("num_urls", 0) == 0 and not reasons:
        reasons.append("No strong individual red flags detected — verdict based on overall pattern")

    return reasons
