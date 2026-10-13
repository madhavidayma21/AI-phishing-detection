"""
Text/SMS phishing feature extraction.

Same reuse principle as email_features.py: any URL found inside the
message is analyzed through the EXISTING url extractor, never a
duplicate implementation.
"""
import re

from url_finder import find_urls, find_phone_numbers, find_email_addresses
from app.features.extractor import extract_all_features

URGENCY_WORDS = [
    "urgent", "immediately", "act now", "expire", "expiring", "suspended",
    "locked", "restricted", "final notice", "warning", "alert", "deadline",
    "right away", "asap"
]

THREAT_WORDS = [
    "suspended", "blocked", "terminated", "penalty", "fine", "legal action",
    "unauthorized", "closed", "frozen"
]

FINANCIAL_WORDS = [
    "bank", "payment", "refund", "billing", "transaction", "card", "account",
    "transfer", "wire", "paypal", "upi", "otp"
]

CREDENTIAL_WORDS = [
    "password", "pin", "otp", "verify your account", "login", "ssn", "cvv"
]

REWARD_WORDS = [
    "winner", "congratulations", "prize", "reward", "free", "gift",
    "claim now", "you have won", "lottery", "cashback"
]

SUSPICIOUS_KEYWORDS = URGENCY_WORDS + THREAT_WORDS + CREDENTIAL_WORDS + REWARD_WORDS


def _keyword_count(text: str, keywords: list) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_text_features(text: str, timeout: int = 3) -> dict:
    text = text or ""

    message_length = len(text)
    word_count = len(text.split())
    char_count = len(text.replace(" ", ""))

    urls_in_text = find_urls(text)
    num_urls = len(urls_in_text)

    phone_numbers = find_phone_numbers(text)
    email_addresses = find_email_addresses(text)

    exclamation_count = text.count("!")
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_chars / len(text) if text else 0
    digit_count = sum(c.isdigit() for c in text)

    urgency_score = _keyword_count(text, URGENCY_WORDS)
    threat_score = _keyword_count(text, THREAT_WORDS)
    financial_count = _keyword_count(text, FINANCIAL_WORDS)
    credential_count = _keyword_count(text, CREDENTIAL_WORDS)
    reward_count = _keyword_count(text, REWARD_WORDS)
    suspicious_keyword_count = _keyword_count(text, SUSPICIOUS_KEYWORDS)

    url_feature_list = []
    for u in urls_in_text[:5]:
        try:
            url_feature_list.append(extract_all_features(u, include_content=False, timeout=timeout))
        except Exception:
            pass

    num_shortened_urls = sum(1 for f in url_feature_list if f.get("is_shortened") == 1)
    any_url_is_ip_based = 1 if any(f.get("has_ip_address") == 1 for f in url_feature_list) else 0
    any_url_suspicious_word = 1 if any(f.get("has_suspicious_word") == 1 for f in url_feature_list) else 0
    any_url_no_https = 1 if any(f.get("has_https") == 0 for f in url_feature_list) else 0

    features = {
        "message_length": message_length,
        "word_count": word_count,
        "char_count": char_count,
        "num_urls": num_urls,
        "num_shortened_urls": num_shortened_urls,
        "suspicious_keyword_count": suspicious_keyword_count,
        "urgency_score": urgency_score,
        "threat_score": threat_score,
        "financial_keyword_count": financial_count,
        "credential_request_count": credential_count,
        "reward_keyword_count": reward_count,
        "exclamation_count": exclamation_count,
        "uppercase_char_count": uppercase_chars,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "digit_count": digit_count,
        "has_phone_number": 1 if phone_numbers else 0,
        "has_email_address": 1 if email_addresses else 0,
        "has_url": 1 if num_urls > 0 else 0,
        "any_url_is_ip_based": any_url_is_ip_based,
        "any_url_suspicious_word": any_url_suspicious_word,
        "any_url_no_https": any_url_no_https,
        "_text": text,
        "_detected_urls": urls_in_text,
    }
    return features


def generate_text_reasons(features: dict) -> list:
    """Rule-based explanation, same pattern as email_features.generate_email_reasons()."""
    reasons = []

    if features.get("urgency_score", 0) > 0:
        reasons.append("Contains urgent or time-pressure language")
    if features.get("threat_score", 0) > 0:
        reasons.append("Contains threatening language (e.g. account suspension, penalty)")
    if features.get("credential_request_count", 0) > 0:
        reasons.append("Requests a password, PIN, OTP, or account verification")
    if features.get("financial_keyword_count", 0) > 0:
        reasons.append("References banking, payment, or financial account details")
    if features.get("reward_keyword_count", 0) > 0:
        reasons.append("Offers a prize, reward, or unexpected winnings")
    if features.get("any_url_is_ip_based"):
        reasons.append("Contains a link using a raw IP address instead of a domain name")
    if features.get("num_shortened_urls", 0) > 0:
        reasons.append("Contains a shortened URL, which can hide the true destination")
    if features.get("any_url_suspicious_word"):
        reasons.append("Contains a link with phishing-associated keywords (e.g. 'login', 'verify')")
    if features.get("any_url_no_https"):
        reasons.append("Contains a link that does not use a secure (HTTPS) connection")
    if features.get("uppercase_ratio", 0) > 0.3 and features.get("message_length", 0) > 10:
        reasons.append("Unusually high use of uppercase letters, often used for urgency")
    if not reasons:
        reasons.append("No strong individual red flags detected — verdict based on overall pattern")

    return reasons
