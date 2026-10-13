"""
Shared helper: finds URLs embedded in free text (email bodies, SMS messages).

Used by BOTH email_features.py and text_features.py so URL-detection logic
exists in exactly one place, and so any URL found inside an email or SMS
is analyzed with the EXACT SAME feature extractor already used for the
standalone URL detector (app.features.extractor.extract_all_features) —
per the project requirement to never duplicate URL-analysis logic.
"""
import re

# Matches http(s)://... and bare www.-prefixed domains, which is the
# common form phishing messages use ("Click www.fake-bank.com/verify").
URL_PATTERN = re.compile(
    r'(?:(?:https?://)|(?:www\.))[^\s<>"\'\)]+',
    re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
)

EMAIL_ADDRESS_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
)


def find_urls(text: str) -> list:
    """Returns a de-duplicated list of URLs found in free text, preserving order."""
    if not text:
        return []
    found = URL_PATTERN.findall(text)
    # Normalize bare "www." matches to a fetchable URL for downstream feature extraction
    normalized = []
    seen = set()
    for u in found:
        u = u.rstrip('.,;:!?')  # strip trailing punctuation caught by the regex
        if u.lower().startswith("www."):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            normalized.append(u)
    return normalized


def find_phone_numbers(text: str) -> list:
    if not text:
        return []
    return PHONE_PATTERN.findall(text)


def find_email_addresses(text: str) -> list:
    if not text:
        return []
    return EMAIL_ADDRESS_PATTERN.findall(text)
