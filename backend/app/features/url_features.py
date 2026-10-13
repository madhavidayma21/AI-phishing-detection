"""
URL-based feature extraction.
These features are computed purely from the URL string itself —
no network calls required, so they're fast and always available
(unlike WHOIS/SSL features which can fail or time out).
"""
import re
import math
from urllib.parse import urlparse
import tldextract

# The robustness lab must remain offline. Disable tldextract's default PSL download
# and use its bundled snapshot when extracting URL-only features.
_OFFLINE_EXTRACT = tldextract.TLDExtract(suffix_list_urls=None)

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "banking",
    "confirm", "signin", "webscr", "password", "suspended", "urgent"
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly"
}


def shannon_entropy(s: str) -> float:
    """Higher entropy = more random-looking string, common in generated phishing domains."""
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    length = len(s)
    return -sum((f / length) * math.log2(f / length) for f in freq.values())


def extract_url_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    ext = _OFFLINE_EXTRACT(url)
    domain = ext.domain
    subdomain = ext.subdomain
    suffix = ext.suffix
    full_domain = f"{domain}.{suffix}" if suffix else domain
    path = parsed.path or ""
    query = parsed.query or ""

    hostname = parsed.hostname or ""

    features = {
        # Length-based
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),

        # Structural
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_at_symbols": url.count("@"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": len(subdomain.split(".")) if subdomain else 0,

        # Suspicious patterns
        "has_ip_address": 1 if re.match(
            r"^(\d{1,3}\.){3}\d{1,3}$", hostname
        ) else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "is_shortened": 1 if full_domain in URL_SHORTENERS else 0,
        "has_suspicious_word": 1 if any(
            w in url.lower() for w in SUSPICIOUS_WORDS
        ) else 0,
        "has_double_slash_redirect": 1 if "//" in path else 0,
        "has_port": 1 if parsed.port else 0,

        # Entropy (randomness of domain name — generated phishing domains score higher)
        "domain_entropy": shannon_entropy(domain),

        # Brand impersonation heuristic (very common: paypal-secure-login.com)
        "hyphen_in_domain": 1 if "-" in domain else 0,

        "prefix_suffix_at_domain": 1 if domain.startswith("www") is False and "-" in domain else 0,
    }

    return features
