"""
Domain-based feature extraction (WHOIS lookups).
These require a network call and can fail/time out/be rate-limited —
always wrap in try/except and return a neutral default rather than
crashing the pipeline. Treat "WHOIS unavailable" as a weak signal,
not a hard failure (GDPR redaction makes this common for legit sites too).
"""
import whois
from datetime import datetime, timezone


def extract_domain_features(domain: str, timeout: int = 5) -> dict:
    """
    domain: registrable domain, e.g. "paypal.com" (not full URL)
    """
    defaults = {
        "domain_age_days": -1,       # -1 = unknown, handle explicitly in model/imputer
        "whois_available": 0,
        "expiry_days_remaining": -1,
        "domain_age_under_30_days": 0,
        "domain_age_under_180_days": 0,
    }

    try:
        w = whois.whois(domain)

        creation_date = w.creation_date
        expiration_date = w.expiration_date

        # whois lib sometimes returns a list of dates instead of one
        if isinstance(creation_date, list):
            creation_date = creation_date[0] if creation_date else None
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0] if expiration_date else None

        if creation_date is None:
            return defaults

        if creation_date.tzinfo is None:
            creation_date = creation_date.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - creation_date).days

        expiry_days = -1
        if expiration_date:
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            if expiration_date.tzinfo is None:
                expiration_date = expiration_date.replace(tzinfo=timezone.utc)
            expiry_days = (expiration_date - now).days

        return {
            "domain_age_days": age_days,
            "whois_available": 1,
            "expiry_days_remaining": expiry_days,
            "domain_age_under_30_days": 1 if 0 <= age_days < 30 else 0,
            "domain_age_under_180_days": 1 if 0 <= age_days < 180 else 0,
        }

    except Exception:
        # Network failure, rate limit, or no WHOIS record found — don't crash the pipeline
        return defaults
