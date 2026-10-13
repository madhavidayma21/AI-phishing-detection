"""
Content-based feature extraction.
Fetches the page HTML and looks for phishing tells: login forms that
submit to a different domain, brand-name mentions that don't match
the actual domain (typosquatting/impersonation), password fields, etc.

This is the most expensive + failure-prone extractor (full HTTP GET),
so it should have the tightest timeout and always degrade gracefully.
"""
import re
import requests
from urllib.parse import urlparse
from rapidfuzz import fuzz

# Small starter list — expand this with more brands relevant to your
# dataset (banks, payment processors, common Indian services, etc.)
KNOWN_BRANDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "instagram", "netflix", "bankofamerica", "chase", "wellsfargo",
    "hdfcbank", "icicibank", "sbi", "axisbank", "paytm", "phonepe",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (phishing-detector-research-project)"}


def extract_content_features(url: str, timeout: int = 5) -> dict:
    defaults = {
        "fetch_successful": 0,
        "has_login_form": 0,
        "has_password_field": 0,
        "form_action_external": 0,
        "brand_mentioned_in_content": 0,
        "brand_domain_mismatch": 0,
        "num_external_links": 0,
        "has_iframe": 0,
        "title_brand_mismatch": 0,
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        html = resp.text.lower()
        actual_domain = urlparse(resp.url).netloc.lower()

        has_login_form = 1 if re.search(r"<form[^>]*>.*?</form>", html, re.DOTALL) and \
            ("login" in html or "signin" in html or "sign-in" in html) else 0
        has_password_field = 1 if re.search(r'type=["\']password["\']', html) else 0
        has_iframe = 1 if "<iframe" in html else 0

        # Form action pointing to a different domain than the page itself = major red flag
        form_actions = re.findall(r'<form[^>]*action=["\']([^"\']+)["\']', html)
        form_action_external = 0
        for action in form_actions:
            if action.startswith("http") and actual_domain not in action:
                form_action_external = 1
                break

        # Brand impersonation: brand name appears in content/title but domain doesn't match it
        mentioned_brand = None
        for brand in KNOWN_BRANDS:
            if brand in html:
                mentioned_brand = brand
                break

        brand_domain_mismatch = 0
        if mentioned_brand and mentioned_brand not in actual_domain:
            # fuzzy check in case domain is a near-miss typosquat, e.g. paypa1.com
            similarity = fuzz.partial_ratio(mentioned_brand, actual_domain)
            if similarity < 85:  # not a close-enough match to be the real brand domain
                brand_domain_mismatch = 1

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL)
        title = title_match.group(1) if title_match else ""
        title_brand_mismatch = 0
        for brand in KNOWN_BRANDS:
            if brand in title and brand not in actual_domain:
                title_brand_mismatch = 1
                break

        num_external_links = len([
            l for l in re.findall(r'href=["\'](https?://[^"\']+)["\']', html)
            if actual_domain not in l
        ])

        return {
            "fetch_successful": 1,
            "has_login_form": has_login_form,
            "has_password_field": has_password_field,
            "form_action_external": form_action_external,
            "brand_mentioned_in_content": 1 if mentioned_brand else 0,
            "brand_domain_mismatch": brand_domain_mismatch,
            "num_external_links": num_external_links,
            "has_iframe": has_iframe,
            "title_brand_mismatch": title_brand_mismatch,
        }

    except Exception:
        # Site unreachable, timeout, SSL error, etc. — common for dead phishing links in datasets
        return defaults
