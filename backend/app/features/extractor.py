"""
Main orchestrator: runs all feature extractors and merges results
into a single flat dict, ready to feed into the ML model.

Design choice: URL features are always computed (fast, no network).
Domain/SSL/content features are computed with individual try/except
already baked into each module, so a failure in one extractor never
blocks the others. This matters a lot for training on a dataset —
a chunk of URLs will always be dead links or rate-limited lookups,
and the pipeline needs to keep going instead of crashing mid-run.
"""
from urllib.parse import urlparse
import tldextract

from app.features.url_features import extract_url_features
from app.features.domain_features import extract_domain_features
from app.features.ssl_features import extract_ssl_features
from app.features.content_features import extract_content_features


def extract_all_features(url: str, include_content: bool = True, timeout: int = 5) -> dict:
    """
    Extract the full feature set for a single URL.

    include_content: set False during bulk dataset processing if you want
    a faster first pass (content fetching is the slowest extractor by far).
    You can run a second pass later to backfill content features.
    """
    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    ext = tldextract.extract(url)
    registrable_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    features = {}
    features.update(extract_url_features(url))
    features.update(extract_domain_features(registrable_domain, timeout=timeout))
    features.update(extract_ssl_features(hostname, timeout=timeout))

    if include_content:
        features.update(extract_content_features(url, timeout=timeout))

    return features


if __name__ == "__main__":
    # Quick manual test
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure-login-verify.tk/signin",
    ]
    for u in test_urls:
        print(f"\n--- {u} ---")
        result = extract_all_features(u, include_content=False)  # skip content for quick test
        for k, v in result.items():
            print(f"  {k}: {v}")
