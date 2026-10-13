"""
SSL/TLS certificate feature extraction.
Also network-dependent — same rule as domain_features: never let a
timeout or handshake failure crash the pipeline, just return defaults
that signal "no valid cert observed" (which is itself a useful signal).
"""
import ssl
import socket
from datetime import datetime, timezone


def extract_ssl_features(hostname: str, port: int = 443, timeout: int = 5) -> dict:
    defaults = {
        "has_valid_ssl": 0,
        "ssl_cert_age_days": -1,
        "ssl_days_until_expiry": -1,
        "ssl_issuer_is_known_ca": 0,
    }

    known_cas = [
        "Let's Encrypt", "DigiCert", "Sectigo", "GlobalSign",
        "GoDaddy", "Comodo", "Amazon", "Google Trust Services", "Cloudflare"
    ]

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        not_before = datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        issuer = dict(x[0] for x in cert.get("issuer", []))
        issuer_org = issuer.get("organizationName", "")

        return {
            "has_valid_ssl": 1,
            "ssl_cert_age_days": (now - not_before).days,
            "ssl_days_until_expiry": (not_after - now).days,
            "ssl_issuer_is_known_ca": 1 if any(ca in issuer_org for ca in known_cas) else 0,
        }

    except Exception:
        # Includes: no SSL at all, expired/invalid cert, handshake failure, timeout
        return defaults
