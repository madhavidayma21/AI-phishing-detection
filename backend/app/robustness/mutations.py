"""Generate deterministic, offline-only URL mutations for model robustness tests."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MUTATION_TYPES = {
    "character_substitution": "Character substitution",
    "subdomain_manipulation": "Subdomain manipulation",
    "hyphen_manipulation": "Hyphen manipulation",
    "path_manipulation": "Path manipulation",
    "query_manipulation": "Query manipulation",
    "token_manipulation": "Token manipulation",
}


def _normalise_url(url: str) -> str:
    return url.strip() if "://" in url else f"https://{url.strip()}"


def _parts(url: str):
    value = _normalise_url(url)
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        _ = parts.port  # Force validation of malformed ports.
    except ValueError as exc:
        raise ValueError("Enter a valid HTTP or HTTPS URL") from exc
    if parts.scheme not in {"http", "https"} or not hostname or parts.username or parts.password:
        raise ValueError("Enter a URL with a valid hostname")
    if any(ch.isspace() for ch in value):
        raise ValueError("URL cannot contain whitespace")
    return parts


def _with_host(parts, hostname: str) -> str:
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{hostname}{port}", parts.path, parts.query, parts.fragment))


def _character_substitutions(parts) -> list[str]:
    hostname = parts.hostname
    generated = []
    for old, new in (("a", "4"), ("e", "3"), ("i", "1"), ("l", "1"), ("o", "0"), ("s", "5")):
        if old in hostname:
            generated.append(_with_host(parts, hostname.replace(old, new, 1)))
        if len(generated) == 5:
            break
    # Short or vowel-poor hostnames still need a balanced five-case test set.
    # Use plain ASCII substitutions only so this remains a syntactic, offline test.
    for index, character in enumerate(hostname):
        if len(generated) == 5:
            break
        if character.isalpha():
            replacement = "a" if character == "z" else chr(ord(character) + 1)
            candidate = _with_host(parts, hostname[:index] + replacement + hostname[index + 1:])
            if candidate not in generated:
                generated.append(candidate)
    return generated


def _subdomain_manipulations(parts) -> list[str]:
    return [_with_host(parts, f"{prefix}.{parts.hostname}") for prefix in ("secure", "login", "account", "verify", "support")]


def _hyphen_manipulations(parts) -> list[str]:
    labels = parts.hostname.split(".")
    first = labels[0]
    positions = sorted({max(1, min(len(first) - 1, round(len(first) * fraction))) for fraction in (.2, .35, .5, .65, .8)})
    generated = [_with_host(parts, ".".join([first[:position] + "-" + first[position:]] + labels[1:])) for position in positions if len(first) > 1]
    for index, character in enumerate(first):
        if character == "-":
            generated.append(_with_host(parts, ".".join([first[:index] + first[index + 1:]] + labels[1:])))
    return generated


def _path_manipulations(parts) -> list[str]:
    base = parts.path.rstrip("/")
    generated = [urlunsplit((parts.scheme, parts.netloc, f"{base}/{suffix}", parts.query, parts.fragment)) for suffix in ("login", "account", "verify", "secure", "signin")]
    segments = [segment for segment in parts.path.split("/") if segment]
    for index in range(len(segments)):
        remaining = segments[:index] + segments[index + 1:]
        generated.append(urlunsplit((parts.scheme, parts.netloc, "/" + "/".join(remaining), parts.query, parts.fragment)))
    return generated


def _query_manipulations(parts) -> list[str]:
    base = parse_qsl(parts.query, keep_blank_values=True)
    additions = (("source", "email"), ("session", "verify"), ("redirect", "account"), ("continue", "login"), ("ref", "secure"))
    generated = [urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(base + [addition]), parts.fragment)) for addition in additions]
    for index in range(len(base)):
        generated.append(urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(base[:index] + base[index + 1:]), parts.fragment)))
    return generated


def _token_manipulations(parts) -> list[str]:
    tokens = ("login", "verify", "secure", "account", "update")
    base = parts.path.rstrip("/")
    return [urlunsplit((parts.scheme, parts.netloc, f"{base}/{token}", parts.query, parts.fragment)) for token in tokens]


GENERATORS = {
    "character_substitution": _character_substitutions,
    "subdomain_manipulation": _subdomain_manipulations,
    "hyphen_manipulation": _hyphen_manipulations,
    "path_manipulation": _path_manipulations,
    "query_manipulation": _query_manipulations,
    "token_manipulation": _token_manipulations,
}


def generate_mutations(url: str, mutation_types: list[str], limit: int | None = None) -> list[dict]:
    """Generate synthetic URLs only; this module never resolves or requests them."""
    parts = _parts(url)
    unknown = set(mutation_types) - set(GENERATORS)
    if unknown:
        raise ValueError("Unknown mutation type")
    seen, mutations = {urlunsplit(parts)}, []
    for mutation_type in mutation_types:
        for variant in GENERATORS[mutation_type](parts):
            if variant not in seen:
                seen.add(variant)
                mutations.append({"url": variant, "mutation_type": mutation_type, "mutation_label": MUTATION_TYPES[mutation_type]})
                if limit is not None and len(mutations) >= limit:
                    return mutations
    return mutations
