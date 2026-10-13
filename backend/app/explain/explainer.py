"""
Month 3: LLM-based explanation agent.

Takes the ML model's raw output (prediction + feature values) and
generates a plain-English explanation of WHY a URL was flagged.

IMPORTANT DESIGN PRINCIPLE: the ML model does the classification —
this agent NEVER re-decides phishing/legitimate itself, it only
explains a decision that's already been made. This separation matters:
classifiers don't hallucinate, LLMs can. Keeping them separate means
a bad LLM explanation can never flip your actual security decision.

Two-node graph:
  1. select_signals   — rule-based (no LLM). Picks which raw feature
                         values are actually worth mentioning, based
                         on simple thresholds. Grounds the explanation
                         in real numbers instead of letting the LLM
                         invent or misremember them.
  2. generate_explanation — calls the LLM with ONLY the selected,
                         verified signals and asks for a plain-English
                         explanation. The LLM is not shown the full
                         feature vector, so it can't reference numbers
                         we didn't hand it.

Requires Ollama running locally with a model pulled, e.g.:
    ollama pull llama3.1
    ollama serve   (usually already running as a background service)

Usage as a module:
    from app.explain.explainer import explain_result
    text = explain_result(url, prediction, confidence, raw_features)
"""
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


OLLAMA_MODEL = "llama3.1"  # change if you're using a different pulled model


class ExplanationState(TypedDict):
    url: str
    prediction: str  # "phishing" or "legitimate"
    confidence: float
    raw_features: dict
    selected_signals: List[dict]
    explanation: str


def select_signals(state: ExplanationState) -> ExplanationState:
    """
    Rule-based signal selection. No LLM call here — this is deliberate.
    We only want to hand the LLM signals that are ACTUALLY notable,
    each with a short plain-language description of what it means,
    so the LLM has grounded, pre-verified facts to work from instead
    of interpreting raw numbers itself (which risks misreading them).
    """
    f = state["raw_features"]
    signals = []

    if f.get("has_valid_ssl") == 0:
        signals.append({"signal": "no_valid_ssl", "detail": "The site does not have a valid SSL/TLS certificate."})
    elif f.get("has_valid_ssl") == 1:
        signals.append({"signal": "valid_ssl", "detail": "The site has a valid SSL/TLS certificate."})

    if f.get("whois_available") == 0:
        signals.append({"signal": "no_whois", "detail": "Domain registration information could not be verified (WHOIS lookup failed or was withheld)."})
    elif f.get("domain_age_days") is not None and f.get("domain_age_days", -1) >= 0:
        age = f["domain_age_days"]
        if age < 30:
            signals.append({"signal": "very_new_domain", "detail": f"The domain was registered only {age} days ago."})
        elif age < 180:
            signals.append({"signal": "new_domain", "detail": f"The domain was registered {age} days ago, relatively recently."})
        elif age > 1000:
            signals.append({"signal": "established_domain", "detail": f"The domain has existed for {age} days ({age // 365} years), suggesting an established site."})

    if f.get("has_suspicious_word") == 1:
        signals.append({"signal": "suspicious_keywords", "detail": "The URL contains words commonly used in phishing attempts (e.g. 'login', 'verify', 'secure', 'account')."})

    if f.get("has_ip_address") == 1:
        signals.append({"signal": "ip_address_url", "detail": "The URL uses a raw IP address instead of a domain name, which is unusual for legitimate sites."})

    if f.get("is_shortened") == 1:
        signals.append({"signal": "shortened_url", "detail": "The URL uses a link-shortening service, which can be used to hide the true destination."})

    num_digits = f.get("num_digits", 0)
    if num_digits and num_digits > 5:
        signals.append({"signal": "many_digits", "detail": f"The URL contains an unusually high number of digits ({num_digits})."})

    entropy = f.get("domain_entropy", 0)
    if entropy and entropy > 3.5:
        signals.append({"signal": "high_entropy_domain", "detail": "The domain name looks randomly generated rather than a real word or brand name."})

    if f.get("hyphen_in_domain") == 1:
        signals.append({"signal": "hyphen_domain", "detail": "The domain name contains hyphens, sometimes used to mimic a real brand (e.g. 'paypal-secure.com')."})

    state["selected_signals"] = signals
    return state


def generate_explanation(state: ExplanationState) -> ExplanationState:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)

    signal_lines = "\n".join(f"- {s['detail']}" for s in state["selected_signals"])
    if not signal_lines:
        signal_lines = "- No strongly notable signals were found; classification relied on a broader combination of weaker signals."

    verdict = "flagged as PHISHING" if state["prediction"] == "phishing" else "classified as LEGITIMATE"

    prompt = f"""You are explaining a phishing-detection result to a non-technical user.

URL: {state['url']}
Verdict: {verdict}
Confidence: {state['confidence']*100:.1f}%

Verified signals found by the detection system (use ONLY these facts, do not invent or add anything not listed here):
{signal_lines}

Write a short, clear, 2-4 sentence explanation of why this URL received this verdict. Speak directly to the user. Do not mention "the model" or "the classifier" — speak as if you personally reviewed the site. Do not add any facts, numbers, or claims beyond what's listed above."""

    response = llm.invoke(prompt)
    state["explanation"] = response.content.strip()
    return state


def build_graph():
    graph = StateGraph(ExplanationState)
    graph.add_node("select_signals", select_signals)
    graph.add_node("generate_explanation", generate_explanation)
    graph.set_entry_point("select_signals")
    graph.add_edge("select_signals", "generate_explanation")
    graph.add_edge("generate_explanation", END)
    return graph.compile()


_compiled_graph = None


def explain_result(url: str, prediction: str, confidence: float, raw_features: dict) -> str:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: ExplanationState = {
        "url": url,
        "prediction": prediction,
        "confidence": confidence,
        "raw_features": raw_features,
        "selected_signals": [],
        "explanation": "",
    }
    final_state = _compiled_graph.invoke(initial_state)
    return final_state["explanation"]


if __name__ == "__main__":
    # Quick manual test — requires Ollama running locally
    fake_features = {
        "has_valid_ssl": 0,
        "whois_available": 0,
        "domain_age_days": -1,
        "has_suspicious_word": 1,
        "has_ip_address": 0,
        "is_shortened": 0,
        "num_digits": 2,
        "domain_entropy": 3.9,
        "hyphen_in_domain": 1,
    }
    result = explain_result(
        url="http://secure-paypal-verify-account.tk/login",
        prediction="phishing",
        confidence=0.9999,
        raw_features=fake_features,
    )
    print(result)
