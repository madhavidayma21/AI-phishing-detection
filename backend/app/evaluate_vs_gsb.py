"""
Week 15-16: Benchmark evaluation — our model vs Google Safe Browsing API.

Takes a labeled set of test URLs (ideally ones your model hasn't seen
during training — held-out or freshly pulled from OpenPhish/PhishTank),
runs each one through BOTH your own /analyze endpoint AND Google's
Safe Browsing API, and reports how they compare.

This matters for your report: it's the difference between "my model
gets 97% on my own test set" (self-graded) and "here's how my model
compares to an industry-standard tool" (independently benchmarked).

Setup:
    1. Get a free API key: https://console.cloud.google.com/
       -> Enable "Safe Browsing API" -> Credentials -> Create API Key
    2. Your FastAPI server must be running (uvicorn ... on port 8000)

Usage:
    python evaluate_vs_gsb.py --input eval_urls.csv --gsb-key YOUR_KEY_HERE

Expected input CSV format: url,label  (label: 1=phishing, 0=legit)
"""
import argparse
import time
import requests
import pandas as pd

GSB_API_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
OUR_API_URL = "http://localhost:8000/analyze"


def check_google_safe_browsing(url: str, api_key: str) -> int:
    """
    Returns 1 if Google flags the URL as any kind of threat, 0 if not,
    -1 if the request itself failed (rate limit, network issue, etc.)
    """
    payload = {
        "client": {"clientId": "phishing-detector-project", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(f"{GSB_API_URL}?key={api_key}", json=payload, timeout=10)
        if resp.status_code != 200:
            return -1
        data = resp.json()
        # Empty response {} means no match found = not flagged
        return 1 if data.get("matches") else 0
    except Exception:
        return -1


def check_our_model(url: str) -> tuple:
    """Returns (prediction: 0/1, confidence: float) or (-1, 0) on failure."""
    try:
        resp = requests.post(OUR_API_URL, json={"url": url}, timeout=15)
        if resp.status_code != 200:
            return -1, 0.0
        data = resp.json()
        pred = 1 if data["prediction"] == "phishing" else 0
        return pred, data["confidence"]
    except Exception:
        return -1, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with columns: url,label")
    parser.add_argument("--gsb-key", required=True, help="Google Safe Browsing API key")
    parser.add_argument("--output", default="gsb_comparison_results.csv")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between GSB requests (avoid rate limits)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Evaluating {len(df)} URLs against both our model and Google Safe Browsing...\n")

    results = []
    for i, row in df.iterrows():
        url, true_label = row["url"], row["label"]

        our_pred, our_conf = check_our_model(url)
        gsb_pred = check_google_safe_browsing(url, args.gsb_key)

        results.append({
            "url": url,
            "true_label": true_label,
            "our_prediction": our_pred,
            "our_confidence": our_conf,
            "gsb_prediction": gsb_pred,
        })

        status = f"true={true_label} ours={our_pred}({our_conf:.2f}) gsb={gsb_pred}"
        print(f"  [{i+1}/{len(df)}] {url[:50]:<50} {status}")

        time.sleep(args.delay)  # be polite to Google's rate limits

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    # --- Compute comparison metrics ---
    valid = results_df[(results_df["our_prediction"] != -1) & (results_df["gsb_prediction"] != -1)]
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY  ({len(valid)}/{len(results_df)} URLs successfully checked by both)")
    print(f"{'='*70}")

    def accuracy(pred_col):
        return (valid["true_label"] == valid[pred_col]).mean()

    def recall_on_phishing(pred_col):
        phish = valid[valid["true_label"] == 1]
        if len(phish) == 0:
            return float("nan")
        return (phish[pred_col] == 1).mean()

    print(f"\nAccuracy on this test set:")
    print(f"  Our model:            {accuracy('our_prediction'):.4f}")
    print(f"  Google Safe Browsing: {accuracy('gsb_prediction'):.4f}")

    print(f"\nRecall on phishing URLs only (did it catch the actual phish?):")
    print(f"  Our model:            {recall_on_phishing('our_prediction'):.4f}")
    print(f"  Google Safe Browsing: {recall_on_phishing('gsb_prediction'):.4f}")

    # Agreement between the two systems
    agreement = (valid["our_prediction"] == valid["gsb_prediction"]).mean()
    print(f"\nAgreement between our model and GSB: {agreement:.4f}")

    # Cases where we caught something GSB missed (interesting for your report)
    we_caught_gsb_missed = valid[(valid["true_label"] == 1) & (valid["our_prediction"] == 1) & (valid["gsb_prediction"] == 0)]
    gsb_caught_we_missed = valid[(valid["true_label"] == 1) & (valid["gsb_prediction"] == 1) & (valid["our_prediction"] == 0)]
    print(f"\nPhishing URLs WE caught but GSB missed: {len(we_caught_gsb_missed)}")
    print(f"Phishing URLs GSB caught but WE missed: {len(gsb_caught_we_missed)}")

    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
