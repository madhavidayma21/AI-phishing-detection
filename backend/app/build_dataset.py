"""
Builds the training dataset by running feature extraction over a
list of labeled URLs (phishing + legitimate) and saving the result
as a CSV ready for model training.

IMPROVED VERSION: writes results incrementally (one row at a time)
instead of holding everything in memory and writing once at the end.
This means a WiFi drop, Ctrl+C, or crash only loses a few seconds of
work, not the whole run. It's also resumable: if --output already
exists, already-processed URLs are skipped automatically.

Usage:
    python app/build_dataset.py --input data/raw/urls.csv --output data/processed/features.csv --workers 8

    If interrupted, just run the exact same command again — it will
    pick up where it left off.
"""
import argparse
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from app.features.extractor import extract_all_features

write_lock = threading.Lock()


def process_row(url: str, label: int, include_content: bool) -> dict:
    try:
        features = extract_all_features(url, include_content=include_content)
        features["url"] = url
        features["label"] = label
        features["extraction_error"] = 0
        return features
    except Exception as e:
        return {"url": url, "label": label, "extraction_error": 1, "error_msg": str(e)}


def load_already_processed(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    try:
        df_existing = pd.read_csv(output_path)
        return set(df_existing["url"].tolist())
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with columns: url,label")
    parser.add_argument("--output", required=True, help="Where to save the feature CSV")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-content", action="store_true", help="Also fetch page content (slower)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows (for testing)")
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    if args.limit:
        df_in = df_in.head(args.limit)

    # Resume support: skip URLs already present in the output file
    already_done = load_already_processed(args.output)
    if already_done:
        print(f"Found existing output with {len(already_done)} rows already processed — resuming, will skip those.")
        df_in = df_in[~df_in["url"].isin(already_done)]

    total = len(df_in)
    if total == 0:
        print("Nothing left to process — output file already has everything from the input.")
        return

    print(f"Processing {total} remaining URLs with {args.workers} workers "
          f"(include_content={args.include_content})...")

    # Determine fieldnames from a single test extraction so the CSV header is consistent
    file_exists = os.path.exists(args.output) and len(already_done) > 0
    fieldnames = None
    if file_exists:
        with open(args.output, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)

    write_mode = "a" if file_exists else "w"
    csv_file = open(args.output, write_mode, newline="", encoding="utf-8")
    writer = None
    if fieldnames:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    done = 0
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_row, row["url"], row["label"], args.include_content): row["url"]
                for _, row in df_in.iterrows()
            }
            for future in as_completed(futures):
                result = future.result()
                with write_lock:
                    if writer is None:
                        fieldnames = list(result.keys())
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                    # Guard against any row missing a key the header expects
                    row_to_write = {k: result.get(k, "") for k in fieldnames}
                    writer.writerow(row_to_write)
                    csv_file.flush()  # force write to disk immediately, don't buffer

                done += 1
                if result.get("extraction_error"):
                    errors += 1
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} processed")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {done}/{total}. Progress is saved — "
              f"just re-run the same command to resume.")
    finally:
        csv_file.close()

    print(f"\nDone. {errors}/{done} rows in this run had extraction errors.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
