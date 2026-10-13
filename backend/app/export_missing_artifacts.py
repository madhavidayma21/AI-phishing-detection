"""
One-time fix: your best_model.joblib was already trained and saved,
but the run happened before train_baseline.py was updated to also
save imputation_medians / dropped_features / best_model_name. This
script recomputes just those three small artifacts from your existing
features_full.csv, without retraining anything (fast).

Usage (run from backend/app/):
    python export_missing_artifacts.py --input ../../data/processed/features_full.csv --model-dir ../data/processed
"""
import argparse
import joblib
import pandas as pd

DROPPED_FEATURES = [
    "path_length", "num_underscores", "num_slashes", "num_at_symbols",
    "num_question_marks", "num_equal_signs", "has_https"
]
KNOWN_MISSING_COLS = [
    "domain_age_days", "expiry_days_remaining",
    "ssl_cert_age_days", "ssl_days_until_expiry"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "extraction_error" in df.columns:
        df = df[df["extraction_error"] == 0].copy()

    # Recompute the exact same medians train_baseline.py would have computed
    medians = {}
    for col in KNOWN_MISSING_COLS:
        if col in df.columns:
            median_val = df.loc[df[col] != -1, col].median()
            if pd.isna(median_val):
                median_val = 0.0
            medians[col] = float(median_val)

    joblib.dump(medians, f"{args.model_dir}/imputation_medians.joblib")
    joblib.dump(DROPPED_FEATURES, f"{args.model_dir}/dropped_features.joblib")

    # Figure out which model type was actually saved, from its class name
    model = joblib.load(f"{args.model_dir}/best_model.joblib")
    class_name = type(model).__name__
    name_map = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest",
        "XGBClassifier": "XGBoost",
    }
    best_model_name = name_map.get(class_name, class_name)
    joblib.dump(best_model_name, f"{args.model_dir}/best_model_name.joblib")

    print(f"Detected saved model type: {best_model_name}")
    print(f"Imputation medians: {medians}")
    print(f"Dropped features: {DROPPED_FEATURES}")
    print(f"\nSaved all three missing artifacts to {args.model_dir}/")


if __name__ == "__main__":
    main()
