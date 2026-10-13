"""
Week 3-4: Baseline model training.

Loads the feature CSV built by build_dataset.py, cleans it, trains
three baseline classifiers (Logistic Regression, Random Forest,
XGBoost), and compares them on precision/recall/F1 — with precision
and recall reported separately (not just accuracy) because in
phishing detection the two error types have very different costs:

  - False positive (legit site flagged as phishing) = annoying
  - False negative (phishing site passed as safe)   = dangerous

Usage:
    python train_baseline.py --input ../data/processed/features_full.csv
"""
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


NON_FEATURE_COLS = ["url", "label", "extraction_error", "error_msg"]


def load_and_clean(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Drop rows where feature extraction itself failed entirely (rare, but don't train on garbage)
    if "extraction_error" in df.columns:
        before = len(df)
        df = df[df["extraction_error"] == 0].copy()
        print(f"Dropped {before - len(df)} rows with extraction_error=1")

    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # -1 is used throughout the feature extractors as "unknown" (WHOIS/SSL/content
    # lookup failed). Leaving it as a literal -1 would mislead the model into treating
    # "unknown" as an extreme numeric value. Two-step fix:
    #   1. Add a companion *_missing flag so the model can still learn from "WHOIS was unavailable"
    #      as its own signal (which is itself informative, not noise)
    #   2. Replace -1 with the column median so the raw value doesn't distort splits
    known_missing_cols = [
        "domain_age_days", "expiry_days_remaining",
        "ssl_cert_age_days", "ssl_days_until_expiry"
    ]
    for col in known_missing_cols:
        if col in X.columns:
            X[f"{col}_missing"] = (X[col] == -1).astype(int)
            median_val = X.loc[X[col] != -1, col].median()
            X[col] = X[col].replace(-1, median_val)

    # Any remaining non-numeric columns (shouldn't be any, but guard against it)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"Dropping non-numeric columns not handled above: {non_numeric}")
        X = X.drop(columns=non_numeric)

    X = X.fillna(0)

    return X, y


def evaluate(name: str, y_test, y_pred, y_proba):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted Legit  Predicted Phishing")
    print(f"Actual Legit          {cm[0][0]:<6}          {cm[0][1]:<6}")
    print(f"Actual Phishing       {cm[1][0]:<6}          {cm[1][1]:<6}")

    fn = cm[1][0]  # phishing predicted as legit — the dangerous error
    fp = cm[0][1]  # legit predicted as phishing — the annoying error
    print(f"\n  False Negatives (phishing missed):  {fn}  <- most dangerous error")
    print(f"  False Positives (legit flagged):    {fp}  <- annoying but safe")

    auc = roc_auc_score(y_test, y_proba)
    print(f"  ROC-AUC: {auc:.4f}")

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": auc,
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def diagnose_leakage(X: pd.DataFrame, y: pd.Series, threshold: float = 0.01):
    """
    Flags features that are suspiciously clean separators between classes —
    specifically, features with near-zero variance within one class (a strong
    sign the value is a dataset-collection artifact rather than real signal,
    as we saw with path_length/num_slashes: legit URLs in PhiUSIIL are all
    bare homepages, so anything measuring 'stuff after the domain' leaks
    the label instead of describing phishing behavior).
    """
    print(f"\n{'='*70}\nLEAKAGE DIAGNOSTIC — checking every feature for suspiciously clean splits\n{'='*70}")
    suspects = []
    for col in X.columns:
        if col.endswith("_missing"):
            continue  # these are meant to be near-binary, skip
        std_class0 = X.loc[y == 0, col].std()
        std_class1 = X.loc[y == 1, col].std()
        mean_diff = abs(X.loc[y == 0, col].mean() - X.loc[y == 1, col].mean())
        # Flag: near-zero variance in either class AND a real gap between class means
        if (std_class0 < threshold or std_class1 < threshold) and mean_diff > threshold:
            suspects.append({
                "feature": col,
                "legit_mean": round(X.loc[y == 0, col].mean(), 3),
                "legit_std": round(std_class0, 4),
                "phish_mean": round(X.loc[y == 1, col].mean(), 3),
                "phish_std": round(std_class1, 4),
            })

    if suspects:
        print(f"Found {len(suspects)} suspect feature(s) with near-zero variance in one class:\n")
        print(pd.DataFrame(suspects).to_string(index=False))
        print("\nThese likely reflect how the dataset was COLLECTED (e.g. legit URLs "
              "scraped as bare homepages) rather than genuine phishing behavior. "
              "Consider excluding them with --drop-features, or better, sourcing "
              "legit URLs that include real subpages so the feature reflects reality.")
    else:
        print("No obvious zero-variance-in-one-class artifacts found.")
    print(f"{'='*70}\n")
    return [s["feature"] for s in suspects]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="../data/processed")
    parser.add_argument("--drop-features", nargs="*", default=[],
                         help="Feature columns to exclude, e.g. --drop-features path_length "
                              "(use this to remove features identified as dataset artifacts/leakage)")
    parser.add_argument("--cv-folds", type=int, default=5,
                         help="Number of cross-validation folds to run on each model (0 to skip)")
    parser.add_argument("--use-smote", action="store_true",
                         help="Apply SMOTE oversampling to the training set before fitting")
    parser.add_argument("--tune-xgb", action="store_true",
                         help="Run RandomizedSearchCV hyperparameter tuning on XGBoost (slower)")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    X, y = load_and_clean(args.input)

    if args.drop_features:
        present = [c for c in args.drop_features if c in X.columns]
        missing = [c for c in args.drop_features if c not in X.columns]
        if missing:
            print(f"Warning: these --drop-features weren't found in the data: {missing}")
        if present:
            X = X.drop(columns=present)
            print(f"Dropped features (manual exclusion): {present}")
    print(f"Final feature matrix: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Features used: {list(X.columns)}")

    diagnose_leakage(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")

    if args.use_smote:
        print(f"Applying SMOTE oversampling to training set only "
              f"(test set stays untouched so evaluation reflects real-world data)...")
        before_counts = y_train.value_counts().to_dict()
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        after_counts = pd.Series(y_train).value_counts().to_dict()
        print(f"  Before: {before_counts}  ->  After: {after_counts}")

    results = []

    # --- Logistic Regression (needs scaled features) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_proba = lr.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, y_pred, y_proba))

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, y_pred, y_proba))

    # --- XGBoost (optionally tuned) ---
    if args.tune_xgb:
        print(f"\n{'='*50}\nRunning RandomizedSearchCV on XGBoost (this takes a few minutes)...\n{'='*50}")
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        base_xgb = XGBClassifier(random_state=42, eval_metric="logloss")
        search = RandomizedSearchCV(
            base_xgb, param_distributions=param_dist, n_iter=25,
            scoring="f1", cv=3, random_state=42, n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        xgb = search.best_estimator_
        print(f"Best params found: {search.best_params_}")
        print(f"Best CV F1 during search: {search.best_score_:.4f}")
    else:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", y_test, y_pred, y_proba))

    # --- Cross-validation (on the whole X,y — confirms results aren't a lucky train/test split) ---
    if args.cv_folds > 0:
        print(f"\n{'='*70}")
        print(f"{args.cv_folds}-FOLD CROSS-VALIDATION (F1 score, run on full dataset)")
        print(f"{'='*70}")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

        cv_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            "XGBoost": xgb,
        }
        for name, model in cv_models.items():
            if name == "Logistic Regression":
                # LR needs scaled input — build a quick pipeline-equivalent manually
                X_scaled_full = StandardScaler().fit_transform(X)
                scores = cross_val_score(model, X_scaled_full, y, cv=skf, scoring="f1", n_jobs=-1)
            else:
                scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
            print(f"  {name}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})  "
                  f"[folds: {np.round(scores, 4)}]")

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(results).set_index("model")
    print(summary_df.round(4).to_string())

    # --- Feature importance from the best tree model (helps for your report + explanation layer) ---
    print(f"\n{'='*50}\nTop 10 Most Important Features (XGBoost)\n{'='*50}")
    importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(10).to_string())

    # --- Save the best model (by F1) + scaler + feature list ---
    best_model_name = summary_df["f1"].idxmax()
    print(f"\nBest model by F1 score: {best_model_name}")

    models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
    best_model = models[best_model_name]

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(best_model, f"{args.output_dir}/best_model.joblib")
    joblib.dump(scaler, f"{args.output_dir}/scaler.joblib")  # only actually needed if best is LR
    joblib.dump(list(X.columns), f"{args.output_dir}/feature_columns.joblib")
    print(f"Saved best model to {args.output_dir}/best_model.joblib")


if __name__ == "__main__":
    main()
