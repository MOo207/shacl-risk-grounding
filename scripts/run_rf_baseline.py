"""
RF Baseline Runner — CSV-backed wrapper for T-A2
=================================================
Loads data/processed/cleaned_dataset.csv (produced by T-A1),
trains a Random Forest with the same hyperparameters documented in
baseline/ml_only/random_forest_baseline.py (n_estimators=100, max_depth=20),
and saves canonical results to results/rf_baseline.json.

NOTE: baseline/ml_only/random_forest_baseline.py is the PostgreSQL-backed
      implementation (requires psycopg2 + live DB).  This wrapper uses the
      identical model configuration but reads from the pre-processed CSV so
      it can run without a database.

Usage:
    python scripts/run_rf_baseline.py
    python scripts/run_rf_baseline.py --data data/processed/cleaned_dataset.csv
    python scripts/run_rf_baseline.py --n-estimators 200 --max-depth 30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_dataset(csv_path: Path) -> pd.DataFrame:
    print(f"[T-A2] Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    # Handle both 'Label' and 'label' column names
    label_col = 'Label' if 'Label' in df.columns else 'label'
    print(f"  Label distribution:\n{df[label_col].value_counts().to_string()}")
    return df


def prepare_features(df: pd.DataFrame):
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP", "Destination IP",
               "Source Port", "Destination Port", "Protocol"}
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in (float, int, "float64", "int64")]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    # Handle both 'Label' and 'label' column names
    label_col = 'Label' if 'Label' in df.columns else 'label'
    y = df[label_col]
    return X, y, feature_cols


def run_rf_baseline(data_path: Path, n_estimators: int, max_depth: int,
                    random_state: int, output_dir: Path,
                    dump_predictions: Path | None = None) -> dict:
    df = load_dataset(data_path)
    X, y, feature_cols = prepare_features(df)

    # Chronological 70/15/15 split (respects temporal ordering)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

    print(f"\n[T-A2] Split sizes — train:{len(X_train):,}  val:{len(X_val):,}  test:{len(X_test):,}")

    print(f"[T-A2] Training RF (n_estimators={n_estimators}, max_depth={max_depth}) ...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    def evaluate(X, y_true, split_name):
        y_pred = clf.predict(X)
        return {
            "split": split_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique())).tolist(),
            "per_class_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            "samples": len(y_true),
        }

    val_results  = evaluate(X_val,  y_val,  "val")
    test_results = evaluate(X_test, y_test, "test")

    # Dump per-row test predictions if requested
    if dump_predictions is not None:
        y_pred_test = clf.predict(X_test)
        dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with dump_predictions.open("w") as _f:
            for _i, (_true, _pred) in enumerate(zip(y_test, y_pred_test)):
                _idx = int(X_test.index[_i])
                _f.write(json.dumps({
                    "flow_id": _idx,
                    "feature_row_index": _idx,
                    "true_label": str(_true),
                    "predicted_label": str(_pred),
                }) + "\n")
        print(f"[T-A2] Dumped {len(y_test)} predictions to {dump_predictions}")

    print(f"\n[T-A2] Val  accuracy={val_results['accuracy']:.4f}  f1_macro={val_results['f1_macro']:.4f}")
    print(f"[T-A2] Test accuracy={test_results['accuracy']:.4f}  f1_macro={test_results['f1_macro']:.4f}")

    # Infer dataset name from data_path
    dataset_name = data_path.stem.replace('_cleaned', '').replace('_dataset', '')

    results = {
        "model_type": "RandomForest",
        "dataset": dataset_name,
        "data_source": str(data_path),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
        "split_type": "chronological_70_15_15",
        "features_used": len(feature_cols),
        "val": val_results,
        "test": test_results,
        "timestamp": datetime.now().isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_rf_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[T-A2] Saved results to {out_path}")

    # Also persist the fitted model for T-A5/T-A6
    import joblib
    model_path = output_dir / f"{dataset_name}_rf_model.joblib"
    joblib.dump({"model": clf, "feature_cols": feature_cols}, model_path)
    print(f"[T-A2] Saved fitted RF model to {model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="RF baseline runner (T-A2)")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--dump-predictions",
        default=None,
        help="If set, write per-row test predictions JSONL to this path",
    )
    args = parser.parse_args()

    run_rf_baseline(
        data_path=Path(args.data),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        output_dir=Path(args.output_dir),
        dump_predictions=Path(args.dump_predictions) if args.dump_predictions else None,
    )


if __name__ == "__main__":
    main()
