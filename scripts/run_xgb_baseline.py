"""
XGBoost Baseline Runner — CSV-backed wrapper for T-A3
======================================================
Loads data/processed/cleaned_dataset.csv (produced by T-A1),
trains XGBoost with the same hyperparameters documented in
baseline/ml_only/xgboost_baseline.py (max_depth=6, lr=0.1, n_estimators=500),
and saves canonical results to results/xgb_baseline.json.

NOTE: baseline/ml_only/xgboost_baseline.py is the PostgreSQL-backed
      implementation.  This wrapper uses identical model configuration
      but reads from the pre-processed CSV.

Requires: pip install xgboost

Usage:
    python scripts/run_xgb_baseline.py
    python scripts/run_xgb_baseline.py --data data/processed/cleaned_dataset.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_dataset(csv_path: Path) -> pd.DataFrame:
    print(f"[T-A3] Loading {csv_path} ...")
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


def run_xgb_baseline(data_path: Path, max_depth: int, learning_rate: float,
                     n_estimators: int, random_state: int, output_dir: Path) -> dict:
    df = load_dataset(data_path)
    X, y, feature_cols = prepare_features(df)

    # Encode labels for XGBoost multi-class
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    X_train, y_train_enc = X.iloc[:train_end].values, y_enc[:train_end]
    X_val,   y_val_enc   = X.iloc[train_end:val_end].values, y_enc[train_end:val_end]
    X_test,  y_test_enc  = X.iloc[val_end:].values, y_enc[val_end:]
    y_val_str  = y.iloc[train_end:val_end]
    y_test_str = y.iloc[val_end:]

    print(f"\n[T-A3] Split sizes — train:{len(X_train):,}  val:{len(X_val):,}  test:{len(X_test):,}")

    dtrain = xgb.DMatrix(X_train, label=y_train_enc)
    dval   = xgb.DMatrix(X_val,   label=y_val_enc)
    dtest  = xgb.DMatrix(X_test,  label=y_test_enc)

    params = {
        "max_depth": max_depth,
        "eta": learning_rate,
        "objective": "multi:softmax",
        "num_class": len(le.classes_),
        "eval_metric": "mlogloss",
        "seed": random_state,
        "tree_method": "hist",
    }

    print(f"[T-A3] Training XGBoost (max_depth={max_depth}, lr={learning_rate}, n={n_estimators}) ...")
    booster = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=20,
        verbose_eval=100,
    )

    def evaluate(dmat, y_true_str, split_name):
        y_pred_enc = booster.predict(dmat).astype(int)
        y_pred_str = le.inverse_transform(y_pred_enc)
        return {
            "split": split_name,
            "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
            "f1_macro": float(f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_true_str, y_pred_str, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true_str, y_pred_str, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true_str, y_pred_str, labels=sorted(y_true_str.unique())).tolist(),
            "per_class_report": classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0),
            "samples": len(y_true_str),
        }

    val_results  = evaluate(dval,  y_val_str,  "val")
    test_results = evaluate(dtest, y_test_str, "test")

    print(f"\n[T-A3] Val  accuracy={val_results['accuracy']:.4f}  f1_macro={val_results['f1_macro']:.4f}")
    print(f"[T-A3] Test accuracy={test_results['accuracy']:.4f}  f1_macro={test_results['f1_macro']:.4f}")

    # Infer dataset name from data_path
    dataset_name = data_path.stem.replace('_cleaned', '').replace('_dataset', '')

    results = {
        "model_type": "XGBoost",
        "dataset": dataset_name,
        "data_source": str(data_path),
        "hyperparameters": {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "random_state": random_state,
        },
        "split_type": "chronological_70_15_15",
        "features_used": len(feature_cols),
        "label_classes": list(le.classes_),
        "val": val_results,
        "test": test_results,
        "timestamp": datetime.now().isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_xgb_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[T-A3] Saved results to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="XGBoost baseline runner (T-A3)")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_xgb_baseline(
        data_path=Path(args.data),
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
