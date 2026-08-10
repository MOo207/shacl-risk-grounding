"""
Rule Baseline Runner — CSV-backed wrapper for T-A4
===================================================
Loads data/processed/cleaned_dataset.csv (produced by T-A1),
applies the Sigma-style threshold rules from
baseline/ml_only/sigma_rules_baseline.py to each flow,
and saves canonical results to results/rule_baseline.json.

Metrics:
  - coverage      : fraction of flows where at least one rule fires
  - precision     : of covered flows, how many are correctly labelled
  - recall        : of all attack flows, how many are covered
  - f1_macro      : macro-F1 on label predictions (uncovered → "Benign")

NOTE: baseline/ml_only/sigma_rules_baseline.py is the PostgreSQL-backed
      implementation.  The rule conditions are replicated here verbatim
      so we can run against the CSV without a database.

Usage:
    python scripts/run_rule_baseline.py
    python scripts/run_rule_baseline.py --data data/processed/cleaned_dataset.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Sigma rules (verbatim conditions from sigma_rules_baseline.py — no DB needed)
# ---------------------------------------------------------------------------
SIGMA_RULES = [
    {
        "rule_id": "SIG-001",
        "attack_type": "BruteForce",
        "conditions": [
            ("Total Fwd Packets", ">", 100),
            ("SYN Flag Count",    ">", 10),
            ("Flow Duration",     "<", 60),
        ],
    },
    {
        "rule_id": "SIG-002",
        "attack_type": "DoS",
        "conditions": [
            ("Total Fwd Packets",            ">", 1000),
            ("Flow Duration",                "<", 30),
            ("Flow Packets/s",               ">", 100),
        ],
    },
    {
        "rule_id": "SIG-002b",
        "attack_type": "DDoS",
        "conditions": [
            ("Total Fwd Packets",            ">", 500),
            ("Flow Duration",                "<", 60),
            ("Flow Packets/s",               ">", 50),
        ],
    },
    {
        "rule_id": "SIG-003",
        "attack_type": "WebAttack",
        "conditions": [
            ("Flow Packets/s", ">", 50),
            ("PSH Flag Count",  ">", 5),
        ],
    },
    {
        "rule_id": "SIG-004",
        "attack_type": "Infiltration",
        "conditions": [
            ("Total Length of Fwd Packets", ">", 50000),
            ("ACK Flag Count",              ">", 20),
            ("Flow Duration",               ">", 300),
        ],
    },
    {
        "rule_id": "SIG-005",
        "attack_type": "Bot",
        "conditions": [
            ("Flow IAT Std",   "<", 10),
            ("Flow Duration",  "<", 120),
        ],
    },
]

OPS = {">": float.__gt__, "<": float.__lt__, ">=": float.__ge__,
       "<=": float.__le__, "==": float.__eq__}


def match_row(row: pd.Series) -> Optional[str]:
    """Return the first matching attack_type, or None if no rule fires."""
    for rule in SIGMA_RULES:
        ok = True
        for col, op, val in rule["conditions"]:
            try:
                cell = float(row.get(col, np.nan))
                if np.isnan(cell) or not OPS[op](cell, val):
                    ok = False
                    break
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            return rule["attack_type"]
    return None


def run_rule_baseline(data_path: Path, output_dir: Path) -> dict:
    print(f"[T-A4] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # Use test portion only (last 15%)
    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].copy()
    print(f"[T-A4] Evaluating on test split: {len(test_df):,} rows")

    print("[T-A4] Applying Sigma rules ...")
    test_df["pred"] = test_df.apply(match_row, axis=1)
    covered_mask = test_df["pred"].notna()
    test_df["pred_label"] = test_df["pred"].fillna("Normal")

    # Handle both 'Label' and 'label' column names
    label_col = 'Label' if 'Label' in test_df.columns else 'label'
    y_true = test_df[label_col]
    y_pred = test_df["pred_label"]

    coverage = float(covered_mask.sum() / len(test_df))
    print(f"[T-A4] Coverage: {coverage:.4f}  ({covered_mask.sum():,}/{len(test_df):,} flows covered by rules)")

    # Infer dataset name from data_path
    dataset_name = data_path.stem.replace('_cleaned', '').replace('_dataset', '')

    results = {
        "baseline_type": "SigmaRules",
        "dataset": dataset_name,
        "data_source": str(data_path),
        "n_rules": len(SIGMA_RULES),
        "test": {
            "split": "test",
            "samples": len(test_df),
            "coverage": coverage,
            "covered_flows": int(covered_mask.sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "per_class_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
            # Precision on covered flows only
            "covered_precision_macro": float(
                precision_score(
                    y_true[covered_mask], y_pred[covered_mask],
                    average="macro", zero_division=0
                )
            ) if covered_mask.sum() > 0 else 0.0,
        },
        "timestamp": datetime.now().isoformat(),
    }

    print(f"[T-A4] f1_macro={results['test']['f1_macro']:.4f}  "
          f"precision_macro={results['test']['precision_macro']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_rule_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[T-A4] Saved results to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sigma rule baseline runner (T-A4)")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_rule_baseline(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
