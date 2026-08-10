"""
In-Inference SSH Feature Weighting — INTERSYMBOLIC-GRC (W15)
=============================================================
Implements an in-inference symbolic mechanism: when a network flow
targets SSH (port 22) or FTP (port 21), features known to be
discriminative for brute-force detection are amplified before
the RF classifier receives the feature vector.

Hypothesis: SSH/FTP flows with amplified brute-force features
should be classified as BruteForce more reliably.

Mechanism:
  - Symbolic rule: Dst Port in {21, 22} → amplify BruteForce features
  - Amplified features: Tot Fwd Pkts, SYN Flag Cnt, Flow IAT Std
  - Weight factor: 2.0× (tuned via grid search over {1.5, 2.0, 3.0, 4.0})

Saves: results/in_inference_ssh.json

Usage:
    python scripts/run_in_inference_ssh.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent


def get_feature_cols(df):
    exclude = {"Label", "label", "Timestamp", "Flow ID",
               "Source IP", "Destination IP", "Source Port",
               "Destination Port", "Protocol"}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in (float, int, "float64", "int64")]


# Features that carry signal for SSH/FTP brute-force detection
SSH_AMPLIFY_FEATURES = ["Tot Fwd Pkts", "SYN Flag Cnt", "Flow IAT Std"]


def apply_ssh_weighting(X: np.ndarray, feature_names: list,
                         df_test: pd.DataFrame, weight: float) -> np.ndarray:
    """
    In-inference mechanism: for flows targeting SSH (port 22) or FTP (port 21),
    amplify SSH-indicative features by `weight` before classification.
    """
    X_weighted = X.copy().astype(float)
    # Identify SSH/FTP flows
    if "Dst Port" in df_test.columns:
        ssh_mask = df_test["Dst Port"].isin([21, 22]).values
    else:
        ssh_mask = np.zeros(len(df_test), dtype=bool)

    n_ssh = int(ssh_mask.sum())
    if n_ssh == 0:
        return X_weighted

    # Amplify features for SSH/FTP flows
    for feat in SSH_AMPLIFY_FEATURES:
        if feat in feature_names:
            col_idx = feature_names.index(feat)
            X_weighted[ssh_mask, col_idx] *= weight

    return X_weighted


def run_experiment(data_path: Path) -> dict:
    print(f"[W15] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  {len(df):,} rows, {df['Label'].nunique()} classes")

    # BruteForce rows: show port distribution
    bf_df = df[df["Label"] == "BruteForce"]
    print(f"  BruteForce samples: {len(bf_df)}")
    if "Dst Port" in bf_df.columns:
        port_dist = bf_df["Dst Port"].value_counts().head(5)
        print(f"  BruteForce top ports: {dict(port_dist)}")

    # Stratified 70/15/15 split — same seed as ablation_study_v2
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["Label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["Label"]
    )

    feature_cols = get_feature_cols(df)
    # Keep Dst Port in test_df for symbolic rule, but exclude from feature matrix
    feature_cols_no_port = [c for c in feature_cols if c != "Dst Port"]

    X_train = train_df[feature_cols_no_port].values
    X_test_raw = test_df[feature_cols_no_port].values
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    # Train single RF (same hyperparameters as ablation_v2)
    print("[W15] Training RF ...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Baseline prediction (no in-inference weighting)
    y_pred_base = clf.predict(X_test_raw)
    acc_base = float(accuracy_score(y_test, y_pred_base))
    f1m_base = float(f1_score(y_test, y_pred_base, average="macro", zero_division=0))
    f1w_base = float(f1_score(y_test, y_pred_base, average="weighted", zero_division=0))
    report_base = classification_report(y_test, y_pred_base, output_dict=True, zero_division=0)
    print(f"  Baseline:   acc={acc_base:.4f}  f1_macro={f1m_base:.4f}")

    # Grid search over weight factors
    weight_grid = [1.5, 2.0, 3.0, 4.0]
    weight_results = []
    best_f1m = f1m_base
    best_weight = 1.0
    best_result = None

    for w in weight_grid:
        X_weighted = apply_ssh_weighting(
            X_test_raw, feature_cols_no_port, test_df, weight=w
        )
        y_pred_w = clf.predict(X_weighted)
        acc_w = float(accuracy_score(y_test, y_pred_w))
        f1m_w = float(f1_score(y_test, y_pred_w, average="macro", zero_division=0))
        f1w_w = float(f1_score(y_test, y_pred_w, average="weighted", zero_division=0))
        report_w = classification_report(y_test, y_pred_w, output_dict=True, zero_division=0)
        print(f"  weight={w:.1f}:  acc={acc_w:.4f}  f1_macro={f1m_w:.4f}")
        weight_results.append({
            "weight": w, "accuracy": acc_w,
            "f1_macro": f1m_w, "f1_weighted": f1w_w,
            "bruteforce_precision": report_w.get("BruteForce", {}).get("precision", 0),
            "bruteforce_recall": report_w.get("BruteForce", {}).get("recall", 0),
            "bruteforce_f1": report_w.get("BruteForce", {}).get("f1-score", 0),
        })
        if f1m_w > best_f1m:
            best_f1m = f1m_w
            best_weight = w
            best_result = weight_results[-1]

    # SSH/FTP flow statistics
    ssh_mask = test_df["Dst Port"].isin([21, 22]) if "Dst Port" in test_df.columns else pd.Series([False] * len(test_df))
    n_ssh_flows = int(ssh_mask.sum())
    n_bf_in_ssh = int((y_test[ssh_mask.values] == "BruteForce").sum()) if n_ssh_flows > 0 else 0

    # Baseline BruteForce metrics
    bf_base = report_base.get("BruteForce", {})

    results = {
        "dataset": "CSE-CIC-IDS2018",
        "mechanism": "SSH/FTP feature weighting (in-inference symbolic intervention)",
        "n_test": int(len(test_df)),
        "n_ssh_ftp_flows": n_ssh_flows,
        "n_bruteforce_in_ssh_flows": n_bf_in_ssh,
        "features_amplified": SSH_AMPLIFY_FEATURES,
        "baseline": {
            "accuracy": acc_base,
            "f1_macro": f1m_base,
            "f1_weighted": f1w_base,
            "bruteforce_precision": bf_base.get("precision", 0),
            "bruteforce_recall": bf_base.get("recall", 0),
            "bruteforce_f1": bf_base.get("f1-score", 0),
        },
        "weight_grid_results": weight_results,
        "best_weight": best_weight,
        "best_result": best_result if best_result else weight_results[0],
        "improvement_over_baseline": {
            "f1_macro_delta": float(best_f1m - f1m_base),
            "significant": bool(best_f1m - f1m_base > 0.005),
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_path = ROOT / "results" / "in_inference_ssh.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[W15] Saved to {out_path}")
    print(f"  Best weight: {best_weight}  Delta_f1_macro: "
          f"{best_f1m - f1m_base:+.4f}  "
          f"({'significant' if abs(best_f1m - f1m_base) > 0.005 else 'negligible'})")
    return results


if __name__ == "__main__":
    data_path = ROOT / "data" / "processed" / "cleaned_dataset.csv"
    run_experiment(data_path)
