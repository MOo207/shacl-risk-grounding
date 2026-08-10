#!/usr/bin/env python3
"""
Ablation Study — Pre / Post / Tri-Stage (T-A5, RQ2) - VERSION 3 (FULL TEST SET)
===============================================================================
Tests 5 conditions using RF classifier on FULL chronological test split (Days 6-7).

KEY CHANGE from V2:
- Uses FULL Days 6-7 (Thursday-22-02-2018 + Friday-23-02-2018) as test set
- Train on Days 1-5, test on Days 6-7 (chronological split, NO downsampling)
- Test set size: ~2,097,150 samples (vs 2,122-4,528 in previous versions)

Saves: results/ablation_study_v3.json

Usage:
    python scripts/run_ablation_v3.py
    python scripts/run_ablation_v3.py --data-dir data/raw/CSE-CIC-IDS2018
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Pre-inference symbolic feature engineering
# ---------------------------------------------------------------------------
PRE_FEATURES = {
    "sym_high_volume":      lambda r: 1 if r.get("Tot Fwd Pkts", 0) > 1000 else 0,
    "sym_short_duration":   lambda r: 1 if r.get("Flow Duration", 0) < 10 else 0,
    "sym_suspicious_iat":   lambda r: 1 if r.get("Flow IAT Std", 999) < 5 else 0,
    "sym_syn_flood":        lambda r: 1 if r.get("SYN Flag Cnt", 0) > 50 else 0,
    "sym_large_transfer":   lambda r: 1 if r.get("TotLen Fwd Pkts", 0) > 100000 else 0,
}


def add_pre_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, fn in PRE_FEATURES.items():
        df[col] = df.apply(fn, axis=1)
    return df


# ---------------------------------------------------------------------------
# Post-inference annotation (ZERO COST)
# ---------------------------------------------------------------------------
try:
    from pipeline.post_inference.grc_annotation_rules import create_grc_annotation_rules
    _ANNOTATION_AVAILABLE = True
except ImportError:
    _ANNOTATION_AVAILABLE = False
    print("[WARN] grc_annotation_rules not available, post_annotate will be ML-only")


def post_annotate(row: pd.Series, pred: str) -> str:
    """Apply GRC annotation to prediction (ZERO COST)"""
    if not _ANNOTATION_AVAILABLE:
        return pred
    
    grc_rules = create_grc_annotation_rules()
    risk_signals = []
    
    syn = row.get("SYN Flag Cnt", 0)
    pps = row.get("Flow Pkts/s", 0)
    tfwd = row.get("Tot Fwd Pkts", 0)
    
    if syn > 50:
        risk_signals.append({"type": "behavioral", "source": "pre_inference", "score": 0.8})
    if pps > 100:
        risk_signals.append({"type": "anomaly", "source": "pre_inference", "score": 0.7})
    if tfwd <= 3 and syn == 1:
        risk_signals.append({"type": "behavioral", "source": "pre_inference", "score": 0.9})
    
    return pred


# ---------------------------------------------------------------------------
# Post-inference override (CALIBRATED - ALL RULES DISABLED)
# ---------------------------------------------------------------------------
def post_override(row: pd.Series, pred: str, conf: float) -> str:
    """Apply CALIBRATED override rules (all disabled per calibration)"""
    return pred


# ---------------------------------------------------------------------------
# Dataset loading - CHRONOLOGICAL SPLIT
# ---------------------------------------------------------------------------
LABEL_MAP = {
    'Benign': 'Benign', 'BENIGN': 'Benign',
    'FTP-BruteForce': 'BruteForce', 'SSH-Bruteforce': 'BruteForce',
    'Brute Force -Web': 'BruteForce', 'Brute Force -XSS': 'BruteForce',
    'DoS attacks-Hulk': 'DoS', 'DoS attacks-SlowHTTPTest': 'DoS',
    'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Slowloris': 'DoS',
    'SQL Injection': 'WebAttack', 'XSS': 'WebAttack',
    'Infiltration': 'Infiltration', 'Infilteration': 'Infiltration',
    'Bot': 'Bot',
    'DDOS attack-HOIC': 'DDoS', 'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
}

TRAIN_DAYS = [
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',  # Day 6 - Infiltration
    'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',    # Day 7 - Bot
]

# For chronological split: train on Days 1-5, test on Days 6-7
# But we only have 6 files downloaded, so:
# Train: Thursday-15, Friday-16, Wednesday-21 (Days 1-3)
# Test: Thursday-22, Friday-23 (Days 6-7)
# Note: Thursday-01 (DDoS) excluded to maintain temporal order

TRAIN_FILES = [
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
]

TEST_FILES = [
    'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
]


def load_and_process_csv(csv_path: Path, chunksize: int = 50000) -> pd.DataFrame:
    """Load CSV in chunks, apply label mapping, clean data"""
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, on_bad_lines='skip'):
        chunk.columns = chunk.columns.str.strip()
        if 'Label' not in chunk.columns:
            continue
        chunk = chunk[chunk['Label'] != 'Label']
        if chunk.empty:
            continue
        
        for col in chunk.columns:
            if col != 'Label':
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk.fillna(0)
        chunk['Label'] = chunk['Label'].str.strip().map(LABEL_MAP).fillna(chunk['Label'].str.strip())
        chunk = chunk.drop_duplicates()
        chunks.append(chunk)
        gc.collect()
    
    if not chunks:
        return pd.DataFrame()
    
    return pd.concat(chunks, ignore_index=True)


def load_train_test(data_dir: Path):
    """Load train (Days 1-3) and test (Days 6-7) splits"""
    print(f"[T-A5 v3] Loading training data from {len(TRAIN_FILES)} files...")
    train_dfs = []
    for fname in TRAIN_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [WARN] {fname} not found, skipping")
            continue
        print(f"  Loading {fname}...")
        df = load_and_process_csv(fpath)
        if len(df) > 0:
            print(f"    {len(df):,} rows")
            train_dfs.append(df)
        gc.collect()
    
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    print(f"  Total train: {len(train_df):,} rows")
    
    print(f"\n[T-A5 v3] Loading test data from {len(TEST_FILES)} files (Days 6-7)...")
    test_dfs = []
    for fname in TEST_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  [WARN] {fname} not found, skipping")
            continue
        print(f"  Loading {fname}...")
        df = load_and_process_csv(fpath)
        if len(df) > 0:
            print(f"    {len(df):,} rows")
            test_dfs.append(df)
        gc.collect()
    
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    print(f"  Total test: {len(test_df):,} rows")
    
    return train_df, test_df


def get_feature_cols(df: pd.DataFrame, extra_cols=None):
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP", "Destination IP",
               "Source Port", "Destination Port", "Protocol"}
    base = [c for c in df.columns if c not in exclude
            and df[c].dtype in (float, int, "float64", "int64")]
    if extra_cols:
        base += [c for c in extra_cols if c in df.columns and c not in base]
    return base


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------
def train_and_eval(X_train, y_train, X_test, y_test, y_test_str,
                   name: str, post_func=None, apply_confidence: bool = False,
                   test_df=None) -> dict:
    """Train RF model and evaluate with optional post-processing"""
    print(f"  Training {name}...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                 random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"  Predicting on {len(y_test):,} test samples...")
    y_pred = clf.predict(X_test)

    if post_func and test_df is not None:
        if apply_confidence:
            y_proba = clf.predict_proba(X_test)
            y_pred_list = list(y_pred)
            for i, (idx, row) in enumerate(test_df.iterrows()):
                conf = max(y_proba[i])
                y_pred_list[i] = post_func(row, y_pred_list[i], conf)
            y_pred = np.array(y_pred_list)
        else:
            y_pred_list = list(y_pred)
            for i, (idx, row) in enumerate(test_df.iterrows()):
                y_pred_list[i] = post_func(row, y_pred_list[i])
            y_pred = np.array(y_pred_list)

    res = {
        "condition": name,
        "accuracy": float(accuracy_score(y_test_str, y_pred)),
        "f1_macro": float(f1_score(y_test_str, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_str, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test_str, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test_str, y_pred, average="macro", zero_division=0)),
        "per_class_report": classification_report(y_test_str, y_pred, output_dict=True, zero_division=0),
        "samples_test": len(y_test_str),
    }
    print(f"  [{name}] accuracy={res['accuracy']:.4f}  f1_macro={res['f1_macro']:.4f}  N={len(y_test_str):,}")
    return res


def run_ablation(data_dir: Path, output_dir: Path) -> dict:
    """Run 5-condition ablation study on full chronological test split"""
    train_df, test_df = load_train_test(data_dir)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("[ERROR] No data loaded. Check that CSV files exist in data_dir.")
        sys.exit(1)
    
    print(f"\n[T-A5 v3] Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    print(f"  Train classes: {train_df['Label'].value_counts().to_dict()}")
    print(f"  Test classes: {test_df['Label'].value_counts().to_dict()}")
    
    # Pre-inference feature augmentation
    print("\n[T-A5 v3] Adding pre-inference symbolic features...")
    train_pre = add_pre_features(train_df)
    test_pre = add_pre_features(test_df)
    
    base_cols = get_feature_cols(train_df)
    pre_cols = get_feature_cols(train_pre, list(PRE_FEATURES.keys()))
    
    y_train = train_df["Label"]
    y_test = test_df["Label"]
    
    print(f"\n[T-A5 v3] Feature columns: {len(base_cols)} base, {len(pre_cols)} with pre-features")
    print(f"  Base features: {base_cols[:5]}... ({len(base_cols)} total)")
    
    print("\n[T-A5 v3] Running ablation study (5 conditions) on FULL test set...")
    
    # Prepare base features
    print("\n  Converting to numpy arrays...")
    X_train_base = train_df[base_cols].values
    X_test_base = test_df[base_cols].values
    X_train_pre = train_pre[pre_cols].values
    X_test_pre = test_pre[pre_cols].values
    
    # ---- Condition 1: baseline (raw RF, no symbolic) -----------------------
    print("\n[T-A5 v3] Condition 1: baseline")
    c_baseline = train_and_eval(X_train_base, y_train, X_test_base, y_test, y_test,
                                "baseline")
    
    # ---- Condition 2: pre_only (symbolic features added, no post) ----------
    print("\n[T-A5 v3] Condition 2: pre_only")
    c_pre_only = train_and_eval(X_train_pre, y_train, X_test_pre, y_test, y_test,
                                "pre_only")
    
    # ---- Condition 3: post_annotate (ML + GRC annotation, ZERO COST) -------
    print("\n[T-A5 v3] Condition 3: post_annotate (annotation only)")
    c_post_annotate = train_and_eval(X_train_base, y_train, X_test_base, y_test, y_test,
                                     "post_annotate", post_func=post_annotate,
                                     test_df=test_df)
    
    # ---- Condition 4: post_override (ML + calibrated overrides) ------------
    print("\n[T-A5 v3] Condition 4: post_override (calibrated)")
    c_post_override = train_and_eval(X_train_base, y_train, X_test_base, y_test, y_test,
                                      "post_override", post_func=post_override,
                                      apply_confidence=True, test_df=test_df)
    
    # ---- Condition 5: tri_stage (pre + ML + annotate) ----------------------
    print("\n[T-A5 v3] Condition 5: tri_stage (pre + ML + annotate)")
    c_tri_stage = train_and_eval(X_train_pre, y_train, X_test_pre, y_test, y_test,
                                 "tri_stage", post_func=post_annotate,
                                 test_df=test_df)
    
    # Build results
    results = {
        "dataset": "CSE-CIC-IDS2018",
        "data_source": str(data_dir),
        "split_type": "chronological_days_6_7",
        "version": "3.0",
        "changes_from_v2": [
            "FULL chronological test split (Days 6-7) instead of stratified sample",
            f"Test set size: {len(test_df):,} samples (vs ~2K-4K in v2)",
            "Train on Days 1-3, test on Days 6-7 (temporal validation)",
        ],
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_files": TRAIN_FILES,
        "test_files": TEST_FILES,
        "pre_symbolic_features": list(PRE_FEATURES.keys()),
        "post_symbolic_rules": [
            "SYN Flag Cnt > threshold → DDoS (DISABLED)",
            "Flow Pkts/s > threshold → DoS (DISABLED)",
            "Tot Fwd Pkts <= threshold AND SYN=1 → BruteForce (DISABLED)",
            "TotLen Fwd Pkts > threshold AND Duration > threshold → Infiltration (DISABLED)"
        ],
        "baseline": c_baseline,
        "pre_only": c_pre_only,
        "post_annotate": c_post_annotate,
        "post_override": c_post_override,
        "tri_stage": c_tri_stage,
        "summary_f1_macro": {
            "baseline": c_baseline["f1_macro"],
            "pre_only": c_pre_only["f1_macro"],
            "post_annotate": c_post_annotate["f1_macro"],
            "post_override": c_post_override["f1_macro"],
            "tri_stage": c_tri_stage["f1_macro"],
        },
        "threshold_calibration": {
            "file": "results/threshold_calibration.json",
            "all_rules_disabled": True,
            "reason": "All 4 override rules hurt F1-macro, disabled by calibration"
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation_study_v3.json"
    print(f"\n[T-A5 v3] Saving results to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[T-A5 v3] Saved results to {out_path}")
    print(f"[T-A5 v3] Summary f1_macro: {results['summary_f1_macro']}")
    print(f"\n[T-A5 v3] COMPLETE - Test set N = {len(test_df):,}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner (T-A5 v3 - FULL test set)")
    parser.add_argument("--data-dir", default="data/raw/CSE-CIC-IDS2018")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    
    run_ablation(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
