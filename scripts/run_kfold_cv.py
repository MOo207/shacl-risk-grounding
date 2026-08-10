"""
5-Fold Stratified Cross-Validation — RF Baseline and Tri-Stage (W14)
=====================================================================
Evaluates generalisability of results from ablation_study_v2.json
by running k-fold CV instead of a single 70/15/15 split.

Saves: results/kfold_cv.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent.parent

# Pre-inference symbolic features (same as run_ablation_v2.py)
PRE_FEATURES = {
    "sym_high_volume":    lambda r: 1 if r.get("Tot Fwd Pkts", 0) > 1000 else 0,
    "sym_short_duration": lambda r: 1 if r.get("Flow Duration", 0) < 10 else 0,
    "sym_suspicious_iat": lambda r: 1 if r.get("Flow IAT Std", 999) < 5 else 0,
    "sym_syn_flood":      lambda r: 1 if r.get("SYN Flag Cnt", 0) > 50 else 0,
    "sym_large_transfer": lambda r: 1 if r.get("TotLen Fwd Pkts", 0) > 100000 else 0,
}


def add_pre_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, fn in PRE_FEATURES.items():
        df[col] = df.apply(fn, axis=1)
    return df


def get_feature_cols(df, extra_cols=None):
    exclude = {"Label", "label", "Timestamp", "Flow ID",
               "Source IP", "Destination IP", "Source Port",
               "Destination Port", "Protocol"}
    base = [c for c in df.columns if c not in exclude
            and df[c].dtype in (float, int, "float64", "int64")]
    if extra_cols:
        base += [c for c in extra_cols if c in df.columns and c not in base]
    return base


def run_kfold(data_path: Path, k: int = 5, seed: int = 42) -> dict:
    print(f"[W14] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  {len(df):,} rows, {df['Label'].nunique()} classes")

    df_pre = add_pre_features(df)

    base_cols = get_feature_cols(df)
    pre_cols = get_feature_cols(df_pre, list(PRE_FEATURES.keys()))

    X_base = df[base_cols].values
    X_pre = df_pre[pre_cols].values
    y = df["Label"].values

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_base, y)):
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Baseline (RF, no symbolic features) ---
        clf_base = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1
        )
        clf_base.fit(X_base[train_idx], y_train)
        y_pred_base = clf_base.predict(X_base[test_idx])

        acc_base = float(accuracy_score(y_test, y_pred_base))
        f1m_base = float(f1_score(y_test, y_pred_base, average="macro", zero_division=0))
        f1w_base = float(f1_score(y_test, y_pred_base, average="weighted", zero_division=0))

        # --- Tri-stage (pre-inference symbolic features + RF + annotation) ---
        clf_tri = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=seed, n_jobs=-1
        )
        clf_tri.fit(X_pre[train_idx], y_train)
        y_pred_tri = clf_tri.predict(X_pre[test_idx])

        acc_tri = float(accuracy_score(y_test, y_pred_tri))
        f1m_tri = float(f1_score(y_test, y_pred_tri, average="macro", zero_division=0))
        f1w_tri = float(f1_score(y_test, y_pred_tri, average="weighted", zero_division=0))

        fold_results.append({
            "fold": fold_i + 1,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "baseline": {"accuracy": acc_base, "f1_macro": f1m_base, "f1_weighted": f1w_base},
            "tri_stage": {"accuracy": acc_tri, "f1_macro": f1m_tri, "f1_weighted": f1w_tri},
        })
        print(f"  Fold {fold_i+1}: baseline acc={acc_base:.4f} f1m={f1m_base:.4f} "
              f"| tri acc={acc_tri:.4f} f1m={f1m_tri:.4f}")

    # Aggregate
    def agg(cond, metric):
        vals = [f[cond][metric] for f in fold_results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)), "values": vals}

    summary = {
        "dataset": "CSE-CIC-IDS2018",
        "data_source": str(data_path),
        "k_folds": k,
        "random_seed": seed,
        "n_total": int(len(df)),
        "n_classes": int(df["Label"].nunique()),
        "baseline": {
            "accuracy":    agg("baseline", "accuracy"),
            "f1_macro":    agg("baseline", "f1_macro"),
            "f1_weighted": agg("baseline", "f1_weighted"),
        },
        "tri_stage": {
            "accuracy":    agg("tri_stage", "accuracy"),
            "f1_macro":    agg("tri_stage", "f1_macro"),
            "f1_weighted": agg("tri_stage", "f1_weighted"),
        },
        "fold_results": fold_results,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = ROOT / "results" / "kfold_cv.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[W14] Saved to {out_path}")
    print(f"  Baseline:  acc={summary['baseline']['accuracy']['mean']:.4f}±"
          f"{summary['baseline']['accuracy']['std']:.4f}  "
          f"f1m={summary['baseline']['f1_macro']['mean']:.4f}±"
          f"{summary['baseline']['f1_macro']['std']:.4f}")
    print(f"  Tri-stage: acc={summary['tri_stage']['accuracy']['mean']:.4f}±"
          f"{summary['tri_stage']['accuracy']['std']:.4f}  "
          f"f1m={summary['tri_stage']['f1_macro']['mean']:.4f}±"
          f"{summary['tri_stage']['f1_macro']['std']:.4f}")
    return summary


if __name__ == "__main__":
    data_path = ROOT / "data" / "processed" / "cleaned_dataset.csv"
    run_kfold(data_path)
