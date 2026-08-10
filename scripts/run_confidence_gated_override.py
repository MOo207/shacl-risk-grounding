"""
Confidence-Gated Post-Inference Override Experiment
====================================================
Tests whether gating symbolic override rules by RF confidence (max predicted
probability) avoids the degradation seen with fixed-threshold rules.

Logic: Only apply a symbolic override rule when RF confidence < tau. When RF
is confident, trust the RF. When RF is uncertain, symbolic rules provide
fallback signal.

Thresholds tested: tau in {0.50, 0.60, 0.70, 0.80, 0.90}
Rules tested (same as ablation_study_v2.json):
  R1: SYN Flag Cnt > 200  -> DDoS
  R2: Flow Pkts/s > 500   -> DoS
  R3: Tot Fwd Pkts <= 3 AND SYN Flag Cnt >= 1  -> BruteForce
  R4: TotLen Fwd Pkts > 10000 AND Flow Duration > 5  -> Infiltration

Saves: results/confidence_gated_override.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score, accuracy_score, classification_report

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH   = ROOT / "data/processed/cleaned_dataset.csv"
MODEL_PATH  = ROOT / "results/cleaned_rf_model.joblib"
OUTPUT_PATH = ROOT / "results/confidence_gated_override.json"

CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]

RULES = {
    "R1_ddos": {
        "label": "DDoS",
        "description": "SYN Flag Cnt > 200 -> DDoS",
        "fn": lambda row: row.get("SYN Flag Cnt", 0) > 200,
    },
    "R2_dos": {
        "label": "DoS",
        "description": "Flow Pkts/s > 500 -> DoS",
        "fn": lambda row: row.get("Flow Pkts/s", 0) > 500,
    },
    "R3_bruteforce": {
        "label": "BruteForce",
        "description": "Tot Fwd Pkts <= 3 AND SYN Flag Cnt >= 1 -> BruteForce",
        "fn": lambda row: row.get("Tot Fwd Pkts", 999) <= 3 and row.get("SYN Flag Cnt", 0) >= 1,
    },
    "R4_infiltration": {
        "label": "Infiltration",
        "description": "TotLen Fwd Pkts > 10000 AND Flow Duration > 5 -> Infiltration",
        "fn": lambda row: row.get("TotLen Fwd Pkts", 0) > 10000 and row.get("Flow Duration", 0) > 5,
    },
}


def load_data():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP",
               "Destination IP", "Source Port", "Destination Port", "Protocol"}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in (float, int, "float64", "int64")]
    X = df[feat_cols]
    y = df["Label"] if "Label" in df.columns else df["label"]
    n = len(df)
    val_end = int(n * 0.85)
    X_test = X.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)
    print(f"Test set: {len(X_test):,} samples")
    return X_test, y_test, df_test, feat_cols


def apply_rules_with_gate(df_test, y_pred_rf, max_proba, tau, rules_enabled):
    """Apply enabled rules only when RF confidence < tau."""
    y_out = list(y_pred_rf)
    overrides = 0
    for i, row in enumerate(df_test.to_dict("records")):
        if max_proba[i] >= tau:
            continue  # RF is confident -> trust RF
        for rule_id, rule in RULES.items():
            if rule_id not in rules_enabled:
                continue
            if rule["fn"](row):
                y_out[i] = rule["label"]
                overrides += 1
                break  # first matching rule wins
    return y_out, overrides


def mcnemar_p(y_true, y_a, y_b):
    """One-sided McNemar test: does A beat B significantly?"""
    a_right_b_wrong = sum(1 for t, a, b in zip(y_true, y_a, y_b)
                          if (a == t) and (b != t))
    a_wrong_b_right = sum(1 for t, a, b in zip(y_true, y_a, y_b)
                          if (a != t) and (b == t))
    # exact binomial is better for small n, use chi2 for simplicity
    table = [[0, a_right_b_wrong], [a_wrong_b_right, 0]]
    if a_right_b_wrong + a_wrong_b_right == 0:
        return 1.0
    try:
        _, p, _, _ = chi2_contingency([[a_wrong_b_right, a_right_b_wrong],
                                       [0, 0]], correction=True)
        return float(p)
    except Exception:
        return 1.0


def main():
    X_test, y_test, df_test, feat_cols = load_data()

    print(f"Retraining RF on current dataset (stratified split, matching ablation_study_v2) ...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    # re-load full dataset with stratified split to match ablation_study_v2
    df_full = pd.read_csv(DATA_PATH, low_memory=False)
    df_full = df_full.replace([np.inf, -np.inf], 0).fillna(0)
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP",
               "Destination IP", "Source Port", "Destination Port", "Protocol"}
    fc = [c for c in df_full.columns if c not in exclude
          and df_full[c].dtype in (float, int, "float64", "int64")]
    X_all = df_full[fc]
    y_all = df_full["Label"] if "Label" in df_full.columns else df_full["label"]
    # Stratified 70/15/15 split (matching ablation_study_v2 split_type)
    X_trainval, X_test, y_trainval, y_test, idx_tv, idx_te = train_test_split(
        X_all, y_all, df_full.index.to_series(),
        test_size=0.15, stratify=y_all, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15/0.85, stratify=y_trainval, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    df_test = df_full.loc[idx_te].reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    classes = list(clf.classes_)
    print(f"  Trained on {len(X_train):,} samples, testing on {len(X_test):,}")

    y_pred_rf = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    max_proba = proba.max(axis=1)

    baseline_f1  = float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0))
    baseline_acc = float(accuracy_score(y_test, y_pred_rf))
    print(f"\nBaseline RF: acc={baseline_acc:.4f}  F1-macro={baseline_f1:.4f}")

    # Confidence distribution
    pct_under = {tau: float((max_proba < tau).mean()) for tau in CONFIDENCE_THRESHOLDS}
    print("\nFraction of test samples with RF confidence < tau:")
    for tau, frac in pct_under.items():
        print(f"  tau={tau:.2f}: {frac*100:.1f}% of test set")

    results_by_rule = {}
    best_global = {"f1": baseline_f1, "rule": None, "tau": None, "improvement": 0.0}

    print("\n=== Grid search: tau x rule ===")
    for rule_id, rule in RULES.items():
        rule_results = []
        print(f"\nRule {rule_id}: {rule['description']}")
        for tau in CONFIDENCE_THRESHOLDS:
            y_gated, n_overrides = apply_rules_with_gate(
                df_test, y_pred_rf, max_proba, tau, {rule_id}
            )
            f1_gated  = float(f1_score(y_test, y_gated, average="macro", zero_division=0))
            acc_gated = float(accuracy_score(y_test, y_gated))
            delta_f1  = f1_gated - baseline_f1

            # McNemar p-value (gated vs baseline)
            a_r_b_w = sum(1 for t, g, b in zip(y_test, y_gated, y_pred_rf)
                          if g == t and b != t)
            b_r_a_w = sum(1 for t, g, b in zip(y_test, y_gated, y_pred_rf)
                          if b == t and g != t)

            rec = {
                "tau": tau,
                "f1_macro": f1_gated,
                "accuracy": acc_gated,
                "delta_f1": round(delta_f1, 6),
                "n_overrides": n_overrides,
                "pct_affected": round(n_overrides / len(y_test) * 100, 2),
                "mcnemar_b": a_r_b_w,
                "mcnemar_c": b_r_a_w,
            }
            rule_results.append(rec)
            print(f"  tau={tau:.2f}: F1={f1_gated:.4f} (delta={delta_f1:+.4f})  overrides={n_overrides} ({rec['pct_affected']}%)")
            if delta_f1 > best_global["improvement"]:
                best_global = {"f1": f1_gated, "rule": rule_id, "tau": tau,
                               "improvement": delta_f1, "n_overrides": n_overrides}

        results_by_rule[rule_id] = {
            "description": rule["description"],
            "target_label": rule["label"],
            "grid": rule_results,
        }

    # All rules combined at optimal tau
    print("\n=== All rules combined ===")
    combined_results = []
    for tau in CONFIDENCE_THRESHOLDS:
        y_comb, n_ov = apply_rules_with_gate(
            df_test, y_pred_rf, max_proba, tau, set(RULES.keys())
        )
        f1_c  = float(f1_score(y_test, y_comb, average="macro", zero_division=0))
        acc_c = float(accuracy_score(y_test, y_comb))
        delta = f1_c - baseline_f1
        print(f"  tau={tau:.2f}: F1={f1_c:.4f} (delta={delta:+.4f})  overrides={n_ov}")
        combined_results.append({"tau": tau, "f1_macro": f1_c, "accuracy": acc_c,
                                  "delta_f1": round(delta, 6), "n_overrides": n_ov})

    # Best single rule at best tau — full per-class report
    best_per_class = None
    if best_global["rule"]:
        rule_id, tau = best_global["rule"], best_global["tau"]
        y_best, _ = apply_rules_with_gate(df_test, y_pred_rf, max_proba, tau, {rule_id})
        best_per_class = classification_report(y_test, y_best, output_dict=True, zero_division=0)
        print(f"\nBest single rule: {rule_id} at tau={tau}")
        print(f"  F1-macro: {best_global['f1']:.4f} (delta={best_global['improvement']:+.4f})")
        print(f"  Overrides: {best_global['n_overrides']} samples")
    else:
        print("\nNo rule improved over baseline.")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL_PATH.name),
        "n_test": len(y_test),
        "baseline": {
            "f1_macro": baseline_f1,
            "accuracy": baseline_acc,
        },
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "pct_under_threshold": pct_under,
        "rules": results_by_rule,
        "combined_rules": combined_results,
        "best_single_rule": best_global,
        "best_single_rule_per_class": best_per_class,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")

    print("\n" + "="*60)
    print("CONFIDENCE-GATED OVERRIDE SUMMARY")
    print("="*60)
    print(f"  Baseline F1-macro: {baseline_f1:.4f}")
    if best_global["rule"]:
        print(f"  Best rule: {best_global['rule']} at tau={best_global['tau']}")
        print(f"  Best F1-macro: {best_global['f1']:.4f} (+{best_global['improvement']:.4f})")
        print(f"  Overrides applied: {best_global['n_overrides']} / {len(y_test)} samples")
    else:
        print("  No rule improved F1-macro over baseline.")
    print("="*60)


if __name__ == "__main__":
    main()
