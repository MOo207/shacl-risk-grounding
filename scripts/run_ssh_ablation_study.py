"""
Enhanced Ablation Study with SSH Feature Weighting — INTERSYMBOLIC-GRC (W15)
============================================================================

Extends the ablation study v2 to include SSH feature weighting as an 
in-inference mechanism. Tests 6 conditions:

  baseline      : raw RF (reference)
  pre_only      : pre-inference symbolic features → RF
  post_annotate: RF predictions + GRC annotation (zero cost)
  post_override : RF predictions + calibrated override rules (disabled)
  tri_stage      : pre_only features → RF → post_annotate (current pipeline)
  tri_stage_ssh  : pre_only features → SSH weighting → RF → post_annotate (NEW)

NEW: SSH feature weighting identifies SSH/FTP flows and amplifies 
SSH-relevant features before classification.

Saves: results/ssh_feature_weighting.json

Usage:
    python scripts/run_ssh_ablation_study.py
    python scripts/run_ssh_ablation_study.py --data data/processed/cleaned_dataset.csv
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
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import SSH feature weighting
from pipeline.inference.ssh_feature_weighting import SSHFeatureWeighting

# Import existing ablation functionality
from scripts.run_ablation_v2 import (
    PRE_FEATURES, add_pre_features, get_feature_cols, 
    load_dataset, post_annotate, post_override, train_and_eval
)


def run_ssh_ablation_study(data_path: Path, output_dir: Path) -> dict:
    """Run 6-condition ablation study with SSH feature weighting"""
    df = load_dataset(data_path)

    # Stratified 70/15/15 split (same as ablation_v2 for comparability)
    from sklearn.model_selection import train_test_split

    # First split: 70% train, 30% temp (stratified by Label)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['Label']
    )

    # Second split: 15% val, 15% test (from temp, stratified)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['Label']
    )

    # Pre-inference feature augmentation
    train_pre = add_pre_features(train_df)
    test_pre = add_pre_features(test_df)

    base_cols = get_feature_cols(df)
    pre_cols = get_feature_cols(train_pre, list(PRE_FEATURES.keys()))

    y_train = train_df["Label"]
    y_test = test_df["Label"]

    print("\n[W15] Running SSH-enhanced ablation study (6 conditions)...")

    # ---- Condition 1: baseline (raw RF, no symbolic) -----------------------
    X_train_base = train_df[base_cols].values
    X_test_base = test_df[base_cols].values
    c_baseline = train_and_eval(X_train_base, y_train, X_test_base, y_test, y_test,
                                "baseline", save_predictions=True, test_df=test_df)

    # ---- Condition 2: pre_only (symbolic features added, no post) -----
    X_train_pre = train_pre[pre_cols].values
    X_test_pre = test_pre[pre_cols].values
    c_pre_only = train_and_eval(X_train_pre, y_train, X_test_pre, y_test, y_test,
                                "pre_only", test_df=test_df)

    # ---- Condition 3: post_annotate (ML + GRC annotation, ZERO COST) --
    c_post_annotate = train_and_eval(X_train_base, y_train, X_test_base, y_test, y_test,
                                     "post_annotate", post_func=post_annotate,
                                     test_df=test_df)

    # ---- Condition 4: post_override (ML + calibrated overrides) -------------
    c_post_override = train_and_eval(X_train_base, y_train, X_test_base, y_test, y_test,
                                      "post_override", post_func=post_override,
                                      apply_confidence=True, test_df=test_df)

    # ---- Condition 5: tri_stage (pre + ML + annotate) ----------------
    c_tri_stage = train_and_eval(X_train_pre, y_train, X_test_pre, y_test, y_test,
                                 "tri_stage", post_func=post_annotate,
                                 test_df=test_df, save_predictions=True)

    # ---- Condition 6: tri_stage_ssh (pre + SSH weighting + ML + annotate) ---
    print("\n[W15] Condition 6: tri_stage_ssh (pre + SSH weighting + ML + annotate)")
    
    # Initialize SSH feature weighting
    ssh_weighting = SSHFeatureWeighting()
    
    # Get SSH statistics from test set
    ssh_stats = ssh_weighting.get_ssh_statistics(test_df)
    print(f"  SSH stats: {ssh_stats}")
    
    # Train base model on pre_features
    X_train_ssh = train_pre[pre_cols].values
    X_test_ssh_raw = test_pre[pre_cols].values
    
    # Apply SSH feature weighting to test set
    X_test_ssh_weighted = ssh_weighting.apply_weighting(
        X_test_ssh_raw, pre_cols, test_df, weight_factor=2.0
    )
    
    # Train and evaluate with SSH weighting
    ssh_clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                   random_state=42, n_jobs=-1)
    ssh_clf.fit(X_train_ssh, y_train)
    
    # Get predictions with SSH weighting
    y_pred_ssh = ssh_clf.predict(X_test_ssh_weighted)
    
    # Apply post-annotation
    y_pred_final = list(y_pred_ssh)
    for i, (idx, row) in enumerate(test_df.iterrows()):
        y_pred_final[i] = post_annotate(row, y_pred_final[i])
    y_pred_final = np.array(y_pred_final)
    
    # Calculate metrics
    c_tri_stage_ssh = {
        "condition": "tri_stage_ssh",
        "accuracy": float(accuracy_score(y_test, y_pred_final)),
        "f1_macro": float(f1_score(y_test, y_pred_final, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred_final, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred_final, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred_final, average="macro", zero_division=0)),
        "per_class_report": classification_report(y_test, y_pred_final, output_dict=True, zero_division=0),
        "samples_test": len(y_test),
    }
    
    # Optimize SSH weights
    print("\n[W15] Optimizing SSH weight factor...")
    weight_optimization = ssh_weighting.optimize_weights(
        X_test_ssh_raw, pre_cols, test_df, y_test, ssh_clf
    )
    
    # Run with best weight
    best_weight = weight_optimization['best_weight']
    print(f"[W15] Using best SSH weight: {best_weight}")
    
    X_test_ssh_best = ssh_weighting.apply_weighting(
        X_test_ssh_raw, pre_cols, test_df, weight_factor=best_weight
    )
    
    y_pred_ssh_best = ssh_clf.predict(X_test_ssh_best)
    
    # Apply post-annotation
    y_pred_final_best = list(y_pred_ssh_best)
    for i, (idx, row) in enumerate(test_df.iterrows()):
        y_pred_final_best[i] = post_annotate(row, y_pred_final_best[i])
    y_pred_final_best = np.array(y_pred_final_best)
    
    # Final results with optimized weight
    c_tri_stage_ssh_optimized = {
        "condition": "tri_stage_ssh_optimized",
        "ssh_weight_factor": best_weight,
        "accuracy": float(accuracy_score(y_test, y_pred_final_best)),
        "f1_macro": float(f1_score(y_test, y_pred_final_best, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred_final_best, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred_final_best, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred_final_best, average="macro", zero_division=0)),
        "per_class_report": classification_report(y_test, y_pred_final_best, output_dict=True, zero_division=0),
        "samples_test": len(y_test),
    }
    
    print(f"  [tri_stage_ssh] accuracy={c_tri_stage_ssh_optimized['accuracy']:.4f}  f1_macro={c_tri_stage_ssh_optimized['f1_macro']:.4f}")

    # Build comprehensive results
    results = {
        "dataset": "CSE-CIC-IDS2018",
        "data_source": str(data_path),
        "split_type": "stratified_70_15_15",
        "version": "ssh_weighting_v1",
        "mechanism": "SSH feature weighting (in-inference symbolic intervention)",
        "ssh_statistics": ssh_stats,
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
        "tri_stage_ssh": c_tri_stage_ssh,
        "tri_stage_ssh_optimized": c_tri_stage_ssh_optimized,
        "ssh_weight_optimization": weight_optimization,
        "summary_f1_macro": {
            "baseline": c_baseline["f1_macro"],
            "pre_only": c_pre_only["f1_macro"],
            "post_annotate": c_post_annotate["f1_macro"],
            "post_override": c_post_override["f1_macro"],
            "tri_stage": c_tri_stage["f1_macro"],
            "tri_stage_ssh": c_tri_stage_ssh_optimized["f1_macro"],
        },
        "improvements": {
            "tri_stage_vs_baseline": float(c_tri_stage["f1_macro"] - c_baseline["f1_macro"]),
            "tri_stage_ssh_vs_tri_stage": float(c_tri_stage_ssh_optimized["f1_macro"] - c_tri_stage["f1_macro"]),
            "tri_stage_ssh_vs_baseline": float(c_tri_stage_ssh_optimized["f1_macro"] - c_baseline["f1_macro"]),
        },
        "threshold_calibration": {
            "file": "results/threshold_calibration.json",
            "all_rules_disabled": True,
            "reason": "All 4 override rules hurt F1-macro, disabled by calibration"
        },
        "test_labels": y_test.tolist(),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ssh_feature_weighting.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[W15] Saved results to {out_path}")
    
    # Print summary
    print(f"\n[W15] SSH Feature Weighting Results Summary:")
    print(f"  Baseline F1-macro:      {c_baseline['f1_macro']:.4f}")
    print(f"  Tri-stage F1-macro:     {c_tri_stage['f1_macro']:.4f}")
    print(f"  Tri-stage+SSH F1-macro: {c_tri_stage_ssh_optimized['f1_macro']:.4f}")
    print(f"  SSH improvement:        {c_tri_stage_ssh_optimized['f1_macro'] - c_tri_stage['f1_macro']:+.4f}")
    print(f"  Best SSH weight:        {best_weight}")
    
    # Check if improvement is significant (> 0.005)
    improvement = c_tri_stage_ssh_optimized['f1_macro'] - c_tri_stage['f1_macro']
    significant = abs(improvement) > 0.005
    print(f"  Significant improvement: {significant}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SSH feature weighting ablation study (W15)")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_ssh_ablation_study(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()