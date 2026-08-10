"""
Threshold Calibration for Override Rules (T-1.3)
==============================================
Calibrates post-inference override thresholds on validation set.

For each override rule:
1. Calculate baseline F1-macro (ML only)
2. Test threshold variations (current, ±10%, ±20%, ±30%)
3. Find best threshold that maximizes F1-macro
4. If no threshold improves F1: disable rule (set enabled=False)

Saves: results/threshold_calibration.json

Usage:
    python scripts/calibrate_thresholds.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import override rules
from pipeline.post_inference.grc_override_rules import OverrideRules


def post_override(row: pd.Series, pred: str, thresholds: dict) -> str:
    """Apply override rules with given thresholds."""
    try:
        syn = float(row.get("SYN Flag Cnt", 0))
        pps = float(row.get("Flow Pkts/s", 0))
        tfwd = float(row.get("Tot Fwd Pkts", 0))
        tlen = float(row.get("TotLen Fwd Pkts", 0))
        dur = float(row.get("Flow Duration", 0))
    except (TypeError, ValueError):
        return pred

    # Rule 1: SYN > threshold → DDoS
    if thresholds.get("syn_ddos") is not None and syn > thresholds["syn_ddos"]:
        return "DDoS"

    # Rule 2: Flow Pkts/s > threshold → DoS
    if thresholds.get("flow_dos") is not None and pps > thresholds["flow_dos"]:
        return "DoS"

    # Rule 3: Tot Fwd Pkts <= threshold AND SYN = 1 → BruteForce
    if (thresholds.get("bruteforce_fwd") is not None and
        thresholds.get("bruteforce_syn") is not None):
        if tfwd <= thresholds["bruteforce_fwd"] and syn == thresholds["bruteforce_syn"]:
            return "BruteForce"

    # Rule 4: TotLen > threshold AND Duration > threshold → Infiltration
    if (thresholds.get("infiltration_len") is not None and
        thresholds.get("infiltration_dur") is not None):
        if tlen > thresholds["infiltration_len"] and dur > thresholds["infiltration_dur"]:
            return "Infiltration"

    return pred


def calibrate_rule(
    rule_name: str,
    base_thresholds: dict,
    threshold_variations: list,
    X_val: np.ndarray,
    y_val: pd.Series,
    ml_model: RandomForestClassifier,
    feature_cols: list
) -> dict:
    """
    Calibrate a single rule

    Args:
        rule_name: Name of the rule to calibrate
        base_thresholds: Base threshold values
        threshold_variations: List of (threshold_key, base_value, multipliers) tuples
        X_val: Validation features
        y_val: Validation labels
        ml_model: Trained ML model
        feature_cols: Feature column names

    Returns:
        Calibration results
    """
    print(f"\n[Calibrate] {rule_name}")

    # Get baseline predictions (ML only)
    y_pred_baseline = ml_model.predict(X_val)
    baseline_f1 = f1_score(y_val, y_pred_baseline, average='macro', zero_division=0)

    print(f"  Baseline F1: {baseline_f1:.4f}")

    # Test threshold variations
    best_f1 = baseline_f1
    best_thresholds = {k: base_thresholds.get(k) for k, _, _ in threshold_variations}
    best_enabled = False

    results = []

    # Generate all threshold combinations (simplified: test one rule at a time)
    for threshold_key, base_value, multipliers in threshold_variations:
        print(f"\n  Testing {threshold_key}...")

        for multiplier in multipliers:
            # Calculate new threshold
            new_threshold = base_value * multiplier

            # Create threshold dict with only this rule enabled
            test_thresholds = best_thresholds.copy()
            test_thresholds[threshold_key] = new_threshold

            # Apply override
            df_val = pd.DataFrame(X_val, columns=feature_cols)
            y_pred_override = ml_model.predict(X_val)
            y_pred_override_list = list(y_pred_override)

            for idx, (df_idx, row) in enumerate(df_val.iterrows()):
                y_pred_override_list[idx] = post_override(row, y_pred_override_list[idx], test_thresholds)

            # Calculate F1
            f1 = f1_score(y_val, y_pred_override_list, average='macro', zero_division=0)
            improvement = f1 - baseline_f1

            results.append({
                'threshold_key': threshold_key,
                'multiplier': multiplier,
                'threshold_value': new_threshold,
                'f1': f1,
                'improvement': improvement
            })

            print(f"    Threshold={new_threshold:.1f}  F1={f1:.4f}  Δ={improvement:+.4f}")

            # Track best
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[threshold_key] = new_threshold
                best_enabled = True
                print(f"    ★ NEW BEST!")

    # Build calibration result
    calibration_result = {
        'rule_name': rule_name,
        'description': f"Override rule: {rule_name}",
        'baseline_f1': float(baseline_f1),
        'best_threshold': best_thresholds.get(threshold_variations[0][0]) if best_enabled else None,
        'best_f1': float(best_f1),
        'enabled': best_enabled,
        'improvement': float(best_f1 - baseline_f1),
        'all_results': results
    }

    print(f"\n  Final: {'ENABLED' if best_enabled else 'DISABLED'}")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Improvement: {(best_f1 - baseline_f1):+.4f}")

    return calibration_result


def run_calibration(
    data_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.7
) -> dict:
    """Run full calibration for all rules."""
    print("=" * 70)
    print("THRESHOLD CALIBRATION")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load dataset
    print(f"\n[Load] Loading dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  Loaded {len(df):,} rows")

    # Stratified split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['Label']
    )

    # Split temp into 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['Label']
    )

    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Get feature columns
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP", "Destination IP",
               "Source Port", "Destination Port", "Protocol"}
    feature_cols = [c for c in df.columns if c not in exclude
                   and df[c].dtype in (float, int, "float64", "int64")]

    print(f"  Features: {len(feature_cols)}")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df["Label"]
    X_val = val_df[feature_cols].values
    y_val = val_df["Label"]

    # Train ML model
    print(f"\n[Train] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("  Model trained")

    # Base thresholds from run_ablation.py
    base_thresholds = {
        'syn_ddos': 200.0,
        'flow_dos': 500.0,
        'bruteforce_fwd': 3.0,
        'bruteforce_syn': 1.0,
        'infiltration_len': 200000.0,
        'infiltration_dur': 600.0
    }

    # Define threshold variations to test
    # Format: (threshold_key, base_value, multipliers)
    # Multipliers: 0.7 (30% lower), 0.8, 0.9, 1.0 (current), 1.1, 1.2, 1.3 (30% higher)
    multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    # Calibrate each rule
    all_calibrations = {}

    # Rule 1: SYN > threshold → DDoS
    cal1 = calibrate_rule(
        "rule_1_syn_ddos",
        base_thresholds,
        [('syn_ddos', 200.0, multipliers)],
        X_val, y_val, clf, feature_cols
    )
    all_calibrations['rule_1_syn_ddos'] = cal1

    # Rule 2: Flow Pkts/s > threshold → DoS
    cal2 = calibrate_rule(
        "rule_2_flow_dos",
        base_thresholds,
        [('flow_dos', 500.0, multipliers)],
        X_val, y_val, clf, feature_cols
    )
    all_calibrations['rule_2_flow_dos'] = cal2

    # Rule 3: Tot Fwd Pkts <= threshold AND SYN = 1 → BruteForce
    # Note: Simplified - only test fwd threshold
    cal3 = calibrate_rule(
        "rule_3_bruteforce",
        base_thresholds,
        [('bruteforce_fwd', 3.0, multipliers)],
        X_val, y_val, clf, feature_cols
    )
    all_calibrations['rule_3_bruteforce'] = cal3

    # Rule 4: TotLen > threshold AND Duration > threshold → Infiltration
    # Note: Simplified - only test len threshold
    cal4 = calibrate_rule(
        "rule_4_infiltration",
        base_thresholds,
        [('infiltration_len', 200000.0, multipliers)],
        X_val, y_val, clf, feature_cols
    )
    all_calibrations['rule_4_infiltration'] = cal4

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    enabled_count = sum(1 for c in all_calibrations.values() if c['enabled'])
    print(f"\nRules enabled after calibration: {enabled_count}/{len(all_calibrations)}")

    print("\nPer-rule results:")
    for rule_id, cal in all_calibrations.items():
        status = "✅ ENABLED" if cal['enabled'] else "❌ DISABLED"
        print(f"  {rule_id}: {status}")
        print(f"    Baseline F1: {cal['baseline_f1']:.4f}")
        print(f"    Best F1:     {cal['best_f1']:.4f}")
        print(f"    Improvement:   {cal['improvement']:+.4f}")
        if cal['best_threshold']:
            print(f"    Best threshold: {cal['best_threshold']:.2f}")

    # Build final calibration dict
    calibration_result = {
        'confidence_threshold': confidence_threshold,
        'dataset': str(data_path),
        'baseline_f1_macro': float(f1_score(y_val, clf.predict(X_val), average='macro', zero_division=0)),
        'calibrations': all_calibrations,
        'summary': {
            'total_rules': len(all_calibrations),
            'enabled_rules': enabled_count,
            'disabled_rules': len(all_calibrations) - enabled_count
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "threshold_calibration.json"

    with open(out_path, 'w') as f:
        json.dump(calibration_result, f, indent=2)

    print(f"\n[Save] Saved calibration to {out_path}")

    return calibration_result


def main():
    parser = argparse.ArgumentParser(description="Calibrate override thresholds")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="ML confidence threshold for overrides (default: 0.7)")
    args = parser.parse_args()

    run_calibration(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()
