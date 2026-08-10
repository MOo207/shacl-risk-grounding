#!/usr/bin/env python3
"""
T1 ARG Enrichment — XGBoost with ARG-derived Semantic Features
===============================================================
Adds 2 ARG-derived feature columns to XGBoost by mapping Dst Port to
ARG asset criticality score and max CVSS, reflecting the structured
knowledge in the Asset Relationship Graph.

NOTE: CIC-IDS2018 CICFlowMeter CSVs do NOT contain Src IP / Dst IP columns.
The publicly available version strips them. Port-to-ARG mapping is used
instead: Dst Port -> known service -> ARG criticality + CVSS.

Uses data/processed/cleaned_dataset.csv (same as XGBoost 95.90% baseline).
Split: chronological 70/15/15 (matching xgb_baseline.json).

Features added:
  - dst_criticality  : criticality_score of asset(s) running service on Dst Port
                       (critical=4, high=3, medium=2, low=1, unknown=0)
  - dst_max_cvss     : max CVSS v3 score of CVEs associated with service on Dst Port
                       (0.0 if port not in ARG)

Saves: results/t1_arg_enrichment_results.json

Usage:
    python scripts/run_t1_arg_enrichment.py
"""

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# XGBoost hyperparameters (matching xgb_baseline.json)
XGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

CRITICALITY_MAP = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

# Columns to exclude from features
EXCLUDE_COLS = {"Label", "label", "Timestamp", "Flow ID"}


# ---------------------------------------------------------------------------
# Build ARG port-to-features lookup
# ---------------------------------------------------------------------------
def build_port_lookup(arg_path: Path) -> dict:
    """
    Build {dst_port: {'criticality_score': int, 'max_cvss': float}} from ARG.
    """
    with open(arg_path) as f:
        arg = json.load(f)

    nodes = arg['nodes']
    edges = arg['edges']

    node_by_id = {n['id']: n for n in nodes}

    # Map Asset node id -> max_cvss from EXPOSED_TO_CVE edges
    asset_max_cvss = {}
    for e in edges:
        if e['relation'] == 'EXPOSED_TO_CVE':
            srv_id = e['source']
            cve_id = e['target']
            cve_node = node_by_id.get(cve_id, {})
            cvss = cve_node.get('cvss_v3', 0.0) or 0.0
            asset_max_cvss[srv_id] = max(asset_max_cvss.get(srv_id, 0.0), cvss)

    # Hostname -> standard port mapping (from CIC-IDS2018 testbed knowledge)
    HOSTNAME_PORT = {
        'ftp-srv-01': [21],
        'ftp2-srv-01': [21],
        'ssh-srv-01': [22],
        'telnet-srv-01': [23],
        'smtp-srv-01': [25],
        'dns-srv-01': [53],
        'http-srv-01': [80, 8080],
        'php-srv-01': [80],
        'dvwa-srv-01': [80],
        'mutillidae-srv-01': [80],
        'phpmyadmin-srv-01': [80],
        'twiki-srv-01': [80],
        'tikiwiki-srv-01': [80],
        'web-dos-target-01': [80],
        'infiltration-victim-01': [80, 443],
        'win-srv-01': [80, 443, 445],
        'win-srv-02': [80, 443, 445],
        'rpcbind-srv-01': [111],
        'smb-srv-01': [445, 139],
        'rservices-srv-01': [512, 513, 514],
        'rmi-srv-01': [1099],
        'backdoor-srv-01': [1524, 6200],
        'nfs-srv-01': [2049],
        'mysql-srv-01': [3306],
        'distcc-srv-01': [3632],
        'pgsql-srv-01': [5432],
        'vnc-srv-01': [5900],
        'x11-srv-01': [6000],
        'irc-srv-01': [6667],
        'tomcat-srv-01': [8080, 8009],
        'drb-srv-01': [9999],
        'heartbleed-srv-01': [443],
        'infiltration-victim-02': [80],
    }

    # Build port -> {criticality_score, max_cvss} (take MAX when port is shared)
    port_lookup = {}
    assets = [n for n in nodes if n.get('type') == 'Asset']
    for a in assets:
        hostname = a.get('hostname', '')
        ports = HOSTNAME_PORT.get(hostname, [])
        crit_str = a.get('criticality', 'unknown').lower()
        crit_score = CRITICALITY_MAP.get(crit_str, 0)
        max_cvss = asset_max_cvss.get(a['id'], 0.0)

        for port in ports:
            existing = port_lookup.get(port, {'criticality_score': 0, 'max_cvss': 0.0})
            port_lookup[port] = {
                'criticality_score': max(existing['criticality_score'], crit_score),
                'max_cvss': max(existing['max_cvss'], max_cvss),
            }

    print(f"[ARG] Port lookup built: {len(port_lookup)} ports with ARG features")
    for port in sorted(port_lookup.keys()):
        v = port_lookup[port]
        print(f"  Port {port:5d}: criticality_score={v['criticality_score']}, max_cvss={v['max_cvss']}")

    return port_lookup


# ---------------------------------------------------------------------------
# ARG feature enrichment
# ---------------------------------------------------------------------------
def enrich_with_arg(df: pd.DataFrame, port_lookup: dict) -> pd.DataFrame:
    """Add dst_criticality and dst_max_cvss columns from port_lookup."""
    df = df.copy()
    dst_ports = pd.to_numeric(df['Dst Port'], errors='coerce').fillna(0).astype(int)

    df['dst_criticality'] = dst_ports.map(
        lambda p: port_lookup.get(p, {}).get('criticality_score', 0)
    ).values
    df['dst_max_cvss'] = dst_ports.map(
        lambda p: port_lookup.get(p, {}).get('max_cvss', 0.0)
    ).values
    return df


def compute_coverage(df: pd.DataFrame) -> float:
    """Fraction of rows that have non-zero ARG features."""
    has_arg = (df['dst_criticality'] > 0) | (df['dst_max_cvss'] > 0)
    return float(has_arg.sum() / len(df)) if len(df) > 0 else 0.0


# ---------------------------------------------------------------------------
# McNemar test (continuity-corrected)
# ---------------------------------------------------------------------------
def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar test comparing two classifiers.
    b = samples where a wrong, b correct
    c = samples where a correct, b wrong
    Returns (p_value, b_count, c_count).
    """
    from scipy.stats import chi2 as chi2_dist

    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    b = int(np.sum(~a_correct & b_correct))
    c = int(np.sum(a_correct & ~b_correct))
    n = b + c

    if n == 0:
        return 1.0, b, c

    # Continuity-corrected chi-square
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1.0 - chi2_dist.cdf(chi2, df=1)
    return float(p), b, c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_path = ROOT / 'data' / 'processed' / 'cleaned_dataset.csv'
    arg_path = ROOT / 'results' / 'multisource_arg.json'
    out_path = ROOT / 'results' / 't1_arg_enrichment_results.json'

    print("=" * 70)
    print("T1 ARG Enrichment -- XGBoost + ARG Semantic Features")
    print("=" * 70)
    print(f"Data : {data_path}")
    print(f"ARG  : {arg_path}")
    print()

    # ---------------------------------------------------------------------------
    # Step 1: Build ARG port lookup
    # ---------------------------------------------------------------------------
    print("[Step 1] Building ARG port lookup...")
    port_lookup = build_port_lookup(arg_path)
    print()

    # ---------------------------------------------------------------------------
    # Step 2: Load cleaned dataset
    # ---------------------------------------------------------------------------
    print("[Step 2] Loading cleaned dataset...")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  Total rows: {len(df):,}")
    print(f"  Label distribution: {df['Label'].value_counts().to_dict()}")

    # ---------------------------------------------------------------------------
    # Step 3: Chronological 70/15/15 split (matching xgb_baseline.json)
    # ---------------------------------------------------------------------------
    print("\n[Step 3] Applying chronological 70/15/15 split...")
    n = len(df)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    # Chronological: first 70% train, next 15% val, last 15% test
    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train: n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val:].copy()
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    print(f"  Train labels: {train_df['Label'].value_counts().to_dict()}")
    print(f"  Test labels:  {test_df['Label'].value_counts().to_dict()}")

    # ---------------------------------------------------------------------------
    # Step 4: Enrich with ARG features
    # ---------------------------------------------------------------------------
    print("\n[Step 4] Enriching with ARG features...")
    train_enr = enrich_with_arg(train_df, port_lookup)
    test_enr  = enrich_with_arg(test_df, port_lookup)

    coverage = compute_coverage(test_enr)
    print(f"  ARG coverage on test set: {coverage:.2%} of flows have non-zero ARG features")
    print(f"  Train dst_criticality dist: {train_enr['dst_criticality'].value_counts().to_dict()}")
    print(f"  Test  dst_criticality dist: {test_enr['dst_criticality'].value_counts().to_dict()}")

    # Coverage by class in test set
    print("  ARG coverage by class:")
    for cls in sorted(test_enr['Label'].unique()):
        sub = test_enr[test_enr['Label'] == cls]
        cov = ((sub['dst_criticality'] > 0) | (sub['dst_max_cvss'] > 0)).mean()
        print(f"    {cls:20s}: {cov:.1%} ({len(sub)} samples)")

    # ---------------------------------------------------------------------------
    # Step 5: Feature columns
    # ---------------------------------------------------------------------------
    base_feat_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS
                      and train_df[c].dtype in (float, int, np.float64, np.int64,
                                                 'float64', 'int64')]
    enr_feat_cols = base_feat_cols + ['dst_criticality', 'dst_max_cvss']

    print(f"\n  Base feature count    : {len(base_feat_cols)}")
    print(f"  Enriched feature count: {len(enr_feat_cols)}")

    # Labels
    y_train = train_df['Label'].values
    y_test  = test_df['Label'].values
    all_classes = sorted(set(y_train) | set(y_test))
    train_classes = sorted(set(y_train))
    print(f"  All classes: {all_classes}")
    print(f"  Train classes: {train_classes}")

    # Label encoder fitted on TRAINING classes only (XGBoost needs 0-based contiguous)
    le = LabelEncoder()
    le.fit(train_classes)
    y_train_enc = le.transform(y_train)
    # Map test labels: classes not in train will be marked, predicted as wrong
    y_test_for_eval = y_test  # use raw strings for sklearn metrics

    X_train_base = train_df[base_feat_cols].values.astype(np.float32)
    X_test_base  = test_df[base_feat_cols].values.astype(np.float32)
    X_train_enr  = train_enr[enr_feat_cols].values.astype(np.float32)
    X_test_enr   = test_enr[enr_feat_cols].values.astype(np.float32)

    # ---------------------------------------------------------------------------
    # Step 6: Train BASELINE XGBoost (no ARG features)
    # ---------------------------------------------------------------------------
    print("\n[Step 6] Training BASELINE XGBoost (no ARG features)...")
    clf_base = xgb.XGBClassifier(**XGB_PARAMS)
    clf_base.fit(X_train_base, y_train_enc)
    y_pred_base_enc = clf_base.predict(X_test_base)
    y_pred_base = le.inverse_transform(y_pred_base_enc)

    acc_base = float(accuracy_score(y_test_for_eval, y_pred_base))
    f1_base  = float(f1_score(y_test_for_eval, y_pred_base, average='macro', zero_division=0))
    report_base = classification_report(y_test_for_eval, y_pred_base,
                                         output_dict=True, zero_division=0)

    print(f"  Baseline accuracy : {acc_base:.4f} ({acc_base*100:.2f}%)")
    print(f"  Baseline F1-macro : {f1_base:.4f}")
    print("  Per-class F1:")
    for cls in all_classes:
        r = report_base.get(cls, {})
        print(f"    {cls:20s}: F1={r.get('f1-score', 0):.4f}, "
              f"support={int(r.get('support', 0))}")
    print()
    gc.collect()

    # ---------------------------------------------------------------------------
    # Step 7: Train ENRICHED XGBoost (+ dst_criticality, dst_max_cvss)
    # ---------------------------------------------------------------------------
    print("[Step 7] Training ENRICHED XGBoost (+ dst_criticality, dst_max_cvss)...")
    clf_enr = xgb.XGBClassifier(**XGB_PARAMS)
    clf_enr.fit(X_train_enr, y_train_enc)
    y_pred_enr_enc = clf_enr.predict(X_test_enr)
    y_pred_enr = le.inverse_transform(y_pred_enr_enc)

    acc_enr = float(accuracy_score(y_test_for_eval, y_pred_enr))
    f1_enr  = float(f1_score(y_test_for_eval, y_pred_enr, average='macro', zero_division=0))
    report_enr = classification_report(y_test_for_eval, y_pred_enr,
                                        output_dict=True, zero_division=0)

    print(f"  Enriched accuracy : {acc_enr:.4f} ({acc_enr*100:.2f}%)")
    print(f"  Enriched F1-macro : {f1_enr:.4f}")
    print("  Per-class F1:")
    for cls in all_classes:
        r_b = report_base.get(cls, {})
        r_e = report_enr.get(cls, {})
        delta = float(r_e.get('f1-score', 0)) - float(r_b.get('f1-score', 0))
        print(f"    {cls:20s}: {r_b.get('f1-score', 0):.4f} -> "
              f"{r_e.get('f1-score', 0):.4f} ({delta:+.4f}), "
              f"support={int(r_e.get('support', 0))}")
    print()
    gc.collect()

    # ---------------------------------------------------------------------------
    # Step 8: McNemar test
    # ---------------------------------------------------------------------------
    print("[Step 8] McNemar test (enriched vs baseline)...")
    p_val, b_count, c_count = mcnemar_test(y_test_for_eval, y_pred_base, y_pred_enr)
    delta_acc = acc_enr - acc_base
    delta_f1  = f1_enr - f1_base

    print(f"  b (baseline wrong, enriched correct): {b_count}")
    print(f"  c (baseline correct, enriched wrong): {c_count}")
    print(f"  McNemar p-value                     : {p_val:.6f}")
    print(f"  Accuracy delta                      : {delta_acc:+.6f} ({delta_acc*100:+.4f}pp)")
    print(f"  F1-macro delta                      : {delta_f1:+.6f}")
    print()

    # ---------------------------------------------------------------------------
    # Step 9: Feature importance of ARG features
    # ---------------------------------------------------------------------------
    print("[Step 9] Feature importance analysis...")
    fi = clf_enr.feature_importances_
    fi_dict = dict(zip(enr_feat_cols, fi.tolist()))
    arg_fi = {k: fi_dict.get(k, 0.0) for k in ['dst_criticality', 'dst_max_cvss']}
    total_fi = float(sum(fi))
    arg_fi_pct = sum(arg_fi.values()) / total_fi * 100 if total_fi > 0 else 0.0

    print(f"  dst_criticality importance : {arg_fi['dst_criticality']:.6f}")
    print(f"  dst_max_cvss importance    : {arg_fi['dst_max_cvss']:.6f}")
    print(f"  Combined ARG importance    : {arg_fi_pct:.2f}% of total")

    top_features = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    print("  Top 15 features by importance:")
    for feat, imp in top_features:
        marker = " <-- ARG" if feat in ('dst_criticality', 'dst_max_cvss') else ""
        print(f"    {feat:40s}: {imp:.6f}{marker}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "T1 ARG Enrichment -- XGBoost + dst_criticality + dst_max_cvss",
        "data_split": {
            "source": "data/processed/cleaned_dataset.csv",
            "split_type": "chronological_70_15_15",
            "train_n": int(len(train_df)),
            "val_n":   int(len(val_df)),
            "test_n":  int(len(test_df)),
            "train_classes": train_classes,
            "all_classes": all_classes,
        },
        "arg_enrichment": {
            "port_lookup_size": len(port_lookup),
            "features_added": ["dst_criticality", "dst_max_cvss"],
            "feature_source": "Dst Port -> ARG Asset (criticality score, max CVSSv3)",
            "note": (
                "CIC-IDS2018 CSVs have no Src/Dst IP columns (stripped by CICFlowMeter). "
                "Port-to-ARG mapping used: each Dst Port is mapped to the ARG asset(s) "
                "that run the corresponding service, taking MAX criticality/CVSS when multiple assets share a port."
            ),
        },
        "baseline_accuracy":   round(acc_base, 6),
        "enriched_accuracy":   round(acc_enr, 6),
        "delta":               round(delta_acc, 6),
        "f1_macro_baseline":   round(f1_base, 6),
        "f1_macro_enriched":   round(f1_enr, 6),
        "f1_macro_delta":      round(delta_f1, 6),
        "mcnemar_p":           round(p_val, 6),
        "mcnemar_b":           b_count,
        "mcnemar_c":           c_count,
        "arg_coverage_pct":    round(coverage * 100, 2),
        "per_class_f1_baseline": {
            cls: round(float(report_base.get(cls, {}).get('f1-score', 0)), 4)
            for cls in all_classes
        },
        "per_class_f1_enriched": {
            cls: round(float(report_enr.get(cls, {}).get('f1-score', 0)), 4)
            for cls in all_classes
        },
        "arg_feature_importance": {
            "dst_criticality":       float(arg_fi['dst_criticality']),
            "dst_max_cvss":          float(arg_fi['dst_max_cvss']),
            "combined_pct_of_total": round(arg_fi_pct, 4),
        },
        "top15_features": [
            {"feature": feat, "importance": round(float(imp), 6),
             "is_arg": feat in ('dst_criticality', 'dst_max_cvss')}
            for feat, imp in top_features
        ],
        "xgb_params": XGB_PARAMS,
        "interpretation": (
            f"Enriched XGBoost: {acc_enr*100:.2f}% vs baseline {acc_base*100:.2f}% "
            f"(delta={delta_acc*100:+.2f}pp). "
            f"F1-macro: {f1_enr:.4f} vs {f1_base:.4f} ({delta_f1:+.4f}). "
            f"McNemar p={p_val:.4f} ({'significant p<0.05' if p_val < 0.05 else 'not significant p>0.05'}). "
            f"ARG features covered {coverage*100:.1f}% of test flows. "
            f"Combined ARG feature importance: {arg_fi_pct:.1f}% of total."
        )
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results written to {out_path}")

    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Dataset             : data/processed/cleaned_dataset.csv (N={len(df):,})")
    print(f"  Split               : chronological 70/15/15")
    print(f"  Test N              : {len(test_df):,}")
    print(f"  Baseline accuracy   : {acc_base*100:.2f}%")
    print(f"  Enriched accuracy   : {acc_enr*100:.2f}%")
    print(f"  Delta               : {delta_acc*100:+.2f}pp")
    print(f"  F1-macro baseline   : {f1_base:.4f}")
    print(f"  F1-macro enriched   : {f1_enr:.4f}")
    print(f"  F1-macro delta      : {delta_f1:+.4f}")
    print(f"  McNemar p-value     : {p_val:.6f}")
    print(f"  ARG coverage        : {coverage*100:.1f}%")
    print(f"  ARG feature imp.    : {arg_fi_pct:.1f}% combined")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
