"""Build NSL-KDD Unified Ablation Test Set (N=50, 10 per class).

NSL-KDD classes (5): Normal, DoS, Probe, R2L, U2R
NFCRM §6.9 mapping:  Benign, DoS, Infiltration, WebAttack, BruteForce

Zero-leakage design: eval drawn from KDDTest+.txt, ML trained on KDDTrain+.txt.

Usage:
    python scripts/build_nslkdd_ablation_testset.py
    python scripts/build_nslkdd_ablation_testset.py --n-per-class 5 --out results/nslkdd_ablation/test_set.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── NSL-KDD column names ───────────────────────────────────────────────────────
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty',
]

ATTACK_MAPPING = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
    'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
}

CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# NSL-KDD class → NFCRM §6.9 analog (for risk level computation)
NSLKDD_TO_NFCRM = {
    "Normal":  "Benign",
    "DoS":     "DoS",
    "Probe":   "Infiltration",
    "R2L":     "WebAttack",
    "U2R":     "BruteForce",
}

CRITICALITIES = ["Low", "Medium", "High", "Critical"]
CRIT_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}

CONTROL_BY_NFCRM = {
    "Benign":       "SS6.1-MON-1",
    "BruteForce":   "SS6.7-AC-1",
    "DoS":          "SS6.7-NET-1",
    "Infiltration": "SS6.5-NET-1",
    "WebAttack":    "SS6.7-APP-1",
}

# CVE pool mapped via NFCRM analog (reuse ARG CVEs)
CVE_BY_NFCRM: dict[str, list[str]] = {
    "Benign":       [],
    "DoS":          ["CVE-2007-6750", "CVE-2009-0696", "CVE-2011-3192"],
    "Infiltration": ["CVE-2011-2523", "CVE-2010-2075", "CVE-2014-0160"],
    "WebAttack":    ["CVE-2012-1823", "CVE-2009-1151", "CVE-2007-5423"],
    "BruteForce":   ["CVE-2008-0166", "CVE-2006-2369"],
}


def load_raw(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, names=COLUMN_NAMES, header=None)
    df['label'] = df['label'].str.strip().str.rstrip('.').str.lower()
    df['label'] = df['label'].map(ATTACK_MAPPING).fillna('Normal')
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)
    df = df[df['label'].isin(CLASSES)].reset_index(drop=True)
    return df


def encode_for_ml(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical columns using combined vocab, split back."""
    CAT_COLS = ['protocol_type', 'service', 'flag']
    combined = pd.concat(
        [train_raw.assign(_split='train'), test_raw.assign(_split='test')],
        ignore_index=True,
    )
    combined_enc = pd.get_dummies(combined, columns=CAT_COLS, drop_first=False)
    # Convert all non-label, non-split columns to numeric
    feature_cols = [c for c in combined_enc.columns if c not in ('label', '_split')]
    combined_enc[feature_cols] = combined_enc[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    train_enc = combined_enc[combined_enc['_split'] == 'train'].drop('_split', axis=1).reset_index(drop=True)
    test_enc  = combined_enc[combined_enc['_split'] == 'test'].drop('_split', axis=1).reset_index(drop=True)
    return train_enc, test_enc


def train_ml_models(train_enc: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [c for c in train_enc.columns if c != 'label']
    X = train_enc[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(train_enc['label'].values)

    print("Training RF on KDDTrain+ ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print(f"  RF train classes: {le.classes_}")

    try:
        from xgboost import XGBClassifier
        print("Training XGBoost on KDDTrain+ ...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss',
                            use_label_encoder=False, verbosity=0)
        xgb.fit(X, y)
        print("  XGBoost done.")
    except ImportError:
        print("  XGBoost not available; using RF as fallback")
        xgb = rf

    return rf, xgb, le, feature_cols


def load_cve_index(arg_path: Path) -> dict[str, dict]:
    arg = json.loads(arg_path.read_text())
    return {n["id"]: n for n in arg["nodes"] if n.get("type") == "CVE"}


def pick_cve(nfcrm_class: str, cve_index: dict, rng: random.Random) -> Optional[dict]:
    candidates = CVE_BY_NFCRM.get(nfcrm_class, [])
    if not candidates:
        return None
    return cve_index.get(rng.choice(candidates))


def build_test_set(seed: int = 42, n_per_class: int = 10) -> dict:
    from run_slm_nslkdd_classification import features_to_nl_nslkdd
    from pipeline.nfcrm.risk_score import compute_risk_score

    rng = random.Random(seed)
    np.random.seed(seed)

    train_path = ROOT / "data" / "raw" / "NSL-KDD" / "KDDTrain+.txt"
    test_path  = ROOT / "data" / "raw" / "NSL-KDD" / "KDDTest+.txt"
    arg_path   = ROOT / "results" / "multisource_arg.json"

    print(f"Loading KDDTrain+ ({train_path}) ...")
    train_raw = load_raw(train_path)
    print(f"  {len(train_raw):,} rows | {train_raw['label'].value_counts().to_dict()}")

    print(f"Loading KDDTest+ ({test_path}) ...")
    test_raw = load_raw(test_path)
    print(f"  {len(test_raw):,} rows | {test_raw['label'].value_counts().to_dict()}")

    # Check U2R count in test (smallest class)
    for cls in CLASSES:
        n = (test_raw['label'] == cls).sum()
        if n < n_per_class:
            raise ValueError(f"Class '{cls}' has only {n} rows in KDDTest+ — cannot sample {n_per_class}. "
                             f"Reduce --n-per-class to {n}.")

    # Encode for ML
    print("Encoding for ML ...")
    train_enc, test_enc = encode_for_ml(train_raw, test_enc_tmp := test_raw.copy())
    # test_enc_tmp was just a copy var, real test_enc is returned from encode_for_ml
    print(f"  Encoded shapes: train={train_enc.shape}, test={test_enc.shape}")

    # Train ML models
    rf_model, xgb_model, le, feature_cols = train_ml_models(train_enc)

    # Sample eval set from KDDTest+ (stratified, 10/class)
    eval_raw_dfs = []
    eval_indices = []
    for cls in CLASSES:
        cls_idxs = test_raw.index[test_raw['label'] == cls].tolist()
        rng_sample = random.Random(seed)
        chosen = rng_sample.sample(cls_idxs, n_per_class)
        eval_raw_dfs.append(test_raw.loc[chosen])
        eval_indices.extend(chosen)

    eval_raw_df = pd.concat(eval_raw_dfs).sample(frac=1, random_state=seed).reset_index()
    eval_orig_indices = eval_raw_df['index'].tolist()
    eval_raw_df = eval_raw_df.drop('index', axis=1)

    # Get encoded rows for these eval cases (same positions in test_enc)
    eval_enc_df = test_enc.loc[eval_orig_indices].reset_index(drop=True)

    # ML predictions on eval
    X_eval = eval_enc_df[feature_cols].values.astype(float)
    rf_preds  = le.inverse_transform(rf_model.predict(X_eval))
    xgb_preds = le.inverse_transform(xgb_model.predict(X_eval))
    rf_proba  = rf_model.predict_proba(X_eval).max(axis=1)
    xgb_proba = xgb_model.predict_proba(X_eval).max(axis=1)

    cve_index = load_cve_index(arg_path)
    print(f"Loaded {len(cve_index)} CVEs from ARG")

    # Build cases
    cases = []
    for i, (_, row) in enumerate(eval_raw_df.iterrows()):
        true_cls  = row['label']
        rf_cls    = rf_preds[i]
        xgb_cls   = xgb_preds[i]
        nfcrm_cls = NSLKDD_TO_NFCRM[true_cls]
        criticality = rng.choice(CRITICALITIES)
        c_val = CRIT_MAP[criticality]

        gt_score = compute_risk_score(nfcrm_cls, c_override=c_val, i_override=c_val, a_override=c_val)
        gt_risk_level = gt_score.level_en if gt_score.level_en != "N/A (non-attack)" else "Very Low"

        nfcrm_rf = NSLKDD_TO_NFCRM.get(rf_cls, "Benign")
        rf_score = compute_risk_score(nfcrm_rf, c_override=c_val, i_override=c_val, a_override=c_val)
        rf_risk_level = rf_score.level_en if rf_score.level_en != "N/A (non-attack)" else "Very Low"

        nfcrm_xgb = NSLKDD_TO_NFCRM.get(xgb_cls, "Benign")
        xgb_score = compute_risk_score(nfcrm_xgb, c_override=c_val, i_override=c_val, a_override=c_val)
        xgb_risk_level = xgb_score.level_en if xgb_score.level_en != "N/A (non-attack)" else "Very Low"

        paired_cve = pick_cve(nfcrm_cls, cve_index, rng)

        nl_desc = features_to_nl_nslkdd(row)

        asset = {
            "id": f"nsl_{i:03d}",
            "hostname": f"host-{i:03d}",
            "criticality": criticality,
            "asset_type": rng.choice(["server", "workstation", "iot", "firewall"]),
            "os": rng.choice(["Linux", "Windows Server", "Windows 10", "Ubuntu"]),
        }

        feature_subset = {
            "protocol_type": str(row.get("protocol_type", "tcp")),
            "service":        str(row.get("service", "other")),
            "flag":           str(row.get("flag", "SF")),
            "duration":       float(row.get("duration", 0)),
            "src_bytes":      int(row.get("src_bytes", 0)),
            "dst_bytes":      int(row.get("dst_bytes", 0)),
            "count":          int(row.get("count", 0)),
            "diff_srv_rate":  float(row.get("diff_srv_rate", 0.0)),
            "dst_host_count": int(row.get("dst_host_count", 0)),
            "root_shell":     int(row.get("root_shell", 0)),
            "su_attempted":   int(row.get("su_attempted", 0)),
            "num_root":       int(row.get("num_root", 0)),
            "num_shells":     int(row.get("num_shells", 0)),
            "num_failed_logins": int(row.get("num_failed_logins", 0)),
            "wrong_fragment": int(row.get("wrong_fragment", 0)),
        }

        cases.append({
            "case_id":                  f"nsl_{i:03d}",
            "dataset":                  "NSL-KDD",
            "true_attack_class":        true_cls,
            "true_nfcrm_class":         nfcrm_cls,
            "asset":                    asset,
            "paired_cve":               paired_cve,
            "nl_description":           nl_desc,
            "feature_subset":           feature_subset,
            "rf_predicted_class":       rf_cls,
            "rf_nfcrm_class":           nfcrm_rf,
            "rf_confidence":            float(rf_proba[i]),
            "rf_risk_level":            rf_risk_level,
            "xgb_predicted_class":      xgb_cls,
            "xgb_nfcrm_class":          nfcrm_xgb,
            "xgb_confidence":           float(xgb_proba[i]),
            "xgb_risk_level":           xgb_risk_level,
            "ground_truth_risk_level":  gt_risk_level,
            "criticality":              criticality,
        })

    rf_correct  = sum(1 for c in cases if c["rf_risk_level"]  == c["ground_truth_risk_level"])
    xgb_correct = sum(1 for c in cases if c["xgb_risk_level"] == c["ground_truth_risk_level"])
    n = len(cases)
    print(f"\nML risk-level accuracy on NSL-KDD eval set ({n} cases):")
    print(f"  RF : {rf_correct}/{n} = {rf_correct/n:.1%}")
    print(f"  XGB: {xgb_correct}/{n} = {xgb_correct/n:.1%}")

    class_dist = {}
    for cls in CLASSES:
        class_dist[cls] = sum(1 for c in cases if c["true_attack_class"] == cls)
    print(f"  Class distribution: {class_dist}")

    return {
        "meta": {
            "description": "NSL-KDD Unified Ablation Test Set — all paradigms evaluated on same N=50 flows",
            "dataset": "NSL-KDD",
            "n": n,
            "n_per_class": n_per_class,
            "seed": seed,
            "classes": CLASSES,
            "nfcrm_mapping": NSLKDD_TO_NFCRM,
            "criticalities": CRITICALITIES,
            "rf_risk_accuracy":  rf_correct / n,
            "xgb_risk_accuracy": xgb_correct / n,
            "created": datetime.now().isoformat(),
            "train_source": "data/raw/NSL-KDD/KDDTrain+.txt",
            "eval_source":  "data/raw/NSL-KDD/KDDTest+.txt",
        },
        "cases": cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nslkdd_ablation/test_set.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-class", type=int, default=10)
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_set = build_test_set(seed=args.seed, n_per_class=args.n_per_class)
    out_path.write_text(json.dumps(test_set, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(test_set['cases'])} cases to {out_path}")
    print("Next: python scripts/run_nslkdd_ablation.py --all-nonllm")


if __name__ == "__main__":
    main()
