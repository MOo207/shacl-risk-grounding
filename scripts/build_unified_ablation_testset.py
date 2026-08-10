"""Build the Unified Ablation Test Set (N=180, 30 per attack class).

Creates a single shared test set where ALL paradigms (Rule, ML, LLM, combinations)
can be evaluated on the SAME 180 flows with IDENTICAL inputs. This resolves the
apples-to-oranges problem in the original table (ML at N=2,122 vs LLM at N=180).

Each case includes:
  - Raw tabular features       → for Rule / ML paradigms
  - NL description             → for LLM paradigms
  - RF and XGBoost predictions → for Tri-Stage/LLM (ML provides attack class to LLM)
  - Asset metadata + criticality → for NFCRM §6.9 risk level computation
  - Paired CVE from ARG        → for LLM GRC narrative
  - Ground truth risk level    → metric for all paradigms

Usage:
    python scripts/build_unified_ablation_testset.py
    python scripts/build_unified_ablation_testset.py --out results/unified_ablation/test_set.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Attack-class → CVE mapping (from ARG, matched by attack type) ─────────────
CVE_BY_CLASS: dict[str, list[str]] = {
    "DoS":          ["CVE-2007-6750", "CVE-2009-0696", "CVE-2011-3192"],
    "DDoS":         ["CVE-2017-0144", "CVE-2017-7679"],
    "BruteForce":   ["CVE-2008-0166", "CVE-2006-2369"],
    "WebAttack":    ["CVE-2012-1823", "CVE-2009-1151", "CVE-2007-5423"],
    "Infiltration": ["CVE-2011-2523", "CVE-2010-2075", "CVE-2014-0160"],
    "Benign":       [],
}

CRITICALITIES = ["Low", "Medium", "High", "Critical"]

CONTROL_BY_CLASS: dict[str, str] = {
    "Benign":       "SS6.1-MON-1",
    "BruteForce":   "SS6.7-AC-1",
    "DoS":          "SS6.7-NET-1",
    "DDoS":         "SS6.7-NET-2",
    "Infiltration": "SS6.5-NET-1",
    "WebAttack":    "SS6.7-APP-1",
}


def load_cve_index(arg_path: Path) -> dict[str, dict]:
    """Return {cve_id: cve_node_dict} from the ARG JSON."""
    arg = json.loads(arg_path.read_text())
    return {n["id"]: n for n in arg["nodes"] if n.get("type") == "CVE"}


def pick_cve(attack_class: str, cve_index: dict, rng: random.Random) -> Optional[dict]:
    """Pick a random CVE for this attack class, return None for Benign."""
    candidates = CVE_BY_CLASS.get(attack_class, [])
    if not candidates:
        return None
    chosen_id = rng.choice(candidates)
    return cve_index.get(chosen_id)


def train_ml_models(df: pd.DataFrame, label_col: str = "Label"):
    """Train RF and XGBoost on 70% train split. Returns (rf_model, xgb_model, feature_cols)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values.astype(float)
    y_raw = df[label_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"  RF test accuracy: {rf_acc:.4f}")

    try:
        from xgboost import XGBClassifier
        print("Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss",
                            use_label_encoder=False, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_acc = xgb.score(X_test, y_test)
        print(f"  XGBoost test accuracy: {xgb_acc:.4f}")
    except ImportError:
        print("  XGBoost not available; using RF copy as fallback")
        xgb = rf
        xgb_acc = rf_acc

    return rf, xgb, le, feature_cols, X_test, y_test


def features_to_nl(row: pd.Series) -> str:
    """Import NL conversion from the classification script."""
    # Inline import avoids circular path issues
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_slm_nl_classification import features_to_nl as _fn
    return _fn(row)


def build_test_set(seed: int = 42, n_per_class: int = 30) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)

    data_path = ROOT / "data" / "processed" / "cleaned_dataset.csv"
    arg_path  = ROOT / "results" / "multisource_arg.json"

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    classes = sorted(df["Label"].unique().tolist())
    print(f"  {len(df):,} rows | classes: {classes}")

    cve_index = load_cve_index(arg_path)
    print(f"Loaded {len(cve_index)} CVEs from ARG")

    # ── Reserve exactly n_per_class rows per class as evaluation set first,
    #    then train ML on the remaining rows. Zero leakage by construction.
    eval_dfs, train_dfs = [], []
    for cls in classes:
        cls_df = df[df["Label"] == cls]
        if len(cls_df) < n_per_class:
            raise ValueError(
                f"Class '{cls}' has only {len(cls_df)} rows — cannot reserve {n_per_class}. "
                f"Reduce --n-per-class to {len(cls_df)}."
            )
        eval_sample = cls_df.sample(n=n_per_class, random_state=seed)
        eval_dfs.append(eval_sample)
        train_dfs.append(cls_df.drop(eval_sample.index))

    sample    = pd.concat(eval_dfs,  ignore_index=True).sample(frac=1, random_state=seed)
    train_df  = pd.concat(train_dfs, ignore_index=True)
    print(f"Evaluation set: {len(sample)} flows ({n_per_class}/class) | Training pool: {len(train_df)} rows")

    # ── Train ML on the remaining data (no evaluation rows included) ──────────
    rf_model, xgb_model, le, feature_cols, _, _ = train_ml_models(train_df, label_col="Label")

    # ── Get ML predictions for each sampled flow ───────────────────────────────
    X_sample = sample[feature_cols].values.astype(float)
    rf_preds  = le.inverse_transform(rf_model.predict(X_sample))
    xgb_preds = le.inverse_transform(xgb_model.predict(X_sample))
    rf_proba  = rf_model.predict_proba(X_sample).max(axis=1)
    xgb_proba = xgb_model.predict_proba(X_sample).max(axis=1)

    # ── Build cases ────────────────────────────────────────────────────────────
    from pipeline.nfcrm.risk_score import compute_risk_score
    CRIT_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}

    cases = []
    for i, (idx, row) in enumerate(sample.iterrows()):
        true_cls  = row["Label"]
        rf_cls    = rf_preds[i]
        xgb_cls   = xgb_preds[i]
        criticality = rng.choice(CRITICALITIES)
        c_val = CRIT_MAP[criticality]

        # Ground truth risk level from NFCRM §6.9
        gt_score = compute_risk_score(true_cls, c_override=c_val, i_override=c_val, a_override=c_val)
        gt_risk_level = gt_score.level_en if gt_score.level_en != "N/A (non-attack)" else "Very Low"

        # RF-derived risk level (what a pure-ML pipeline would produce)
        rf_score = compute_risk_score(rf_cls, c_override=c_val, i_override=c_val, a_override=c_val)
        rf_risk_level = rf_score.level_en if rf_score.level_en != "N/A (non-attack)" else "Very Low"

        # XGBoost-derived risk level
        xgb_score = compute_risk_score(xgb_cls, c_override=c_val, i_override=c_val, a_override=c_val)
        xgb_risk_level = xgb_score.level_en if xgb_score.level_en != "N/A (non-attack)" else "Very Low"

        paired_cve = pick_cve(true_cls, cve_index, rng)

        asset = {
            "id": f"asset_{i:03d}",
            "hostname": f"host-{i:03d}",
            "criticality": criticality,
            "asset_type": rng.choice(["server", "workstation", "iot", "firewall"]),
            "os": rng.choice(["Linux", "Windows Server", "Windows 10", "Ubuntu"]),
        }

        # Minimal ARG neighborhood
        arg_neighborhood = {
            "related_cves": [{"id": paired_cve["id"], "cvss_v3": paired_cve.get("cvss_v3")}]
            if paired_cve else []
        }

        # NL description for LLM paradigms
        nl_desc = features_to_nl(row)

        # Feature dict (subset of key features for readability)
        feature_subset = {
            "Dst Port": int(row.get("Dst Port", 0)),
            "Protocol": int(row.get("Protocol", 0)),
            "Flow Duration": float(row.get("Flow Duration", 0)),
            "Tot Fwd Pkts": int(row.get("Tot Fwd Pkts", 0)),
            "Tot Bwd Pkts": int(row.get("Tot Bwd Pkts", 0)),
            "Init Fwd Win Byts": int(row.get("Init Fwd Win Byts", 0)),
            "Flow Pkts/s": float(row.get("Flow Pkts/s", 0)),
        }

        cases.append({
            "case_id": f"ua_{i:03d}",
            "true_attack_class": true_cls,
            "asset": asset,
            "paired_cve": paired_cve,
            "arg_neighborhood": arg_neighborhood,
            "controls": [CONTROL_BY_CLASS.get(true_cls, "SS6.1-MON-1")],
            "nl_description": nl_desc,
            "feature_subset": feature_subset,
            # ML predictions and their derived risk levels
            "rf_predicted_class": rf_cls,
            "rf_confidence": float(rf_proba[i]),
            "rf_risk_level": rf_risk_level,
            "xgb_predicted_class": xgb_cls,
            "xgb_confidence": float(xgb_proba[i]),
            "xgb_risk_level": xgb_risk_level,
            # Ground truth
            "ground_truth_risk_level": gt_risk_level,
            "criticality": criticality,
        })

    # ── Summary stats ──────────────────────────────────────────────────────────
    rf_correct  = sum(1 for c in cases if c["rf_risk_level"]  == c["ground_truth_risk_level"])
    xgb_correct = sum(1 for c in cases if c["xgb_risk_level"] == c["ground_truth_risk_level"])
    print(f"\nML risk-level accuracy on unified set:")
    print(f"  RF : {rf_correct}/{len(cases)} = {rf_correct/len(cases):.1%}")
    print(f"  XGB: {xgb_correct}/{len(cases)} = {xgb_correct/len(cases):.1%}")
    print(f"  (compare to classification accuracy: RF~87.7%, XGB~95.9%)")
    print(f"  Note: risk-level accuracy != classification accuracy due to criticality mapping")

    output = {
        "meta": {
            "description": "Unified Ablation Test Set — all paradigms evaluated on same N=180 flows",
            "n": len(cases),
            "n_per_class": n_per_class,
            "seed": seed,
            "classes": classes,
            "criticalities": CRITICALITIES,
            "rf_risk_accuracy": rf_correct / len(cases),
            "xgb_risk_accuracy": xgb_correct / len(cases),
            "created": datetime.now().isoformat(),
            "source": "data/processed/cleaned_dataset.csv",
            "cve_source": "results/multisource_arg.json",
        },
        "cases": cases,
    }
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/unified_ablation/test_set.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-class", type=int, default=30)
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_set = build_test_set(seed=args.seed, n_per_class=args.n_per_class)

    out_path.write_text(json.dumps(test_set, indent=2, ensure_ascii=False), encoding="utf-8")
    n = len(test_set["cases"])
    print(f"\nSaved {n} cases to {out_path}")
    print("Next: run scripts/run_unified_ablation.py --paradigm all")


if __name__ == "__main__":
    main()
