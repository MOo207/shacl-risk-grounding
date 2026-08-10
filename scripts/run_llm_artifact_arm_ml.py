"""Pure ML arm — XGBoost trained on Set A (rule labels), tested on full 80 cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.llm_artifact_lib import RISK_LEVELS, stratified_slice
from scripts.run_llm_artifact_arm_rule import ATTACK_TO_CONTROL

CLASS_IDX = {"Benign": 0, "BruteForce": 1, "DoS": 2, "DDoS": 3, "Infiltration": 4}
CRIT_IDX = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
AV_IDX = {"NETWORK": 0, "ADJACENT_NETWORK": 1, "LOCAL": 2, "PHYSICAL": 3}


def featurize_case(case: dict) -> list[float]:
    masked = set(case.get("masked_fields", []))
    attack_idx = (
        -1.0 if "true_attack_class" in masked
        else float(CLASS_IDX.get(case["true_attack_class"], -1))
    )
    crit = case.get("asset", {}).get("criticality")
    crit_idx = (
        -1.0 if "asset_criticality" in masked or crit is None
        else float(CRIT_IDX.get(crit.upper(), -1))
    )
    cvss = case.get("paired_cve") or {}
    cvss_v3 = (
        0.0 if "cvss_components" in masked or not cvss
        else float(cvss.get("cvss_v3") or 0.0)
    )
    av_idx = (
        -1.0 if "cvss_components" in masked or not cvss
        else float(AV_IDX.get(cvss.get("attack_vector") or "", -1))
    )
    expl = (
        0.0 if "cvss_components" in masked or not cvss
        else float(cvss.get("exploitability_score") or 0.0)
    )
    return [attack_idx, crit_idx, cvss_v3, av_idx, expl]


def train_ml_arm(set_a_cases: list[dict]):
    """Train XGBoost on Set A cases.

    Returns a tuple (model, label_map) where label_map maps contiguous
    integer predictions back to RISK_LEVELS indices.
    """
    raw_labels = [RISK_LEVELS.index(c["ground_truth_risk_level"]) for c in set_a_cases]
    # Build a contiguous 0-based encoding to satisfy XGBoost's class constraint.
    unique_labels = sorted(set(raw_labels))
    enc = {lbl: i for i, lbl in enumerate(unique_labels)}
    dec = {i: lbl for lbl, i in enc.items()}  # contiguous idx → RISK_LEVELS idx

    X = np.array([featurize_case(c) for c in set_a_cases])
    y = np.array([enc[l] for l in raw_labels])
    model = xgb.XGBClassifier(
        tree_method="hist", max_depth=4, n_estimators=100,
        random_state=42, eval_metric="mlogloss",
    )
    model.fit(X, y)
    return model, dec


def predict_with_ml_arm(model_bundle, cases: list[dict]) -> list[dict]:
    model, dec = model_bundle
    X = np.array([featurize_case(c) for c in cases])
    preds = model.predict(X).tolist()
    out = []
    for case, p in zip(cases, preds):
        risk_level_idx = dec.get(int(p), int(p))  # map back through decode table
        out.append({
            "case_id": case["case_id"],
            "arm": "ml",
            "risk_level": RISK_LEVELS[risk_level_idx],
            "recommended_control_id": ATTACK_TO_CONTROL.get(case["true_attack_class"], "SS6.1-MON-1"),
            "nfcrm_clauses": ["§6.9"],
            "narrative": "",
            "evidence_refs": [],
            "ground_truth_risk_level": case.get("ground_truth_risk_level", ""),
            "true_attack_class": case.get("true_attack_class", ""),
            "subset": case.get("subset", ""),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="results/llm_experiment/test_set.json")
    ap.add_argument("--out", default="results/llm_experiment/raw/run_1_arm_ml.jsonl")
    ap.add_argument("--n-per-class", type=int, default=None)
    args = ap.parse_args()

    all_cases = json.loads(Path(args.test_set).read_text())["cases"]
    cases = stratified_slice(all_cases, args.n_per_class)
    set_a = [c for c in cases if c["subset"] == "A"]
    print(f"ML arm | Cases: {len(cases)} (Set A for training: {len(set_a)})")
    model_bundle = train_ml_arm(set_a)
    artifacts = predict_with_ml_arm(model_bundle, cases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for a in artifacts:
            a["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(json.dumps(a) + "\n")
    correct = sum(1 for a in artifacts if a["risk_level"] == a.get("ground_truth_risk_level"))
    print(f"Wrote {len(artifacts)} ml-arm artifacts to {args.out} | acc={correct/len(artifacts)*100:.1f}%")


if __name__ == "__main__":
    main()
