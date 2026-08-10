"""Matched-information XGBoost baseline: plain features + binary cve_present.

Reviewer-requested control (DKE round 5, item 3 / methodology W2). The LLM
arms receive a class-paired CVE block in-prompt; the CVE channel is
class-indexed BY TEST-SET CONSTRUCTION (attack cases are paired with a CVE
for their mapped NFCRM class, Normal cases receive none -- verified:
results/nslkdd_unified_rerun/test_sample.json has paired_cve non-null iff
true_attack_class != Normal). This script gives XGBoost the same information
channel as one extra binary feature, cve_present = (label != 'Normal') at
train time, mirroring the pairing rule, and (paired_cve is not None) at test
time.

Pre-registered interpretation rule, fixed before running: if XGBoost+CVE
matches the tri-stage arm's safety metrics, the non-channel-independent share
of the safety headline is fully explained by the presence bit and the LLM's
use of the channel is not shown to be non-trivial; if it does not match, the
LLM extracts more than the presence bit. Either outcome is reported verbatim.

Training/eval reuse scripts/build_nslkdd_unified_rerun.py's exact pipeline:
same load_raw/encode_for_ml, same XGBClassifier hyperparameters
(n_estimators=100, random_state=42, tree_method='hist', n_jobs=1), same
Sec 6.9 risk mapping via compute_risk_score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.build_nslkdd_ablation_testset import (  # noqa: E402
    load_raw, encode_for_ml, NSLKDD_TO_NFCRM,
)
from scripts.build_nslkdd_unified_rerun import (  # noqa: E402
    train_xgb, risk_level, FEATURE_COLS_FOR_FINGERPRINT, LEVELS, IDX, SEVERE,
    TRAIN_PATH, TEST_PATH,
)

TEST_SAMPLE = ROOT / "results/nslkdd_unified_rerun/test_sample.json"
OUT = ROOT / "results/nslkdd_xgb_cve_feature.json"

FP_COLS = [c for c in FEATURE_COLS_FOR_FINGERPRINT if c != "label"]


def fingerprint(vals) -> tuple:
    out = []
    for v in vals:
        if isinstance(v, (int, np.integer)):
            out.append(float(v))
        elif isinstance(v, (float, np.floating)):
            out.append(round(float(v), 6))
        else:
            out.append(str(v))
    return tuple(out)


def main() -> None:
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    train_enc, test_enc = encode_for_ml(train_raw, test_raw.copy())

    # Augment with the presence bit, mirroring the test-set pairing rule.
    train_enc = train_enc.copy()
    train_enc["cve_present"] = (train_raw["label"].values != "Normal").astype(float)

    cases = json.load(open(TEST_SAMPLE))["cases"]

    # Map each case to its KDDTest+ row via full raw-feature fingerprint.
    fp_to_idx: dict[tuple, int] = {}
    for i, row in test_raw[FP_COLS].iterrows():
        fp_to_idx.setdefault(fingerprint(row.values), i)

    case_rows = []
    for c in cases:
        fp = fingerprint([c["raw_features"][col] for col in FP_COLS])
        if fp not in fp_to_idx:
            raise SystemExit(f"fingerprint miss for {c['case_id']}")
        case_rows.append(fp_to_idx[fp])

    xgb, le, feature_cols = train_xgb(train_enc)

    X_test = test_enc[[c for c in test_enc.columns if c != "label"]].copy()
    X_test["cve_present"] = 0.0   # per-case value set below
    X_test = X_test[feature_cols]  # training column order

    records = []
    for c, ridx in zip(cases, case_rows):
        xrow = X_test.iloc[[ridx]].copy()
        xrow["cve_present"] = 1.0 if c["paired_cve"] is not None else 0.0
        xrow = xrow[feature_cols].values.astype(float)
        pred_cls = le.inverse_transform(xgb.predict(xrow))[0]
        pred_nfcrm = NSLKDD_TO_NFCRM.get(pred_cls, "Benign")
        pred_risk = risk_level(pred_nfcrm, c["criticality"])
        records.append({
            "case_id": c["case_id"],
            "true_class": c["true_attack_class"],
            "criticality": c["criticality"],
            "cve_present": c["paired_cve"] is not None,
            "plain_xgb_risk": c["xgb_risk_level"],
            "xgb_cve_pred_class": pred_cls,
            "xgb_cve_risk": pred_risk,
            "ground_truth": c["ground_truth_risk_level"],
        })

    def metrics(key: str) -> dict:
        n = len(records)
        under = sum(IDX[r[key]] < IDX[r["ground_truth"]] for r in records)
        over = sum(IDX[r[key]] > IDX[r["ground_truth"]] for r in records)
        exact = sum(r[key] == r["ground_truth"] for r in records)
        sev = [r for r in records if r["ground_truth"] in SEVERE]
        srec = sum(r[key] in SEVERE for r in sev)
        return {"n": n, "under_escalation_pct": under / n * 100,
                "over_escalation_pct": over / n * 100,
                "exact_pct": exact / n * 100,
                "severe_recall_pct": srec / len(sev) * 100, "n_severe": len(sev)}

    # McNemar (exact-level correctness) augmented vs plain XGBoost.
    from scipy.stats import binomtest
    b = sum(r["plain_xgb_risk"] == r["ground_truth"] and r["xgb_cve_risk"] != r["ground_truth"]
            for r in records)
    c_ = sum(r["plain_xgb_risk"] != r["ground_truth"] and r["xgb_cve_risk"] == r["ground_truth"]
             for r in records)
    p = binomtest(min(b, c_), b + c_, 0.5).pvalue if (b + c_) else 1.0

    out = {
        "description": __doc__.strip(),
        "xgb_params": {"n_estimators": 100, "random_state": 42,
                       "tree_method": "hist", "n_jobs": 1,
                       "extra_feature": "cve_present (binary)"},
        "train_rule": "cve_present = (label != 'Normal'), mirroring test-set CVE pairing",
        "metrics_xgb_cve": metrics("xgb_cve_risk"),
        "metrics_plain_xgb_reference": metrics("plain_xgb_risk"),
        "mcnemar_cve_vs_plain": {"b_plain_correct_cve_wrong": b,
                                 "c_plain_wrong_cve_correct": c_, "p_value": p},
        "records": records,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("metrics_xgb_cve", "metrics_plain_xgb_reference",
                                          "mcnemar_cve_vs_plain")}, indent=2))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
