"""Class-weighted XGBoost baseline on the NSL-KDD unified-rerun N=100 pool.

Addresses review finding M11 ("XGBoost ~56%/49% accuracy may be a strawman;
no class-weighted or SMOTE variant was run") for the CLEAN leak-free pool
(results/nslkdd_unified_rerun/test_sample.json, N=100, 20/class) used by
reconciled_tristage_haiku_results.json.

Protocol:
  1. Rebuild the EXACT same train/eval pipeline as
     scripts/build_nslkdd_unified_rerun.py: same load_raw/encode_for_ml
     feature pipeline, same seed-42 sampling (exclude old compromised N=50
     set, draw calibration 10/class, then held-out test 20/class from
     KDDTest+.txt), same XGBClassifier(n_estimators=100, random_state=42,
     eval_metric='mlogloss', tree_method='hist', n_jobs=1).
  2. VALIDITY CHECK: the default-loss retrain must reproduce, per case, the
     frozen xgb_predicted_class in test_sample.json AND the published
     baseline metrics (exact 49.0%, under-escalation 48.0%, severe recall
     42.9%). If not, the script aborts without reporting weighted numbers.
  3. Retrain with class-imbalance handling, changing ONLY the weighting:
       - 'balanced'      : sklearn compute_sample_weight(class_weight='balanced')
                           (inverse-frequency, Elkan 2001 cost-sensitive)
       - 'sqrt_balanced' : sqrt of the balanced weights (milder reweighting)
     SMOTE variant is attempted only if imbalanced-learn is installed;
     otherwise it is recorded as skipped.
  4. Score every variant on the same N=100 pool with the identical NFCRM
     SS6.9 risk mapping + safety_table_haiku.py::metrics() computation.

Output: results/nslkdd_ablation/xgb_weighted.json

Usage:
    python -m scripts.run_nslkdd_xgb_weighted
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.build_nslkdd_ablation_testset import (  # noqa: E402
    load_raw, encode_for_ml, NSLKDD_TO_NFCRM, CLASSES, CRIT_MAP,
)
from pipeline.nfcrm.risk_score import compute_risk_score  # noqa: E402

SEED = 42
N_CALIBRATION_PER_CLASS = 10
N_TEST_PER_CLASS = 20

TRAIN_PATH = ROOT / "data/raw/NSL-KDD/KDDTrain+.txt"
TEST_PATH = ROOT / "data/raw/NSL-KDD/KDDTest+.txt"
OLD_TEST_SET = ROOT / "results/nslkdd_ablation/test_set.json"
POOL_PATH = ROOT / "results/nslkdd_unified_rerun/test_sample.json"
BASELINE_PATH = ROOT / "results/nslkdd_unified_rerun/xgboost_baseline_results.json"
OUT_PATH = ROOT / "results/nslkdd_ablation/xgb_weighted.json"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

XGB_PARAMS = dict(n_estimators=100, random_state=SEED, eval_metric="mlogloss",
                  verbosity=0, tree_method="hist", n_jobs=1)


def risk_level(nfcrm_class: str, criticality: str) -> str:
    c_val = CRIT_MAP.get(criticality, 3)
    score = compute_risk_score(nfcrm_class, c_override=c_val, i_override=c_val, a_override=c_val)
    return "Very Low" if score.level_en == "N/A (non-attack)" else score.level_en


def metrics(pairs: list[tuple[str, str]]) -> dict:
    """(pred_risk, gt_risk) pairs -> safety_table_haiku.py::metrics()-identical."""
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for p, g in pairs:
        n += 1
        if p not in IDX or g not in IDX:
            if g in SEVERE:
                sev_tot += 1
            under += 1
            worst = max(worst, IDX.get(g, 0))
            continue
        d = IDX[p] - IDX[g]
        if d == 0:
            exact += 1
        if abs(d) <= 1:
            within1 += 1
        if d < 0:
            under += 1
            worst = max(worst, -d)
        elif d > 0:
            over += 1
        if g in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_rec += 1
    pct = lambda a, b: round(a / b * 100, 1) if b else 0.0
    return dict(n=n, exact_pct=pct(exact, n), within1_pct=pct(within1, n),
                under_escalation_pct=pct(under, n), over_escalation_pct=pct(over, n),
                severe_recall_pct=pct(sev_rec, sev_tot), severe_total=sev_tot,
                worst_miss_levels=worst)


def rebuild_test_indices(test_raw: pd.DataFrame) -> list[int]:
    """Re-derive the exact KDDTest+ row indices of test_sample.json using the
    same sampling code path as build_nslkdd_unified_rerun.py."""
    old_data = json.loads(OLD_TEST_SET.read_text(encoding="utf-8"))
    old_seed = old_data["meta"]["seed"]
    old_n_per_class = old_data["meta"]["n_per_class"]

    excluded: set[int] = set()
    for cls in CLASSES:
        cls_idxs = test_raw.index[test_raw["label"] == cls].tolist()
        excluded.update(random.Random(old_seed).sample(cls_idxs, old_n_per_class))

    remaining = test_raw.drop(index=list(excluded))
    calib: list[int] = []
    for cls in CLASSES:
        cls_idxs = remaining.index[remaining["label"] == cls].tolist()
        calib.extend(random.Random(SEED).sample(cls_idxs, N_CALIBRATION_PER_CLASS))

    pool = remaining.drop(index=calib)
    test: list[int] = []
    for cls in CLASSES:
        cls_idxs = pool.index[pool["label"] == cls].tolist()
        test.extend(random.Random(SEED).sample(cls_idxs, N_TEST_PER_CLASS))

    # same final shuffle -> defines case_id order nslr_test_000..099
    test_df = test_raw.loc[test].sample(frac=1, random_state=SEED)
    return test_df.index.tolist()


def train_and_predict(train_enc: pd.DataFrame, X_eval: np.ndarray,
                      weighting: str | None):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    feature_cols = [c for c in train_enc.columns if c != "label"]
    X = train_enc[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(train_enc["label"].values)

    sw = None
    if weighting == "balanced":
        sw = compute_sample_weight(class_weight="balanced", y=y)
    elif weighting == "sqrt_balanced":
        sw = np.sqrt(compute_sample_weight(class_weight="balanced", y=y))

    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X, y, sample_weight=sw)
    preds = le.inverse_transform(xgb.predict(X_eval))
    return preds, le, y


def try_smote(train_enc: pd.DataFrame, X_eval: np.ndarray):
    """SMOTE oversampling variant, only if imbalanced-learn is installed."""
    try:
        from imblearn.over_sampling import SMOTE  # noqa: F401
    except ImportError:
        return None, "imbalanced-learn (imblearn) not installed in this environment; SMOTE variant skipped."
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE

    feature_cols = [c for c in train_enc.columns if c != "label"]
    X = train_enc[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(train_enc["label"].values)
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X, y)
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_res, y_res)
    return le.inverse_transform(xgb.predict(X_eval)), None


def score_variant(preds, cases: list[dict]) -> dict:
    pairs, records = [], []
    n_cls_correct = 0
    for i, case in enumerate(cases):
        pred_nsl = preds[i]
        pred_nfcrm = NSLKDD_TO_NFCRM.get(pred_nsl, "Benign")
        pred_risk = risk_level(pred_nfcrm, case["criticality"])
        gt = case["ground_truth_risk_level"]
        pairs.append((pred_risk, gt))
        if pred_nsl == case["true_attack_class"]:
            n_cls_correct += 1
        records.append({"case_id": case["case_id"], "true_class": case["true_attack_class"],
                        "pred_class": pred_nsl, "risk_level": pred_risk,
                        "ground_truth": gt, "correct": pred_risk == gt})
    m = metrics(pairs)
    m["attack_class_accuracy_pct"] = round(100 * n_cls_correct / len(cases), 1)
    return {"metrics": m, "records": records}


def main() -> None:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    cases = pool["cases"]
    assert len(cases) == 100, f"expected N=100 pool, got {len(cases)}"

    published = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    pub_m = published["held_out_test_xgboost_baseline_metrics"]

    print("Rebuilding feature pipeline + exact N=100 eval indices ...")
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    train_enc, test_enc = encode_for_ml(train_raw, test_raw.copy())
    feature_cols = [c for c in train_enc.columns if c != "label"]

    test_indices = rebuild_test_indices(test_raw)
    assert len(test_indices) == 100
    X_eval = test_enc.loc[test_indices][feature_cols].values.astype(float)

    # sanity: row order must match case order (true labels align)
    row_labels = test_raw.loc[test_indices]["label"].tolist()
    for i, case in enumerate(cases):
        assert row_labels[i] == case["true_attack_class"], (
            f"row/case misalignment at {i}: {row_labels[i]} != {case['true_attack_class']}")

    # ── VALIDITY CHECK: default-loss retrain must reproduce frozen baseline ──
    print("Training default-loss XGBoost (reproduction check) ...")
    default_preds, le, y_train = train_and_predict(train_enc, X_eval, weighting=None)
    n_match = sum(1 for i, c in enumerate(cases)
                  if default_preds[i] == c["xgb_predicted_class"])
    default_scored = score_variant(default_preds, cases)
    dm = default_scored["metrics"]
    repro_preds = n_match == len(cases)
    repro_metrics = (dm["exact_pct"] == pub_m["exact_pct"]
                     and dm["under_escalation_pct"] == pub_m["under_escalation_pct"]
                     and dm["severe_recall_pct"] == pub_m["severe_recall_pct"])
    print(f"  per-case prediction match vs frozen test_sample.json: {n_match}/{len(cases)}")
    print(f"  retrain metrics: exact={dm['exact_pct']} under={dm['under_escalation_pct']} "
          f"sev_recall={dm['severe_recall_pct']}  "
          f"(published: {pub_m['exact_pct']}/{pub_m['under_escalation_pct']}/{pub_m['severe_recall_pct']})")
    if not (repro_preds and repro_metrics):
        print("\nREPRODUCTION CHECK FAILED -- aborting without weighted conclusions.")
        sys.exit(1)
    print("  REPRODUCTION CHECK PASSED.")

    # class weights actually applied (for disclosure)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_map = {le.inverse_transform([int(c)])[0]: round(float(w), 4)
                        for c, w in zip(np.unique(y_train), weights)}
    train_dist = {le.inverse_transform([int(c)])[0]: int(n)
                  for c, n in zip(*np.unique(y_train, return_counts=True))}

    variants: dict[str, dict] = {"default_unweighted": default_scored}

    for w in ("balanced", "sqrt_balanced"):
        print(f"Training {w} XGBoost ...")
        preds, _, _ = train_and_predict(train_enc, X_eval, weighting=w)
        variants[w] = score_variant(preds, cases)
        m = variants[w]["metrics"]
        print(f"  exact={m['exact_pct']}%  under={m['under_escalation_pct']}%  "
              f"over={m['over_escalation_pct']}%  sev_recall={m['severe_recall_pct']}%  "
              f"cls_acc={m['attack_class_accuracy_pct']}%")

    smote_preds, smote_skip_reason = try_smote(train_enc, X_eval)
    if smote_preds is not None:
        variants["smote"] = score_variant(smote_preds, cases)
        m = variants["smote"]["metrics"]
        print(f"SMOTE: exact={m['exact_pct']}%  under={m['under_escalation_pct']}%  "
              f"sev_recall={m['severe_recall_pct']}%")
    else:
        print(f"SMOTE skipped: {smote_skip_reason}")

    import xgboost as xgb_mod
    import sklearn
    out = {
        "meta": {
            "description": (
                "Class-weighted XGBoost baselines on the clean leak-free NSL-KDD "
                "N=100 held-out pool (results/nslkdd_unified_rerun/test_sample.json), "
                "addressing review finding M11 (unweighted XGBoost as possible strawman). "
                "Identical train data, feature pipeline, seed, XGB params, eval pool, "
                "NFCRM SS6.9 risk mapping, and safety metrics as "
                "scripts/build_nslkdd_unified_rerun.py; ONLY the class-imbalance "
                "handling differs between variants."
            ),
            "xgb_params": XGB_PARAMS,
            "train_source": str(TRAIN_PATH.relative_to(ROOT)),
            "eval_pool": str(POOL_PATH.relative_to(ROOT)),
            "train_class_distribution": train_dist,
            "balanced_class_weights_applied": class_weight_map,
            "weighting_definitions": {
                "default_unweighted": "no sample_weight (published baseline)",
                "balanced": "sklearn compute_sample_weight(class_weight='balanced') -> "
                            "w_i = n_samples / (n_classes * count(class_i)); Elkan 2001",
                "sqrt_balanced": "element-wise sqrt of the 'balanced' weights (milder reweighting)",
            },
            "smote": {"run": smote_preds is not None,
                      "skip_reason": smote_skip_reason},
            "library_versions": {"xgboost": xgb_mod.__version__,
                                 "sklearn": sklearn.__version__},
            "created": datetime.now().isoformat(),
        },
        "reproduction_check": {
            "per_case_prediction_match": f"{n_match}/{len(cases)}",
            "published_baseline_metrics": pub_m,
            "retrained_default_metrics": dm,
            "passed": bool(repro_preds and repro_metrics),
        },
        "variants": {name: v["metrics"] for name, v in variants.items()},
        "per_case_records": {name: v["records"] for name, v in variants.items()
                             if name != "default_unweighted"},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
