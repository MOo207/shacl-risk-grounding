"""Cost-sensitive (class-weighted) XGBoost baseline for the NSL-KDD N=50 safety
evaluation — addresses Construct-Validity Threat CV-4 ("No Cost-Sensitive ML
Baseline") disclosed in the thesis.

CV-4 argues the safest rival explanation for "why ML looks unsafe" on NSL-KDD
is that the XGBoost baseline used a standard, default-loss objective (uniform
class weights) rather than a cost-sensitive one. This script retrains XGBoost
on the identical KDDTrain+/KDDTest+ split, features, seed, and eval subset as
scripts/build_nslkdd_ablation_testset.py, changing ONLY the class-weighting
strategy (Elkan 2001 cost-sensitive learning via inverse-frequency sample
weights, cf. thesis references.bib), then applies the *same* NFCRM SS6.9
risk-level mapping and under-escalation / severe-recall computation as
scripts/safety_table_haiku.py.

Reproducibility: the eval subset is regenerated (not re-read) from the raw
NSL-KDD files using the exact same stratified-sampling code (seed=42,
10/class) as build_nslkdd_ablation_testset.py. A sanity check confirms the
regenerated default-loss XGBoost predictions match the ones frozen in
results/nslkdd_ablation/test_set.json (xgb_predicted_class) before the
cost-sensitive comparison is trusted.

Output: results/nslkdd_ablation/xgb_cost_sensitive_baseline.json

Usage:
    python -m scripts.run_nslkdd_cost_sensitive_baseline
"""
from __future__ import annotations

import json
import sys
import warnings
from math import comb
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

SEED          = 42
N_PER_CLASS   = 10
TEST_SET_PATH = ROOT / "results/nslkdd_ablation/test_set.json"
TRISTAGE_PATH = ROOT / "results/nslkdd_ablation/tristage_llm_haiku.jsonl"
OUT_PATH      = ROOT / "results/nslkdd_ablation/xgb_cost_sensitive_baseline.json"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX    = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

# Thesis Table~\ref{tab:safety-synthesis} / Section~\ref{sec:nslkdd-ablation}
# reference numbers (default-loss XGBoost and Tri-stage LLM Haiku), for the
# plain-language comparison written into the output JSON.
EXISTING_XGB_UNDER_ESC   = 0.44
EXISTING_XGB_SEVERE_REC  = 0.45
EXISTING_LLM_UNDER_ESC   = 0.16
EXISTING_LLM_SEVERE_REC  = 0.825
EXISTING_MCNEMAR_B       = 14
EXISTING_MCNEMAR_C       = 0


def risk_level(nfcrm_class: str, criticality: str) -> str:
    c_val = CRIT_MAP.get(criticality, 3)
    score = compute_risk_score(nfcrm_class, c_override=c_val, i_override=c_val, a_override=c_val)
    return "Very Low" if score.level_en == "N/A (non-attack)" else score.level_en


def mcnemar_1s(b: int, c: int) -> float:
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) * (0.5 ** n) if n else 1.0


def metrics_from_records(recs: list[dict]) -> dict:
    """Mirrors scripts/safety_table_haiku.py::metrics() exactly."""
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for r in recs:
        p, g = r["risk_level"], r["ground_truth"]
        if p not in IDX or g not in IDX:
            n += 1
            if g in SEVERE:
                sev_tot += 1
            under += 1
            worst = max(worst, IDX.get(g, 0))
            continue
        n += 1
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
    pct = lambda a, b: (a / b * 100) if b else 0.0
    return dict(n=n, exact_pct=pct(exact, n), within1_pct=pct(within1, n),
                under_pct=pct(under, n), over_pct=pct(over, n),
                severe_recall_pct=pct(sev_rec, sev_tot), sev_tot=sev_tot,
                worst_miss_levels=worst)


def build_eval_set():
    """Regenerate the identical N=50 (10/class) eval subset from raw NSL-KDD
    files, using the same code path as build_nslkdd_ablation_testset.py."""
    train_path = ROOT / "data/raw/NSL-KDD/KDDTrain+.txt"
    test_path  = ROOT / "data/raw/NSL-KDD/KDDTest+.txt"

    train_raw = load_raw(train_path)
    test_raw  = load_raw(test_path)
    train_enc, test_enc = encode_for_ml(train_raw, test_raw.copy())

    import random
    rng = random.Random(SEED)
    np.random.seed(SEED)

    eval_raw_dfs, eval_indices = [], []
    for cls in CLASSES:
        cls_idxs = test_raw.index[test_raw["label"] == cls].tolist()
        rng_sample = random.Random(SEED)
        chosen = rng_sample.sample(cls_idxs, N_PER_CLASS)
        eval_raw_dfs.append(test_raw.loc[chosen])
        eval_indices.extend(chosen)

    eval_raw_df = pd.concat(eval_raw_dfs).sample(frac=1, random_state=SEED).reset_index()
    eval_orig_indices = eval_raw_df["index"].tolist()
    eval_raw_df = eval_raw_df.drop("index", axis=1)
    eval_enc_df = test_enc.loc[eval_orig_indices].reset_index(drop=True)

    return train_enc, eval_raw_df, eval_enc_df


def train_xgb(train_enc: pd.DataFrame, cost_sensitive: bool):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    feature_cols = [c for c in train_enc.columns if c != "label"]
    X = train_enc[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(train_enc["label"].values)

    xgb = XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="mlogloss",
                        use_label_encoder=False, verbosity=0,
                        tree_method="exact", n_jobs=1)
    sw = compute_sample_weight(class_weight="balanced", y=y) if cost_sensitive else None
    xgb.fit(X, y, sample_weight=sw)
    return xgb, le, feature_cols


def main() -> None:
    print(f"Rebuilding NSL-KDD eval subset (seed={SEED}, {N_PER_CLASS}/class) ...")
    train_enc, eval_raw_df, eval_enc_df = build_eval_set()
    print(f"  eval rows: {len(eval_raw_df)}")

    if not TEST_SET_PATH.exists():
        print(f"ERROR: {TEST_SET_PATH} not found; cannot align case_ids / criticality.")
        sys.exit(1)
    frozen_cases = json.loads(TEST_SET_PATH.read_text())["cases"]
    if len(frozen_cases) != len(eval_raw_df):
        print(f"ERROR: regenerated eval set (n={len(eval_raw_df)}) does not match "
              f"frozen test_set.json (n={len(frozen_cases)}) -- sampling has diverged.")
        sys.exit(1)

    # --- default-loss XGBoost: reproducibility sanity check -----------------
    print("Training default-loss XGBoost (reproducibility check) ...")
    xgb_default, le, feature_cols = train_xgb(train_enc, cost_sensitive=False)
    X_eval = eval_enc_df[feature_cols].values.astype(float)
    default_preds = le.inverse_transform(xgb_default.predict(X_eval))
    n_match = sum(1 for i, c in enumerate(frozen_cases) if default_preds[i] == c["xgb_predicted_class"])
    match_rate = n_match / len(frozen_cases)
    print(f"  Match vs frozen test_set.json xgb_predicted_class: {n_match}/{len(frozen_cases)} = {match_rate:.1%}")
    reproducible = match_rate == 1.0
    if not reproducible:
        print("  WARNING: default-loss retrain does not exactly reproduce the frozen "
              "predictions (library-version drift is the likely cause). Proceeding, "
              "but treat the cost-sensitive comparison as approximate, not exact.")

    # --- cost-sensitive XGBoost -----------------------------------------------
    print("Training cost-sensitive (class-weighted, sklearn 'balanced') XGBoost ...")
    xgb_cs, le_cs, feature_cols_cs = train_xgb(train_enc, cost_sensitive=True)
    cs_preds = le_cs.inverse_transform(xgb_cs.predict(X_eval))
    cs_proba = xgb_cs.predict_proba(X_eval).max(axis=1)

    # class weights actually applied (for methodology disclosure)
    from sklearn.utils.class_weight import compute_class_weight
    y_all = le_cs.transform(train_enc["label"].values)
    weights = compute_class_weight("balanced", classes=np.unique(y_all), y=y_all)
    class_weight_map = {le_cs.inverse_transform([int(c)])[0]: round(float(w), 4)
                        for c, w in zip(np.unique(y_all), weights)}

    # --- build per-case records, same schema as ml_xgb.jsonl -----------------
    records = []
    for i, case in enumerate(frozen_cases):
        crit      = case["criticality"]
        gt        = case["ground_truth_risk_level"]
        pred_nsl  = cs_preds[i]
        pred_nfcrm = NSLKDD_TO_NFCRM.get(pred_nsl, "Benign")
        pred_risk = risk_level(pred_nfcrm, crit)
        records.append({
            "case_id":       case["case_id"],
            "paradigm":      "ml_xgb_cost_sensitive",
            "true_class":    case["true_attack_class"],
            "pred_class":    pred_nsl,
            "confidence":    float(cs_proba[i]),
            "risk_level":    pred_risk,
            "ground_truth":  gt,
            "correct":       pred_risk == gt,
            "criticality":   crit,
        })

    n_correct = sum(r["correct"] for r in records)
    print(f"  Cost-sensitive XGBoost risk-level accuracy: {n_correct}/{len(records)} = {n_correct/len(records):.1%}")

    m_cs = metrics_from_records(records)
    print(f"  Under-escalation: {m_cs['under_pct']:.1f}%   Severe recall: {m_cs['severe_recall_pct']:.1f}%   "
          f"Worst miss: {m_cs['worst_miss_levels']} level(s)")

    # --- default-loss metrics (recomputed here too, for a same-run comparison) ---
    default_records = []
    for i, case in enumerate(frozen_cases):
        crit = case["criticality"]
        gt   = case["ground_truth_risk_level"]
        pred_nfcrm = NSLKDD_TO_NFCRM.get(default_preds[i], "Benign")
        pred_risk  = risk_level(pred_nfcrm, crit)
        default_records.append({
            "case_id": case["case_id"], "risk_level": pred_risk, "ground_truth": gt,
        })
    m_default_retrain = metrics_from_records(default_records)

    # --- also load the ORIGINAL frozen ml_xgb.jsonl metrics for direct compare ---
    orig_ml_path = ROOT / "results/nslkdd_ablation/ml_xgb.jsonl"
    orig_records = []
    if orig_ml_path.exists():
        for ln in orig_ml_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                orig_records.append(json.loads(ln))
        m_orig = metrics_from_records(orig_records)
    else:
        m_orig = None

    # --- McNemar vs Tri-stage LLM Haiku per-case predictions ------------------
    mcnemar_result = None
    if TRISTAGE_PATH.exists():
        llm_rows = [json.loads(l) for l in TRISTAGE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        llm_by_id = {r["case_id"]: r for r in llm_rows}
        cs_by_id  = {r["case_id"]: r for r in records}

        def unsafe(r):
            p, g = r.get("risk_level", ""), r.get("ground_truth", "")
            if p not in IDX or g not in IDX:
                return True
            return IDX[p] < IDX[g]

        b = c = 0
        n_pairs = 0
        for cid in cs_by_id:
            if cid in llm_by_id:
                n_pairs += 1
                lu, mu = unsafe(llm_by_id[cid]), unsafe(cs_by_id[cid])
                if mu and not lu:
                    b += 1
                elif lu and not mu:
                    c += 1
        p_val = mcnemar_1s(b, c)
        mcnemar_result = {"n_pairs": n_pairs, "b_ml_unsafe_llm_safe": b,
                          "c_llm_unsafe_ml_safe": c, "p_one_sided": p_val}
        print(f"  McNemar (Tri-stage LLM Haiku vs cost-sensitive XGBoost): "
              f"b={b} c={c} p={p_val:.4g} (n_pairs={n_pairs})")
    else:
        print(f"  WARNING: {TRISTAGE_PATH} not found -- cannot compute paired McNemar.")

    # --- assemble output -------------------------------------------------------
    out = {
        "meta": {
            "description": (
                "Cost-sensitive (class-weighted) XGBoost baseline for NSL-KDD N=50 "
                "safety evaluation, addressing thesis Construct-Validity Threat CV-4 "
                "('No Cost-Sensitive ML Baseline')."
            ),
            "methodology": {
                "model": "XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')",
                "class_weighting": "sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced', y) "
                                   "-> per-sample weight = n_samples / (n_classes * class_count), passed as "
                                   "sample_weight to XGBClassifier.fit(); cost-sensitive reweighting per Elkan 2001",
                "class_weights_applied": class_weight_map,
                "train_source": "data/raw/NSL-KDD/KDDTrain+.txt",
                "eval_source":  "data/raw/NSL-KDD/KDDTest+.txt",
                "eval_subset":  "identical N=50 (10/class) stratified sample, seed=42, as "
                               "results/nslkdd_ablation/test_set.json",
                "seed": SEED,
                "features": "same one-hot-encoded feature set (protocol_type/service/flag + "
                           "numeric columns) as build_nslkdd_ablation_testset.py::encode_for_ml",
                "risk_level_mapping": "identical NFCRM SS6.9 lookup (pipeline.nfcrm.risk_score.compute_risk_score) "
                                     "as the existing ml_xgb / tristage arms",
                "under_over_escalation_metric": "identical to scripts/safety_table_haiku.py::metrics() "
                                                "(under = predicted risk level index < ground-truth index)",
            },
            "reproducibility": {
                "default_loss_retrain_matches_frozen_predictions": reproducible,
                "match_rate_vs_frozen_xgb_predicted_class": round(match_rate, 4),
                "note": ("A default-loss XGBoost was retrained on the identical data/seed/features "
                         "and its per-case predicted class compared against the frozen "
                         "xgb_predicted_class field in test_set.json, to confirm the pipeline is "
                         "reproduced exactly before trusting the cost-sensitive comparison." if reproducible else
                         "Retrain did NOT exactly match frozen predictions (see match_rate); treat "
                         "cost-sensitive vs baseline delta as directionally informative, not exact."),
            },
        },
        "cost_sensitive_xgboost": {
            "n": m_cs["n"],
            "risk_level_accuracy_pct": round(100 * n_correct / len(records), 1),
            "under_escalation_pct": round(m_cs["under_pct"], 1),
            "over_escalation_pct": round(m_cs["over_pct"], 1),
            "severe_recall_pct": round(m_cs["severe_recall_pct"], 1),
            "exact_pct": round(m_cs["exact_pct"], 1),
            "within1_pct": round(m_cs["within1_pct"], 1),
            "worst_miss_levels": m_cs["worst_miss_levels"],
        },
        "default_loss_xgboost_retrained_this_run": {
            "under_escalation_pct": round(m_default_retrain["under_pct"], 1),
            "severe_recall_pct": round(m_default_retrain["severe_recall_pct"], 1),
            "worst_miss_levels": m_default_retrain["worst_miss_levels"],
        },
        "default_loss_xgboost_original_frozen_file": (
            {
                "source": "results/nslkdd_ablation/ml_xgb.jsonl",
                "under_escalation_pct": round(m_orig["under_pct"], 1),
                "severe_recall_pct": round(m_orig["severe_recall_pct"], 1),
                "worst_miss_levels": m_orig["worst_miss_levels"],
            } if m_orig is not None else None
        ),
        "thesis_reference_numbers": {
            "xgboost_default_loss_under_escalation_pct": 100 * EXISTING_XGB_UNDER_ESC,
            "xgboost_default_loss_severe_recall_pct": 100 * EXISTING_XGB_SEVERE_REC,
            "tristage_llm_haiku_under_escalation_pct": 100 * EXISTING_LLM_UNDER_ESC,
            "tristage_llm_haiku_severe_recall_pct": 100 * EXISTING_LLM_SEVERE_REC,
            "existing_mcnemar_tristage_llm_vs_xgboost": {
                "b": EXISTING_MCNEMAR_B, "c": EXISTING_MCNEMAR_C, "p": 6.1e-5,
            },
        },
        "mcnemar_tristage_llm_haiku_vs_cost_sensitive_xgboost": mcnemar_result,
        "plain_language_comparison": (
            f"Class-weighted (cost-sensitive) XGBoost under-escalates "
            f"{round(m_cs['under_pct'], 1)}% of cases with {round(m_cs['severe_recall_pct'], 1)}% severe recall, "
            f"vs the existing default-loss XGBoost baseline's 44% under-escalation / 45% severe recall, "
            f"and vs the tri-stage LLM (Haiku)'s 16% under-escalation / 82.5% severe recall. "
            + (
                "Cost-sensitive reweighting narrows the gap materially versus default-loss XGBoost but does "
                "not close it: the LLM-grounded pipeline remains the safer paradigm on this severely "
                "class-imbalanced dataset."
                if m_cs["under_pct"] < EXISTING_XGB_UNDER_ESC * 100 - 5 and m_cs["under_pct"] > EXISTING_LLM_UNDER_ESC * 100
                else (
                    "Cost-sensitive reweighting does not meaningfully narrow the safety gap versus "
                    "default-loss XGBoost; the LLM-grounded pipeline remains substantially safer."
                    if m_cs["under_pct"] >= EXISTING_XGB_UNDER_ESC * 100 - 5
                    else "Cost-sensitive reweighting closes most or all of the gap to the tri-stage LLM arm "
                         "on this narrow under-escalation metric -- CV-4's rival explanation is supported."
                )
            )
        ),
    }

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
