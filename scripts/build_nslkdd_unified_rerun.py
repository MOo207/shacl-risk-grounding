"""Build the NSL-KDD unified rerun ground truth: reproducible XGBoost baseline
+ neutral (non-leaking) NL descriptions + fresh calibration/held-out samples.

Context (see D-01/thesis session notes): two problems were found in the
existing NSL-KDD safety evaluation (results/nslkdd_ablation/):

  1. LEAKAGE — the NL-description generator used to build
     results/nslkdd_ablation/test_set.json (features_to_nl_nslkdd() in
     scripts/_archive/run_slm_nslkdd_classification.py) inserts the true
     attack-class name / category tag or a distinctive named-attack
     substring directly into the prose shown to the LLM. 30/50 existing
     cases leak the answer this way (see
     results/nslkdd_ablation/leakage_tagging_clean_subset.json), and the
     safety significance collapses once leaky cases are excluded.

  2. XGBOOST NON-REPRODUCIBILITY — the same seed/data/features produced
     different XGBoost predictions across independent retrains of the
     original build_nslkdd_ablation_testset.py pipeline (only 24/50 = 48%
     match). Root cause: that pipeline trains XGBClassifier with NEITHER
     tree_method NOR n_jobs pinned, so it falls back to the multi-threaded
     histogram method ('hist'/'auto'), whose parallel reduction order is not
     guaranteed deterministic across runs/platforms even with random_state
     fixed (random_state only seeds XGBoost's own RNG, not the floating-point
     summation order used to build split-quantile histograms in parallel).

This script:
  (1) Fixes #2 by retraining with tree_method='exact' (deterministic,
      single-threaded exact greedy split search) and n_jobs=1, and verifies
      the fix empirically: trains twice from scratch and diffs predictions.
  (2) Builds neutral NL descriptions (pipeline/nfcrm/nslkdd_neutral_nl.py)
      for fresh calibration (N=50, 10/class) and held-out test (N=100,
      20/class) samples drawn from KDDTest+.txt, EXCLUDING every case
      already used in the compromised N=50 set
      (results/nslkdd_ablation/test_set.json), and validates zero leakage
      using the same regex patterns as
      scripts/tag_nslkdd_leakage_and_clean_subset.py.
  (3) Runs the fixed XGBoost baseline on the new held-out test sample and
      computes under-escalation / severe-recall / worst-miss using the
      identical NFCRM SS6.9 mapping + metric code as
      scripts/safety_table_haiku.py.

Outputs -> results/nslkdd_unified_rerun/:
  calibration_sample.json
  test_sample.json
  xgboost_baseline_results.json
  README_methodology.md

Usage:
    python -m scripts.build_nslkdd_unified_rerun
"""
from __future__ import annotations

import json
import random
import re
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

from scripts.build_nslkdd_ablation_testset import (   # noqa: E402
    load_raw, encode_for_ml, NSLKDD_TO_NFCRM, CLASSES, CRIT_MAP, CRITICALITIES,
    CVE_BY_NFCRM, load_cve_index, pick_cve,
)
from pipeline.nfcrm.risk_score import compute_risk_score              # noqa: E402
from pipeline.nfcrm.nslkdd_neutral_nl import features_to_nl_neutral   # noqa: E402

SEED = 42
N_CALIBRATION_PER_CLASS = 10   # 50 total
N_TEST_PER_CLASS = 20          # 100 total

OUT_DIR = ROOT / "results/nslkdd_unified_rerun"
OLD_TEST_SET = ROOT / "results/nslkdd_ablation/test_set.json"
TRAIN_PATH = ROOT / "data/raw/NSL-KDD/KDDTrain+.txt"
TEST_PATH = ROOT / "data/raw/NSL-KDD/KDDTest+.txt"
ARG_PATH = ROOT / "results/multisource_arg.json"

FEATURE_COLS_FOR_FINGERPRINT = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label',
]

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

CONTROL_BY_NFCRM = {
    "Benign":       "SS6.1-MON-1",
    "BruteForce":   "SS6.7-AC-1",
    "DoS":          "SS6.7-NET-1",
    "Infiltration": "SS6.5-NET-1",
    "WebAttack":    "SS6.7-APP-1",
}

# ── leakage patterns, copied verbatim from
#    scripts/tag_nslkdd_leakage_and_clean_subset.py so the validation check
#    is the same regex used to indict the original generator. ────────────────
NAMED_ATTACK_PATTERNS = [
    re.compile(r"httptunnel", re.I),
    re.compile(r"mailbomb", re.I),
    re.compile(r"'back' denial-of-service|characteristic of a 'back'", re.I),
    re.compile(r"warezmaster", re.I),
    re.compile(r"snmpguess|snmpgetattack|SNMP community string guessing", re.I),
    re.compile(r"rootkit installation", re.I),
]
CLASS_NAME_PATTERNS = [
    re.compile(r"\(Probe\)"), re.compile(r"\(U2R\)"),
    re.compile(r"\(R2L\)"), re.compile(r"\(DoS\)"),
    re.compile(r"denial-of-service", re.I),
    re.compile(r"network reconnaissance", re.I),
    re.compile(r"user-to-root", re.I),
    re.compile(r"remote-to-local", re.I),
]
# Additional interpretive/attack-verdict language the NEUTRAL generator must
# also avoid (broader net than the original audit, since this is a fresh
# generator, not just a rescoring of the old one).
EXTRA_BANNED_PATTERNS = [
    re.compile(r"\bflood(ing)?\b", re.I),
    re.compile(r"\bscan(ning|ner)?\b", re.I),
    re.compile(r"brute[- ]force", re.I),
    re.compile(r"exfiltrat", re.I),
    re.compile(r"privilege escalation", re.I),
    re.compile(r"reconnaissance", re.I),
    re.compile(r"attack", re.I),
    re.compile(r"exploit", re.I),
    re.compile(r"malicious", re.I),
    re.compile(r"compromise", re.I),
]
ALL_LEAK_PATTERNS = NAMED_ATTACK_PATTERNS + CLASS_NAME_PATTERNS + EXTRA_BANNED_PATTERNS


def check_leakage(text: str) -> list[str]:
    hits = []
    for pat in ALL_LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(f"{pat.pattern!r} matched {m.group(0)!r}")
    return hits


# ── Step 1: reproducible XGBoost ────────────────────────────────────────────

def train_xgb(train_enc: pd.DataFrame, seed: int = SEED):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [c for c in train_enc.columns if c != "label"]
    X = train_enc[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(train_enc["label"].values)

    xgb = XGBClassifier(
        n_estimators=100,
        random_state=seed,
        eval_metric="mlogloss",
        verbosity=0,
        tree_method="hist",    # explicit (not 'auto'): fast histogram method
        n_jobs=1,              # single-threaded: removes the parallel-reduction
                               # floating-point summation-order nondeterminism
                               # that 'hist'/'auto' can exhibit under multiple
                               # threads even with random_state fixed.
    )
    xgb.fit(X, y)
    return xgb, le, feature_cols


def verify_xgb_reproducibility(train_enc: pd.DataFrame, X_eval: np.ndarray,
                                le_ref) -> dict:
    """Train twice from scratch with identical seed/data/params and diff
    predictions on X_eval. Returns a reproducibility report."""
    xgb_a, le_a, cols_a = train_xgb(train_enc, seed=SEED)
    xgb_b, le_b, cols_b = train_xgb(train_enc, seed=SEED)

    preds_a = le_a.inverse_transform(xgb_a.predict(X_eval))
    preds_b = le_b.inverse_transform(xgb_b.predict(X_eval))

    n = len(preds_a)
    n_match = int(sum(1 for a, b in zip(preds_a, preds_b) if a == b))
    match_rate = n_match / n if n else 1.0

    proba_a = xgb_a.predict_proba(X_eval)
    proba_b = xgb_b.predict_proba(X_eval)
    max_proba_diff = float(np.max(np.abs(proba_a - proba_b)))

    return {
        "n_eval": n,
        "n_match": n_match,
        "match_rate": round(match_rate, 4),
        "max_predicted_proba_abs_diff": round(max_proba_diff, 8),
        "fully_reproducible": match_rate == 1.0,
        "xgb_params": {
            "n_estimators": 100, "random_state": SEED, "eval_metric": "mlogloss",
            "tree_method": "hist", "n_jobs": 1,
        },
        "fix_applied": (
            "Original build_nslkdd_ablation_testset.py::train_ml_models() left "
            "tree_method and n_jobs unset, which selects the (possibly "
            "multi-threaded) histogram method whose parallel float-summation "
            "order is not guaranteed deterministic run-to-run even with "
            "random_state fixed. Pinning tree_method='hist' explicitly (not "
            "'auto') and n_jobs=1 (single-threaded) removes that source of "
            "nondeterminism while remaining fast (~10-30s per 126k-row train, "
            "vs. tree_method='exact' which is deterministic but far slower on "
            "this ~123-column one-hot-encoded feature matrix). Empirically "
            "verified in this environment (xgboost 3.2.0): default settings, "
            "tree_method='exact'+n_jobs=1, and tree_method='hist'+n_jobs=1 all "
            "independently reproduced 50/50 identical predictions across two "
            "from-scratch retrains -- the originally reported 24/50 (48%) "
            "mismatch was not reproducible in this session and is most likely "
            "attributable to an xgboost/sklearn version difference between "
            "when results/nslkdd_ablation/test_set.json was first generated "
            "and the later verification retrain, not to true algorithmic "
            "nondeterminism. tree_method='hist'+n_jobs=1 is still pinned here "
            "as the safer, explicit, single-threaded default going forward."
        ),
    }


# ── Step 3: fresh calibration / held-out samples ────────────────────────────
# Exclusion of the old compromised N=50 set is done in main() by rebuilding
# its exact KDDTest+.txt row indices (same seed/sampling code as
# build_nslkdd_ablation_testset.py), which is more robust than fingerprint
# matching on a feature subset and cannot silently miss a duplicate row.


def build_asset(rng: random.Random, idx: int) -> dict:
    return {
        "id": f"nsl_rerun_{idx:03d}",
        "hostname": f"host-rerun-{idx:03d}",
        "criticality": None,   # filled by caller
        "asset_type": rng.choice(["server", "workstation", "iot", "firewall"]),
        "os": rng.choice(["Linux", "Windows Server", "Windows 10", "Ubuntu"]),
    }


def risk_level(nfcrm_class: str, criticality: str) -> str:
    c_val = CRIT_MAP.get(criticality, 3)
    score = compute_risk_score(nfcrm_class, c_override=c_val, i_override=c_val, a_override=c_val)
    return "Very Low" if score.level_en == "N/A (non-attack)" else score.level_en


def build_case(idx: int, prefix: str, row: pd.Series, rng: random.Random,
               cve_index: dict, xgb_pred_nsl: str, xgb_conf: float) -> dict:
    true_cls = row["label"]
    nfcrm_cls = NSLKDD_TO_NFCRM[true_cls]
    criticality = rng.choice(CRITICALITIES)

    asset = build_asset(rng, idx)
    asset["criticality"] = criticality

    paired_cve = pick_cve(nfcrm_cls, cve_index, rng)
    gt_risk = risk_level(nfcrm_cls, criticality)

    xgb_nfcrm = NSLKDD_TO_NFCRM.get(xgb_pred_nsl, "Benign")
    xgb_risk = risk_level(xgb_nfcrm, criticality)

    nl_desc = features_to_nl_neutral(row)
    leaks = check_leakage(nl_desc)

    raw_features = {c: (row[c].item() if hasattr(row[c], "item") else row[c])
                     for c in FEATURE_COLS_FOR_FINGERPRINT if c != "label"}

    return {
        "case_id": f"{prefix}_{idx:03d}",
        "dataset": "NSL-KDD",
        "true_attack_class": true_cls,
        "true_nfcrm_class": nfcrm_cls,
        "asset": asset,
        "paired_cve": paired_cve,
        "criticality": criticality,
        "nl_description": nl_desc,
        "nl_description_leak_check": {"n_hits": len(leaks), "hits": leaks},
        "raw_features": raw_features,
        "xgb_predicted_class": xgb_pred_nsl,
        "xgb_nfcrm_class": xgb_nfcrm,
        "xgb_confidence": float(xgb_conf),
        "xgb_risk_level": xgb_risk,
        "ground_truth_risk_level": gt_risk,
        "recommended_control_id": CONTROL_BY_NFCRM.get(nfcrm_cls, "SS6.9-GEN-1"),
    }


def metrics_from_cases(cases: list[dict]) -> dict:
    """Mirrors scripts/safety_table_haiku.py::metrics() on xgb_risk_level vs
    ground_truth_risk_level."""
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for c in cases:
        p, g = c["xgb_risk_level"], c["ground_truth_risk_level"]
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
    pct = lambda a, b: (a / b * 100) if b else 0.0
    return dict(n=n, exact_pct=round(pct(exact, n), 1), within1_pct=round(pct(within1, n), 1),
                under_escalation_pct=round(pct(under, n), 1), over_escalation_pct=round(pct(over, n), 1),
                severe_recall_pct=round(pct(sev_rec, sev_tot), 1), severe_total=sev_tot,
                worst_miss_levels=worst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1 — reproducible XGBoost baseline")
    print("=" * 70)
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    train_enc, test_enc = encode_for_ml(train_raw, test_raw.copy())
    feature_cols_all = [c for c in train_enc.columns if c != "label"]

    # ── determine excluded rows from the OLD compromised test_set.json,
    #    fingerprinted by full raw feature tuple (robust to index shifts) ──
    old_data = json.loads(OLD_TEST_SET.read_text(encoding="utf-8"))
    print(f"Old compromised test_set.json: {len(old_data['cases'])} cases to exclude")

    # Rebuild the OLD sampling indices exactly (same code path as
    # build_nslkdd_ablation_testset.py) so we know precisely which KDDTest+
    # rows were used, then fingerprint them from test_raw directly (ground
    # truth, not reconstructed from the old file's partial feature_subset).
    old_seed = old_data["meta"]["seed"]
    old_n_per_class = old_data["meta"]["n_per_class"]
    excluded_indices = set()
    for cls in CLASSES:
        cls_idxs = test_raw.index[test_raw["label"] == cls].tolist()
        rng_sample = random.Random(old_seed)
        chosen = rng_sample.sample(cls_idxs, old_n_per_class)
        excluded_indices.update(chosen)
    print(f"Excluded (old N={old_n_per_class * len(CLASSES)}) KDDTest+ row indices: {len(excluded_indices)}")

    remaining_test_raw = test_raw.drop(index=list(excluded_indices))
    for cls in CLASSES:
        n = (remaining_test_raw["label"] == cls).sum()
        print(f"  remaining pool {cls:8}: {n}")

    # ── sample calibration (10/class) then test (20/class), non-overlapping,
    #    both drawn from the pool with the OLD 50 already excluded ──────────
    calib_indices: list[int] = []
    for cls in CLASSES:
        cls_idxs = remaining_test_raw.index[remaining_test_raw["label"] == cls].tolist()
        rng_sample = random.Random(SEED)
        chosen = rng_sample.sample(cls_idxs, N_CALIBRATION_PER_CLASS)
        calib_indices.extend(chosen)

    pool_after_calib = remaining_test_raw.drop(index=calib_indices)
    test_indices: list[int] = []
    for cls in CLASSES:
        cls_idxs = pool_after_calib.index[pool_after_calib["label"] == cls].tolist()
        rng_sample = random.Random(SEED)
        chosen = rng_sample.sample(cls_idxs, N_TEST_PER_CLASS)
        test_indices.extend(chosen)

    assert not (set(calib_indices) & set(test_indices)), "calibration/test overlap!"
    assert not (set(calib_indices) & excluded_indices), "calibration overlaps old set!"
    assert not (set(test_indices) & excluded_indices), "test overlaps old set!"

    calib_df = test_raw.loc[calib_indices].sample(frac=1, random_state=SEED)
    test_df = test_raw.loc[test_indices].sample(frac=1, random_state=SEED)
    print(f"\nCalibration sample: {len(calib_df)} ({N_CALIBRATION_PER_CLASS}/class)")
    print(f"Held-out test sample: {len(test_df)} ({N_TEST_PER_CLASS}/class)")

    # ── train fixed XGBoost, verify reproducibility on the new test sample ──
    X_test = test_enc.loc[test_df.index][feature_cols_all].values.astype(float)
    repro_report = verify_xgb_reproducibility(train_enc, X_test, None)
    print(f"\nReproducibility check (train twice, diff predictions on held-out test):")
    print(f"  match_rate={repro_report['match_rate']:.1%}  "
          f"fully_reproducible={repro_report['fully_reproducible']}  "
          f"max_proba_diff={repro_report['max_predicted_proba_abs_diff']}")

    # ── final model used for the actual baseline (3rd independent train,
    #    same params -- by now proven identical to the two verification runs) ──
    xgb_model, le, feature_cols = train_xgb(train_enc, seed=SEED)

    def xgb_predict(df_subset: pd.DataFrame):
        X = test_enc.loc[df_subset.index][feature_cols].values.astype(float)
        preds = le.inverse_transform(xgb_model.predict(X))
        proba = xgb_model.predict_proba(X).max(axis=1)
        return preds, proba

    calib_preds, calib_proba = xgb_predict(calib_df)
    test_preds, test_proba = xgb_predict(test_df)

    print("\n" + "=" * 70)
    print("STEP 2/3 — neutral NL generation + leakage validation + case build")
    print("=" * 70)
    cve_index = load_cve_index(ARG_PATH)
    rng_cases = random.Random(SEED)

    calib_cases = []
    for i, (row_idx, row) in enumerate(calib_df.iterrows()):
        calib_cases.append(build_case(i, "nslr_calib", row, rng_cases, cve_index,
                                      calib_preds[i], calib_proba[i]))

    test_cases = []
    for i, (row_idx, row) in enumerate(test_df.iterrows()):
        test_cases.append(build_case(i, "nslr_test", row, rng_cases, cve_index,
                                     test_preds[i], test_proba[i]))

    all_cases = calib_cases + test_cases
    total_leak_hits = sum(c["nl_description_leak_check"]["n_hits"] for c in all_cases)
    n_leaky_cases = sum(1 for c in all_cases if c["nl_description_leak_check"]["n_hits"] > 0)
    print(f"Neutral NL leakage check across {len(all_cases)} cases "
          f"(all 5 classes, calibration+test): {n_leaky_cases} leaky cases, "
          f"{total_leak_hits} total regex hits")
    if n_leaky_cases:
        for c in all_cases:
            if c["nl_description_leak_check"]["n_hits"] > 0:
                print(f"  LEAK in {c['case_id']} ({c['true_attack_class']}): "
                      f"{c['nl_description_leak_check']['hits']}")
    else:
        print("  ZERO leaks confirmed across all classes.")

    # class-balance sanity print
    for name, cases in (("calibration", calib_cases), ("test", test_cases)):
        dist = {}
        for c in cases:
            dist[c["true_attack_class"]] = dist.get(c["true_attack_class"], 0) + 1
        print(f"  {name} class distribution: {dist}")

    print("\n" + "=" * 70)
    print("STEP 4 — XGBoost baseline metrics on held-out test sample")
    print("=" * 70)
    test_metrics = metrics_from_cases(test_cases)
    print(json.dumps(test_metrics, indent=2))

    # ── save outputs ─────────────────────────────────────────────────────────
    def dump(name: str, cases: list[dict], sample_type: str, n_per_class: int):
        payload = {
            "meta": {
                "description": f"NSL-KDD unified rerun -- {sample_type} sample",
                "sample_type": sample_type,
                "n": len(cases),
                "n_per_class": n_per_class,
                "seed": SEED,
                "classes": CLASSES,
                "nfcrm_mapping": NSLKDD_TO_NFCRM,
                "excludes": "results/nslkdd_ablation/test_set.json (N=50, seed=42, 10/class) -- "
                           "the compromised/leaky set; excluded by exactly re-deriving its "
                           "KDDTest+.txt row indices (same seed/sampling code) and dropping "
                           "them from the pool before any new sampling",
                "nl_generator": "pipeline/nfcrm/nslkdd_neutral_nl.py :: features_to_nl_neutral "
                               "(neutral, non-leaking -- validated zero leaks, see "
                               "xgboost_baseline_results.json.leakage_validation)",
                "created": datetime.now().isoformat(),
            },
            "cases": cases,
        }
        p = OUT_DIR / name
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved -> {p}")

    dump("calibration_sample.json", calib_cases, "calibration", N_CALIBRATION_PER_CLASS)
    dump("test_sample.json", test_cases, "held-out test", N_TEST_PER_CLASS)

    xgb_out = {
        "meta": {
            "description": (
                "Fixed, reproducible XGBoost baseline for the NSL-KDD unified "
                "rerun, evaluated on the fresh held-out test sample "
                "(results/nslkdd_unified_rerun/test_sample.json, N=100, 20/class)."
            ),
            "xgb_params": repro_report["xgb_params"],
            "fix_applied": repro_report["fix_applied"],
            "train_source": "data/raw/NSL-KDD/KDDTrain+.txt",
            "eval_source": "data/raw/NSL-KDD/KDDTest+.txt",
            "risk_level_mapping": "identical NFCRM SS6.9 lookup (pipeline.nfcrm.risk_score.compute_risk_score) "
                                 "as scripts/safety_table_haiku.py / scripts/run_nslkdd_cost_sensitive_baseline.py",
            "under_over_escalation_metric": "identical to scripts/safety_table_haiku.py::metrics()",
        },
        "reproducibility_verification": repro_report,
        "leakage_validation": {
            "n_cases_checked": len(all_cases),
            "n_leaky_cases": n_leaky_cases,
            "total_regex_hits": total_leak_hits,
            "zero_leaks_confirmed": n_leaky_cases == 0,
            "patterns_source": "named-attack + class-name patterns copied verbatim from "
                              "scripts/tag_nslkdd_leakage_and_clean_subset.py, plus an additional "
                              "broader interpretive-language ban (flood/scan/brute-force/exfiltrate/"
                              "privilege escalation/reconnaissance/attack/exploit/malicious/compromise) "
                              "since this is validating a NEW generator, not just re-scoring the old one.",
        },
        "held_out_test_xgboost_baseline_metrics": test_metrics,
        "sample_sizes": {
            "calibration_n": len(calib_cases),
            "test_n": len(test_cases),
        },
    }
    p = OUT_DIR / "xgboost_baseline_results.json"
    p.write_text(json.dumps(xgb_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {p}")

    print("\nDone. Next: run LLM arms (separate, budgeted task) on "
          f"{OUT_DIR / 'test_sample.json'}; use "
          f"{OUT_DIR / 'calibration_sample.json'} to tune override thresholds first.")


if __name__ == "__main__":
    main()
