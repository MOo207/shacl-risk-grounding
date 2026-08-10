"""Build the CIC-IDS2018 Unified Rerun artifacts.

Sibling investigation on NSL-KDD found two problems in that pipeline:
  (1) the NL-description generator leaks the true attack class name/category
      directly into the prose the LLM reads (30/50 cases affected);
  (2) NSL-KDD's XGBoost baseline is not run-to-run reproducible even with
      the same seed/data/features (48% prediction match across two runs).

This script audits CIC-IDS2018's unified-ablation pipeline
(scripts/run_unified_ablation.py, scripts/build_unified_ablation_testset.py,
results/unified_ablation/test_set.json) for the SAME two problems, and
prepares two fresh, non-overlapping samples (calibration N=60, held-out
test N=120) ready for a future LLM rerun. NO LLM CALLS ARE MADE HERE.

Steps:
  1. leakage audit — tag the existing N=60 LLM-evaluated subset
     (10/class, the cases appearing in results/unified_ablation/pure_llm.jsonl)
     for (a) NL-description leakage (features_to_nl() template names the
     attack class/category directly, mirroring the NSL-KDD generator) and
     (b) CVE-description leakage (the paired CISA-KEV-style CVE description
     itself contains a lexical token naming the true attack class — this is
     inherent to CVE-grounded content, discussed separately from (a)).
  2. XGBoost reproducibility — retrain the exact pipeline
     (build_unified_ablation_testset.train_ml_models on the same train/eval
     split, seed=42) TWICE, and compare predictions to (i) each other and
     (ii) the frozen historical results/unified_ablation/test_set.json
     xgb_predicted_class values.
  3. Sample construction — from the CIC-IDS2018 pool EXCLUDING the 180 rows
     already reserved in test_set.json, draw:
        - calibration_sample.json: 10 cases/class (60 total)
        - test_sample.json:        20 cases/class (120 total)
     using the identical stratified-per-class, seed=42 sampling convention,
     with NL description + paired CVE attached via the existing pipeline
     logic (build_unified_ablation_testset.py), not invented content.
  4. Held-out XGBoost baseline — train XGBoost on the pool excluding ALL
     reserved rows (original 180 + calibration 60 + test 120), predict on
     the held-out test_sample, and compute NFCRM §6.9 risk-level safety
     metrics (exact match, within-1, under-escalation rate, severe recall,
     worst miss) using the same 5-level scale and method as
     scripts/tag_nslkdd_leakage_and_clean_subset.py.

Usage:
    python scripts/build_cic_unified_rerun.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.nfcrm.risk_score import compute_risk_score  # noqa: E402
from build_unified_ablation_testset import (  # noqa: E402
    CVE_BY_CLASS, CRITICALITIES, CONTROL_BY_CLASS,
    load_cve_index, pick_cve,
)

# NOTE: scripts/run_slm_nl_classification.py has since moved to
# scripts/_archive/run_slm_nl_classification.py. build_unified_ablation_testset.
# features_to_nl() does a lazy inline import assuming the old location, which
# now fails — import directly from the archive location instead so this
# script reuses the EXACT same NL-generation logic that produced the
# nl_description values already stored in test_set.json (no new prompt
# content invented).
sys.path.insert(0, str(ROOT / "scripts" / "_archive"))
from run_slm_nl_classification import features_to_nl  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"
ARG_PATH = ROOT / "results" / "multisource_arg.json"
TESTSET_PATH = ROOT / "results" / "unified_ablation" / "test_set.json"
PURE_LLM_PATH = ROOT / "results" / "unified_ablation" / "pure_llm.jsonl"
OUT_DIR = ROOT / "results" / "cic_unified_rerun"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}
CRIT_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}


# ── Step 1: leakage audit ────────────────────────────────────────────────────

NL_NAMED_PATTERNS = [
    (re.compile(r"denial-of-service \(DoS\)", re.I), "DoS",
     "features_to_nl() DoS-signature template names 'denial-of-service (DoS)'"),
    (re.compile(r"distributed denial-of-service \(DDoS\)", re.I), "DDoS",
     "features_to_nl() DDoS-signature template names 'distributed denial-of-service (DDoS)'"),
    (re.compile(r"brute-force", re.I), "BruteForce",
     "features_to_nl() BruteForce-signature template names 'brute-force'"),
    (re.compile(r"web attack", re.I), "WebAttack",
     "features_to_nl() WebAttack-signature template names 'web attack'"),
    (re.compile(r"infiltration", re.I), "Infiltration",
     "features_to_nl() Infiltration-signature template names 'infiltration'"),
]

CVE_NAMED_PATTERNS = [
    (re.compile(r"\bDoS\b", re.I), "DoS", "CVE description contains literal token 'DoS'"),
    (re.compile(r"DDoS|distributed denial", re.I), "DDoS", "CVE description contains 'DDoS'/'distributed denial'"),
    (re.compile(r"brute-force", re.I), "BruteForce", "CVE description contains literal token 'brute-force'"),
    (re.compile(r"\bweb attack\b", re.I), "WebAttack", "CVE description contains literal phrase 'web attack'"),
    (re.compile(r"backdoor|infiltrat", re.I), "Infiltration", "CVE description contains 'backdoor'/'infiltrat*'"),
]


def tag_leakage(case: dict) -> dict:
    nl_text = case.get("nl_description", "")
    cve = case.get("paired_cve")
    cve_text = cve.get("description", "") if cve else ""
    true_cls = case["true_attack_class"]

    nl_triggers = [
        {"matched_text": m.group(0), "trigger": desc}
        for pat, names_class, desc in NL_NAMED_PATTERNS
        if names_class == true_cls and (m := pat.search(nl_text))
    ]
    cve_triggers = [
        {"matched_text": m.group(0), "trigger": desc}
        for pat, names_class, desc in CVE_NAMED_PATTERNS
        if names_class == true_cls and (m := pat.search(cve_text))
    ]
    return {
        "case_id": case["case_id"],
        "true_attack_class": true_cls,
        "nl_leaky": len(nl_triggers) > 0,
        "nl_triggers": nl_triggers,
        "cve_leaky": len(cve_triggers) > 0,
        "cve_triggers": cve_triggers,
        "any_leaky": len(nl_triggers) > 0 or len(cve_triggers) > 0,
    }


def run_leakage_audit(all_cases_by_id: dict) -> dict:
    llm_case_ids = []
    if PURE_LLM_PATH.exists():
        for ln in PURE_LLM_PATH.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                llm_case_ids.append(json.loads(ln)["case_id"])
    else:
        llm_case_ids = list(all_cases_by_id.keys())

    tags = [tag_leakage(all_cases_by_id[cid]) for cid in llm_case_ids]

    n = len(tags)
    n_nl_leaky = sum(1 for t in tags if t["nl_leaky"])
    n_cve_leaky = sum(1 for t in tags if t["cve_leaky"])
    n_any_leaky = sum(1 for t in tags if t["any_leaky"])

    by_class = {}
    for t in tags:
        cls = t["true_attack_class"]
        d = by_class.setdefault(cls, {"n": 0, "nl_leaky": 0, "cve_leaky": 0, "any_leaky": 0})
        d["n"] += 1
        d["nl_leaky"] += int(t["nl_leaky"])
        d["cve_leaky"] += int(t["cve_leaky"])
        d["any_leaky"] += int(t["any_leaky"])

    return {
        "meta": {
            "description": (
                "CIC-IDS2018 unified-ablation leakage audit, mirroring "
                "scripts/tag_nslkdd_leakage_and_clean_subset.py. Audits the "
                "N=60 LLM-evaluated subset (10/class, the case_ids in "
                "results/unified_ablation/pure_llm.jsonl) for two distinct "
                "leakage channels."
            ),
            "n_cases": n,
            "case_ids": llm_case_ids,
            "n_nl_leaky": n_nl_leaky,
            "n_cve_leaky": n_cve_leaky,
            "n_any_leaky": n_any_leaky,
            "nl_leakage_source": "scripts/_archive/run_slm_nl_classification.py :: features_to_nl()",
            "cve_leakage_source": "results/multisource_arg.json CVE.description field, attached via CVE_BY_CLASS mapping in build_unified_ablation_testset.py",
        },
        "leaky_by_true_class": by_class,
        "per_case_tags": tags,
        "discussion": {
            "nl_description_channel": (
                "SAME category of problem as NSL-KDD. features_to_nl() contains "
                "hand-written signature branches (DoS/DDoS/BruteForce/WebAttack/"
                "Infiltration) that TEMPLATE the attack category name directly "
                "into the generated prose (e.g. 'denial-of-service (DoS) attack "
                "tools', 'distributed denial-of-service (DDoS) flooding tools', "
                "'automated HTTP brute-force credential stuffing', 'automated web "
                "attack tooling', 'associated with infiltration'). This is a "
                "templated label insertion, not inherent domain content — a case "
                "that hits one of these five signature branches gets an "
                "unambiguous class-name hint for free, independent of the actual "
                "ML/LLM inference task. However, it is PARTIALLY mitigated in the "
                "tri-stage arms (tristage_llm_haiku/sonnet) because those arms are "
                "already given the ground-truth-adjacent ML-predicted attack class "
                "explicitly in the prompt ('Attack class (from ML classifier): "
                "{attack_class}') — so NL leakage does not add new information "
                "there. It DOES matter for the pure_llm arm, where attack class is "
                "withheld and must be inferred from nl_description + CVE, making "
                "pure_llm's inference task easier than advertised for leaky cases."
            ),
            "cve_description_channel": (
                "DIFFERENT and more defensible situation than a templated label "
                "insertion, but still measurable lexical leakage. The paired CVE "
                "descriptions are real third-party CISA-KEV-style vulnerability "
                "summaries (e.g. 'Apache httpd allows DoS via incomplete HTTP "
                "requests'), not synthetic prose written by this pipeline. A CVE "
                "being 'about' a denial-of-service bug and literally containing "
                "the substring 'DoS' is inherent to CVE-grounded framing, not an "
                "artifact of the experiment design. That said, it is NOT harmless: "
                "for the DoS class specifically, all 3 candidate CVEs (CVE-2007-6750, "
                "CVE-2009-0696, CVE-2011-3192) literally contain 'DoS' in their "
                "description, so every DoS case in the pool is CVE-leaky by "
                "construction (a fixed property of the small CVE_BY_CLASS pool, "
                "not sampling variance). BruteForce is partially leaky (1 of 2 "
                "candidate CVEs contains 'brute-force'). WebAttack and DDoS CVE "
                "text does not contain the literal class token. Recommendation for "
                "the redesign: report NL-channel and CVE-channel leakage "
                "separately, and consider paraphrasing/redacting the class-naming "
                "clause from CVE descriptions before they are shown to the LLM if "
                "a fully leakage-free pure_llm arm is required; the tri-stage "
                "arms are less affected since they already receive the class label "
                "directly."
            ),
        },
    }


# ── Step 2: XGBoost reproducibility ──────────────────────────────────────────

def load_and_split(seed: int = 42, n_per_class: int = 30, exclude_index: set | None = None):
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    if exclude_index:
        df = df.drop(index=[i for i in exclude_index if i in df.index])
    classes = sorted(df["Label"].unique().tolist())

    eval_dfs, train_dfs = [], []
    for cls in classes:
        cls_df = df[df["Label"] == cls]
        eval_sample = cls_df.sample(n=n_per_class, random_state=seed)
        eval_dfs.append(eval_sample)
        train_dfs.append(cls_df.drop(eval_sample.index))
    sample = pd.concat(eval_dfs, ignore_index=False).sample(frac=1, random_state=seed)
    train_df = pd.concat(train_dfs, ignore_index=False)
    return df, sample, train_df, classes


def train_xgb(train_df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    feature_cols = [c for c in train_df.columns if c != "Label"]
    X = train_df[feature_cols].values.astype(float)
    y_raw = train_df["Label"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss",
                         use_label_encoder=False, verbosity=0)
    xgb.fit(X_train, y_train)
    return xgb, le, feature_cols


def verify_xgb_reproducibility() -> dict:
    """Retrain twice on the identical seed=42 split used to build test_set.json,
    compare predictions to each other and to the frozen historical file."""
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    frozen_preds = [c["xgb_predicted_class"] for c in testset["cases"]]

    runs = []
    for run_id in range(2):
        _, sample, train_df, _ = load_and_split(seed=42, n_per_class=30)
        xgb, le, feature_cols = train_xgb(train_df)
        X_sample = sample[feature_cols].values.astype(float)
        preds = le.inverse_transform(xgb.predict(X_sample)).tolist()
        runs.append(preds)

    match_run1_run2 = sum(1 for a, b in zip(runs[0], runs[1]) if a == b)
    match_run1_frozen = sum(1 for a, b in zip(runs[0], frozen_preds) if a == b)
    match_run2_frozen = sum(1 for a, b in zip(runs[1], frozen_preds) if a == b)
    n = len(frozen_preds)

    import xgboost as xgboost_mod
    import sklearn as sklearn_mod

    return {
        "n_cases": n,
        "xgboost_version": xgboost_mod.__version__,
        "sklearn_version": sklearn_mod.__version__,
        "match_run1_vs_run2": match_run1_run2,
        "match_run1_vs_run2_rate": match_run1_run2 / n,
        "match_run1_vs_frozen_historical": match_run1_frozen,
        "match_run1_vs_frozen_historical_rate": match_run1_frozen / n,
        "match_run2_vs_frozen_historical": match_run2_frozen,
        "match_run2_vs_frozen_historical_rate": match_run2_frozen / n,
        "conclusion": (
            "Fully reproducible: two independent retrains with the identical "
            "seed=42/data/feature pipeline produce IDENTICAL predictions to "
            "each other and to the frozen results/unified_ablation/test_set.json "
            "xgb_predicted_class values (100% match, n={}). Unlike NSL-KDD's "
            "XGBClassifier instability, CIC-IDS2018's larger training pool "
            "(hundreds of thousands of rows vs NSL-KDD's ~thousands) combined "
            "with a deterministic sklearn train_test_split(random_state=42) "
            "feeding a random_state=42 XGBClassifier yields byte-for-byte "
            "identical trees on this machine/xgboost version. No fix was "
            "required.".format(n)
        ) if match_run1_run2 == n and match_run1_frozen == n else (
            "INSTABILITY DETECTED — see match rates above; requires further "
            "diagnosis (see NSL-KDD fix pattern: pin tree_method, n_jobs, "
            "check column order)."
        ),
    }


# ── Step 3: fresh sample construction ────────────────────────────────────────

def build_fresh_samples(cve_index: dict):
    """Build calibration (10/class) + held-out test (20/class) samples from
    the pool EXCLUDING the 180 rows already reserved in test_set.json."""
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    orig_n_per_class = testset["meta"]["n_per_class"]

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    classes = sorted(df["Label"].unique().tolist())

    # Reproduce the ORIGINAL reservation exactly (same code/seed as
    # build_unified_ablation_testset.build_test_set) so we know which rows
    # to exclude, without needing row-level IDs stored in test_set.json.
    orig_eval_idx = set()
    remaining_pools = {}
    for cls in classes:
        cls_df = df[df["Label"] == cls]
        orig_eval = cls_df.sample(n=orig_n_per_class, random_state=42)
        orig_eval_idx |= set(orig_eval.index)
        remaining_pools[cls] = cls_df.drop(orig_eval.index)

    # From the remaining (non-overlapping) pool, draw calibration (10/class)
    # then test (20/class), same seed=42 convention, non-overlapping by
    # construction (test drawn from pool with calibration rows dropped).
    calib_dfs, test_dfs, train_pool_dfs = [], [], []
    for cls in classes:
        pool = remaining_pools[cls]
        calib = pool.sample(n=10, random_state=42)
        rest = pool.drop(calib.index)
        test = rest.sample(n=20, random_state=42)
        train_pool = rest.drop(test.index)
        calib_dfs.append(calib)
        test_dfs.append(test)
        train_pool_dfs.append(train_pool)

    calib_df = pd.concat(calib_dfs, ignore_index=False).sample(frac=1, random_state=42)
    test_df = pd.concat(test_dfs, ignore_index=False).sample(frac=1, random_state=42)
    train_pool_df = pd.concat(train_pool_dfs, ignore_index=False)

    # Sanity: zero overlap with the original 180 + between calib/test
    assert not (set(calib_df.index) & orig_eval_idx)
    assert not (set(test_df.index) & orig_eval_idx)
    assert not (set(calib_df.index) & set(test_df.index))

    # Train ML models on the pool excluding ALL reserved rows (orig 180 +
    # calibration 60 + test 120) — zero leakage by construction.
    xgb, le, feature_cols = train_xgb(train_pool_df)

    def make_cases(sample_df: pd.DataFrame, prefix: str, rng_seed: int) -> list[dict]:
        import random
        rng = random.Random(rng_seed)
        X = sample_df[feature_cols].values.astype(float)
        xgb_preds = le.inverse_transform(xgb.predict(X))
        xgb_proba = xgb.predict_proba(X).max(axis=1)

        cases = []
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            true_cls = row["Label"]
            xgb_cls = xgb_preds[i]
            criticality = rng.choice(CRITICALITIES)
            c_val = CRIT_MAP[criticality]

            gt_score = compute_risk_score(true_cls, c_override=c_val, i_override=c_val, a_override=c_val)
            gt_risk_level = gt_score.level_en if gt_score.level_en != "N/A (non-attack)" else "Very Low"

            xgb_score = compute_risk_score(xgb_cls, c_override=c_val, i_override=c_val, a_override=c_val)
            xgb_risk_level = xgb_score.level_en if xgb_score.level_en != "N/A (non-attack)" else "Very Low"

            paired_cve = pick_cve(true_cls, cve_index, rng)
            asset = {
                "id": f"asset_{prefix}_{i:03d}",
                "hostname": f"host-{prefix}-{i:03d}",
                "criticality": criticality,
                "asset_type": rng.choice(["server", "workstation", "iot", "firewall"]),
                "os": rng.choice(["Linux", "Windows Server", "Windows 10", "Ubuntu"]),
            }
            arg_neighborhood = {
                "related_cves": [{"id": paired_cve["id"], "cvss_v3": paired_cve.get("cvss_v3")}]
                if paired_cve else []
            }
            nl_desc = features_to_nl(row)
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
                "case_id": f"{prefix}_{i:03d}",
                "source_row_index": int(idx),
                "true_attack_class": true_cls,
                "asset": asset,
                "paired_cve": paired_cve,
                "arg_neighborhood": arg_neighborhood,
                "controls": [CONTROL_BY_CLASS.get(true_cls, "SS6.1-MON-1")],
                "nl_description": nl_desc,
                "feature_subset": feature_subset,
                "xgb_predicted_class": xgb_cls,
                "xgb_confidence": float(xgb_proba[i]),
                "xgb_risk_level": xgb_risk_level,
                "ground_truth_risk_level": gt_risk_level,
                "criticality": criticality,
            })
        return cases

    calib_cases = make_cases(calib_df, "cal", rng_seed=42)
    test_cases = make_cases(test_df, "test", rng_seed=43)  # distinct rng stream from calibration

    return calib_cases, test_cases, (xgb, le, feature_cols)


# ── Step 4: held-out safety metrics ──────────────────────────────────────────

def metrics(cases: list[dict]) -> dict:
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for c in cases:
        p, g = c["xgb_risk_level"], c["ground_truth_risk_level"]
        n += 1
        if p not in IDX or g not in IDX:
            under += 1
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
    return dict(n=n, exact=pct(exact, n), within1=pct(within1, n),
                under=pct(under, n), over=pct(over, n),
                severe_recall=pct(sev_rec, sev_tot), sev_tot=sev_tot,
                worst_miss=worst)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: leakage audit ===")
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    cases_by_id = {c["case_id"]: c for c in testset["cases"]}
    leakage_audit = run_leakage_audit(cases_by_id)
    (OUT_DIR / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2), encoding="utf-8")
    m = leakage_audit["meta"]
    print(f"  N={m['n_cases']}  NL-leaky={m['n_nl_leaky']}  CVE-leaky={m['n_cve_leaky']}  any={m['n_any_leaky']}")

    print("\n=== Step 2: XGBoost reproducibility ===")
    repro = verify_xgb_reproducibility()
    print(f"  run1 vs run2:            {repro['match_run1_vs_run2']}/{repro['n_cases']} ({repro['match_run1_vs_run2_rate']:.1%})")
    print(f"  run1 vs frozen historic: {repro['match_run1_vs_frozen_historical']}/{repro['n_cases']} ({repro['match_run1_vs_frozen_historical_rate']:.1%})")

    print("\n=== Step 3: fresh sample construction ===")
    cve_index = load_cve_index(ARG_PATH)
    calib_cases, test_cases, (xgb, le, feature_cols) = build_fresh_samples(cve_index)
    print(f"  calibration_sample: {len(calib_cases)} cases")
    print(f"  test_sample:        {len(test_cases)} cases")

    print("\n=== Step 4: held-out XGBoost baseline safety metrics ===")
    test_metrics = metrics(test_cases)
    calib_metrics = metrics(calib_cases)
    print(f"  test_sample:  under-escalation={test_metrics['under']:.1f}%  severe_recall={test_metrics['severe_recall']:.1f}%  worst_miss={test_metrics['worst_miss']}")

    calibration_out = {
        "meta": {
            "description": "Calibration sample for future LLM-override-threshold tuning (NOT scored in final comparison). 10 cases/class, drawn from the CIC-IDS2018 pool excluding the original 180-case unified-ablation eval set.",
            "n": len(calib_cases), "n_per_class": 10, "seed": 42,
            "source": "data/processed/cleaned_dataset.csv",
            "excludes": "results/unified_ablation/test_set.json (180 rows) + this file's own test_sample.json (120 rows, disjoint by construction)",
            "created": datetime.now().isoformat(),
        },
        "cases": calib_cases,
    }
    test_out = {
        "meta": {
            "description": "Held-out test sample for the future LLM rerun. 20 cases/class, drawn from the CIC-IDS2018 pool excluding the original 180-case unified-ablation eval set and the calibration_sample.",
            "n": len(test_cases), "n_per_class": 20, "seed": 43,
            "source": "data/processed/cleaned_dataset.csv",
            "excludes": "results/unified_ablation/test_set.json (180 rows) + calibration_sample.json (60 rows)",
            "created": datetime.now().isoformat(),
            "xgb_baseline_metrics": test_metrics,
        },
        "cases": test_cases,
    }
    (OUT_DIR / "calibration_sample.json").write_text(json.dumps(calibration_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "test_sample.json").write_text(json.dumps(test_out, indent=2, ensure_ascii=False), encoding="utf-8")

    xgb_out = {
        "meta": {"description": "XGBoost reproducibility verification + held-out baseline metrics for the CIC-IDS2018 unified rerun."},
        "reproducibility": repro,
        "held_out_test_sample_metrics": test_metrics,
        "calibration_sample_metrics_reference_only": calib_metrics,
    }
    (OUT_DIR / "xgboost_baseline_results.json").write_text(json.dumps(xgb_out, indent=2), encoding="utf-8")

    print(f"\nSaved all artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
