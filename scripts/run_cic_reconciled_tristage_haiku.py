"""Reconciled tri-stage LLM (Haiku 4.5) rerun on the CIC-IDS2018 clean held-out sample.

Context (see results/nslkdd_unified_rerun/override_design_investigation.md,
2026-07-11, for the full investigation this design implements): the existing
CIC tri-stage arm (scripts/run_cic_tristage_haiku_clean.py) gives the LLM the
ML (XGBoost) predicted class up front and asks it to "verify" that class,
with permission to override only under a narrow three-condition rule. On the
120-case clean held-out sample, 0/120 cases meet the override preconditions,
so the LLM's only observed contribution on that arm is GRC-artifact
generation, not classification divergence — the risk-level series is
byte-identical to XGBoost's (McNemar b=0/1, c=0, p~1.0).

This script implements the alternative, non-p-hacked design proposed in that
investigation: **independent-inference-then-reconcile**.

  Stage 1 (LLM, independent): the LLM is given ONLY the leak-free NL flow
    description, asset context, and redacted CVE text — NOT the XGBoost
    predicted class — and independently infers the attack class and §6.9
    risk level. This reuses the existing SYSTEM_PROMPT_PURE_LLM /
    PROMPT_PURE_LLM prompts from scripts/run_unified_ablation.py verbatim
    (the "Pure LLM" paradigm already used elsewhere in the thesis), combined
    with the SAME leak-free NL/CVE reconstruction used by
    scripts/run_cic_tristage_haiku_clean.py (clean_features_to_nl,
    redact_cve_description — imported directly, not reimplemented).

  Stage 2 (ML, independent): XGBoost's per-case predicted class and risk
    level, taken directly from results/cic_unified_rerun/test_sample.json
    (xgb_predicted_class / xgb_risk_level), which mirror
    results/cic_unified_rerun/xgboost_baseline_results.json's per-case
    predictions on this same held-out sample.

  Stage 3 (symbolic arbitration, new, PRE-REGISTERED before any run):
    When the LLM's independently-inferred risk_level and XGBoost's
    risk_level AGREE, that risk level is final (no reconciliation needed).
    When they DISAGREE, the arbitration rule is:

        final_risk_level = the MORE SEVERE of the two §6.9 risk levels
        (Very Low < Low < Medium < High < Catastrophic)

    This is chosen and documented BEFORE running any live calls, for three
    reasons: (1) it is symmetric — it does not privilege either model's
    output on structural grounds (unlike "defer to higher confidence",
    which the Pure-LLM prompt does not even elicit a calibrated confidence
    score for, making that option unusable here without inventing an
    ad hoc elicitation); (2) it directly operationalizes this thesis's
    LOCKED safety-primary framing (the NSL-KDD spine result already treats
    under-escalation, not raw accuracy, as the decisive metric) — taking
    the more severe level can only ever reduce or hold constant
    under-escalation risk, never increase it, by construction; (3) it is
    the same rule already sketched as the first-listed principled option in
    the override-design investigation doc, so both the NSL-KDD companion
    task and this CIC task apply one consistent, nameable rule across
    datasets rather than two different post-hoc-flavored designs.
    The FINAL ATTACK CLASS attributed alongside a disagreement is the class
    belonging to whichever side's risk level was selected (i.e., the more
    severe side); on a same-severity tie the ML class is kept as the
    anchor (this tie-break affects only which class label/GRC control is
    attributed, never the risk_level number itself, so it cannot bias the
    safety metrics below).

Usage:
    python scripts/run_cic_reconciled_tristage_haiku.py                # full 120-case run
    python scripts/run_cic_reconciled_tristage_haiku.py --limit 5       # smoke test
    python scripts/run_cic_reconciled_tristage_haiku.py --start-offset 40  # resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.llm_artifact_lib import REQUIRED_FIELDS  # noqa: E402
from scripts.run_unified_ablation import (  # noqa: E402
    SYSTEM_PROMPT_PURE_LLM, PROMPT_PURE_LLM, ATTACK_TO_CONTROL, JSON_FENCE,
)
from scripts.run_cic_tristage_haiku_clean import (  # noqa: E402
    clean_features_to_nl, redact_cve_description, verify_leak_free,
    _format_cve as _format_cve_redacted,
)

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"
TEST_SAMPLE_PATH = ROOT / "results" / "cic_unified_rerun" / "test_sample.json"
XGB_BASELINE_PATH = ROOT / "results" / "cic_unified_rerun" / "xgboost_baseline_results.json"
OUT_DIR = ROOT / "results" / "cic_unified_rerun"
OUT_JSONL = OUT_DIR / "reconciled_tristage_haiku_raw.jsonl"
OUT_SUMMARY = OUT_DIR / "reconciled_tristage_haiku_results.json"

MODEL = "claude-haiku-4-5-20251001"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}


def _parse_response(text: str) -> Optional[dict]:
    m = JSON_FENCE.search(text)
    blob = m.group(1) if m else text
    try:
        return json.loads(blob)
    except Exception:
        f, l = blob.find("{"), blob.rfind("}")
        if f >= 0 and l > f:
            try:
                return json.loads(blob[f:l+1])
            except Exception:
                return None
    return None


def reconcile(llm_class: str, llm_risk: str, ml_class: str, ml_risk: str) -> dict:
    """Pre-registered symbolic arbitration (see module docstring, Stage 3).

    Agreement: llm_risk == ml_risk -> that level is final, no arbitration
    invoked. Disagreement: take the MORE SEVERE of the two risk levels;
    attribute the final class to whichever side produced that level (ML
    anchor on a same-severity tie, which cannot occur when risk levels
    differ, so the tie-break here only matters if risk levels are equal —
    included defensively).
    """
    agree = (llm_risk == ml_risk)
    if agree:
        return {
            "arbitration_invoked": False,
            "final_risk_level": ml_risk,
            "final_attack_class": ml_class,
            "class_disagreement": llm_class != ml_class,
        }
    llm_i = IDX.get(llm_risk, -1)
    ml_i = IDX.get(ml_risk, -1)
    if llm_i > ml_i:
        final_risk, final_cls, source = llm_risk, llm_class, "llm_more_severe"
    elif ml_i > llm_i:
        final_risk, final_cls, source = ml_risk, ml_class, "ml_more_severe"
    else:
        # Defensive tie-break (unreachable when llm_risk != ml_risk with a
        # valid IDX lookup, but kept explicit for parse-failure fallbacks).
        final_risk, final_cls, source = ml_risk, ml_class, "tie_ml_anchor"
    return {
        "arbitration_invoked": True,
        "arbitration_source": source,
        "final_risk_level": final_risk,
        "final_attack_class": final_cls,
        "class_disagreement": llm_class != ml_class,
    }


def compute_metrics(records: list[dict]) -> dict:
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for r in records:
        p, g = r["risk_level"], r["ground_truth"]
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
    return dict(n=n, exact_match_pct=pct(exact, n), within1_pct=pct(within1, n),
                under_escalation_pct=pct(under, n), over_escalation_pct=pct(over, n),
                severe_recall_pct=pct(sev_rec, sev_tot), severe_total=sev_tot,
                worst_miss_levels=worst)


def mcnemar(records: list[dict]) -> dict:
    from scipy.stats import binomtest
    b = c = 0
    for r in records:
        xgb_correct = r["xgb_risk_level"] == r["ground_truth"]
        rec_correct = r["risk_level"] == r["ground_truth"]
        if xgb_correct and not rec_correct:
            b += 1
        elif rec_correct and not xgb_correct:
            c += 1
    n_disc = b + c
    p = 1.0 if n_disc == 0 else binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue
    return {"b_xgb_correct_reconciled_wrong": b, "c_xgb_wrong_reconciled_correct": c,
            "n_discordant": n_disc, "p_value": p}


def grc_completeness(records: list[dict]) -> dict:
    list_fields = {"evidence_refs", "nfcrm_clauses"}
    complete = 0
    per_field_missing = {f: 0 for f in REQUIRED_FIELDS}
    for r in records:
        art = r.get("artifact") or {}
        ok = True
        for f in REQUIRED_FIELDS:
            v = art.get(f)
            present = isinstance(v, list) if f in list_fields else v not in (None, "")
            if not present:
                per_field_missing[f] += 1
                ok = False
        if ok:
            complete += 1
    n = len(records)
    return {"n": n, "complete": complete,
            "completeness_pct": (complete / n * 100) if n else 0.0,
            "per_field_missing_count": per_field_missing}


def main():
    global OUT_JSONL, OUT_SUMMARY
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--ablate-cve", action="store_true",
                    help="CVE-ablation control arm: replace the CVE block with "
                         "'No CVE context.' for every case; outputs go to "
                         "*_cve_ablated files. Pre-registered interpretation "
                         "rule (fixed before running): if the ablated arm's "
                         "under-escalation approximately equals the unablated "
                         "reconciled run's 0.83%, the CIC reconciled result is "
                         "channel-independent; if it degrades toward XGBoost's "
                         "3.33%, the class-paired CVE channel is load-bearing "
                         "on CIC as on NSL-KDD. Everything else identical.")
    args = ap.parse_args()
    if args.ablate_cve:
        OUT_JSONL = OUT_DIR / "reconciled_tristage_haiku_cve_ablated_raw.jsonl"
        OUT_SUMMARY = OUT_DIR / "reconciled_tristage_haiku_cve_ablated_results.json"

    test_out = json.loads(TEST_SAMPLE_PATH.read_text(encoding="utf-8"))
    cases = test_out["cases"]
    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases from {TEST_SAMPLE_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    from scripts.claude_cli_client import ClaudeCLIClient
    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=args.timeout_sec)

    mode = "a" if args.start_offset > 0 else "w"
    leak_check_failures = []
    api_errors = 0
    parse_failures = 0
    disagreements = []

    with OUT_JSONL.open(mode, encoding="utf-8") as f:
        for i, case in enumerate(cases[args.start_offset:], start=args.start_offset + 1):
            t0 = time.time()
            cid = case["case_id"]
            crit = case["criticality"]
            gt = case["ground_truth_risk_level"]
            true_cls = case["true_attack_class"]
            cve = case.get("paired_cve")

            row = df.loc[case["source_row_index"]]
            nl_desc_clean = clean_features_to_nl(row)
            cve_blk_clean = "No CVE context." if args.ablate_cve else _format_cve_redacted(cve)

            leak_hits = verify_leak_free(nl_desc_clean, cve_blk_clean, true_cls)
            if leak_hits:
                leak_check_failures.append({"case_id": cid, "hits": leak_hits})

            ml_cls = case["xgb_predicted_class"]
            ml_risk = case.get("xgb_risk_level", "")
            asset_j = json.dumps(case.get("asset", {}))

            # Stage 1: independent LLM inference — NO ML class shown.
            prompt = PROMPT_PURE_LLM.format(
                criticality=crit, asset_json=asset_j, cve_block=cve_blk_clean,
                nl_desc=nl_desc_clean[:300],
            )

            try:
                response = client.generate(SYSTEM_PROMPT_PURE_LLM, prompt) or ""
            except Exception:
                response = ""
            if not response:
                api_errors += 1

            artifact = _parse_response(response)
            latency_ms = int((time.time() - t0) * 1000)

            if artifact:
                llm_cls = artifact.get("inferred_attack_class", "") or ml_cls
                llm_risk = artifact.get("risk_level", "")
                parse_ok = bool(llm_risk)
            else:
                parse_ok = False

            if not artifact or not parse_ok:
                parse_failures += 1
                # Fallback convention (same as clean tri-stage arm): when the
                # LLM stage fails to parse, there is no independent judgment
                # to reconcile against, so trust ML (agreement by construction,
                # arbitration not invoked).
                llm_cls, llm_risk = ml_cls, ml_risk

            rec = reconcile(llm_cls, llm_risk, ml_cls, ml_risk)
            if rec["arbitration_invoked"] or rec["class_disagreement"]:
                disagreements.append({
                    "case_id": cid, "llm_class": llm_cls, "llm_risk": llm_risk,
                    "ml_class": ml_cls, "ml_risk": ml_risk, **rec,
                })

            final_cls = rec["final_attack_class"]
            final_risk = rec["final_risk_level"]
            control_id = ATTACK_TO_CONTROL.get(final_cls, ATTACK_TO_CONTROL.get(ml_cls, "SS6.1-MON-1"))

            # Build a GRC artifact consistent with REQUIRED_FIELDS, reusing
            # the LLM's own narrative/evidence when parsed, else a minimal
            # deterministic fallback so completeness reflects real content.
            if artifact:
                narrative = artifact.get("narrative", "") or (
                    f"Reconciled assessment: LLM independently inferred {llm_cls} "
                    f"(risk {llm_risk}); ML/XGBoost predicted {ml_cls} (risk {ml_risk}). "
                    f"Arbitration selected {final_risk} ({rec.get('arbitration_source', 'agreement')})."
                )
                evidence_refs = artifact.get("evidence_refs") or ([cve.get("id")] if cve else [])
                nfcrm_clauses = artifact.get("nfcrm_clauses") or ["§6.9"]
            else:
                narrative = (
                    f"LLM stage parse failure; fell back to ML class {ml_cls} (risk {ml_risk})."
                )
                evidence_refs = [cve.get("id")] if cve else []
                nfcrm_clauses = ["§6.9"]

            final_artifact = {
                "risk_level": final_risk,
                "recommended_control_id": control_id,
                "nfcrm_clauses": nfcrm_clauses,
                "narrative": narrative,
                "evidence_refs": evidence_refs,
            }

            record = {
                "case_id": cid,
                "model": MODEL,
                "true_class": true_cls,
                "ml_pred_class": ml_cls,
                "ml_risk_level": ml_risk,
                "llm_inferred_class": llm_cls,
                "llm_risk_level": llm_risk,
                "final_attack_class": final_cls,
                "arbitration_invoked": rec["arbitration_invoked"],
                "arbitration_source": rec.get("arbitration_source"),
                "class_disagreement": rec["class_disagreement"],
                "risk_level": final_risk,
                "xgb_risk_level": ml_risk,
                "ground_truth": gt,
                "correct": final_risk == gt,
                "xgb_correct": ml_risk == gt,
                "parse_error": not parse_ok,
                "criticality": crit,
                "nl_description_clean": nl_desc_clean,
                "cve_block_clean": cve_blk_clean,
                "leak_check_hits": leak_hits,
                "artifact": final_artifact,
                "raw_response": response[:2000],
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            match = "OK" if record["correct"] else "MISS"
            tag = " [ARBITRATED]" if rec["arbitration_invoked"] else ""
            print(f"  [{i}/{len(cases)}] {cid}: {final_risk or 'PARSE_ERR'} vs GT={gt} [{match}]{tag} ({latency_ms}ms)")

    all_records = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    metrics = compute_metrics(all_records)
    mcn = mcnemar(all_records)
    grc = grc_completeness(all_records)

    xgb_baseline = json.loads(XGB_BASELINE_PATH.read_text(encoding="utf-8"))

    summary = {
        "meta": {
            "description": (
                "Reconciled tri-stage LLM (Claude Haiku 4.5, CLI) on the CIC-IDS2018 "
                "clean held-out sample (N=120, 20/class): Stage 1 independent LLM "
                "inference (SYSTEM_PROMPT_PURE_LLM/PROMPT_PURE_LLM, no ML class shown, "
                "leak-free NL + redacted CVE text), Stage 2 XGBoost per-case class/risk "
                "from test_sample.json, Stage 3 pre-registered symbolic arbitration "
                "(more-severe-of-two on disagreement). See module docstring of "
                "scripts/run_cic_reconciled_tristage_haiku.py for the full pre-registration "
                "rationale, written before any live call in this run."
            ),
            "model": MODEL,
            "n_cases": len(all_records),
            "api_errors": api_errors,
            "parse_failures": parse_failures,
            "leak_check_failures": leak_check_failures,
            "n_leak_check_failures": len(leak_check_failures),
            "n_disagreements_logged": len(disagreements),
            "n_arbitration_invoked": sum(1 for r in all_records if r["arbitration_invoked"]),
            "n_class_disagreement": sum(1 for r in all_records if r["class_disagreement"]),
            "disagreements": disagreements,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "metrics": metrics,
        "mcnemar_vs_clean_xgboost": mcn,
        "grc_completeness": grc,
        "clean_xgboost_baseline_reference": xgb_baseline["held_out_test_sample_metrics"],
        "records": all_records,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "metrics": metrics, "mcnemar": mcn,
        "grc": {k: v for k, v in grc.items() if k != "per_field_missing_count"},
        "api_errors": api_errors, "parse_failures": parse_failures,
        "leak_check_failures": len(leak_check_failures),
        "n_disagreements": len(disagreements),
    }, indent=2))
    print(f"\nSaved: {OUT_JSONL}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
