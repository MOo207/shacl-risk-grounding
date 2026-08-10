"""Tri-stage LLM (Haiku) safety rerun on the clean, leak-free NSL-KDD held-out
test sample (results/nslkdd_unified_rerun/test_sample.json, N=100, 20/class).

This reproduces -- unchanged -- the EXACT mechanism in
scripts/run_nslkdd_ablation.py::run_llm_paradigm(paradigm="tristage_llm_haiku"),
which is the script that actually generated
results/nslkdd_ablation/tristage_llm_haiku.jsonl (verified by matching its
record schema: ml_pred_class / grc / control / narrative / model fields) --
the file consumed by scripts/safety_table_haiku.py to produce the thesis
Table~\\ref{tab:safety-synthesis} "Tri-stage LLM (Haiku)" row and the
b=14,c=0,p=6.1e-5 McNemar spine test vs XGBoost.

Mechanism (unchanged from run_nslkdd_ablation.py; NOT re-derived):
  Stage 1 (ML, authoritative): attack_class = case["xgb_predicted_class"]
    (raw NSL-KDD label from the fixed/reproducible XGBoost baseline in
    results/nslkdd_unified_rerun/test_sample.json). The tri-stage design
    does NOT re-infer the class from the LLM and does NOT apply any
    CVSS/confidence-gated override -- the system prompt is explicit:
    "The ML classifier's attack class is authoritative -- use it as the
    row key" / "DO NOT override the table". (This was checked against
    scripts/run_confidence_gated_override.py and pipeline symbolic_override()
    in scripts/run_nslkdd_risk_ablation.py -- neither is wired into this
    paradigm; those belong to different ablations. No new override logic
    is introduced here.)
  Stage 2 (symbolic, MANDATORY table): the same NFCRM SS6.9 lookup table
    (NSL-KDD class x criticality -> risk_level) is embedded verbatim in the
    system prompt, and independently verified here to reproduce
    pipeline.nfcrm.risk_score.compute_risk_score() exactly for all 20
    (class, criticality) combinations -- see smoke-test output in the
    calibration sanity-check section below.
  Stage 3 (LLM narrative + self-applied lookup): Haiku is given the
    authoritative class + criticality + CVE + neutral NL description and
    must (a) apply the mandatory table to emit risk_level, (b) write a
    2-3 sentence narrative, (c) recommend a control. The record's
    risk_level is the LLM's own output (trusted, as in the original
    script) -- not overwritten by a second Python-side computation.

Only change vs the original: the NL description comes from the NEUTRAL,
leak-validated generator (pipeline/nfcrm/nslkdd_neutral_nl.py, already baked
into results/nslkdd_unified_rerun/test_sample.json) instead of the old
leaky features_to_nl_nslkdd(). No threshold, prompt-logic, or lookup-table
change relative to run_nslkdd_ablation.py.

Usage:
    python -m scripts.run_nslkdd_tristage_haiku_clean --calibration-check
    python -m scripts.run_nslkdd_tristage_haiku_clean --run-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from math import comb
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.claude_cli_client import ClaudeCLIClient  # noqa: E402

CALIB_PATH = ROOT / "results/nslkdd_unified_rerun/calibration_sample.json"
TEST_PATH = ROOT / "results/nslkdd_unified_rerun/test_sample.json"
XGB_BASELINE_PATH = ROOT / "results/nslkdd_unified_rerun/xgboost_baseline_results.json"
OUT_PATH = ROOT / "results/nslkdd_unified_rerun/tristage_haiku_clean_results.json"

MODEL = "claude-haiku-4-5-20251001"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

# ── Verbatim from scripts/run_nslkdd_ablation.py ─────────────────────────────

SYSTEM_PROMPT_TRISTAGE = """You are a GRC analyst applying NFCRM-1:2025 (Saudi Arabia National Cybersecurity Risk Management framework).

MANDATORY RULE — §6.9 Risk Level Lookup Table for NSL-KDD traffic (you MUST follow this exactly):

  Criticality values: Low | Medium | High | Critical

  NSL-KDD class | Low      | Medium | High          | Critical
  --------------|----------|--------|---------------|----------
  Normal        | Very Low | Very Low | Very Low    | Very Low
  DoS           | Medium   | High   | Catastrophic  | Catastrophic
  Probe         | Low      | Medium | Medium        | High
  R2L           | Medium   | High   | Catastrophic  | Catastrophic
  U2R           | Medium   | High   | Catastrophic  | Catastrophic

DO NOT override the table. The ML classifier's attack class is authoritative — use it as the row key.
Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_TRISTAGE = """Security event (NSL-KDD network flow):
- Attack class (from ML classifier): {attack_class}
- Asset criticality: {criticality}
- Asset: {asset_json}
- CVE context: {cve_block}
- Flow description: {nl_desc}

Step 1 — look up row "{attack_class}" and column "{criticality}" in the §6.9 table → that is the risk_level.
Step 2 — write 2–3 sentence narrative citing the attack class, CVE, and asset criticality.

```json
{{
  "risk_level": "<exact value from §6.9 table>",
  "recommended_control_id": "<NFCRM control>",
  "nfcrm_clauses": ["§6.9"],
  "narrative": "<2-3 sentences>",
  "evidence_refs": ["{attack_class}", "<cve_id>"]
}}```"""

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def _format_cve(cve: Optional[dict]) -> str:
    """Same shape as run_nslkdd_ablation.py._format_cve, adapted to the new
    case schema's CVE field names (id/description instead of
    id/shortDescription -- both are the same CVE dict shape used elsewhere,
    just with 'description' not 'shortDescription')."""
    if not cve:
        return "No CVE context."
    desc = cve.get("shortDescription") or cve.get("description") or ""
    return f"CVE: {cve.get('id')}  CVSS: {cve.get('cvss_v3')}  Desc: {desc[:120]}"


def _parse_response(text: str) -> Optional[dict]:
    m = JSON_FENCE.search(text or "")
    blob = m.group(1) if m else (text or "")
    try:
        return json.loads(blob)
    except Exception:
        f, l = blob.find("{"), blob.rfind("}")
        if f >= 0 and l > f:
            try:
                return json.loads(blob[f:l + 1])
            except Exception:
                return None
    return None


def run_tristage_llm_haiku(cases: list[dict], client: ClaudeCLIClient,
                           label: str, out_jsonl: Optional[Path] = None) -> list[dict]:
    records = []
    fout = out_jsonl.open("w", encoding="utf-8") if out_jsonl else None
    try:
        for i, case in enumerate(cases, 1):
            t0 = time.time()
            cid = case["case_id"]
            crit = case["criticality"]
            gt = case["ground_truth_risk_level"]
            cve = case.get("paired_cve")
            asset_j = json.dumps(case.get("asset", {}))
            cve_blk = _format_cve(cve)
            nl_desc = case.get("nl_description", "")
            ml_cls = case["xgb_predicted_class"]

            prompt = PROMPT_TRISTAGE.format(
                attack_class=ml_cls, criticality=crit,
                asset_json=asset_j, cve_block=cve_blk,
                nl_desc=nl_desc[:300],
            )

            try:
                response = client.generate(SYSTEM_PROMPT_TRISTAGE, prompt) or ""
            except Exception as e:
                response = ""
                print(f"  [{i}/{len(cases)}] {cid}: EXCEPTION during client.generate: {e}")

            artifact = _parse_response(response)
            latency = int((time.time() - t0) * 1000)

            if artifact:
                risk_lvl = artifact.get("risk_level", "")
                correct = risk_lvl == gt
                parse_ok = True
            else:
                risk_lvl = ""
                correct = False
                parse_ok = False

            d = (IDX.get(risk_lvl, None) - IDX[gt]) if risk_lvl in IDX and gt in IDX else None
            under_escalated = (d is not None and d < 0) or (risk_lvl not in IDX)
            over_escalated = d is not None and d > 0

            record = {
                "case_id": cid,
                "paradigm": "tristage_llm_haiku_clean",
                "model": MODEL,
                "true_class": case["true_attack_class"],
                "true_nfcrm_class": case.get("true_nfcrm_class", ""),
                "ml_pred_class": ml_cls,
                "xgb_risk_level": case.get("xgb_risk_level", ""),
                "risk_level": risk_lvl,
                "ground_truth": gt,
                "correct": correct,
                "under_escalated": bool(under_escalated),
                "over_escalated": bool(over_escalated),
                "severe_gt": gt in SEVERE,
                "severe_hit": (gt in SEVERE) and (risk_lvl in SEVERE),
                "parse_error": not parse_ok,
                "criticality": crit,
                "grc": "Full",
                "narrative": (artifact or {}).get("narrative", ""),
                "control": (artifact or {}).get("recommended_control_id", ""),
                "raw_response": response,
                "prompt_system": SYSTEM_PROMPT_TRISTAGE,
                "prompt_user": prompt,
                "latency_ms": latency,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            records.append(record)
            if fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            match = "OK" if correct else ("PARSE_ERR" if not parse_ok else "MISS")
            print(f"  [{label} {i}/{len(cases)}] {cid} ({case['true_attack_class']:6}): "
                  f"ml_cls={ml_cls:6} risk={risk_lvl or '?':12} GT={gt:12} [{match}] ({latency}ms)")
    finally:
        if fout:
            fout.close()
    return records


# ── Metrics (verbatim logic from scripts/safety_table_haiku.py::metrics()) ──

def metrics(records: list[dict]) -> dict:
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for r in records:
        p, g = r.get("risk_level", ""), r.get("ground_truth", "")
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


def xgb_metrics_from_cases(cases: list[dict]) -> dict:
    """Per-case XGBoost baseline metrics using xgb_risk_level already stored
    in test_sample.json (identical field/logic to
    scripts/build_nslkdd_unified_rerun.py::metrics_from_cases())."""
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


def mcnemar_1s(b, c):
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) * (0.5 ** n) if n else 1.0


def unsafe(risk_level: str, gt: str) -> bool:
    if risk_level not in IDX or gt not in IDX:
        return True
    return IDX[risk_level] < IDX[gt]


def paired_mcnemar(llm_records: list[dict], xgb_cases: list[dict]) -> dict:
    """b = XGB unsafe & LLM safe; c = LLM unsafe & XGB safe. H1: b>c (LLM safer).
    Identical semantics to scripts/safety_table_haiku.py::unsafe_mcnemar()."""
    xgb_by_id = {c["case_id"]: c for c in xgb_cases}
    b = c = 0
    for r in llm_records:
        cid = r["case_id"]
        if cid not in xgb_by_id:
            continue
        xc = xgb_by_id[cid]
        gt = r["ground_truth"]
        llm_unsafe = unsafe(r.get("risk_level", ""), gt)
        xgb_unsafe = unsafe(xc["xgb_risk_level"], xc["ground_truth_risk_level"])
        if xgb_unsafe and not llm_unsafe:
            b += 1
        elif llm_unsafe and not xgb_unsafe:
            c += 1
    p = mcnemar_1s(b, c)
    return {"b": b, "c": c, "p": p}


def verify_lookup_table_matches_engine():
    """Sanity check: confirm the hardcoded §6.9 table in SYSTEM_PROMPT_TRISTAGE
    matches pipeline.nfcrm.risk_score.compute_risk_score() exactly (it must,
    since the original run_nslkdd_ablation.py was designed so the LLM's
    self-applied table lookup reproduces the symbolic engine)."""
    from pipeline.nfcrm.risk_score import compute_risk_score
    NSLKDD_TO_NFCRM = {"Normal": "Benign", "DoS": "DoS", "Probe": "Infiltration",
                       "R2L": "WebAttack", "U2R": "BruteForce"}
    CRIT_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}
    EXPECTED = {
        ("Normal", "Low"): "Very Low", ("Normal", "Medium"): "Very Low",
        ("Normal", "High"): "Very Low", ("Normal", "Critical"): "Very Low",
        ("DoS", "Low"): "Medium", ("DoS", "Medium"): "High",
        ("DoS", "High"): "Catastrophic", ("DoS", "Critical"): "Catastrophic",
        ("Probe", "Low"): "Low", ("Probe", "Medium"): "Medium",
        ("Probe", "High"): "Medium", ("Probe", "Critical"): "High",
        ("R2L", "Low"): "Medium", ("R2L", "Medium"): "High",
        ("R2L", "High"): "Catastrophic", ("R2L", "Critical"): "Catastrophic",
        ("U2R", "Low"): "Medium", ("U2R", "Medium"): "High",
        ("U2R", "High"): "Catastrophic", ("U2R", "Critical"): "Catastrophic",
    }
    mismatches = []
    for (cls, crit), expected in EXPECTED.items():
        c_val = CRIT_MAP[crit]
        s = compute_risk_score(NSLKDD_TO_NFCRM[cls], c_override=c_val, i_override=c_val, a_override=c_val)
        actual = "Very Low" if s.level_en == "N/A (non-attack)" else s.level_en
        if actual != expected:
            mismatches.append((cls, crit, expected, actual))
    return mismatches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-check", action="store_true",
                    help="Sanity-check run over the 50-case calibration_sample.json (no threshold tuning).")
    ap.add_argument("--run-test", action="store_true",
                    help="Run the real 100-case held-out test_sample.json evaluation.")
    args = ap.parse_args()

    mismatches = verify_lookup_table_matches_engine()
    print(f"Lookup-table vs symbolic-engine check: {'OK, zero mismatches' if not mismatches else mismatches}")

    if not args.calibration_check and not args.run_test:
        args.calibration_check = args.run_test = True

    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=180)

    calib_summary = None
    if args.calibration_check:
        print("\n" + "=" * 70)
        print("CALIBRATION SANITY CHECK (N=50, NOT used for threshold tuning)")
        print("=" * 70)
        calib_cases = json.loads(CALIB_PATH.read_text(encoding="utf-8"))["cases"]
        calib_records = run_tristage_llm_haiku(calib_cases, client, "calib")
        calib_m = metrics(calib_records)
        n_parse_err = sum(1 for r in calib_records if r["parse_error"])
        print(f"\nCalibration sanity metrics: {json.dumps(calib_m, indent=2)}")
        print(f"Parse errors: {n_parse_err}/{len(calib_records)}")
        # crude sanity: check class-conditional risk-level distribution isn't degenerate
        from collections import Counter
        by_class = {}
        for r in calib_records:
            by_class.setdefault(r["true_class"], Counter())[r["risk_level"] or "PARSE_ERR"] += 1
        for cls, ctr in by_class.items():
            print(f"  {cls:8}: {dict(ctr)}")
        calib_summary = {"metrics": calib_m, "n_parse_errors": n_parse_err,
                         "risk_level_distribution_by_true_class":
                             {k: dict(v) for k, v in by_class.items()}}

    test_summary = None
    if args.run_test:
        print("\n" + "=" * 70)
        print("HELD-OUT TEST (N=100, real evaluation)")
        print("=" * 70)
        test_data = json.loads(TEST_PATH.read_text(encoding="utf-8"))
        test_cases = test_data["cases"]
        jsonl_path = ROOT / "results/nslkdd_unified_rerun/tristage_haiku_clean_test.jsonl"
        test_records = run_tristage_llm_haiku(test_cases, client, "test", out_jsonl=jsonl_path)

        llm_m = metrics(test_records)
        xgb_m = xgb_metrics_from_cases(test_cases)
        spine = paired_mcnemar(test_records, test_cases)
        n_parse_err = sum(1 for r in test_records if r["parse_error"])

        print(f"\nTri-stage LLM (Haiku, clean N=100) metrics: {json.dumps(llm_m, indent=2)}")
        print(f"XGBoost baseline (clean N=100) metrics:      {json.dumps(xgb_m, indent=2)}")
        print(f"Parse errors: {n_parse_err}/{len(test_records)}")
        print(f"\nSPINE TEST — under-escalation McNemar, Tri-LLM(Haiku,clean) vs XGBoost(clean): "
              f"b(XGB-unsafe,LLM-safe)={spine['b']}  c(LLM-unsafe,XGB-safe)={spine['c']}  "
              f"p(1-sided)={spine['p']:.4g}")

        test_summary = {
            "n": len(test_records),
            "tristage_llm_haiku_clean_metrics": llm_m,
            "xgboost_clean_baseline_metrics": xgb_m,
            "unsafe_mcnemar_llm_vs_xgb": spine,
            "n_parse_errors": n_parse_err,
            "records": test_records,
        }

    out = {
        "meta": {
            "description": ("Tri-stage LLM (Haiku 4.5) safety rerun on the clean, "
                            "leak-free NSL-KDD held-out test sample. Reproduces the "
                            "identical mechanism (ML-authoritative class + mandatory "
                            "§6.9 lookup table + LLM narrative) from "
                            "scripts/run_nslkdd_ablation.py's tristage_llm_haiku "
                            "paradigm -- no override thresholds re-tuned."),
            "model": MODEL,
            "test_sample_source": str(TEST_PATH.relative_to(ROOT)),
            "calibration_sample_source": str(CALIB_PATH.relative_to(ROOT)),
            "lookup_table_vs_engine_mismatches": mismatches,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "calibration_sanity_check": calib_summary,
        "held_out_test": test_summary,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
