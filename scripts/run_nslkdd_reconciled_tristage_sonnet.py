"""Reconciled tri-stage LLM (Haiku) safety rerun on the clean, leak-free
NSL-KDD held-out test sample (results/nslkdd_unified_rerun/test_sample.json,
N=100, 20/class).

Background (see results/nslkdd_unified_rerun/override_design_investigation.md):
the existing tri-stage design feeds the LLM the ML's predicted class and
instructs it "DO NOT override the table ... authoritative" -- there is no
override mechanism at all. On clean data this produces a result byte-identical
to XGBoost (b=0, c=0, p=1.0): not evidence the LLM has no reasoning value, but
a structural consequence of never letting it disagree. The investigation's
proposed fix ("independent-inference-then-reconcile") is implemented here.

Design (PRE-REGISTERED BEFORE ANY CASE IN THIS 100-CASE TEST SET WAS RUN
THROUGH THIS SCRIPT -- the arbitration rule below is fixed by the reasoning
in this docstring, not fit to this test set's outcomes):

  Stage 1a (LLM, independent): Haiku receives ONLY the neutral NL flow
    description, asset, and CVE context -- NOT xgb_predicted_class -- and
    independently infers the NSL-KDD attack class and the §6.9 risk_level.
    This reuses SYSTEM_PROMPT_PURE_LLM / PROMPT_PURE_LLM verbatim from
    scripts/run_nslkdd_ablation.py (the "Pure LLM" arm that already exists
    in this codebase; not a new prompt design).
  Stage 1b (ML, independent): xgb_predicted_class / xgb_risk_level, already
    computed and stored per-case in test_sample.json (identical values used
    by the clean XGBoost baseline and by the old "obey-the-table" tristage
    rerun -- no ML re-training or re-scoring here).
  Stage 2 (symbolic arbitration, NEW): a deterministic reconciliation
    function compares the two INDEPENDENTLY-derived risk_levels.
      - If they agree: use that risk_level (no arbitration needed).
      - If they disagree: ESCALATE -- take the more severe (higher on the
        Very Low < Low < Medium < High < Catastrophic ladder) of the two.
    This is an "escalate-on-disagreement" rule: symmetric (it does not favor
    the LLM or the ML a priori), safety-first (standard practice in
    safety-critical / GRC risk management: when two independent assessors
    disagree, adopt the more conservative reading pending further review),
    and simple (a single min/max comparison, no free parameters to tune).

  WHY NOT confidence-gated tie-break (the other option considered): NSL-KDD
  XGBoost is documented elsewhere in this codebase (see the comment above
  run_tristage_llm_arm in scripts/run_nslkdd_risk_ablation.py) to be
  confidently WRONG -- it mispredicts R2L/U2R as Benign at 99-100% reported
  confidence. A rule of the form "defer to whichever model reports higher
  confidence" would systematically defer to XGBoost's mis-calibrated
  confidence exactly in the cases that matter most (severe under-called
  attacks), defeating the purpose of adding independent LLM reasoning. A
  confidence-based rule is therefore not defensible here and was rejected
  BEFORE running, not after seeing results. Escalate-on-disagreement has no
  such dependency on ML confidence calibration.

Stage 3 (LLM narrative + control, over the FINAL reconciled risk_level):
  a short GRC narrative + control recommendation is generated over the
  reconciled risk_level (re-using the Stage-1a LLM artifact's narrative when
  its own risk_level was the one selected by arbitration; when the ML's
  risk_level was selected instead, a lightweight narrative call embeds the
  reconciled level -- see build_narrative_prompt below). This preserves the
  "GRC completeness" value-add of the tri-stage design without re-injecting
  the ML class as an authoritative anchor into the classification step.

Metrics: identical logic (under/over-escalation, severe recall, worst-miss,
paired McNemar vs the clean XGBoost baseline) as
scripts/run_nslkdd_tristage_haiku_clean.py, so results are directly
comparable to the b=0,c=0,p=1.0 "obey-the-table" clean rerun and to the old
leaky b=14,c=0,p=6.1e-5 spine test.

Usage:
    python -m scripts.run_nslkdd_reconciled_tristage_haiku --run-test
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

TEST_PATH = ROOT / "results/nslkdd_unified_rerun/test_sample.json"
OUT_PATH = ROOT / "results/nslkdd_unified_rerun/reconciled_tristage_sonnet_results.json"
OUT_JSONL = ROOT / "results/nslkdd_unified_rerun/reconciled_tristage_sonnet_test.jsonl"

MODEL = "claude-sonnet-4-6"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

VALID_CLASSES = {"Normal", "DoS", "Probe", "R2L", "U2R"}

# ── Stage 1a: independent LLM inference (verbatim from
#    scripts/run_nslkdd_ablation.py::SYSTEM_PROMPT_PURE_LLM /
#    PROMPT_PURE_LLM -- the "Pure LLM" arm already in this codebase) ─────────

SYSTEM_PROMPT_PURE_LLM = """You are a GRC analyst applying NFCRM-1:2025.
The attack classification is NOT available — infer the most likely NSL-KDD attack class from the CVE, flow description, and asset criticality.

MANDATORY — §6.9 Risk Level Lookup Table for NSL-KDD traffic:

  Criticality values: Low | Medium | High | Critical

  NSL-KDD class | Low      | Medium | High          | Critical
  --------------|----------|--------|---------------|----------
  Normal        | Very Low | Very Low | Very Low    | Very Low
  DoS           | Medium   | High   | Catastrophic  | Catastrophic
  Probe         | Low      | Medium | Medium        | High
  R2L           | Medium   | High   | Catastrophic  | Catastrophic
  U2R           | Medium   | High   | Catastrophic  | Catastrophic

Steps: (1) infer NSL-KDD class (Normal/DoS/Probe/R2L/U2R) from flow+CVE, (2) look up risk_level.
Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_PURE_LLM = """Security event (NSL-KDD network flow, attack class UNKNOWN):
- Asset criticality: {criticality}
- Asset: {asset_json}
- CVE context: {cve_block}
- Network flow NL description: {nl_desc}

Step 1 — infer the most likely NSL-KDD attack class (Normal/DoS/Probe/R2L/U2R) from the CVE and flow.
Step 2 — look up that class + "{criticality}" in the §6.9 table to get risk_level.

```json
{{
  "risk_level": "<exact value from §6.9 table>",
  "recommended_control_id": "<NFCRM control>",
  "nfcrm_clauses": ["§6.9"],
  "inferred_attack_class": "<your inferred class>",
  "narrative": "<2-3 sentences>",
  "evidence_refs": ["<cve_id if any>"]
}}```"""

# ── Stage 3: narrative-only re-grounding when the ML's risk_level was the
#    one selected by arbitration (LLM's own artifact narrative disagreed
#    with the reconciled level, so it can't be reused verbatim) ────────────

SYSTEM_PROMPT_RECONCILE_NARRATIVE = """You are a GRC analyst writing the audit narrative for a NFCRM-1:2025 §6.9 risk assessment on NSL-KDD data.
Two independent assessments (an ML classifier and an independent LLM inference) DISAGREED on the risk level for this event.
Per the NFCRM symbolic arbitration rule, the MORE SEVERE of the two assessments is authoritative and is given to you below — do not change it.
Write a concise 2-3 sentence narrative that (a) states the reconciled risk level, (b) notes the disagreement between the two independent assessments and that the more severe one was adopted as a safety-first measure, (c) cites the CVE and asset criticality. Recommend one NFCRM-1:2025 control (starting SS5 or SS6).
Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_RECONCILE_NARRATIVE = """RECONCILED RESULT (authoritative — narrate only):
- ML-predicted class: {ml_class} -> ML risk_level: {ml_risk}
- LLM-inferred class: {llm_class} -> LLM risk_level: {llm_risk}
- Reconciled (escalate-on-disagreement) risk_level: {final_risk}
- Asset criticality: {criticality}
- CVE context: {cve_block}
- Flow description: {nl_desc}

```json
{{
  "risk_level": "{final_risk}",
  "recommended_control_id": "<NFCRM control starting SS5 or SS6>",
  "nfcrm_clauses": ["§6.9"],
  "narrative": "<2-3 sentences per instructions>",
  "evidence_refs": ["<cve_id if any>"]
}}```"""

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def _format_cve(cve: Optional[dict]) -> str:
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


def reconcile(llm_risk: str, ml_risk: str) -> tuple[str, str]:
    """Pre-registered escalate-on-disagreement arbitration.

    Returns (final_risk_level, source) where source in
    {"agree", "escalate_to_llm", "escalate_to_ml"}.
    If either level fails to parse into the known ladder, fall back to
    whichever one DOES parse (safe default); if neither parses, "" (parse
    error, handled upstream as an under-escalation / unsafe case).
    """
    llm_ok = llm_risk in IDX
    ml_ok = ml_risk in IDX
    if not llm_ok and not ml_ok:
        return "", "both_unparseable"
    if not llm_ok:
        return ml_risk, "ml_only_parseable"
    if not ml_ok:
        return llm_risk, "llm_only_parseable"
    if llm_risk == ml_risk:
        return llm_risk, "agree"
    if IDX[llm_risk] > IDX[ml_risk]:
        return llm_risk, "escalate_to_llm"
    else:
        return ml_risk, "escalate_to_ml"


def run_reconciled_tristage(cases: list[dict], client: ClaudeCLIClient,
                            label: str, out_jsonl: Optional[Path] = None) -> list[dict]:
    # Resume support: if the output JSONL already holds completed cases, keep
    # them, skip their case_ids, and append the remainder (kill-safe reruns).
    records = []
    done_ids = set()
    if out_jsonl and out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records.append(r)
                done_ids.add(r["case_id"])
        if done_ids:
            print(f"[resume] {len(done_ids)} cases already in {out_jsonl.name}; skipping them")
    cases = [c for c in cases if c["case_id"] not in done_ids]
    fout = out_jsonl.open("a" if done_ids else "w", encoding="utf-8") if out_jsonl else None
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

            ml_class = case["xgb_predicted_class"]
            ml_risk = case["xgb_risk_level"]

            # Stage 1a: independent LLM inference (no ML class shown).
            infer_prompt = PROMPT_PURE_LLM.format(
                criticality=crit, asset_json=asset_j, cve_block=cve_blk,
                nl_desc=nl_desc[:300],
            )
            llm_calls = 0
            try:
                infer_resp = client.generate(SYSTEM_PROMPT_PURE_LLM, infer_prompt) or ""
            except Exception as e:
                infer_resp = ""
                print(f"  [{i}/{len(cases)}] {cid}: EXCEPTION during Stage-1a generate: {e}")
            llm_calls += 1
            infer_art = _parse_response(infer_resp)

            llm_class = (infer_art or {}).get("inferred_attack_class", "")
            llm_risk = (infer_art or {}).get("risk_level", "")
            llm_narrative = (infer_art or {}).get("narrative", "")
            llm_control = (infer_art or {}).get("recommended_control_id", "")
            infer_parse_ok = infer_art is not None

            # Stage 2: symbolic arbitration over the two INDEPENDENT judgments.
            final_risk, arb_source = reconcile(llm_risk, ml_risk)
            class_disagree = (llm_class in VALID_CLASSES) and (llm_class != ml_class)

            # Stage 3: narrative + control over the reconciled level.
            if arb_source in ("agree", "escalate_to_llm", "llm_only_parseable") and infer_parse_ok:
                narrative = llm_narrative
                control = llm_control
                narrative_calls = 0
            else:
                # ML's risk_level was selected (or LLM parse failed) -- generate
                # a fresh narrative grounded in the reconciled level.
                narr_prompt = PROMPT_RECONCILE_NARRATIVE.format(
                    ml_class=ml_class, ml_risk=ml_risk,
                    llm_class=llm_class or "unparsed", llm_risk=llm_risk or "unparsed",
                    final_risk=final_risk or ml_risk, criticality=crit,
                    cve_block=cve_blk, nl_desc=nl_desc[:300],
                )
                try:
                    narr_resp = client.generate(SYSTEM_PROMPT_RECONCILE_NARRATIVE, narr_prompt) or ""
                except Exception as e:
                    narr_resp = ""
                    print(f"  [{i}/{len(cases)}] {cid}: EXCEPTION during Stage-3 generate: {e}")
                narrative_calls = 1
                narr_art = _parse_response(narr_resp)
                narrative = (narr_art or {}).get("narrative", "")
                control = (narr_art or {}).get("recommended_control_id", "")
            llm_calls += narrative_calls

            risk_lvl = final_risk
            correct = risk_lvl == gt
            parse_error = not infer_parse_ok and arb_source == "both_unparseable"

            d = (IDX.get(risk_lvl) - IDX[gt]) if risk_lvl in IDX and gt in IDX else None
            under_escalated = (d is not None and d < 0) or (risk_lvl not in IDX)
            over_escalated = d is not None and d > 0

            record = {
                "case_id": cid,
                "paradigm": "reconciled_tristage_llm_sonnet",
                "model": MODEL,
                "true_class": case["true_attack_class"],
                "true_nfcrm_class": case.get("true_nfcrm_class", ""),
                "ml_pred_class": ml_class,
                "ml_risk_level": ml_risk,
                "llm_inferred_class": llm_class,
                "llm_risk_level": llm_risk,
                "class_disagreement": bool(class_disagree),
                "risk_level_disagreement": bool(arb_source in ("escalate_to_llm", "escalate_to_ml")),
                "arbitration_source": arb_source,
                "risk_level": risk_lvl,
                "ground_truth": gt,
                "correct": correct,
                "under_escalated": bool(under_escalated),
                "over_escalated": bool(over_escalated),
                "severe_gt": gt in SEVERE,
                "severe_hit": (gt in SEVERE) and (risk_lvl in SEVERE),
                "parse_error": bool(parse_error),
                "infer_parse_error": not infer_parse_ok,
                "criticality": crit,
                "grc": "Full",
                "narrative": narrative,
                "control": control,
                "llm_calls": llm_calls,
                "raw_response_stage1a": infer_resp,
                "latency_ms": int((time.time() - t0) * 1000),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            records.append(record)
            if fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            match = "OK" if correct else ("PARSE_ERR" if parse_error else "MISS")
            dis = "" if arb_source == "agree" else f" DISAGREE({arb_source})"
            print(f"  [{label} {i}/{len(cases)}] {cid} ({case['true_attack_class']:6}): "
                  f"ml={ml_class:6}/{ml_risk:10} llm={llm_class or '?':6}/{llm_risk or '?':10} "
                  f"-> final={risk_lvl or '?':12} GT={gt:12} [{match}]{dis} ({int((time.time()-t0)*1000)}ms)")
    finally:
        if fout:
            fout.close()
    return records


# ── Metrics (verbatim logic from
#    scripts/run_nslkdd_tristage_haiku_clean.py::metrics()) ─────────────────

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
    """b = XGB unsafe & Reconciled-LLM safe; c = Reconciled-LLM unsafe & XGB safe.
    H1: b>c (reconciled tri-stage safer). Identical semantics to
    scripts/run_nslkdd_tristage_haiku_clean.py::paired_mcnemar()."""
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


def main() -> None:
    global OUT_PATH, OUT_JSONL
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-test", action="store_true",
                    help="Run the real 100-case held-out test_sample.json evaluation.")
    ap.add_argument("--run-tag", default=None,
                    help="Optional replicate tag (e.g. 'run2'): outputs go to "
                         "results/nslkdd_unified_rerun/replicates/ with the tag "
                         "suffixed. Default (no tag) preserves original paths. "
                         "Prompts, seeds, and scoring are untouched.")
    args = ap.parse_args()
    if not args.run_test:
        args.run_test = True
    if args.run_tag:
        rep_dir = ROOT / "results/nslkdd_unified_rerun/replicates"
        rep_dir.mkdir(parents=True, exist_ok=True)
        OUT_PATH = rep_dir / f"reconciled_tristage_sonnet_results_{args.run_tag}.json"
        OUT_JSONL = rep_dir / f"reconciled_tristage_sonnet_test_{args.run_tag}.jsonl"

    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=180)

    print("\n" + "=" * 70)
    print("RECONCILED TRI-STAGE (HELD-OUT TEST, N=100, real evaluation)")
    print("=" * 70)
    test_data = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    test_cases = test_data["cases"]
    test_records = run_reconciled_tristage(test_cases, client, "test", out_jsonl=OUT_JSONL)

    llm_m = metrics(test_records)
    xgb_m = xgb_metrics_from_cases(test_cases)
    spine = paired_mcnemar(test_records, test_cases)
    n_parse_err = sum(1 for r in test_records if r["parse_error"])
    n_class_disagree = sum(1 for r in test_records if r["class_disagreement"])
    n_risk_disagree = sum(1 for r in test_records if r["risk_level_disagreement"])
    n_escalate_llm = sum(1 for r in test_records if r["arbitration_source"] == "escalate_to_llm")
    n_escalate_ml = sum(1 for r in test_records if r["arbitration_source"] == "escalate_to_ml")

    # Of the cases where LLM and ML disagreed on risk level, how many times
    # did the resulting escalation move the answer CLOSER to ground truth
    # (a "safer and correct-direction" pick) vs further from it?
    disagreement_diagnostics = []
    for r in test_records:
        if not r["risk_level_disagreement"]:
            continue
        gt = r["ground_truth"]
        picked = r["risk_level"]
        other = r["ml_risk_level"] if r["arbitration_source"] == "escalate_to_llm" else r["llm_risk_level"]
        picked_dist = abs(IDX.get(picked, 0) - IDX.get(gt, 0)) if picked in IDX and gt in IDX else None
        other_dist = abs(IDX.get(other, 0) - IDX.get(gt, 0)) if other in IDX and gt in IDX else None
        disagreement_diagnostics.append({
            "case_id": r["case_id"], "true_class": r["true_class"], "ground_truth": gt,
            "ml_class": r["ml_pred_class"], "ml_risk": r["ml_risk_level"],
            "llm_class": r["llm_inferred_class"], "llm_risk": r["llm_risk_level"],
            "picked": picked, "picked_source": r["arbitration_source"],
            "picked_was_closer_to_gt": (picked_dist is not None and other_dist is not None
                                        and picked_dist < other_dist),
        })

    print(f"\nReconciled tri-stage (Haiku, N=100) metrics: {json.dumps(llm_m, indent=2)}")
    print(f"XGBoost clean baseline metrics:               {json.dumps(xgb_m, indent=2)}")
    print(f"Parse errors: {n_parse_err}/{len(test_records)}")
    print(f"Class disagreement (LLM vs ML): {n_class_disagree}/{len(test_records)}")
    print(f"Risk-level disagreement requiring arbitration: {n_risk_disagree}/{len(test_records)} "
          f"(escalated_to_llm={n_escalate_llm}, escalated_to_ml={n_escalate_ml})")
    print(f"\nSPINE TEST — under-escalation McNemar, Reconciled-Tristage(Haiku) vs XGBoost(clean): "
          f"b(XGB-unsafe,Reconciled-safe)={spine['b']}  c(Reconciled-unsafe,XGB-safe)={spine['c']}  "
          f"p(1-sided)={spine['p']:.4g}")

    out = {
        "meta": {
            "description": ("Reconciled tri-stage LLM (Haiku 4.5) safety rerun on the clean, "
                             "leak-free NSL-KDD held-out test sample. Stage 1a: LLM independently "
                             "infers class + risk_level from the neutral NL description ALONE "
                             "(no ML class shown -- reuses the existing Pure-LLM prompt verbatim). "
                             "Stage 1b: ML class + risk_level from the fixed XGBoost baseline "
                             "(test_sample.json, unchanged). Stage 2: pre-registered "
                             "escalate-on-disagreement symbolic arbitration (take the more severe "
                             "of the two independent risk_levels when they disagree; use the "
                             "agreed level otherwise) -- chosen and documented BEFORE this run, "
                             "not tuned to its outcome. Stage 3: narrative + control over the "
                             "final reconciled risk_level."),
            "arbitration_rule": "escalate_on_disagreement",
            "arbitration_rule_rationale": (
                "Symmetric, safety-first, parameter-free: when two independently-derived risk "
                "assessments disagree, adopt the more severe one pending review (standard "
                "safety-critical practice). Rejected alternative: confidence-gated tie-break, "
                "because XGBoost's reported confidence is documented elsewhere in this codebase "
                "to be poorly calibrated for exactly the classes that matter (R2L/U2R "
                "confidently mispredicted as Benign at 99-100% confidence), so deferring to "
                "'higher reported confidence' would systematically defer to the miscalibrated "
                "ML confidence on the highest-stakes cases."),
            "model": MODEL,
            "test_sample_source": str(TEST_PATH.relative_to(ROOT)),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "held_out_test": {
            "n": len(test_records),
            "reconciled_tristage_llm_sonnet_metrics": llm_m,
            "xgboost_clean_baseline_metrics": xgb_m,
            "unsafe_mcnemar_reconciled_vs_xgb": spine,
            "n_parse_errors": n_parse_err,
            "n_class_disagreement": n_class_disagree,
            "n_risk_level_disagreement": n_risk_disagree,
            "n_escalated_to_llm": n_escalate_llm,
            "n_escalated_to_ml": n_escalate_ml,
            "disagreement_diagnostics": disagreement_diagnostics,
            "records": test_records,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
