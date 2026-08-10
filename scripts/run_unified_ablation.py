"""Unified Ablation Runner — all paradigms on the same N=180 test set.

Evaluates six paradigms on the unified test set produced by
build_unified_ablation_testset.py. Each paradigm produces a risk_level
prediction so accuracy can be compared on a single common metric.

Paradigms:
  rule          Pure symbolic rule lookup (no ML, no LLM)
  ml_rf         Pure ML — RF-predicted attack class → §6.9 lookup
  ml_xgb        Pure ML — XGBoost-predicted attack class → §6.9 lookup
  tristage_rf   SHACL+RF → §6.9 lookup + SHACL annotation (GRC=Partial)
  tristage_xgb  SHACL+XGBoost → §6.9 lookup + SHACL annotation (GRC=Partial)
  pure_llm      NL description + CVE → LLM → risk level (no ML attack class)
  tristage_llm_haiku   ML attack class + NL + CVE → Claude Haiku → risk level + GRC
  tristage_llm_sonnet  ML attack class + NL + CVE → Claude Sonnet → risk level + GRC

Usage:
    # Non-LLM paradigms (fast, no API cost):
    python scripts/run_unified_ablation.py --paradigm rule
    python scripts/run_unified_ablation.py --paradigm ml_rf ml_xgb tristage_rf tristage_xgb

    # LLM paradigms (requires Claude CLI auth, ~40 min per arm):
    python scripts/run_unified_ablation.py --paradigm pure_llm --model claude-haiku-4-5-20251001
    python scripts/run_unified_ablation.py --paradigm tristage_llm_haiku
    python scripts/run_unified_ablation.py --paradigm tristage_llm_sonnet

    # All non-LLM at once:
    python scripts/run_unified_ablation.py --all-nonllm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.nfcrm.risk_score import compute_risk_score
from scripts.llm_artifact_lib import RISK_LEVELS, validate_artifact_schema

RISK_LEVELS_SET = set(RISK_LEVELS)
CRITICALITY_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}

ATTACK_TO_CONTROL = {
    "Benign":       "SS6.1-MON-1",
    "BruteForce":   "SS6.7-AC-1",
    "DoS":          "SS6.7-NET-1",
    "DDoS":         "SS6.7-NET-2",
    "Infiltration": "SS6.5-NET-1",
    "WebAttack":    "SS6.7-APP-1",
}

# ── §6.9 rule lookup ──────────────────────────────────────────────────────────

def rule_lookup(attack_class: str, criticality: str) -> str:
    """NFCRM §6.9 deterministic risk level from attack class + criticality."""
    c_val = CRITICALITY_MAP.get(criticality, 3)
    score = compute_risk_score(attack_class, c_override=c_val, i_override=c_val, a_override=c_val)
    raw = score.level_en
    return "Very Low" if raw == "N/A (non-attack)" else raw


def symbolic_rule_predict_class(case: dict) -> str:
    """Simple signature-based attack class from feature_subset (no ML)."""
    f = case.get("feature_subset", {})
    port     = int(f.get("Dst Port", 0))
    win_fwd  = int(f.get("Init Fwd Win Byts", 0))
    dur_us   = float(f.get("Flow Duration", 0))
    dur_s    = dur_us / 1e6
    fwd_pkts = int(f.get("Tot Fwd Pkts", 0))
    bwd_pkts = int(f.get("Tot Bwd Pkts", 0))

    if win_fwd == 225 and port == 80:
        return "DoS"
    if win_fwd == 65535 and port == 80 and dur_s < 0.1:
        return "DDoS"
    if port == 80 and dur_s > 30 and fwd_pkts > 50 and bwd_pkts > 30:
        return "BruteForce"
    if port == 80 and 3 <= dur_s <= 10 and 2 <= fwd_pkts <= 15 and 1 <= bwd_pkts <= 15:
        return "WebAttack"
    if port not in {80, 443, 8080, 53, 22, 21, 3389, 3306} and port > 1024:
        return "Infiltration"
    return "Benign"


# ── Symbolic SHACL pre-filter (simple approximation) ─────────────────────────

def shacl_prefilter(case: dict) -> bool:
    """Returns True if the flow passes the SHACL monitoring filter."""
    # Very simple: filter out flows with zero packets (noise)
    f = case.get("feature_subset", {})
    fwd = int(f.get("Tot Fwd Pkts", 0))
    bwd = int(f.get("Tot Bwd Pkts", 0))
    return (fwd + bwd) > 0


# ── Non-LLM paradigm runner ───────────────────────────────────────────────────

def run_nonllm_paradigm(cases: list[dict], paradigm: str) -> list[dict]:
    results = []
    for case in cases:
        cid     = case["case_id"]
        crit    = case["criticality"]
        gt      = case["ground_truth_risk_level"]
        true_cl = case["true_attack_class"]

        if paradigm == "rule":
            pred_cls = symbolic_rule_predict_class(case)
            risk_lvl = rule_lookup(pred_cls, crit)
            grc      = "None"

        elif paradigm == "ml_rf":
            pred_cls = case["rf_predicted_class"]
            risk_lvl = case["rf_risk_level"]
            grc      = "None"

        elif paradigm == "ml_xgb":
            pred_cls = case["xgb_predicted_class"]
            risk_lvl = case["xgb_risk_level"]
            grc      = "None"

        elif paradigm == "tristage_rf":
            if not shacl_prefilter(case):
                pred_cls = "Benign"
                risk_lvl = "Very Low"
            else:
                pred_cls = case["rf_predicted_class"]
                risk_lvl = case["rf_risk_level"]
            grc = "Partial"

        elif paradigm == "tristage_xgb":
            if not shacl_prefilter(case):
                pred_cls = "Benign"
                risk_lvl = "Very Low"
            else:
                pred_cls = case["xgb_predicted_class"]
                risk_lvl = case["xgb_risk_level"]
            grc = "Partial"

        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

        results.append({
            "case_id":      cid,
            "paradigm":     paradigm,
            "true_class":   true_cl,
            "pred_class":   pred_cls,
            "risk_level":   risk_lvl,
            "ground_truth": gt,
            "correct":      risk_lvl == gt,
            "grc":          grc,
            "criticality":  crit,
        })

    correct = sum(1 for r in results if r["correct"])
    print(f"  {paradigm}: {correct}/{len(results)} = {correct/len(results):.1%}")
    return results


# ── LLM paradigm runner ───────────────────────────────────────────────────────

SYSTEM_PROMPT_TRISTAGE = """You are a GRC analyst applying NFCRM-1:2025 (Saudi Arabia National Cybersecurity Risk Management framework).

An ML classifier has suggested an attack class with a confidence score. Verify the suggestion, then apply §6.9.

MANDATORY — §6.9 Risk Level Lookup Table:

  Criticality values: Low | Medium | High | Critical

  Attack class  | Low      | Medium | High          | Critical
  --------------|----------|--------|---------------|----------
  Benign        | Very Low | Very Low | Very Low    | Very Low
  BruteForce    | Medium   | High   | Catastrophic  | Catastrophic
  DoS           | Medium   | High   | Catastrophic  | Catastrophic
  DDoS          | Medium   | High   | Catastrophic  | Catastrophic
  WebAttack     | Medium   | High   | Catastrophic  | Catastrophic
  Infiltration  | Low      | Medium | Medium        | High

VERIFICATION RULE — override only when ALL three conditions hold:
  (a) ML predicts Benign with confidence < 80%
  (b) The paired CVE has CVSS ≥ 9.5 (critical severity exploit)
  (c) Asset criticality is High or Critical
  → In this case classify as Infiltration (stealthy C2 or data exfiltration missed by ML).

In all other cases trust the ML class exactly as given.
Output ONLY a valid JSON object inside a ```json fenced code block."""

SYSTEM_PROMPT_PURE_LLM = """You are a GRC analyst applying NFCRM-1:2025.
The attack classification is NOT available — infer the most likely attack class from the CVE CVSS score,
flow description, and asset criticality, then apply the §6.9 table below.

MANDATORY — §6.9 Risk Level Lookup Table:

  Criticality values: Low | Medium | High | Critical

  Attack class  | Low      | Medium | High          | Critical
  --------------|----------|--------|---------------|----------
  Benign        | Very Low | Very Low | Very Low    | Very Low
  BruteForce    | Medium   | High   | Catastrophic  | Catastrophic
  DoS           | Medium   | High   | Catastrophic  | Catastrophic
  DDoS          | Medium   | High   | Catastrophic  | Catastrophic
  WebAttack     | Medium   | High   | Catastrophic  | Catastrophic
  Infiltration  | Low      | Medium | Medium        | High

Steps: (1) infer attack class from flow+CVE, (2) look up risk_level using the table above.
Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_TRISTAGE = """Security event:
- Attack class (from ML classifier): {attack_class}
- ML classifier confidence: {confidence}
- Asset criticality: {criticality}
- Asset: {asset_json}
- CVE context: {cve_block}
- Flow description: {nl_desc}

Step 1 — Apply the verification rule: if ML=Benign AND confidence<80% AND CVE CVSS≥9.5 AND criticality is High/Critical → override to Infiltration. Otherwise keep "{attack_class}".
Step 2 — Look up the verified attack class + "{criticality}" in the §6.9 table → risk_level.
Step 3 — Write 2–3 sentence narrative citing the verified class, CVE, and asset criticality.

```json
{{
  "verified_attack_class": "<confirmed or overridden class>",
  "risk_level": "<exact value from §6.9 table>",
  "recommended_control_id": "<NFCRM control>",
  "nfcrm_clauses": ["§6.9"],
  "narrative": "<2-3 sentences>",
  "evidence_refs": ["<cve_id>"]
}}```"""

PROMPT_PURE_LLM = """Security event (attack class UNKNOWN):
- Asset criticality: {criticality}
- Asset: {asset_json}
- CVE context: {cve_block}
- Network flow NL description: {nl_desc}

Step 1 — infer the most likely attack class (Benign/BruteForce/DoS/DDoS/WebAttack/Infiltration) from the CVE and flow.
Step 2 — look up that attack class + "{criticality}" in the §6.9 table to get risk_level.

```json
{{
  "risk_level": "<exact value from §6.9 table>",
  "recommended_control_id": "<NFCRM control>",
  "nfcrm_clauses": ["§6.9"],
  "inferred_attack_class": "<your inferred class>",
  "narrative": "<2-3 sentences>",
  "evidence_refs": ["<cve_id if any>"]
}}```"""

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def _format_cve(cve: Optional[dict]) -> str:
    if not cve:
        return "No CVE context."
    return (f"CVE: {cve.get('id')}  CVSS: {cve.get('cvss_v3')}  "
            f"Desc: {cve.get('shortDescription','')[:120]}")


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


def run_llm_paradigm(cases: list[dict], paradigm: str, model: str,
                     out_path: Path, start_offset: int = 0,
                     client: Any = None) -> None:
    """Stream results to JSONL file as they complete (resumable).

    `client` must expose .generate(system_msg, user_msg) -> str|None.
    Defaults to ClaudeCLIClient when None.
    """
    if client is None:
        from scripts.claude_cli_client import ClaudeCLIClient
        client = ClaudeCLIClient(model=model, max_tokens=600, timeout_sec=180)

    cases_to_run = cases[start_offset:]
    append = start_offset > 0
    mode = "a" if append else "w"
    print(f"Model: {model} | Cases: {len(cases_to_run)} | offset={start_offset} | append={append}")

    with out_path.open(mode, encoding="utf-8") as f:
        for i, case in enumerate(cases_to_run, start_offset + 1):
            t0 = time.time()
            cid      = case["case_id"]
            crit     = case["criticality"]
            gt       = case["ground_truth_risk_level"]
            cve      = case.get("paired_cve")
            asset_j  = json.dumps(case.get("asset", {}))
            cve_blk  = _format_cve(cve)
            nl_desc  = case.get("nl_description", "")

            if "tristage_llm" in paradigm:
                ml_cls  = case["xgb_predicted_class"]
                ml_conf = case.get("xgb_confidence", 1.0)
                prompt  = PROMPT_TRISTAGE.format(
                    attack_class=ml_cls,
                    confidence=f"{ml_conf:.0%}",
                    criticality=crit,
                    asset_json=asset_j, cve_block=cve_blk,
                    nl_desc=nl_desc[:300],
                )
                system = SYSTEM_PROMPT_TRISTAGE
            else:  # pure_llm
                prompt = PROMPT_PURE_LLM.format(
                    criticality=crit, asset_json=asset_j,
                    cve_block=cve_blk, nl_desc=nl_desc[:400],
                )
                system = SYSTEM_PROMPT_PURE_LLM

            try:
                response = client.generate(system, prompt) or ""
            except Exception as e:
                response = ""

            artifact = _parse_response(response)
            latency = int((time.time() - t0) * 1000)

            if artifact:
                # For tri-stage: recompute risk from verified_attack_class if LLM overrode
                if "tristage_llm" in paradigm:
                    verified_cls = artifact.get("verified_attack_class", "")
                    if verified_cls and verified_cls in ATTACK_TO_CONTROL and verified_cls != ml_cls:
                        risk_lvl = rule_lookup(verified_cls, crit)
                    else:
                        risk_lvl = artifact.get("risk_level", "")
                else:
                    risk_lvl = artifact.get("risk_level", "")
                correct  = risk_lvl == gt
                parse_ok = True
            else:
                # Parse failure: tri-stage falls back to ML (no LLM override fired = trust ML).
                # Pure-LLM has no ML upstream so it must be scored as a miss.
                if "tristage_llm" in paradigm:
                    risk_lvl = case.get("xgb_risk_level", "")
                    correct  = risk_lvl == gt
                else:
                    risk_lvl = ""
                    correct  = False
                parse_ok = False

            record = {
                "case_id":          cid,
                "paradigm":         paradigm,
                "model":            model,
                "true_class":       case["true_attack_class"],
                "ml_pred_class":    case.get("xgb_predicted_class", ""),
                "llm_verified_cls": (artifact or {}).get("verified_attack_class", "") if "tristage_llm" in paradigm else "",
                "risk_level":       risk_lvl,
                "ground_truth":     gt,
                "correct":          correct,
                "parse_error":      not parse_ok,
                "criticality":      crit,
                "grc":              "Full",
                "narrative":        (artifact or {}).get("narrative", ""),
                "control":          (artifact or {}).get("recommended_control_id", ""),
                "latency_ms":       latency,
                "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            match = "OK" if correct else "MISS"
            print(f"  [{i}/{len(cases)}] {cid}: {risk_lvl or 'PARSE_ERR'} vs GT={gt} [{match}] ({latency}ms)")


# ── Analysis helper ───────────────────────────────────────────────────────────

def analyze_jsonl(path: Path, paradigm: str) -> dict:
    if not path.exists():
        return {}
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(lines)
    correct = sum(1 for l in lines if l.get("correct"))
    parse_errs = sum(1 for l in lines if l.get("parse_error"))
    grc = lines[0].get("grc", "?") if lines else "?"
    return {
        "paradigm": paradigm, "n": n,
        "accuracy": correct / n if n else 0,
        "grc": grc, "parse_errors": parse_errs,
    }


def print_summary_table(results_dir: Path) -> None:
    """Print comparison table from all completed paradigm results."""
    PARADIGMS = [
        ("rule",                   "results/unified_ablation/rule.jsonl"),
        ("ml_rf",                  "results/unified_ablation/ml_rf.jsonl"),
        ("ml_xgb",                 "results/unified_ablation/ml_xgb.jsonl"),
        ("tristage_rf",            "results/unified_ablation/tristage_rf.jsonl"),
        ("tristage_xgb",           "results/unified_ablation/tristage_xgb.jsonl"),
        ("pure_llm",               "results/unified_ablation/pure_llm.jsonl"),
        ("tristage_llm_haiku",     "results/unified_ablation/tristage_llm_haiku.jsonl"),
        ("tristage_llm_sonnet",    "results/unified_ablation/tristage_llm_sonnet.jsonl"),
    ]
    print(f"\n{'Paradigm':<27} {'Accuracy':>10} {'GRC':>8} {'N':>5} {'ParseErr':>10}")
    print("-" * 65)
    for pname, rel_path in PARADIGMS:
        r = analyze_jsonl(ROOT / rel_path, pname)
        if r:
            print(f"  {pname:<25} {r['accuracy']:>9.1%} {r['grc']:>8} {r['n']:>5} {r['parse_errors']:>10}")
        else:
            print(f"  {pname:<25} {'—':>9}  {'—':>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

NON_LLM_PARADIGMS    = ["rule", "ml_rf", "ml_xgb", "tristage_rf", "tristage_xgb"]
LLM_PARADIGMS        = ["pure_llm", "tristage_llm_haiku", "tristage_llm_sonnet"]
ALL_LLM_PARADIGMS    = LLM_PARADIGMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", nargs="+",
                    choices=NON_LLM_PARADIGMS + ALL_LLM_PARADIGMS)
    ap.add_argument("--all-nonllm", action="store_true", help="Run all non-LLM paradigms")
    ap.add_argument("--summary", action="store_true", help="Print summary table of completed results")
    ap.add_argument("--test-set", default="results/unified_ablation/test_set.json")
    ap.add_argument("--out-dir", default="results/unified_ablation")
    ap.add_argument("--model", default=None,
                    help="Model for LLM paradigms (default: haiku for *_haiku, sonnet for *_sonnet)")
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--n-per-class-llm", type=int, default=None,
                    help="Limit LLM arms to N cases per class (stratified). Default: use all 30/class.")
    ap.add_argument("--out", default=None,
                    help="Override output JSONL filename (basename only, e.g. tristage_llm_sonnet_run2.jsonl). "
                         "Only applies when running a single paradigm.")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_summary_table(out_dir)
        return

    test_set_path = ROOT / args.test_set
    if not test_set_path.exists():
        print(f"ERROR: test set not found at {test_set_path}")
        print("Run: python scripts/build_unified_ablation_testset.py")
        sys.exit(1)

    all_cases = json.loads(test_set_path.read_text())["cases"]
    print(f"Loaded {len(all_cases)} cases from {test_set_path}")

    # Build stratified LLM subset if requested
    if args.n_per_class_llm:
        from collections import defaultdict
        import random as _random
        by_cls: dict[str, list] = defaultdict(list)
        for c in all_cases:
            by_cls[c["true_attack_class"]].append(c)
        llm_cases = []
        for cls, items in sorted(by_cls.items()):
            llm_cases.extend(items[:args.n_per_class_llm])
        _random.seed(42)
        _random.shuffle(llm_cases)  # mix classes so running accuracy isn't class-order biased
        print(f"LLM subset: {len(llm_cases)} cases ({args.n_per_class_llm}/class, shuffled)")
    else:
        llm_cases = all_cases

    to_run = list(args.paradigm or [])
    if args.all_nonllm:
        to_run = NON_LLM_PARADIGMS

    for paradigm in to_run:
        if args.out and len(to_run) == 1:
            out_path = out_dir / args.out
        else:
            out_path = out_dir / f"{paradigm}.jsonl"
        print(f"\n--- Running: {paradigm} ---")

        if paradigm in NON_LLM_PARADIGMS:
            results = run_nonllm_paradigm(all_cases, paradigm)
            mode = "a" if args.start_offset > 0 else "w"
            with out_path.open(mode, encoding="utf-8") as f:
                for r in results[args.start_offset:]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Saved to {out_path}")

        else:  # Claude LLM paradigm
            if args.model:
                model = args.model
            elif "haiku" in paradigm:
                model = "claude-haiku-4-5-20251001"
            else:
                model = "claude-sonnet-4-6"
            run_llm_paradigm(llm_cases, paradigm, model, out_path, args.start_offset)

    if to_run:
        print_summary_table(out_dir)


if __name__ == "__main__":
    main()
