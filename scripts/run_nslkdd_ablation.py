"""NSL-KDD Unified Ablation Runner — same 8-paradigm framework as CIC-IDS2018.

NSL-KDD classes:   Normal, DoS, Probe, R2L, U2R
NFCRM §6.9 analogs: Benign, DoS, Infiltration, WebAttack, BruteForce

Usage:
    # Non-LLM paradigms (fast, no API cost):
    python scripts/run_nslkdd_ablation.py --all-nonllm

    # LLM paradigms:
    python scripts/run_nslkdd_ablation.py --paradigm tristage_llm_haiku
    python scripts/run_nslkdd_ablation.py --paradigm tristage_llm_sonnet
    python scripts/run_nslkdd_ablation.py --paradigm pure_llm

    # Summary table:
    python scripts/run_nslkdd_ablation.py --summary
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

CRITICALITY_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}

# NSL-KDD → NFCRM analog (must match build_nslkdd_ablation_testset.py)
NSLKDD_TO_NFCRM = {
    "Normal":  "Benign",
    "DoS":     "DoS",
    "Probe":   "Infiltration",
    "R2L":     "WebAttack",
    "U2R":     "BruteForce",
}

CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


# ── §6.9 risk level lookup (via NFCRM analog) ─────────────────────────────────

def rule_lookup(nfcrm_class: str, criticality: str) -> str:
    c_val = CRITICALITY_MAP.get(criticality, 3)
    score = compute_risk_score(nfcrm_class, c_override=c_val, i_override=c_val, a_override=c_val)
    return "Very Low" if score.level_en == "N/A (non-attack)" else score.level_en


def symbolic_rule_predict_class(case: dict) -> str:
    """Symbolic rule for NSL-KDD features → NSL-KDD class."""
    f = case.get("feature_subset", {})
    protocol   = str(f.get("protocol_type", "tcp")).lower()
    service    = str(f.get("service", "other")).lower()
    flag       = str(f.get("flag", "SF")).upper()
    duration   = float(f.get("duration", 0))
    src_bytes  = int(f.get("src_bytes", 0))
    dst_bytes  = int(f.get("dst_bytes", 0))
    count      = int(f.get("count", 0))
    diff_srv   = float(f.get("diff_srv_rate", 0.0))
    dst_h_cnt  = int(f.get("dst_host_count", 0))
    root_shell = int(f.get("root_shell", 0))
    su_att     = int(f.get("su_attempted", 0))
    num_root   = int(f.get("num_root", 0))
    num_shells = int(f.get("num_shells", 0))
    num_failed = int(f.get("num_failed_logins", 0))
    wrong_frag = int(f.get("wrong_fragment", 0))

    # U2R: privilege-escalation signals
    if root_shell == 1 or su_att == 1 or num_root > 0 or num_shells > 0:
        return "U2R"

    # R2L: remote-to-local exploitation
    if num_failed > 0:
        return "R2L"
    if service in ("ftp", "ftp_data") and duration >= 200:
        return "R2L"
    if protocol == "udp" and service == "private" and dst_bytes == 0 and src_bytes < 200:
        return "R2L"

    # Probe: scanning/reconnaissance
    if diff_srv >= 0.5:
        return "Probe"
    if protocol == "icmp" and src_bytes <= 28:
        return "Probe"

    # DoS: flooding
    if wrong_frag > 0:
        return "DoS"
    if flag in ("REJ", "S0") and count >= 50 and diff_srv < 0.4:
        return "DoS"
    if service == "http" and src_bytes >= 10000 and dst_h_cnt >= 200 and diff_srv < 0.2:
        return "DoS"

    return "Normal"


def shacl_prefilter(case: dict) -> bool:
    """Simple SHACL pre-filter: pass flows with non-zero data."""
    f = case.get("feature_subset", {})
    src_b = int(f.get("src_bytes", 0))
    dst_b = int(f.get("dst_bytes", 0))
    count = int(f.get("count", 0))
    return (src_b + dst_b + count) > 0


# ── Non-LLM paradigm runner ───────────────────────────────────────────────────

def run_nonllm_paradigm(cases: list[dict], paradigm: str) -> list[dict]:
    results = []
    for case in cases:
        cid      = case["case_id"]
        crit     = case["criticality"]
        gt       = case["ground_truth_risk_level"]
        true_cls = case["true_attack_class"]

        if paradigm == "rule":
            pred_nsl = symbolic_rule_predict_class(case)
            nfcrm_cl = NSLKDD_TO_NFCRM.get(pred_nsl, "Benign")
            risk_lvl = rule_lookup(nfcrm_cl, crit)
            grc      = "None"
            pred_cls = pred_nsl

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
                pred_cls = "Normal"
                risk_lvl = "Very Low"
            else:
                pred_cls = case["rf_predicted_class"]
                risk_lvl = case["rf_risk_level"]
            grc = "Partial"

        elif paradigm == "tristage_xgb":
            if not shacl_prefilter(case):
                pred_cls = "Normal"
                risk_lvl = "Very Low"
            else:
                pred_cls = case["xgb_predicted_class"]
                risk_lvl = case["xgb_risk_level"]
            grc = "Partial"

        else:
            raise ValueError(f"Unknown non-LLM paradigm: {paradigm}")

        results.append({
            "case_id":      cid,
            "paradigm":     paradigm,
            "true_class":   true_cls,
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


# ── LLM paradigm ──────────────────────────────────────────────────────────────

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

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def _format_cve(cve: Optional[dict]) -> str:
    if not cve:
        return "No CVE context."
    return (f"CVE: {cve.get('id')}  CVSS: {cve.get('cvss_v3')}  "
            f"Desc: {cve.get('shortDescription', '')[:120]}")


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
    if client is None:
        from scripts.claude_cli_client import ClaudeCLIClient
        client = ClaudeCLIClient(model=model, max_tokens=600, timeout_sec=180)

    cases_to_run = cases[start_offset:]
    mode = "a" if start_offset > 0 else "w"
    print(f"Model: {model} | Cases: {len(cases_to_run)} | offset={start_offset}")

    with out_path.open(mode, encoding="utf-8") as f:
        for i, case in enumerate(cases_to_run, start_offset + 1):
            t0       = time.time()
            cid      = case["case_id"]
            crit     = case["criticality"]
            gt       = case["ground_truth_risk_level"]
            cve      = case.get("paired_cve")
            asset_j  = json.dumps(case.get("asset", {}))
            cve_blk  = _format_cve(cve)
            nl_desc  = case.get("nl_description", "")

            if "tristage_llm" in paradigm:
                ml_cls = case["xgb_predicted_class"]
                prompt = PROMPT_TRISTAGE.format(
                    attack_class=ml_cls, criticality=crit,
                    asset_json=asset_j, cve_block=cve_blk,
                    nl_desc=nl_desc[:300],
                )
                system = SYSTEM_PROMPT_TRISTAGE
            else:
                prompt = PROMPT_PURE_LLM.format(
                    criticality=crit, asset_json=asset_j,
                    cve_block=cve_blk, nl_desc=nl_desc[:400],
                )
                system = SYSTEM_PROMPT_PURE_LLM

            try:
                response = client.generate(system, prompt) or ""
            except Exception:
                response = ""

            artifact = _parse_response(response)
            latency  = int((time.time() - t0) * 1000)

            if artifact:
                risk_lvl = artifact.get("risk_level", "")
                correct  = risk_lvl == gt
                parse_ok = True
            else:
                risk_lvl = ""
                correct  = False
                parse_ok = False

            record = {
                "case_id":          cid,
                "paradigm":         paradigm,
                "model":            model,
                "true_class":       case["true_attack_class"],
                "true_nfcrm_class": case.get("true_nfcrm_class", ""),
                "ml_pred_class":    case.get("xgb_predicted_class", ""),
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


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_jsonl(path: Path, paradigm: str) -> dict:
    if not path.exists():
        return {}
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(lines)
    correct = sum(1 for l in lines if l.get("correct"))
    parse_errs = sum(1 for l in lines if l.get("parse_error"))
    grc = lines[0].get("grc", "?") if lines else "?"
    return {"paradigm": paradigm, "n": n,
            "accuracy": correct / n if n else 0,
            "grc": grc, "parse_errors": parse_errs}


def print_summary_table(out_dir: Path) -> None:
    PARADIGMS = [
        ("rule",               "rule.jsonl"),
        ("ml_rf",              "ml_rf.jsonl"),
        ("ml_xgb",             "ml_xgb.jsonl"),
        ("tristage_rf",        "tristage_rf.jsonl"),
        ("tristage_xgb",       "tristage_xgb.jsonl"),
        ("pure_llm",           "pure_llm.jsonl"),
        ("tristage_llm_haiku", "tristage_llm_haiku.jsonl"),
        ("tristage_llm_sonnet","tristage_llm_sonnet.jsonl"),
    ]
    print(f"\n{'Paradigm':<27} {'Accuracy':>10} {'GRC':>8} {'N':>5} {'ParseErr':>10}")
    print("-" * 65)
    for pname, fname in PARADIGMS:
        r = analyze_jsonl(out_dir / fname, pname)
        if r:
            print(f"  {pname:<25} {r['accuracy']:>9.1%} {r['grc']:>8} {r['n']:>5} {r['parse_errors']:>10}")
        else:
            print(f"  {pname:<25} {'—':>9}  {'—':>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

NON_LLM_PARADIGMS = ["rule", "ml_rf", "ml_xgb", "tristage_rf", "tristage_xgb"]
LLM_PARADIGMS     = ["pure_llm", "tristage_llm_haiku", "tristage_llm_sonnet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", nargs="+",
                    choices=NON_LLM_PARADIGMS + LLM_PARADIGMS)
    ap.add_argument("--all-nonllm", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--test-set", default="results/nslkdd_ablation/test_set.json")
    ap.add_argument("--out-dir", default="results/nslkdd_ablation")
    ap.add_argument("--model", default=None)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Override output JSONL basename (single paradigm only), "
                         "e.g. pure_llm_haiku.jsonl. Avoids clobbering existing files.")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_summary_table(out_dir)
        return

    test_set_path = ROOT / args.test_set
    if not test_set_path.exists():
        print(f"ERROR: test set not found at {test_set_path}")
        print("Run: python scripts/build_nslkdd_ablation_testset.py")
        sys.exit(1)

    all_cases = json.loads(test_set_path.read_text())["cases"]
    print(f"Loaded {len(all_cases)} NSL-KDD cases from {test_set_path}")

    to_run = list(args.paradigm or [])
    if args.all_nonllm:
        to_run = NON_LLM_PARADIGMS

    for paradigm in to_run:
        if args.out and len(to_run) == 1:
            out_path = out_dir / args.out
        else:
            out_path = out_dir / f"{paradigm}.jsonl"
        print(f"\n--- Running: {paradigm} (NSL-KDD) -> {out_path.name} ---")

        if paradigm in NON_LLM_PARADIGMS:
            results = run_nonllm_paradigm(all_cases, paradigm)
            mode = "a" if args.start_offset > 0 else "w"
            with out_path.open(mode, encoding="utf-8") as fh:
                for r in results[args.start_offset:]:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Saved to {out_path}")
        else:
            if args.model:
                model = args.model
            elif "haiku" in paradigm:
                model = "claude-haiku-4-5-20251001"
            else:
                model = "claude-sonnet-4-6"
            run_llm_paradigm(all_cases, paradigm, model, out_path, args.start_offset)

    if to_run:
        print_summary_table(out_dir)


if __name__ == "__main__":
    main()
