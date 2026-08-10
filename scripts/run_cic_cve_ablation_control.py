"""CIC-IDS2018 CVE-ablation control: tri-stage Haiku with ALL CVE context removed.

CIC counterpart of the NSL-KDD CVE-ablation control
(scripts/run_nslkdd_permuted_cve_control.py's ablation arm). Pre-registered
interpretation rule, fixed in advance of the run: if the ablated arm's
under-escalation rate is approximately equal to the clean run's
(results/cic_unified_rerun/tristage_haiku_clean_results.json), the CIC result
is channel-independent; if it degrades materially, the class-paired CVE
channel is load-bearing on CIC as it is on NSL-KDD, and the paper must say so.

Identical to scripts/run_cic_tristage_haiku_clean.py in every respect --
same leak-free NL descriptions, same prompt template, same override
verification, same Sec 6.9 lookup, same scoring and McNemar pairing --
EXCEPT the CVE block is replaced by "No CVE context." for every case.
Reviewer-requested control (DKE round 5, item 4).

Usage:
    python scripts/run_cic_cve_ablation_control.py            # full 120-case run
    python scripts/run_cic_cve_ablation_control.py --limit 3  # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.run_unified_ablation import (  # noqa: E402
    SYSTEM_PROMPT_TRISTAGE, PROMPT_TRISTAGE, ATTACK_TO_CONTROL, rule_lookup,
)
from scripts.run_cic_tristage_haiku_clean import (  # noqa: E402
    clean_features_to_nl, _parse_response, compute_metrics, mcnemar,
    grc_completeness, DATA_PATH, TEST_SAMPLE_PATH, XGB_BASELINE_PATH, MODEL,
)

OUT_DIR = ROOT / "results" / "cic_unified_rerun"
OUT_JSONL = OUT_DIR / "tristage_haiku_cve_ablated_raw.jsonl"
OUT_SUMMARY = OUT_DIR / "tristage_haiku_cve_ablated_results.json"

CVE_ABLATED_BLOCK = "No CVE context."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=180)
    args = ap.parse_args()

    test_out = json.loads(TEST_SAMPLE_PATH.read_text(encoding="utf-8"))
    cases = test_out["cases"]
    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases from {TEST_SAMPLE_PATH} (CVE-ABLATED arm)")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    from scripts.claude_cli_client import ClaudeCLIClient
    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=args.timeout_sec)

    mode = "a" if args.start_offset > 0 else "w"
    api_errors = parse_failures = 0

    with OUT_JSONL.open(mode, encoding="utf-8") as f:
        for i, case in enumerate(cases[args.start_offset:], start=args.start_offset + 1):
            t0 = time.time()
            cid = case["case_id"]
            crit = case["criticality"]
            gt = case["ground_truth_risk_level"]
            true_cls = case["true_attack_class"]

            row = df.loc[case["source_row_index"]]
            nl_desc_clean = clean_features_to_nl(row)

            ml_cls = case["xgb_predicted_class"]
            ml_conf = case.get("xgb_confidence", 1.0)
            asset_j = json.dumps(case.get("asset", {}))

            prompt = PROMPT_TRISTAGE.format(
                attack_class=ml_cls, confidence=f"{ml_conf:.0%}",
                criticality=crit, asset_json=asset_j, cve_block=CVE_ABLATED_BLOCK,
                nl_desc=nl_desc_clean[:300],
            )

            try:
                response = client.generate(SYSTEM_PROMPT_TRISTAGE, prompt) or ""
            except Exception:
                response = ""
            if not response:
                api_errors += 1

            artifact = _parse_response(response)
            latency_ms = int((time.time() - t0) * 1000)

            if artifact:
                verified_cls = artifact.get("verified_attack_class", "")
                if verified_cls and verified_cls in ATTACK_TO_CONTROL and verified_cls != ml_cls:
                    risk_lvl = rule_lookup(verified_cls, crit)
                else:
                    risk_lvl = artifact.get("risk_level", "")
                parse_ok = True
            else:
                parse_failures += 1
                risk_lvl = case.get("xgb_risk_level", "")
                verified_cls = ml_cls
                parse_ok = False

            record = {
                "case_id": cid,
                "model": MODEL,
                "arm": "tristage_haiku_cve_ablated",
                "true_class": true_cls,
                "ml_pred_class": ml_cls,
                "llm_verified_cls": verified_cls,
                "risk_level": risk_lvl,
                "xgb_risk_level": case.get("xgb_risk_level", ""),
                "ground_truth": gt,
                "correct": risk_lvl == gt,
                "xgb_correct": case.get("xgb_risk_level", "") == gt,
                "parse_error": not parse_ok,
                "criticality": crit,
                "nl_description_clean": nl_desc_clean,
                "cve_block": CVE_ABLATED_BLOCK,
                "artifact": artifact,
                "raw_response": response[:2000],
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            match = "OK" if record["correct"] else "MISS"
            print(f"  [{i}/{len(cases)}] {cid}: {risk_lvl or 'PARSE_ERR'} vs GT={gt} [{match}] ({latency_ms}ms)")

    all_records = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    metrics = compute_metrics(all_records)
    mcn = mcnemar(all_records)
    grc = grc_completeness(all_records)
    xgb_baseline = json.loads(XGB_BASELINE_PATH.read_text(encoding="utf-8"))

    summary = {
        "meta": {
            "description": (
                "CIC CVE-ablation control: identical to tristage_haiku_clean "
                "(leak-free NL, same prompts/override/lookup/scoring) except the "
                "CVE block is 'No CVE context.' for every case. Pre-registered "
                "interpretation rule in module docstring."
            ),
            "model": MODEL,
            "n_cases": len(all_records),
            "api_errors": api_errors,
            "parse_failures": parse_failures,
            "clean_run_reference": "results/cic_unified_rerun/tristage_haiku_clean_results.json",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "metrics": metrics,
        "mcnemar_vs_clean_xgboost": mcn,
        "grc_completeness": grc,
        "clean_xgboost_baseline_reference": xgb_baseline["held_out_test_sample_metrics"],
        "records": all_records,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY (CVE-ABLATED) ===")
    print(json.dumps({"metrics": metrics, "mcnemar": mcn,
                      "api_errors": api_errors, "parse_failures": parse_failures}, indent=2))
    print(f"\nSaved: {OUT_JSONL}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
