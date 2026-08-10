"""Wrapper: run the tristage_llm_sonnet arm on the canonical N=60 test subset.

Usage:
    python scripts/run_sonnet_multirun.py --out tristage_llm_sonnet_run2.jsonl
    python scripts/run_sonnet_multirun.py --out tristage_llm_sonnet_run3.jsonl
    python scripts/run_sonnet_multirun.py --out tristase_llm_sonnet_run4.jsonl

Reads the canonical N=60 case IDs from results/unified_ablation/ml_xgb_n60.jsonl
and runs only those 60 cases, saving to results/unified_ablation/<out>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_unified_ablation import run_llm_paradigm

PARADIGM = "tristage_llm_sonnet"
MODEL = "claude-sonnet-4-6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="Output JSONL filename (basename only, stored in results/unified_ablation/)")
    args = ap.parse_args()

    # Load full test set
    test_set_path = ROOT / "results/unified_ablation/test_set.json"
    all_cases = json.loads(test_set_path.read_text())["cases"]
    print(f"Loaded {len(all_cases)} total cases from test set")

    # Load canonical N=60 IDs from the existing n60 reference file
    ref_path = ROOT / "results/unified_ablation/ml_xgb_n60.jsonl"
    ref_records = [json.loads(l) for l in ref_path.read_text(encoding="utf-8").splitlines()
                   if l.strip() and "\x00" not in l]
    n60_ids = [r["case_id"] for r in ref_records]
    print(f"Canonical N=60 IDs loaded: {len(n60_ids)}")

    # Filter test set to N=60, preserving order from reference
    cases_by_id = {c["case_id"]: c for c in all_cases}
    n60_cases = [cases_by_id[cid] for cid in n60_ids if cid in cases_by_id]
    print(f"Cases to run: {len(n60_cases)}")

    # Verify class distribution
    from collections import Counter
    class_counts = Counter(c["true_attack_class"] for c in n60_cases)
    print(f"Class distribution: {dict(class_counts)}")

    out_dir = ROOT / "results/unified_ablation"
    out_path = out_dir / args.out
    print(f"Output: {out_path}")
    print(f"Model: {MODEL}\n")

    run_llm_paradigm(n60_cases, PARADIGM, MODEL, out_path, start_offset=0)

    # Compute and print accuracy
    results = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()
               if l.strip() and "\x00" not in l]
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    parse_errs = sum(1 for r in results if r.get("parse_error"))
    print(f"\n=== FINAL: {correct}/{n} = {correct/n:.1%} | parse_errors={parse_errs} ===")


if __name__ == "__main__":
    main()
