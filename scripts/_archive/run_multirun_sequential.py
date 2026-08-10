"""Sequential runner: runs 3 and 4 after checking run 2 is done.

Usage:
  python scripts/run_multirun_sequential.py --runs 3 4
  python scripts/run_multirun_sequential.py --runs 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_unified_ablation import run_llm_paradigm

TEST_SET = ROOT / "results/unified_ablation/test_set.json"
OUT_DIR  = ROOT / "results/unified_ablation"
N_PER_CLASS = 10


def get_n60_cases():
    all_cases = json.loads(TEST_SET.read_text(encoding="utf-8"))["cases"]
    by_cls = {}
    for c in all_cases:
        by_cls.setdefault(c["true_attack_class"], []).append(c)
    cases = []
    for cls in sorted(by_cls.keys()):
        cases.extend(by_cls[cls][:N_PER_CLASS])
    return cases


def run_one(run_num: int, model: str = "claude-sonnet-4-6"):
    cases = get_n60_cases()
    out_path = OUT_DIR / f"tristage_llm_sonnet_run{run_num}.jsonl"

    # Check if already complete
    if out_path.exists():
        raw = out_path.read_bytes().decode("utf-8-sig")
        records = [json.loads(l) for l in raw.splitlines() if l.strip()]
        if len(records) >= 60:
            correct = sum(1 for r in records if r.get("correct"))
            print(f"Run {run_num} already complete: {correct}/60 = {correct/60:.1%}")
            return correct / 60

    print(f"\n=== Starting Run {run_num} === {len(cases)} cases -> {out_path}")
    run_llm_paradigm(cases, "tristage_llm_sonnet", model, out_path, start_offset=0)

    # Read results
    raw = out_path.read_bytes().decode("utf-8-sig")
    records = [json.loads(l) for l in raw.splitlines() if l.strip()]
    n60_ids = {c["case_id"] for c in cases}
    n60_recs = [r for r in records if r["case_id"] in n60_ids]
    correct = sum(1 for r in n60_recs if r.get("correct"))
    n = len(n60_recs)
    print(f"\nRun {run_num} RESULT: {correct}/{n} = {correct/n:.1%}")
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=int, required=True,
                    help="Run numbers to execute (e.g. 3 4)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    for run_num in sorted(args.runs):
        acc = run_one(run_num, args.model)
        print(f"  Run {run_num}: {acc:.1%}")


if __name__ == "__main__":
    main()
