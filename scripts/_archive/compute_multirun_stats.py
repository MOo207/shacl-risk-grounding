"""Compute multi-run reproducibility statistics for tristage_llm_sonnet.

Reads run1 (canonical) + run2/3/4 JSONL files, restricts to the N=60
canonical subset, and saves results/t5_multirun_results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
ABLATION_DIR = ROOT / "results/unified_ablation"

# Canonical N=60 IDs from the reference file
REF_PATH = ABLATION_DIR / "ml_xgb_n60.jsonl"
ref_records = [json.loads(l) for l in REF_PATH.read_text(encoding="utf-8").splitlines()
               if l.strip() and "\x00" not in l]
N60_IDS = [r["case_id"] for r in ref_records]
N60_ID_SET = set(N60_IDS)

RUN_FILES = [
    ("run1", ABLATION_DIR / "tristage_llm_sonnet.jsonl"),
    ("run2", ABLATION_DIR / "tristage_llm_sonnet_run2.jsonl"),
    ("run3", ABLATION_DIR / "tristage_llm_sonnet_run3.jsonl"),
    ("run4", ABLATION_DIR / "tristage_llm_sonnet_run4.jsonl"),
]


def load_run(path: Path) -> dict[str, bool]:
    """Load a run's results. Returns {case_id: correct} for N=60 cases only."""
    if not path.exists():
        return {}
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
               if l.strip() and "\x00" not in l]
    return {r["case_id"]: r.get("correct", False)
            for r in records if r["case_id"] in N60_ID_SET}


def main():
    runs: list[tuple[str, dict[str, bool]]] = []
    for name, path in RUN_FILES:
        data = load_run(path)
        if not data:
            print(f"  {name}: MISSING or empty ({path.name})")
        else:
            n = len(data)
            correct = sum(data.values())
            acc = correct / n
            print(f"  {name}: {correct}/{n} = {acc:.1%}")
            runs.append((name, data))

    if len(runs) < 2:
        print("Need at least 2 completed runs to compute stats.")
        return

    accuracies = [sum(d.values()) / len(d) for _, d in runs]
    mean_acc = sum(accuracies) / len(accuracies)
    variance = sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)
    std_acc = math.sqrt(variance)
    range_acc = max(accuracies) - min(accuracies)

    # Find cases that flipped (correct in some runs, wrong in others)
    all_ids = N60_IDS
    flipped = []
    case_detail = {}
    for cid in all_ids:
        results_for_case = [d.get(cid) for _, d in runs if cid in d]
        if len(results_for_case) < 2:
            continue
        if any(results_for_case) and not all(results_for_case):
            flipped.append(cid)
            case_detail[cid] = {name: d.get(cid) for name, d in runs}

    print(f"\nRuns completed: {len(runs)}")
    print(f"Accuracies: {[round(a, 4) for a in accuracies]}")
    print(f"Mean: {mean_acc:.4f} ({mean_acc:.1%})")
    print(f"Std:  {std_acc:.4f}")
    print(f"Range: {range_acc:.4f}")
    print(f"Flipped cases: {flipped}")
    if case_detail:
        for cid, detail in case_detail.items():
            print(f"  {cid}: {detail}")

    output = {
        "runs": [name for name, _ in runs],
        "run_accuracies": [round(a, 6) for a in accuracies],
        "mean": round(mean_acc, 6),
        "std": round(std_acc, 6),
        "range": round(range_acc, 6),
        "flipped_cases": flipped,
        "flipped_case_detail": case_detail,
        "n_cases_per_run": [len(d) for _, d in runs],
    }

    out_path = ROOT / "results/t5_multirun_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
