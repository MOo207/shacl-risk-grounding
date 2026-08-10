"""Analyze multi-run reproducibility results for the tristage_llm_sonnet arm.

Computes:
- Per-run accuracy
- Mean, std, range
- Flipped cases (inconsistent across runs)
- Saves results/t5_multirun_results.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
UA_DIR = ROOT / "results" / "unified_ablation"

# The N=60 canonical case_id set
def get_n60_ids():
    ts = json.loads((ROOT / "results/unified_ablation/test_set.json").read_text(encoding="utf-8"))
    all_cases = ts["cases"]
    by_cls = {}
    for c in all_cases:
        by_cls.setdefault(c["true_attack_class"], []).append(c)
    ids = set()
    for cls in sorted(by_cls.keys()):
        for c in by_cls[cls][:10]:
            ids.add(c["case_id"])
    return ids


def parse_run(path: Path, n60_ids: set) -> dict:
    """Return {case_id: correct} for N=60 subset."""
    if not path.exists():
        return {}
    raw = path.read_bytes().decode("utf-8-sig")
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    # Filter to N=60 canonical subset
    result = {}
    for r in records:
        cid = r["case_id"]
        if cid in n60_ids:
            result[cid] = r.get("correct", False)
    return result


def main():
    n60_ids = get_n60_ids()
    print(f"N=60 canonical subset: {len(n60_ids)} case_ids")

    run_files = [
        ("run1", UA_DIR / "tristage_llm_sonnet.jsonl"),
        ("run2", UA_DIR / "tristage_llm_sonnet_run2.jsonl"),
        ("run3", UA_DIR / "tristage_llm_sonnet_run3.jsonl"),
        ("run4", UA_DIR / "tristage_llm_sonnet_run4.jsonl"),
    ]

    run_results = {}
    run_accuracies = []

    for run_name, path in run_files:
        data = parse_run(path, n60_ids)
        if not data:
            print(f"  {run_name}: NOT FOUND or empty ({path.name})")
            continue
        n = len(data)
        correct = sum(1 for v in data.values() if v)
        acc = correct / n if n else 0
        run_results[run_name] = data
        run_accuracies.append(acc)
        print(f"  {run_name}: {correct}/{n} = {acc:.1%} (file: {path.name})")

    if len(run_accuracies) < 2:
        print("\nNot enough completed runs for analysis.")
        return

    mean_acc = statistics.mean(run_accuracies)
    std_acc = statistics.stdev(run_accuracies) if len(run_accuracies) > 1 else 0.0
    range_acc = max(run_accuracies) - min(run_accuracies)

    print(f"\nMean: {mean_acc:.1%}")
    print(f"Std:  {std_acc:.1%}")
    print(f"Range: {range_acc:.1%} (max={max(run_accuracies):.1%}, min={min(run_accuracies):.1%})")

    # Find flipped cases (correct in some runs, wrong in others)
    all_runs = list(run_results.keys())
    all_ids = n60_ids
    flipped = []
    for cid in sorted(all_ids):
        outcomes = [run_results[r].get(cid) for r in all_runs if cid in run_results.get(r, {})]
        if outcomes and not all(o == outcomes[0] for o in outcomes):
            flipped.append(cid)

    print(f"\nFlipped cases (inconsistent across runs): {len(flipped)}")
    for cid in flipped:
        outcomes_str = " ".join(
            f"{r}={'OK' if run_results.get(r, {}).get(cid) else 'MISS'}"
            for r in all_runs if r in run_results
        )
        print(f"  {cid}: {outcomes_str}")

    # Save summary
    out = {
        "run_accuracies": run_accuracies,
        "run_names": [rn for rn, _ in run_files if rn in run_results],
        "n_runs_completed": len(run_accuracies),
        "n_cases_per_run": 60,
        "mean": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "range": round(range_acc, 4),
        "max": round(max(run_accuracies), 4),
        "min": round(min(run_accuracies), 4),
        "flipped_cases": flipped,
        "n_flipped": len(flipped),
    }
    out_path = ROOT / "results" / "t5_multirun_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
