"""Summarize 3-run variance for the NSL-KDD reconciled tri-stage LLM arms.

Run 1 = original results (results/nslkdd_unified_rerun/reconciled_tristage_*_test.jsonl);
runs 2-3 = replicates under results/nslkdd_unified_rerun/replicates/.
Recomputes the five headline metrics per run from per-case records and emits
mean, sample sd, min, max per metric per arm (DKE round 5, item 6).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/nslkdd_unified_rerun"
REP = BASE / "replicates"
OUT = ROOT / "results/nslkdd_replicate_variance.json"

SEV = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(SEV)}
SEVERE = {"High", "Catastrophic"}

ARMS = {
    "tristage_haiku": [
        BASE / "reconciled_tristage_haiku_test.jsonl",
        REP / "reconciled_tristage_haiku_test_run2.jsonl",
        REP / "reconciled_tristage_haiku_test_run3.jsonl",
    ],
    "tristage_sonnet": [
        BASE / "reconciled_tristage_sonnet_test.jsonl",
        REP / "reconciled_tristage_sonnet_test_run2.jsonl",
        REP / "reconciled_tristage_sonnet_test_run3.jsonl",
    ],
}

METRICS = ["under_escalation_pct", "over_escalation_pct", "exact_pct",
           "severe_recall_pct", "parse_fail_pct"]


def run_metrics(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    under = over = exact = sev_tot = sev_hit = parse = 0
    for r in rows:
        p, g = r["risk_level"], r["ground_truth"]
        d = IDX.get(p, 0) - IDX.get(g, 0)
        if p not in IDX:
            under += 1
        elif d < 0:
            under += 1
        elif d > 0:
            over += 1
        else:
            exact += 1
        if g in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_hit += 1
        if r.get("infer_parse_error") or r.get("parse_error"):
            parse += 1
    return {"n": n,
            "under_escalation_pct": under / n * 100,
            "over_escalation_pct": over / n * 100,
            "exact_pct": exact / n * 100,
            "severe_recall_pct": sev_hit / sev_tot * 100,
            "parse_fail_pct": parse / n * 100}


def main() -> None:
    out = {"description": __doc__.strip(), "arms": {}}
    for arm, paths in ARMS.items():
        present = [p for p in paths if p.exists()]
        runs = [run_metrics(p) for p in present]
        summary = {}
        for m in METRICS:
            vals = [r[m] for r in runs]
            summary[m] = {
                "runs": vals,
                "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals), "max": max(vals),
            }
        out["arms"][arm] = {"n_runs": len(runs),
                            "files": [str(p.relative_to(ROOT)) for p in present],
                            "per_run": runs, "summary": summary}
        print(f"{arm}: {len(runs)} runs")
        for m in METRICS:
            s = summary[m]
            sd = f"{s['sd']:.2f}" if s["sd"] is not None else "n/a"
            print(f"  {m}: mean {s['mean']:.1f}  sd {sd}  runs {['%.1f' % v for v in s['runs']]}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
