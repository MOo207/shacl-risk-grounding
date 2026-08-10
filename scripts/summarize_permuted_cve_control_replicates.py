"""3-run variance summary for the permuted-/ablated-CVE control (headline
safety-effect-survives-CVE-ablation result). Companion to
scripts/summarize_nslkdd_replicates.py, which already covers the two primary
tri-stage arms; this closes the same gap for the CVE-ablation arms, which
previously had n=2 replicates only (run3 added to match convention).

Run 1: results/nslkdd_permuted_cve_control.json
Run 2: results/nslkdd_permuted_cve_control_run2.json
Run 3: results/nslkdd_permuted_cve_control_run3.json
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "results/nslkdd_permuted_cve_control.json",
    ROOT / "results/nslkdd_permuted_cve_control_run2.json",
    ROOT / "results/nslkdd_permuted_cve_control_run3.json",
]
OUT = ROOT / "results/nslkdd_permuted_cve_control_variance.json"

ARMS = ["no_cve", "permuted"]
METRICS_RECONCILED = ["exact_pct", "under_escalation_pct", "over_escalation_pct", "severe_recall_pct"]


def main() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in FILES if p.exists()]
    present = [p for p in FILES if p.exists()]
    out = {"description": __doc__.strip(), "n_runs": len(runs),
           "files": [str(p.relative_to(ROOT)) for p in present], "arms": {}}

    for arm in ARMS:
        if not all(arm in r["arms"] for r in runs):
            continue
        arm_out = {"reconciled": {}, "mcnemar_vs_xgb": {}}
        for m in METRICS_RECONCILED:
            vals = [r["arms"][arm]["reconciled"][m] for r in runs]
            arm_out["reconciled"][m] = {
                "runs": vals, "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals), "max": max(vals),
            }
        for m in ("b", "c", "p_one_sided"):
            vals = [r["arms"][arm]["reconciled_mcnemar_vs_xgb"][m] for r in runs]
            arm_out["mcnemar_vs_xgb"][m] = {
                "runs": vals, "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else None,
                "min": min(vals), "max": max(vals),
            }
        out["arms"][arm] = arm_out
        print(f"{arm}: {len(runs)} runs")
        for m in METRICS_RECONCILED:
            s = arm_out["reconciled"][m]
            sd = f"{s['sd']:.2f}" if s["sd"] is not None else "n/a"
            print(f"  reconciled.{m}: mean {s['mean']:.1f}  sd {sd}  runs {s['runs']}")
        for m in ("b", "c", "p_one_sided"):
            s = arm_out["mcnemar_vs_xgb"][m]
            print(f"  mcnemar.{m}: runs {s['runs']}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
