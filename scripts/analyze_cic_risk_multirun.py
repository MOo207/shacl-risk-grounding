"""Multi-run statistics for the CIC risk ablation LLM arms.

Reads results/cic_risk_ablation/runs/run{1,2,3}/<arm>.jsonl and computes
mean ± SD across runs for each metric, plus Wilson CIs on SHACL conformance.

Usage:
  python -m scripts.analyze_cic_risk_multirun
"""
from __future__ import annotations
import json, math, statistics
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results/cic_risk_ablation/runs"
OUT    = ROOT / "results/cic_risk_ablation/multirun_summary.json"

LLM_ARMS = ["llm", "tristage_llm_shacl"]
ARM_LABELS = {"llm": "LLM", "tristage_llm_shacl": "Tri-LLM+SHACL"}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_run(run_dir: Path, arm: str) -> list[dict]:
    path = run_dir / f"{arm}.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def run_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    acc        = sum(r["correct"] for r in rows) / n
    grc        = sum(bool(r.get("grc_complete")) for r in rows) / n
    parse_errs = sum(r.get("parse_error", False) for r in rows)
    shacl_vals = [r["shacl_conform"] for r in rows if r.get("shacl_conform") is not None]
    shacl_rate = sum(shacl_vals)/len(shacl_vals) if shacl_vals else None
    shacl_n    = len(shacl_vals)
    shacl_k    = int(sum(shacl_vals)) if shacl_vals else 0
    retried    = sum(r.get("shacl_retried", 0) for r in rows)
    depths     = [r["grounding_depth"] for r in rows if r.get("grounding_depth") is not None]
    depth_mean = statistics.mean(depths) if depths else None
    consist    = [r["internal_consistent"] for r in rows if r.get("internal_consistent") is not None]
    consist_rate = sum(consist)/len(consist) if consist else None
    by_class   = {}
    for r in rows:
        cls = r.get("true_class", "?")
        by_class.setdefault(cls, []).append(r["correct"])
    per_class  = {cls: round(sum(v)/len(v), 4) for cls, v in sorted(by_class.items())}
    return {
        "n": n, "acc": round(acc, 4), "grc": round(grc, 4),
        "parse_errs": parse_errs,
        "shacl_rate": round(shacl_rate, 4) if shacl_rate is not None else None,
        "shacl_n": shacl_n, "shacl_k": shacl_k, "retried": retried,
        "depth_mean": round(depth_mean, 3) if depth_mean is not None else None,
        "depths": depths,
        "consist_rate": round(consist_rate, 4) if consist_rate is not None else None,
        "per_class": per_class,
    }


def aggregate(run_data: list[dict]) -> dict:
    """Mean ± SD across runs for scalar metrics."""
    if not run_data:
        return {}
    keys = ["acc", "grc", "shacl_rate", "depth_mean", "consist_rate"]
    result = {}
    for k in keys:
        vals = [r[k] for r in run_data if r.get(k) is not None]
        if not vals:
            result[k] = {"mean": None, "sd": None, "runs": []}
            continue
        mean = statistics.mean(vals)
        sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
        result[k] = {"mean": round(mean, 4), "sd": round(sd, 4), "runs": [round(v, 4) for v in vals]}
    # Pooled SHACL Wilson CI across all runs
    total_k = sum(r.get("shacl_k", 0) for r in run_data)
    total_n = sum(r.get("shacl_n", 0) for r in run_data)
    if total_n > 0:
        lo, hi = wilson_ci(total_k, total_n)
        result["shacl_pooled_wilson_95"] = {
            "k": total_k, "n": total_n,
            "rate": round(total_k/total_n, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        }
    # Parse errors and retries
    result["total_parse_errs"] = sum(r.get("parse_errs", 0) for r in run_data)
    result["total_retried"]    = sum(r.get("retried", 0) for r in run_data)
    # Pooled grounding depth SD
    all_depths = []
    for r in run_data:
        all_depths.extend(r.get("depths", []))
    if all_depths:
        result["depth_pooled"] = {
            "n": len(all_depths),
            "mean": round(statistics.mean(all_depths), 3),
            "sd":   round(statistics.stdev(all_depths) if len(all_depths) > 1 else 0.0, 3),
        }
    # Consistency across runs: did per-class ordering stay the same?
    run_accs = [r.get("acc") for r in run_data if r.get("acc") is not None]
    result["n_runs"] = len(run_data)
    return result


def main() -> None:
    run_dirs = sorted(RUNS_DIR.glob("run*"))
    print(f"Found {len(run_dirs)} runs: {[d.name for d in run_dirs]}\n")

    summary: dict = {"runs": [d.name for d in run_dirs], "arms": {}}

    for arm in LLM_ARMS:
        label = ARM_LABELS[arm]
        per_run: list[dict] = []
        print(f"=== {label} ===")
        for run_dir in run_dirs:
            rows = load_run(run_dir, arm)
            if not rows:
                print(f"  {run_dir.name}: NOT FOUND")
                continue
            m = run_metrics(rows)
            per_run.append(m)
            shacl_str = f"shacl={m['shacl_rate']:.1%}" if m.get("shacl_rate") is not None else "shacl=N/A"
            depth_str = f"depth={m['depth_mean']:.2f}" if m.get("depth_mean") is not None else ""
            print(f"  {run_dir.name}: acc={m['acc']:.1%}  grc={m['grc']:.1%}  "
                  f"{shacl_str}  retried={m['retried']}  parse_err={m['parse_errs']}  {depth_str}")

        if not per_run:
            continue
        agg = aggregate(per_run)
        summary["arms"][label] = {"per_run": per_run, "aggregate": agg}

        print(f"\n  AGGREGATE ({agg['n_runs']} runs):")
        print(f"    risk_level_accuracy : {agg['acc']['mean']:.1%} ± {agg['acc']['sd']:.1%}  "
              f"runs={agg['acc']['runs']}")
        print(f"    grc_completeness    : {agg['grc']['mean']:.1%} ± {agg['grc']['sd']:.1%}")
        if agg.get("shacl_pooled_wilson_95"):
            pw = agg["shacl_pooled_wilson_95"]
            print(f"    SHACL conform (pooled): {pw['k']}/{pw['n']} = {pw['rate']:.1%}  "
                  f"Wilson 95% CI [{pw['ci_lo']:.1%}, {pw['ci_hi']:.1%}]")
        if agg.get("depth_pooled"):
            dp = agg["depth_pooled"]
            print(f"    grounding_depth     : {dp['mean']:.2f} ± {dp['sd']:.2f}  (N={dp['n']})")
        if agg.get("consist_rate", {}).get("mean") is not None:
            cr = agg["consist_rate"]
            print(f"    internal_consistent : {cr['mean']:.1%} ± {cr['sd']:.1%}")
        print(f"    total_parse_errs    : {agg['total_parse_errs']}")
        print(f"    total_retried       : {agg['total_retried']}")
        print()

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
