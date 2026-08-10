"""Metrics + multi-run stats for NSL-KDD 5-paradigm risk ablation.

Usage:
  python -m scripts.analyze_nslkdd_risk_ablation          # single-run summary
  python -m scripts.analyze_nslkdd_risk_ablation --multirun  # 3-run aggregate
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

from scripts.run_cic_risk_ablation import symbolic_risk, cve_info, load_test_set as _cic_load

ROOT     = Path(__file__).resolve().parents[1]
IN_DIR   = ROOT / "results/nslkdd_risk_ablation"
OUT      = IN_DIR / "summary.json"
RUNS_DIR = IN_DIR / "runs"

ARM_ORDER  = ["ml", "shacl", "llm", "tristage_ml_shacl", "tristage_llm_shacl"]
ARM_LABELS = {
    "ml":                 "ML",
    "shacl":              "SHACL",
    "llm":                "LLM",
    "tristage_ml_shacl":  "Tri-ML+SHACL",
    "tristage_llm_shacl": "Tri-LLM+SHACL",
}

NSL_TO_NFCRM = {
    "Normal": "Benign", "DoS": "DoS",
    "Probe": "Infiltration", "U2R": "BruteForce", "R2L": "WebAttack",
}


def build_symgt() -> dict[str, str]:
    """Compute symbolic GT for each NSL-KDD case using true_nfcrm_class."""
    ts = json.loads((ROOT / "results/nslkdd_ablation/test_set.json").read_text())["cases"]
    gt: dict[str, str] = {}
    for c in ts:
        asset  = c.get("asset", {})
        crit   = c.get("criticality", asset.get("criticality", "Medium"))
        cvss, in_kev, _ = cve_info(c)
        n_ctrl = len(c.get("controls") or [])
        sym    = symbolic_risk(c["true_nfcrm_class"], crit, cvss, in_kev, n_ctrl)
        gt[c["case_id"]] = sym["risk_level"]
    return gt


def load_arm(arm: str, run_dir: Path = IN_DIR) -> list[dict]:
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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def arm_metrics(rows: list[dict], symgt: dict[str, str]) -> dict:
    if not rows:
        return {}
    n   = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    sym_correct = [symgt.get(r["case_id"], "") == r["risk_level"] for r in rows]
    sym_acc = sum(sym_correct) / n

    shacl_vals = [r["shacl_conform"] for r in rows if r.get("shacl_conform") is not None]
    shacl_rate = (sum(shacl_vals) / len(shacl_vals)) if shacl_vals else None
    shacl_k    = int(sum(shacl_vals)) if shacl_vals else 0

    grc   = sum(bool(r.get("grc_complete")) for r in rows) / n
    depths = [r["grounding_depth"] for r in rows if r.get("grounding_depth") is not None]
    depth = statistics.mean(depths) if depths else None
    parse_errs = sum(r.get("parse_error", False) for r in rows)
    retried    = sum(r.get("shacl_retried", 0) for r in rows)

    by_class: dict[str, list[bool]] = {}
    for r in rows:
        cls = r.get("true_class", "?")
        by_class.setdefault(cls, []).append(r["correct"])

    return {
        "n": n,
        "risk_level_accuracy":  round(acc, 4),
        "sym_gt_accuracy":      round(sym_acc, 4),
        "shacl_conform_rate":   round(shacl_rate, 4) if shacl_rate is not None else None,
        "shacl_k": shacl_k, "shacl_n": len(shacl_vals),
        "grc_completeness":     round(grc, 4),
        "grounding_depth":      round(depth, 2) if depth is not None else None,
        "depths": depths,
        "parse_errors": parse_errs, "retried": retried,
        "per_class_accuracy": {
            cls: round(sum(v)/len(v), 4) for cls, v in sorted(by_class.items())
        },
    }


def print_table(summary: dict, symgt_dist: dict) -> None:
    cols = ["risk_level_accuracy", "sym_gt_accuracy", "shacl_conform_rate",
            "grc_completeness", "grounding_depth"]
    col_w = 20
    print(f"\nSym GT distribution: {symgt_dist}")
    print()
    header = f"{'Arm':<20}" + "".join(f"{c:<{col_w}}" for c in cols)
    print(header)
    print("-" * len(header))
    for arm in ARM_ORDER:
        label  = ARM_LABELS[arm]
        m_dict = summary.get(label, {})
        row    = f"{label:<20}"
        for c in cols:
            v = m_dict.get(c)
            cell = f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))
            row += f"{cell:<{col_w}}"
        print(row)


def multirun_stats(symgt: dict[str, str]) -> dict:
    run_dirs = sorted(RUNS_DIR.glob("run*")) if RUNS_DIR.exists() else []
    if not run_dirs:
        return {}
    result: dict = {"runs": [d.name for d in run_dirs], "arms": {}}
    for arm in ["llm", "tristage_llm_shacl"]:
        label = ARM_LABELS[arm]
        per_run = []
        for d in run_dirs:
            rows = load_arm(arm, d)
            if rows:
                per_run.append(arm_metrics(rows, symgt))
        if not per_run:
            continue
        acc_vals   = [m["risk_level_accuracy"] for m in per_run]
        sym_vals   = [m["sym_gt_accuracy"] for m in per_run]
        grc_vals   = [m["grc_completeness"] for m in per_run]
        depth_vals = [m["grounding_depth"] for m in per_run if m.get("grounding_depth") is not None]
        tot_k      = sum(m["shacl_k"] for m in per_run)
        tot_n      = sum(m["shacl_n"] for m in per_run)
        all_depths = []
        for m in per_run:
            all_depths.extend(m.get("depths", []))

        def ms(vals: list) -> dict:
            mean = statistics.mean(vals)
            sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return {"mean": round(mean, 4), "sd": round(sd, 4), "runs": [round(v,4) for v in vals]}

        agg: dict = {
            "n_runs": len(per_run),
            "risk_level_accuracy": ms(acc_vals),
            "sym_gt_accuracy":     ms(sym_vals),
            "grc_completeness":    ms(grc_vals),
        }
        if tot_n > 0:
            lo, hi = wilson_ci(tot_k, tot_n)
            agg["shacl_pooled"] = {
                "k": tot_k, "n": tot_n, "rate": round(tot_k/tot_n, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            }
        if all_depths:
            agg["depth_pooled"] = {
                "n": len(all_depths),
                "mean": round(statistics.mean(all_depths), 3),
                "sd":   round(statistics.stdev(all_depths) if len(all_depths) > 1 else 0.0, 3),
            }
        agg["total_parse_errs"] = sum(m["parse_errors"] for m in per_run)
        agg["total_retried"]    = sum(m["retried"] for m in per_run)
        result["arms"][label] = {"per_run": per_run, "aggregate": agg}

        print(f"\n  {label} MULTI-RUN ({agg['n_runs']} runs):")
        print(f"    risk_level_accuracy: {agg['risk_level_accuracy']['mean']:.1%} +/-{agg['risk_level_accuracy']['sd']:.1%}  runs={agg['risk_level_accuracy']['runs']}")
        print(f"    sym_gt_accuracy:     {agg['sym_gt_accuracy']['mean']:.1%} +/-{agg['sym_gt_accuracy']['sd']:.1%}")
        print(f"    grc_completeness:    {agg['grc_completeness']['mean']:.1%} +/-{agg['grc_completeness']['sd']:.1%}")
        if agg.get("shacl_pooled"):
            pw = agg["shacl_pooled"]
            print(f"    SHACL pooled:        {pw['k']}/{pw['n']} = {pw['rate']:.1%}  CI [{pw['ci_lo']:.1%},{pw['ci_hi']:.1%}]")
        if agg.get("depth_pooled"):
            dp = agg["depth_pooled"]
            print(f"    grounding_depth:     {dp['mean']:.2f} +/-{dp['sd']:.2f} (N={dp['n']})")
        print(f"    total_parse_errs:    {agg['total_parse_errs']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multirun", action="store_true")
    args = ap.parse_args()

    symgt = build_symgt()
    dist: dict[str, int] = {}
    for v in symgt.values():
        dist[v] = dist.get(v, 0) + 1

    summary: dict = {"arms": {}}
    for arm in ARM_ORDER:
        rows = load_arm(arm)
        if rows:
            label = ARM_LABELS[arm]
            summary["arms"][label] = arm_metrics(rows, symgt)
            print(f"Loaded {arm}: {len(rows)} records")
        else:
            print(f"SKIP {arm}: no file")

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {OUT}")
    print_table(summary["arms"], dist)

    if args.multirun:
        print("\n=== MULTI-RUN ANALYSIS ===")
        mr = multirun_stats(symgt)
        if mr:
            out_mr = IN_DIR / "multirun_summary.json"
            out_mr.write_text(json.dumps(mr, indent=2, ensure_ascii=False))
            print(f"\nSaved multi-run -> {out_mr}")


if __name__ == "__main__":
    main()
