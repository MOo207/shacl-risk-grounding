"""Reproduce and persist the CIC cross-dataset sym-GT cells in
Table `tab:cross-dataset-tristage` (thesis).

This script does NOT reimplement sym-GT scoring: it reuses the exact
symbolic engine (`symbolic_risk`, `cve_info`, `load_test_set`) from
`scripts/run_cic_risk_ablation.py` -- the same function that produced
`results/cic_risk_ablation/summary_symgt.json` (via
`scripts/analyze_cic_risk_symgt.py`).

Two cells are re-derived here, both over CIC LLM-arm runs 1-3
(`results/cic_risk_ablation/runs/run{1,2,3}/llm.jsonl`, N=60 each,
identical 60 case_ids in identical order across runs -> pooled N=180):

1. "LLM (pure)" cell: sym-GT accuracy of the pure-LLM arm's OWN
   self-reported `risk_level` (the LLM decided the risk level itself;
   see `run_llm_arm()` in run_cic_risk_ablation.py, which sets
   risk_level = art["risk_level"] directly, with NO symbolic engine
   post-processing). This is exactly what `analyze_cic_risk_symgt.py`
   computes per single run (0.7667 for run1's top-level llm.jsonl);
   here we pool it across all 3 runs.

2. "Tri-stage (LLM-class)" CIC cell (printed as 71.7%, 129/180): the
   thesis text (line ~2331) describes this as "the symbolic-engine
   output applied to the pure-LLM arm's measured per-run class
   inferences" -- i.e. take the LLM's inferred class
   (`attack_class_used`, NOT its self-reported risk_level), re-run it
   through the SAME deterministic `symbolic_risk()` engine used to
   build sym-GT (with that case's true criticality/CVSS/KEV/n_controls),
   and compare the re-engineered risk level to sym-GT (which is built
   from the TRUE attack class through the same engine). This is a
   different quantity from (1): it isolates the LLM's classification
   accuracy under the deterministic engine, independent of the LLM's
   own (potentially miscalibrated) risk judgement.

Sym-GT definition (identical to analyze_cic_risk_symgt.py):
    sym_gt_i = symbolic_risk(true_attack_class_i, criticality_i, cvss_i,
                              in_kev_i, n_controls_i)["risk_level"]
An arm's case is "sym-correct" if its measured/re-derived risk_level
equals sym_gt_i.

Usage:
  python -m scripts.compute_cic_crossdataset_symgt
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.run_cic_risk_ablation import symbolic_risk, cve_info, load_test_set

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results/cic_risk_ablation/runs"
OUT = ROOT / "results/cic_risk_ablation/crossdataset_symgt.json"

RUN_NAMES = ["run1", "run2", "run3"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials.

    Identical formula to scripts/analyze_cic_risk_multirun.py::wilson_ci
    (the convention already used for other pooled-CI cells in this
    thesis section), reused verbatim for consistency.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_case_index(cases: list[dict]) -> dict[str, dict]:
    """Map case_id -> {criticality, cvss, in_kev, n_controls, true_attack_class}."""
    idx: dict[str, dict] = {}
    for c in cases:
        cvss, in_kev, _ = cve_info(c)
        crit = c.get("criticality", c.get("asset", {}).get("criticality", "Medium"))
        idx[c["case_id"]] = {
            "criticality":       crit,
            "cvss":              cvss,
            "in_kev":            in_kev,
            "n_controls":        len(c.get("controls", [])),
            "true_attack_class": c["true_attack_class"],
        }
    return idx


def build_symgt(case_idx: dict[str, dict]) -> dict[str, str]:
    """sym_gt_i = symbolic_risk(true_attack_class_i, ...)['risk_level']."""
    gt: dict[str, str] = {}
    for case_id, c in case_idx.items():
        sym = symbolic_risk(c["true_attack_class"], c["criticality"], c["cvss"],
                             c["in_kev"], c["n_controls"])
        gt[case_id] = sym["risk_level"]
    return gt


def main() -> None:
    cases = load_test_set(10)
    case_idx = build_case_index(cases)
    symgt = build_symgt(case_idx)
    print(f"Sym-GT built for {len(symgt)} cases (from load_test_set(10)).")

    per_run_pure: list[dict] = []
    per_run_reengine: list[dict] = []
    pooled_pure_correct = 0
    pooled_pure_n = 0
    pooled_re_correct = 0
    pooled_re_n = 0

    for run in RUN_NAMES:
        rows = load_jsonl(RUNS_DIR / run / "llm.jsonl")
        if not rows:
            print(f"WARNING: no rows for {run}, skipping")
            continue
        n = len(rows)

        # (1) LLM (pure): sym-GT accuracy of the LLM's OWN self-reported risk_level.
        pure_correct = sum(
            1 for r in rows if symgt.get(r["case_id"]) == r.get("risk_level")
        )
        per_run_pure.append({"run": run, "n": n, "correct": pure_correct,
                              "acc": round(pure_correct / n, 4)})
        pooled_pure_correct += pure_correct
        pooled_pure_n += n

        # (2) Tri-stage (LLM-class), CIC cell: re-run the LLM's inferred class
        # (attack_class_used) through the SAME symbolic_risk() engine used to
        # build sym-GT, then compare to sym-GT.
        re_correct = 0
        for r in rows:
            case_id = r["case_id"]
            c = case_idx.get(case_id)
            if c is None:
                continue
            inferred_class = r.get("attack_class_used", "")
            sym = symbolic_risk(inferred_class, c["criticality"], c["cvss"],
                                 c["in_kev"], c["n_controls"])
            if sym["risk_level"] == symgt.get(case_id):
                re_correct += 1
        per_run_reengine.append({"run": run, "n": n, "correct": re_correct,
                                  "acc": round(re_correct / n, 4)})
        pooled_re_correct += re_correct
        pooled_re_n += n

    pure_lo, pure_hi = wilson_ci(pooled_pure_correct, pooled_pure_n)
    re_lo, re_hi = wilson_ci(pooled_re_correct, pooled_re_n)

    # Attempt to reproduce the printed "71.7% (129/180)" figure exactly.
    target_k, target_n = 129, 180
    reproduces_71_7 = (pooled_re_correct == target_k and pooled_re_n == target_n)
    # Also check whether the "pure" pooling accidentally matches (sanity check
    # for a possible mislabelled derivation).
    pure_matches_71_7 = (pooled_pure_correct == target_k and pooled_pure_n == target_n)

    result = {
        "description": (
            "Reproduction of the two CIC cross-dataset sym-GT cells in "
            "tab:cross-dataset-tristage, pooled over CIC LLM-arm runs 1-3 "
            "(results/cic_risk_ablation/runs/run{1,2,3}/llm.jsonl, N=60 each, "
            "identical case_id set/order per run -> pooled N=180). Scoring "
            "reuses symbolic_risk()/cve_info()/load_test_set() from "
            "scripts/run_cic_risk_ablation.py verbatim (same function that "
            "produced results/cic_risk_ablation/summary_symgt.json)."
        ),
        "sym_gt_formula": (
            "sym_gt_i = symbolic_risk(true_attack_class_i, criticality_i, cvss_i, "
            "in_kev_i, n_controls_i)['risk_level']; a row is sym-correct if its "
            "risk_level (case 1) or re-derived risk_level (case 2) equals sym_gt_i."
        ),
        "cell_1_llm_pure": {
            "label": "LLM (pure) -- CIC column, tab:cross-dataset-tristage",
            "formula": (
                "sym-GT accuracy of the pure-LLM arm's OWN self-reported "
                "risk_level field (run_llm_arm() sets risk_level = LLM's "
                "artifact['risk_level'] directly, no symbolic post-processing), "
                "pooled across runs 1-3."
            ),
            "per_run": per_run_pure,
            "pooled_n": pooled_pure_n,
            "pooled_correct": pooled_pure_correct,
            "pooled_acc": round(pooled_pure_correct / pooled_pure_n, 4) if pooled_pure_n else None,
            "wilson_ci_95": [round(pure_lo, 4), round(pure_hi, 4)],
        },
        "cell_2_tristage_llm_class": {
            "label": "Tri-stage (LLM-class) -- CIC column, tab:cross-dataset-tristage",
            "formula": (
                "The LLM arm's inferred class (attack_class_used) is re-run "
                "through symbolic_risk() (the same deterministic engine used to "
                "build sym-GT) using that case's true criticality/CVSS/KEV/"
                "n_controls, and the resulting risk_level is compared to sym-GT. "
                "This differs from cell_1 in that it discards the LLM's own "
                "self-reported risk_level and instead re-derives risk "
                "deterministically from the LLM's classification only."
            ),
            "per_run": per_run_reengine,
            "pooled_n": pooled_re_n,
            "pooled_correct": pooled_re_correct,
            "pooled_acc": round(pooled_re_correct / pooled_re_n, 4) if pooled_re_n else None,
            "wilson_ci_95": [round(re_lo, 4), round(re_hi, 4)],
        },
        "reproduction_of_printed_71_7_pct": {
            "printed_value": "71.7% (129/180)",
            "cell_2_reengine_matches_exactly": reproduces_71_7,
            "cell_1_pure_matches_exactly": pure_matches_71_7,
            "outcome": (
                "MATCHED by cell_2 (re-engineered LLM-class derivation)"
                if reproduces_71_7 else
                "MATCHED by cell_1 (pure self-reported derivation)"
                if pure_matches_71_7 else
                "NOT reproduced exactly by either derivation; see per-run "
                "breakdowns above -- closest derivation's pooled_acc/pooled_correct "
                "should be used as the honest replacement value in the thesis table."
            ),
        },
    }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {OUT}\n")

    print("Cell 1 (LLM pure, self-reported risk_level):")
    for r in per_run_pure:
        print(f"  {r['run']}: {r['correct']}/{r['n']} = {r['acc']:.1%}")
    print(f"  Pooled: {pooled_pure_correct}/{pooled_pure_n} = "
          f"{pooled_pure_correct/pooled_pure_n:.1%} "
          f"[{pure_lo:.1%},{pure_hi:.1%}]")

    print("\nCell 2 (Tri-stage LLM-class, re-engineered from attack_class_used):")
    for r in per_run_reengine:
        print(f"  {r['run']}: {r['correct']}/{r['n']} = {r['acc']:.1%}")
    print(f"  Pooled: {pooled_re_correct}/{pooled_re_n} = "
          f"{pooled_re_correct/pooled_re_n:.1%} "
          f"[{re_lo:.1%},{re_hi:.1%}]")

    print(f"\n71.7% (129/180) reproduction: {result['reproduction_of_printed_71_7_pct']['outcome']}")


if __name__ == "__main__":
    main()
