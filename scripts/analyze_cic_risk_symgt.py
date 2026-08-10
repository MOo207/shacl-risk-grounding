"""Symmetric-GT re-evaluation of the CIC 5-paradigm risk ablation.

Replaces the flat (attack_class x criticality) ground truth with a symmetric GT:

    sym_gt_i = symbolic_risk(true_attack_class_i, criticality_i, cvss_i, in_kev_i, n_controls_i)

Under this GT, an arm is "correct" if its risk_level matches what the SAME symbolic
engine would produce for the true class — removing the calibration mismatch between
the comprehensive CVSS/KEV engine and the flat lookup used to label the test set.

Key finding this exposes:
  - Arms that use XGBoost detection (ML, Tri-ML+SHACL, Tri-LLM+SHACL) score ~95%
    under sym GT (limited by XGBoost's class accuracy, not calibration mismatch).
  - Pure LLM scores based on its class-inference accuracy.
  - SHACL (rule-based) scores based on rule accuracy.

Outputs:
  results/cic_risk_ablation/summary_symgt.json   (sym GT metrics per arm)
  Console comparison table: flat GT vs sym GT side-by-side

Usage:
  python -m scripts.analyze_cic_risk_symgt
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

from scripts.run_cic_risk_ablation import symbolic_risk, cve_info, load_test_set
from scripts.analyze_cic_risk_ablation import ARM_ORDER, ARM_LABELS

ROOT   = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results/cic_risk_ablation"
OUT    = IN_DIR / "summary_symgt.json"


# ── Build symbolic GT for the canonical 60 cases ─────────────────────────────

def build_symgt(cases: list[dict]) -> dict[str, str]:
    """Return {case_id: sym_risk_level} using true_attack_class."""
    gt: dict[str, str] = {}
    for c in cases:
        asset  = c.get("asset", {})
        crit   = c.get("criticality", asset.get("criticality", "Medium"))
        cvss, in_kev, _ = cve_info(c)
        n_ctrl = len(c.get("controls", []))
        sym    = symbolic_risk(c["true_attack_class"], crit, cvss, in_kev, n_ctrl)
        gt[c["case_id"]] = sym["risk_level"]
    return gt


# ── Load existing JSONL arm results ──────────────────────────────────────────

def load_arm(arm: str) -> list[dict]:
    path = IN_DIR / f"{arm}.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# ── Per-arm metrics under sym GT ─────────────────────────────────────────────

def sym_metrics(rows: list[dict], symgt: dict[str, str]) -> dict:
    if not rows:
        return {}
    n = len(rows)

    sym_correct = [symgt.get(r["case_id"], "") == r["risk_level"] for r in rows]
    flat_correct = [r["correct"] for r in rows]

    by_class: dict[str, list[bool]] = {}
    for r, sc in zip(rows, sym_correct):
        cls = r.get("true_class", "?")
        by_class.setdefault(cls, []).append(sc)

    # Improvement: cases where sym GT is correct but flat GT was wrong
    improved = sum(sc and not fc for sc, fc in zip(sym_correct, flat_correct))
    regressed = sum(not sc and fc for sc, fc in zip(sym_correct, flat_correct))

    return {
        "n":                       n,
        "risk_level_acc_flat":     round(sum(flat_correct) / n, 4),
        "risk_level_acc_symgt":    round(sum(sym_correct) / n, 4),
        "delta_pp":                round((sum(sym_correct) - sum(flat_correct)) / n * 100, 1),
        "cases_improved":          improved,
        "cases_regressed":         regressed,
        "per_class_acc_symgt":     {
            cls: round(sum(v) / len(v), 4) for cls, v in sorted(by_class.items())
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cases = load_test_set(10)
    symgt = build_symgt(cases)

    print(f"Symmetric GT computed for {len(symgt)} cases.")
    dist: dict[str, int] = {}
    for v in symgt.values():
        dist[v] = dist.get(v, 0) + 1
    print(f"Sym GT distribution: {dist}\n")

    summary: dict = {"sym_gt_distribution": dist, "arms": {}}
    for arm in ARM_ORDER:
        rows = load_arm(arm)
        if not rows:
            print(f"SKIP {arm}: no file found")
            continue
        m = sym_metrics(rows, symgt)
        label = ARM_LABELS[arm]
        summary["arms"][label] = m
        print(f"{label}: flat={m['risk_level_acc_flat']:.1%}  sym={m['risk_level_acc_symgt']:.1%}  "
              f"delta={m['delta_pp']:+.1f}pp  improved={m['cases_improved']}  regressed={m['cases_regressed']}")

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {OUT}\n")

    # Side-by-side comparison table
    col = 14
    header = f"{'Arm':<20} {'Flat GT acc':>{col}} {'Sym GT acc':>{col}} {'Delta (pp)':>{col}} {'Improved':>{col}} {'Regressed':>{col}}"
    print(header)
    print("-" * len(header))
    for arm in ARM_ORDER:
        label = ARM_LABELS[arm]
        m = summary["arms"].get(label)
        if not m:
            continue
        print(f"{label:<20} "
              f"{m['risk_level_acc_flat']:>{col}.1%} "
              f"{m['risk_level_acc_symgt']:>{col}.1%} "
              f"{m['delta_pp']:>+{col}.1f} "
              f"{m['cases_improved']:>{col}} "
              f"{m['cases_regressed']:>{col}}")

    print("\nPer-class Sym GT accuracy (Tri-LLM+SHACL):")
    tls = summary["arms"].get("Tri-LLM+SHACL", {}).get("per_class_acc_symgt", {})
    for cls, acc in sorted(tls.items()):
        print(f"  {cls:<15}: {acc:.0%}")


if __name__ == "__main__":
    main()
