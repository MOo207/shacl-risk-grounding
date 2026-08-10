"""Natural-distribution under-escalation rate for NSL-KDD XGBoost (Task 13).

The thesis repeatedly asserts (abstract + 3 body sites) that the dramatic
44% under-escalation / 45% severe-risk-recall figures measured on the
N=50 *balanced* NSL-KDD sample (10 cases per class, i.e. minority attack
classes R2L/U2R artificially over-represented at 20% each instead of their
~1-12% natural share) would be "substantially lower on the natural test
distribution" — without ever computing that number. This script computes it.

Method
------
1. Source: results/nslkdd_xgb_baseline.json -> test split confusion_matrix,
   the FULL KDDTest+ (n=22,184) 5-class XGBoost confusion matrix (the same
   matrix used for the 75.43% accuracy figure, Task 12).

2. §6.9 class -> risk-level mapping: reproduces EXACTLY the lookup used to
   produce the N=50 ml_xgb.jsonl figures in scripts/run_nslkdd_ablation.py
   (rule_lookup() / NSLKDD_TO_NFCRM / CRITICALITY_MAP), calling the same
   pipeline.nfcrm.risk_score.compute_risk_score() used there. The resulting
   5x4 (class x criticality) table is byte-identical to the MANDATORY
   §6.9 table hard-coded into the NSL-KDD LLM prompts in that script:

      NSL-KDD class | Low      | Medium | High         | Critical
      --------------|----------|--------|--------------|----------
      Normal        | Very Low | Very Low | Very Low   | Very Low
      DoS           | Medium   | High   | Catastrophic | Catastrophic
      Probe         | Low      | Medium | Medium       | High
      R2L           | Medium   | High   | Catastrophic | Catastrophic
      U2R           | Medium   | High   | Catastrophic | Catastrophic

3. Criticality-invariance property (verified below, not assumed): for
   every one of the 4 criticality columns, the ordinal ranking of the 5
   classes by risk level is IDENTICAL:
        Normal (rank 0)  <  Probe (rank 1)  <  {DoS, R2L, U2R} (rank 2, a
        3-way tie — those rows are literally identical in the table).
   Because a natural-distribution flow's predicted and true class are both
   looked up against the SAME (unknown, per-flow) asset criticality, the
   sign of (predicted_level - true_level) for any confusion-matrix cell
   (true=i, pred=j) is therefore the same for ALL FOUR criticality values
   — under-escalation iff rank(j) < rank(i). This lets us compute an
   exact, criticality-independent under-escalation rate directly from the
   class-level confusion matrix, with no need to assume/impute a specific
   per-flow criticality value (which the raw NSL-KDD flows do not carry).
   This is the same class-only comparison the N=50 balanced-sample metric
   effectively reduces to per case (predicted §6.9 level vs ground-truth
   §6.9 level, both computed against that one case's asset criticality).

4. Under-escalation rate, natural distribution: sum, over all confusion-
   matrix cells (true=i, pred=j) with rank(j) < rank(i), of the raw cell
   COUNT (i.e. weighted by the true natural frequency of class i in the
   full 22,184-flow KDDTest+ set — no balancing/resampling), divided by
   the total N. This mirrors the N=50 metric's definition (fraction of
   cases where predicted risk level < ground-truth risk level) but applied
   to the natural (unbalanced) test distribution instead of the
   10-per-class balanced sample.

Caveats (carried from the risk-level construct disclosure elsewhere in the
thesis, Task 8 Step 7): this is a class-through-lookup computation, not a
re-run of the full tri-stage/LLM pipeline on all 22,184 flows; per-flow
asset criticality is not available for the raw NSL-KDD test set, so the
computation relies on the proven criticality-invariance of the §6.9
under/over-escalation *direction* (Step 3) rather than assuming one
specific criticality value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.nfcrm.risk_score import compute_risk_score  # noqa: E402

CRITICALITY_MAP = {"Low": 2, "Medium": 3, "High": 4, "Critical": 5}
NSLKDD_TO_NFCRM = {
    "Normal": "Benign",
    "DoS": "DoS",
    "Probe": "Infiltration",
    "R2L": "WebAttack",
    "U2R": "BruteForce",
}
LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
LEVEL_IDX = {l: i for i, l in enumerate(LEVELS)}


def rule_lookup(nfcrm_class: str, criticality: str) -> str:
    """Identical to run_nslkdd_ablation.py:rule_lookup — same §6.9 call."""
    c_val = CRITICALITY_MAP.get(criticality, 3)
    score = compute_risk_score(nfcrm_class, c_override=c_val, i_override=c_val, a_override=c_val)
    return "Very Low" if score.level_en == "N/A (non-attack)" else score.level_en


def build_table() -> dict[str, dict[str, str]]:
    table = {}
    for cls, nfcrm_cls in NSLKDD_TO_NFCRM.items():
        table[cls] = {crit: rule_lookup(nfcrm_cls, crit) for crit in CRITICALITY_MAP}
    return table


def verify_criticality_invariant_ranks(table: dict[str, dict[str, str]]) -> dict[str, int]:
    """Verify the class ordinal rank (by risk level) is identical across all
    4 criticality columns; return the (single, shared) rank per class.
    Raises if the invariance does not hold (methodology would then be invalid)."""
    classes = list(table.keys())
    crits = list(CRITICALITY_MAP.keys())
    per_crit_ranks = []
    for crit in crits:
        levels = {c: LEVEL_IDX[table[c][crit]] for c in classes}
        # dense rank: classes with equal level share the same rank
        sorted_levels = sorted(set(levels.values()))
        rank_of_level = {lvl: r for r, lvl in enumerate(sorted_levels)}
        per_crit_ranks.append({c: rank_of_level[levels[c]] for c in classes})
    first = per_crit_ranks[0]
    for other in per_crit_ranks[1:]:
        if other != first:
            raise RuntimeError(
                "Criticality invariance FAILED — under-escalation direction "
                "depends on criticality; natural-distribution rate cannot be "
                "computed without per-flow criticality data."
            )
    return first


def main() -> None:
    baseline_path = ROOT / "results" / "nslkdd_xgb_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    label_classes = baseline["label_classes"]  # confusion-matrix row/col order
    cm = baseline["test"]["confusion_matrix"]
    n_total = baseline["test"]["samples"]
    accuracy = baseline["test"]["accuracy"]

    assert sum(sum(row) for row in cm) == n_total, "confusion matrix does not sum to reported N"

    table = build_table()
    ranks = verify_criticality_invariant_ranks(table)

    under_count = 0
    over_count = 0
    exact_count = 0
    per_true_class = {}
    for i, true_cls in enumerate(label_classes):
        row = cm[i]
        row_total = sum(row)
        row_under = row_over = row_exact = 0
        for j, pred_cls in enumerate(label_classes):
            n_cell = row[j]
            if n_cell == 0:
                continue
            if ranks[pred_cls] < ranks[true_cls]:
                under_count += n_cell
                row_under += n_cell
            elif ranks[pred_cls] > ranks[true_cls]:
                over_count += n_cell
                row_over += n_cell
            else:
                exact_count += n_cell
                row_exact += n_cell
        per_true_class[true_cls] = {
            "n": row_total,
            "share_of_natural_distribution": row_total / n_total,
            "under_escalation_count": row_under,
            "under_escalation_rate_within_class": (row_under / row_total) if row_total else 0.0,
        }

    natural_under_rate = under_count / n_total
    natural_over_rate = over_count / n_total
    natural_exact_rate = exact_count / n_total

    # Balanced N=50 comparator (from results/nslkdd_ablation/ml_xgb.jsonl via
    # scripts/safety_table_haiku.py "Pure ML (XGBoost)" row) — value carried
    # as a fixed reference, NOT recomputed here (out of scope for this script;
    # verify against safety_table_haiku.json / thesis prose before citing).
    balanced_n50_reference = {
        "under_escalation_pct": 44.0,
        "severe_risk_recall_pct": 45.0,
        "n": 50,
        "note": (
            "10 cases per class (Normal/DoS/Probe/R2L/U2R), i.e. minority "
            "classes R2L and U2R each artificially held at 20% of the "
            "sample vs. their natural share of ~12.4% and ~0.9% respectively "
            "(see per_true_class below). Source: results/nslkdd_ablation/"
            "ml_xgb.jsonl via scripts/safety_table_haiku.py."
        ),
    }

    out = {
        "description": (
            "Natural-distribution (unbalanced, full KDDTest+ n=22,184) "
            "XGBoost under-escalation rate under the NFCRM-1:2025 §6.9 "
            "class-through-lookup, computed to quantify the qualitative "
            "'substantially lower on the natural test distribution' claim "
            "made elsewhere in the thesis relative to the N=50 balanced-"
            "sample figure (44% under-escalation)."
        ),
        "source_confusion_matrix": "results/nslkdd_xgb_baseline.json (test split)",
        "source_n50_balanced_figure": "results/nslkdd_ablation/ml_xgb.jsonl, results/safety_table_haiku.json",
        "label_classes": label_classes,
        "n_total": n_total,
        "xgb_test_accuracy": accuracy,
        "sixnine_lookup_table": table,
        "class_ranks_verified_criticality_invariant": ranks,
        "methodology": (
            "For each confusion-matrix cell (true class i, predicted class j), "
            "the case is classed as under-escalated iff the §6.9 risk level of "
            "class j is strictly lower than that of class i, HOLDING CRITICALITY "
            "FIXED (identical for both lookups, since only the predicted class "
            "differs from the true class for a given flow/asset). This ordinal "
            "comparison was verified to be identical across all 4 §6.9 "
            "criticality columns (Low/Medium/High/Critical) — see "
            "class_ranks_verified_criticality_invariant — so no specific "
            "per-flow criticality value needs to be assumed or imputed; the "
            "full natural KDDTest+ set does not carry per-flow asset-"
            "criticality metadata (unlike the N=50 ARG-derived ablation "
            "sample). Rate = (sum of confusion-matrix cell counts with "
            "predicted-class rank < true-class rank) / n_total, i.e. weighted "
            "by each class's actual natural frequency in KDDTest+, not a "
            "balanced/resampled distribution."
        ),
        "natural_distribution_under_escalation_rate": natural_under_rate,
        "natural_distribution_under_escalation_pct": round(natural_under_rate * 100, 1),
        "natural_distribution_over_escalation_pct": round(natural_over_rate * 100, 1),
        "natural_distribution_exact_pct": round(natural_exact_rate * 100, 1),
        "under_escalation_count": under_count,
        "over_escalation_count": over_count,
        "exact_count": exact_count,
        "per_true_class": per_true_class,
        "balanced_n50_reference": balanced_n50_reference,
        "caveats": [
            "This is a class-through-lookup computation on the XGBoost "
            "confusion matrix, not a re-run of the tri-stage/LLM pipeline "
            "on all 22,184 flows.",
            "Per-flow asset criticality is not available for the raw "
            "KDDTest+ set; the computation relies on the verified "
            "criticality-invariance of the §6.9 under/over-escalation "
            "DIRECTION (see class_ranks_verified_criticality_invariant) "
            "rather than assuming one specific criticality value.",
            "The §6.9 lookup used is the NSL-KDD-analog table (Normal-> "
            "Benign, DoS->DoS, Probe->Infiltration, R2L->WebAttack, "
            "U2R->BruteForce) exactly as used to produce the N=50 balanced "
            "-sample figures in scripts/run_nslkdd_ablation.py — same "
            "caveats as the rest of the risk-level construct disclosure.",
            "XGBoost only (matches the natural-distribution accuracy claim "
            "this number is attached to); not computed for the tri-stage "
            "or LLM paradigms, which were never run on the full natural "
            "KDDTest+ set.",
        ],
    }

    out_path = ROOT / "results" / "nslkdd_natural_underescalation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Natural-distribution under-escalation rate: {out['natural_distribution_under_escalation_pct']}% "
          f"(n={n_total}), vs balanced N=50 figure 44.0%")
    print(f"Saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
