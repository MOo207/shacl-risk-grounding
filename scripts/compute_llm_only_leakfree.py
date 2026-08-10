"""Compute STANDALONE-LLM (Stage-1a independent inference) metrics from the
reconciled tri-stage per-case records (leak-free reruns), plus tri-stage
escalation-precision diagnostics.

No new model calls: every number is recomputed from per-case records already
stored in:
  results/nslkdd_unified_rerun/reconciled_tristage_haiku_results.json
  results/nslkdd_unified_rerun/reconciled_tristage_sonnet_results.json
  results/cic_unified_rerun/reconciled_tristage_haiku_results.json
  results/cic_unified_rerun/reconciled_tristage_sonnet_results.json

Each record stores three independent risk levels:
  llm_risk_level  -- Stage-1a standalone LLM inference (no ML class shown)
  ml_risk_level   -- XGBoost baseline (Stage 1b)
  risk_level      -- final reconciled (escalate-on-disagreement) level

Validation: before reporting LLM-only numbers, this script recomputes the
published tri-stage and XGBoost aggregates from the SAME records with the SAME
metric definitions and asserts they match the stored aggregates.

Metric definitions (identical to run_nslkdd_reconciled_tristage_haiku.py):
  ladder Very Low < Low < Medium < High < Catastrophic
  exact            : pred == GT
  under-escalation : pred < GT (or pred unparseable)
  over-escalation  : pred > GT
  severe recall    : GT in {High, Catastrophic} and pred in {High, Catastrophic}
  unsafe(pred, gt) : pred < gt or pred unparseable
  McNemar (1-sided): b = XGB unsafe & other-arm safe; c = other-arm unsafe &
                     XGB safe; p = P(X <= c | X~Bin(b+c, 0.5))

Usage:
    python -m scripts.compute_llm_only_leakfree
Output -> results/llm_only_leakfree.json
"""
from __future__ import annotations

import json
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "results/llm_only_leakfree.json"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

SOURCES = {
    "nslkdd_haiku": ROOT / "results/nslkdd_unified_rerun/reconciled_tristage_haiku_results.json",
    "nslkdd_sonnet": ROOT / "results/nslkdd_unified_rerun/reconciled_tristage_sonnet_results.json",
    "cic_haiku": ROOT / "results/cic_unified_rerun/reconciled_tristage_haiku_results.json",
    "cic_sonnet": ROOT / "results/cic_unified_rerun/reconciled_tristage_sonnet_results.json",
}


def metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(pred, gt), ...]; verbatim logic from
    run_nslkdd_reconciled_tristage_haiku.py::metrics()."""
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for p, g in pairs:
        n += 1
        if p not in IDX or g not in IDX:
            if g in SEVERE:
                sev_tot += 1
            under += 1
            worst = max(worst, IDX.get(g, 0))
            continue
        d = IDX[p] - IDX[g]
        if d == 0:
            exact += 1
        if abs(d) <= 1:
            within1 += 1
        if d < 0:
            under += 1
            worst = max(worst, -d)
        elif d > 0:
            over += 1
        if g in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_rec += 1
    pct = lambda a, b: round(a / b * 100, 1) if b else 0.0
    return dict(n=n, exact_pct=pct(exact, n), within1_pct=pct(within1, n),
                under_escalation_pct=pct(under, n), over_escalation_pct=pct(over, n),
                severe_recall_pct=pct(sev_rec, sev_tot), severe_total=sev_tot,
                worst_miss_levels=worst,
                n_exact=exact, n_under=under, n_over=over, n_severe_hit=sev_rec)


def unsafe(pred: str, gt: str) -> bool:
    if pred not in IDX or gt not in IDX:
        return True
    return IDX[pred] < IDX[gt]


def mcnemar_1s(b: int, c: int) -> float:
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) * (0.5 ** n) if n else 1.0


def paired_mcnemar_unsafe(records: list[dict], arm_field: str) -> dict:
    """Directional McNemar on the unsafe indicator: arm (arm_field) vs XGBoost
    (ml_risk_level), same-case pairing. b = XGB unsafe & arm safe;
    c = arm unsafe & XGB safe. Same semantics as
    run_nslkdd_reconciled_tristage_haiku.py::paired_mcnemar()."""
    b = c = 0
    for r in records:
        gt = r["ground_truth"]
        arm_unsafe = unsafe(r.get(arm_field, ""), gt)
        xgb_unsafe = unsafe(r.get("ml_risk_level", ""), gt)
        if xgb_unsafe and not arm_unsafe:
            b += 1
        elif arm_unsafe and not xgb_unsafe:
            c += 1
    return {"b_xgb_unsafe_arm_safe": b, "c_arm_unsafe_xgb_safe": c,
            "p_1sided": mcnemar_1s(b, c)}


def get_records(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "held_out_test" in data:            # NSL-KDD layout
        ht = data["held_out_test"]
        return ht["records"], {
            "tristage": ht.get("reconciled_tristage_llm_haiku_metrics")
                        or ht.get("reconciled_tristage_llm_sonnet_metrics")
                        or ht.get("reconciled_tristage_metrics"),
            "xgb": ht.get("xgboost_clean_baseline_metrics"),
            "mcnemar_tristage_vs_xgb": ht.get("unsafe_mcnemar_reconciled_vs_xgb"),
        }
    # CIC layout
    return data["records"], {
        "tristage": data.get("metrics"),
        "xgb": data.get("clean_xgboost_baseline_reference"),
        "mcnemar_tristage_vs_xgb": data.get("mcnemar_vs_clean_xgboost"),
    }


def validate(label: str, computed: dict, published: dict | None) -> dict:
    """Compare recomputed aggregate against stored aggregate on shared keys
    (tolerating key-name and rounding differences)."""
    if not published:
        return {"validated": False, "reason": "no stored aggregate found"}
    alias = {"exact_pct": ["exact_pct", "exact_match_pct", "exact"],
             "under_escalation_pct": ["under_escalation_pct"],
             "over_escalation_pct": ["over_escalation_pct"],
             "severe_recall_pct": ["severe_recall_pct"],
             "n": ["n"]}
    mismatches = []
    for ck, pkeys in alias.items():
        pv = next((published[k] for k in pkeys if k in published), None)
        if pv is None:
            continue
        if abs(float(computed[ck]) - float(pv)) > 0.11:  # rounding tolerance
            mismatches.append({"key": ck, "computed": computed[ck], "stored": pv})
    return {"validated": not mismatches, "mismatches": mismatches}


def escalation_precision(records: list[dict]) -> dict:
    """TASK 2: escalations = cases where reconciled risk_level > ml_risk_level."""
    esc = [r for r in records
           if r.get("risk_level") in IDX and r.get("ml_risk_level") in IDX
           and IDX[r["risk_level"]] > IDX[r["ml_risk_level"]]]
    n = len(esc)
    gt_severe = [r for r in esc if r["ground_truth"] in SEVERE]
    exact = [r for r in esc if r["risk_level"] == r["ground_truth"]]
    over = [r for r in esc if r["ground_truth"] in IDX
            and IDX[r["risk_level"]] > IDX[r["ground_truth"]]]
    over_err = [IDX[r["risk_level"]] - IDX[r["ground_truth"]] for r in over]
    pct = lambda a, b: round(a / b * 100, 1) if b else None
    return {
        "n_records": len(records),
        "n_escalations_issued": n,
        "n_gt_severe": len(gt_severe),
        "frac_gt_severe_pct": pct(len(gt_severe), n),
        "n_reconciled_eq_gt": len(exact),
        "frac_reconciled_eq_gt_pct": pct(len(exact), n),
        "n_over_escalated": len(over),
        "frac_over_escalated_pct": pct(len(over), n),
        "mean_level_error_over_escalated": (round(sum(over_err) / len(over_err), 3)
                                            if over_err else None),
        "escalated_case_ids": [r["case_id"] for r in esc],
    }


def main() -> None:
    out = {"meta": {
        "description": ("Standalone-LLM (Stage-1a independent inference) metrics "
                        "recomputed from stored per-case records of the leak-free "
                        "reconciled tri-stage reruns (no new model calls), plus "
                        "tri-stage escalation-precision diagnostics. LLM-only "
                        "prediction = record field 'llm_risk_level'; XGBoost = "
                        "'ml_risk_level'; reconciled tri-stage = 'risk_level'."),
        "metric_definitions": ("identical to scripts/run_nslkdd_reconciled_tristage_"
                               "haiku.py (ladder Very Low<Low<Medium<High<Catastrophic; "
                               "unsafe = pred<GT or unparseable; McNemar 1-sided binomial, "
                               "b = XGB unsafe & arm safe, c = arm unsafe & XGB safe)"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/compute_llm_only_leakfree.py",
    }, "arms": {}}

    for label, path in SOURCES.items():
        if not path.exists():
            out["arms"][label] = {"error": f"missing file {path}"}
            continue
        records, published = get_records(path)

        # -- validation: reproduce published tri-stage + XGB aggregates --------
        tri_m = metrics([(r.get("risk_level", ""), r["ground_truth"]) for r in records])
        xgb_m = metrics([(r.get("ml_risk_level", ""), r["ground_truth"]) for r in records])
        tri_val = validate(label, tri_m, published["tristage"])
        xgb_val = validate(label, xgb_m, published["xgb"])

        # -- TASK 1: standalone LLM ------------------------------------------
        llm_m = metrics([(r.get("llm_risk_level", ""), r["ground_truth"]) for r in records])
        llm_vs_xgb = paired_mcnemar_unsafe(records, "llm_risk_level")
        tri_vs_xgb = paired_mcnemar_unsafe(records, "risk_level")

        arm = {
            "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "n": len(records),
            "validation": {
                "tristage_recomputed": tri_m, "tristage_check": tri_val,
                "xgb_recomputed": xgb_m, "xgb_check": xgb_val,
                "stored_mcnemar_tristage_vs_xgb": published["mcnemar_tristage_vs_xgb"],
                "recomputed_mcnemar_unsafe_tristage_vs_xgb": tri_vs_xgb,
            },
            "llm_only_metrics": llm_m,
            "n_llm_stage1a_unparseable": sum(
                1 for r in records if r.get("llm_risk_level", "") not in IDX),
            "llm_unparseable_caveat": (
                "Records whose Stage-1a llm_risk_level is empty/unparseable are "
                "counted as under-escalation AND unsafe (identical to the tri-stage "
                "scripts' handling of parse failures). Where this count is large "
                "the LLM-only under-escalation figure is dominated by parse "
                "failures, not by substantive risk under-calls."),
            "mcnemar_unsafe_llm_only_vs_xgb": llm_vs_xgb,
            "mcnemar_note": (
                "This McNemar is on the UNSAFE indicator (pred<GT or unparseable), "
                "matching the NSL-KDD scripts' spine-test semantics. The CIC result "
                "files' stored mcnemar_vs_clean_xgboost is a DIFFERENT test "
                "(correct/incorrect accuracy-based), so it is not directly "
                "comparable to the recomputed unsafe-based value for CIC arms."),
        }

        # -- TASK 2: escalation precision (tri-stage) ------------------------
        arm["tristage_escalation_precision"] = escalation_precision(records)
        out["arms"][label] = arm

        print(f"[{label}] n={len(records)} "
              f"tri_ok={tri_val['validated']} xgb_ok={xgb_val['validated']} "
              f"llm_only exact={llm_m['exact_pct']} under={llm_m['under_escalation_pct']} "
              f"over={llm_m['over_escalation_pct']} sevrec={llm_m['severe_recall_pct']} "
              f"| McNemar llm_vs_xgb b={llm_vs_xgb['b_xgb_unsafe_arm_safe']} "
              f"c={llm_vs_xgb['c_arm_unsafe_xgb_safe']} p={llm_vs_xgb['p_1sided']:.4g}")

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
