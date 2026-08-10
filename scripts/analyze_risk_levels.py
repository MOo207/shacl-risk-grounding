"""Risk-level evaluation for the unified ablation (Task ii, canonical N=60).

Reframes the empirical result from attack-class accuracy to NFCRM 6.9 RISK-LEVEL
correctness: per-paradigm risk-level confusion matrices, per-level recall, and a
GRC-meaningful under-escalation (dangerous) vs over-escalation (false-alarm) split.

Output: results/risk_level_analysis.json  (+ console summary)
"""
import json
import os
import collections

UA = os.path.join(os.path.dirname(__file__), "..", "results", "unified_ablation")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "risk_level_analysis.json")

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
ORD = {l: i for i, l in enumerate(LEVELS)}

PARADIGMS = [
    ("Pure Rule", "rule_n60.jsonl", "pred"),
    ("Pure ML (RF)", "ml_rf_n60.jsonl", "pred"),
    ("Pure ML (XGBoost)", "ml_xgb_n60.jsonl", "pred"),
    ("Tri-stage RF", "tristage_rf_n60.jsonl", "pred"),
    ("Tri-stage XGBoost", "tristage_xgb_n60.jsonl", "pred"),
    ("Pure LLM (Haiku)", "pure_llm_haiku.jsonl", "pred"),       # genuine Haiku (fixes mislabel)
    ("Pure LLM (Sonnet)", "pure_llm.jsonl", "pred"),            # Sonnet reference
    ("Tri-stage LLM (Haiku)", "tristage_llm_haiku.jsonl", "llm"),
    ("Tri-stage LLM (Sonnet)", "tristage_llm_sonnet.jsonl", "llm"),
]


def load(fname):
    rows = []
    path = os.path.join(UA, fname)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def canonical_ids():
    return [r["case_id"] for r in load("ml_xgb_n60.jsonl")]


def restrict(rows, ids):
    """Return {case_id: record} for canonical ids, taking the LAST occurrence."""
    by = {}
    idset = set(ids)
    for r in rows:
        if r["case_id"] in idset:
            by[r["case_id"]] = r
    return by


def main():
    ids = canonical_ids()
    gt = {r["case_id"]: r["ground_truth"] for r in load("ml_xgb_n60.jsonl")}
    gt_dist = collections.Counter(gt[i] for i in ids)

    results = {
        "n": len(ids),
        "levels": LEVELS,
        "ground_truth_distribution": dict(gt_dist),
        "paradigms": {},
    }

    print(f"Canonical N = {len(ids)} | ground-truth risk-level distribution: {dict(gt_dist)}\n")
    header = f"{'Paradigm':24s} {'RLacc':>6s} {'High+Cat recall':>16s} {'Under%':>7s} {'Over%':>6s} {'maxUnder':>9s}"
    print(header)
    print("-" * len(header))

    for name, fname, kind in PARADIGMS:
        by = restrict(load(fname), ids)
        cm = collections.defaultdict(lambda: collections.Counter())  # cm[true][pred]
        correct = under = over = 0
        max_under = 0
        severe_total = severe_hit = 0  # recall on High+Catastrophic (severe) truths
        missing = 0
        for cid in ids:
            t = gt[cid]
            r = by.get(cid)
            if r is None:
                missing += 1
                continue
            p = r.get("risk_level")
            if p not in ORD:
                p = "Very Low"  # unparseable -> treat as lowest (conservative for under-escalation)
            cm[t][p] += 1
            d = ORD[p] - ORD[t]
            if d == 0:
                correct += 1
            elif d < 0:
                under += 1
                max_under = max(max_under, -d)
            else:
                over += 1
            if ORD[t] >= ORD["High"]:
                severe_total += 1
                if ORD[p] >= ORD["High"]:
                    severe_hit += 1
        n = len(ids) - missing
        rl_acc = correct / n if n else 0.0
        severe_recall = severe_hit / severe_total if severe_total else None
        under_pct = under / n if n else 0.0
        over_pct = over / n if n else 0.0
        results["paradigms"][name] = {
            "file": fname,
            "missing": missing,
            "risk_level_accuracy": round(rl_acc, 4),
            "severe_recall_high_plus": round(severe_recall, 4) if severe_recall is not None else None,
            "under_escalation_rate": round(under_pct, 4),
            "over_escalation_rate": round(over_pct, 4),
            "max_under_escalation_steps": max_under,
            "confusion_matrix": {t: dict(cm[t]) for t in LEVELS if t in cm},
        }
        sr = f"{severe_recall*100:5.1f}%" if severe_recall is not None else "  n/a"
        print(f"{name:24s} {rl_acc*100:5.1f}% {sr:>16s} {under_pct*100:6.1f}% {over_pct*100:5.1f}% {max_under:>9d}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}")

    # Sanity: risk-level accuracy should match the reported Task-ii accuracy
    print("\nNote: 'RLacc' is NFCRM 6.9 risk-level correctness (predicted vs ground-truth level),")
    print("which is the actual risk-assessment metric. Under% = predicted LOWER than true risk")
    print("(operationally dangerous: under-protection); Over% = predicted higher (false alarm).")


if __name__ == "__main__":
    main()
