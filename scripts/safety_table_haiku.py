"""Operational risk-safety table — Haiku-primary, both datasets, all paradigms.

Primary metrics (the reengineered spine): severe-risk recall, under-escalation
(dangerous under-protection), over-escalation (false alarm), worst-miss.
Secondary: exact risk-level accuracy, within-1-level accuracy.

Also runs the spine's key test: per-case McNemar on "under-escalated? (unsafe)",
Tri-stage LLM (Haiku) vs Pure ML (XGBoost), on each dataset.

Output: results/safety_table_haiku.json (+ console).
"""
from __future__ import annotations
import json
from math import comb, asin, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}


def load(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip().replace("\x00", "")
        if ln:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def gt_of(r):
    return r.get("ground_truth") or r.get("ground_truth_risk_level") or ""


def by_id(rows, ids):
    d = {}
    for r in rows:
        if r["case_id"] in ids:
            d[r["case_id"]] = r  # last occurrence wins
    return d


def metrics(recs):
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for r in recs.values():
        p, g = r.get("risk_level", ""), gt_of(r)
        if p not in IDX or g not in IDX:
            # unparseable prediction counts as a miss; treat as max under-escalation
            n += 1
            if g in SEVERE:
                sev_tot += 1
            under += 1
            worst = max(worst, IDX.get(g, 0))
            continue
        n += 1
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
    pct = lambda a, b: (a / b * 100) if b else 0.0
    return dict(n=n, exact=pct(exact, n), within1=pct(within1, n),
                under=pct(under, n), over=pct(over, n),
                severe_recall=pct(sev_rec, sev_tot), sev_tot=sev_tot,
                worst_miss=worst)


def mcnemar(b, c):
    n = b + c
    return sum(comb(n, k) for k in range(0, min(b, c) + 1)) * (0.5 ** n) * (2 if b != c else 1) if n else 1.0


def mcnemar_1s(b, c):
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) * (0.5 ** n) if n else 1.0


def unsafe_mcnemar(llm, ml, ids):
    """McNemar on 'under-escalated (unsafe)' — LLM safer than ML?
    b = ML unsafe & LLM safe; c = LLM unsafe & ML safe. H1: b>c."""
    def unsafe(r):
        p, g = r.get("risk_level", ""), gt_of(r)
        if p not in IDX or g not in IDX:
            return True
        return IDX[p] < IDX[g]
    b = c = 0
    for i in ids:
        if i in llm and i in ml:
            lu, mu = unsafe(llm[i]), unsafe(ml[i])
            if mu and not lu:
                b += 1
            elif lu and not mu:
                c += 1
    return b, c, mcnemar_1s(b, c)


DATASETS = {
    "CIC-IDS2018 (N=60)": {
        "dir": "results/unified_ablation",
        "canon": "ml_xgb_n60.jsonl",
        "paradigms": [
            ("Pure Rule", "rule_n60.jsonl"),
            ("Pure ML (RF)", "ml_rf_n60.jsonl"),
            ("Pure ML (XGBoost)", "ml_xgb_n60.jsonl"),
            ("Tri-stage RF", "tristage_rf_n60.jsonl"),
            ("Tri-stage XGBoost", "tristage_xgb_n60.jsonl"),
            ("Pure LLM (Haiku)", "pure_llm_haiku.jsonl"),
            ("Tri-stage LLM (Haiku)", "tristage_llm_haiku.jsonl"),
            ("[ref] Pure LLM (Sonnet)", "pure_llm.jsonl"),
            ("[ref] Tri-stage LLM (Sonnet)", "tristage_llm_sonnet.jsonl"),
        ],
        "llm": "tristage_llm_haiku.jsonl", "ml": "ml_xgb_n60.jsonl",
    },
    "NSL-KDD (N=50)": {
        "dir": "results/nslkdd_ablation",
        "canon": "ml_xgb.jsonl",
        "paradigms": [
            ("Pure Rule", "rule.jsonl"),
            ("Pure ML (RF)", "ml_rf.jsonl"),
            ("Pure ML (XGBoost)", "ml_xgb.jsonl"),
            ("Tri-stage RF", "tristage_rf.jsonl"),
            ("Tri-stage XGBoost", "tristage_xgb.jsonl"),
            ("Pure LLM (Haiku)", "pure_llm_haiku.jsonl"),
            ("Tri-stage LLM (Haiku)", "tristage_llm_haiku.jsonl"),
            ("[ref] Pure LLM (Sonnet)", "pure_llm.jsonl"),
            ("[ref] Tri-stage LLM (Sonnet)", "tristage_llm_sonnet.jsonl"),
        ],
        "llm": "tristage_llm_haiku.jsonl", "ml": "ml_xgb.jsonl",
    },
}

out_all = {}
for dname, cfg in DATASETS.items():
    d = cfg["dir"]
    canon = load(f"{d}/{cfg['canon']}")
    ids = set(r["case_id"] for r in canon)
    print("=" * 92)
    print(f"{dname}  — canonical ids n={len(ids)}")
    print("=" * 92)
    hdr = f"{'Paradigm':<30}{'Exact':>7}{'Wn-1':>7}{'Sev.Rec':>9}{'Under':>7}{'Over':>7}{'Worst':>7}"
    print(hdr)
    print("-" * 92)
    rows_out = {}
    for label, fname in cfg["paradigms"]:
        rows = load(f"{d}/{fname}")
        if rows is None:
            print(f"{label:<30}  MISSING ({fname})")
            continue
        m = metrics(by_id(rows, ids))
        rows_out[label] = m
        print(f"{label:<30}{m['exact']:>6.0f}%{m['within1']:>6.0f}%"
              f"{m['severe_recall']:>8.0f}%{m['under']:>6.0f}%{m['over']:>6.0f}%{m['worst_miss']:>6}L")
    # spine test
    llm = by_id(load(f"{d}/{cfg['llm']}"), ids)
    ml = by_id(load(f"{d}/{cfg['ml']}"), ids)
    b, c, p = unsafe_mcnemar(llm, ml, ids)
    print("-" * 92)
    print(f"  SPINE TEST — under-escalation McNemar, Tri-LLM(Haiku) vs XGBoost: "
          f"b(ML-unsafe,LLM-safe)={b}  c={c}  p(1-sided)={p:.4g}")
    out_all[dname] = {"paradigms": rows_out, "unsafe_mcnemar": {"b": b, "c": c, "p": p}}
    print()

(ROOT / "results/safety_table_haiku.json").write_text(json.dumps(out_all, indent=2))
print("Saved -> results/safety_table_haiku.json")
