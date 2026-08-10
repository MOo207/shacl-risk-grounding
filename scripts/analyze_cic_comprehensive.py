"""Analyze the comprehensive NFCRM §6.9 risk-assessment ablation (CIC, Haiku).

Reports, per arm:
  PRIMARY (comparable to run_unified_ablation): exact risk-level accuracy vs §6.9 GT,
    severe-recall, under-/over-escalation, worst-miss.
  PROCESS-VAR DELTA: how the LLM's likelihood x impact reasoning relates to the lookup GT --
    implied-risk accuracy, lookup-vs-process divergence rate + direction, internal consistency
    (stated risk_level vs matrix-implied), and the cases where comprehensive reasoning would
    change the assigned risk level.

Run after the comprehensive arms finish. No args.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/cic_comprehensive"
LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

ARMS = [("Pure LLM (Haiku, comprehensive)", "pure_llm_comprehensive.jsonl"),
        ("Tri-stage LLM (Haiku, comprehensive)", "tristage_llm_comprehensive.jsonl")]


def load(fname):
    p = D / fname
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]


def pct(a, b):
    return f"{a/b*100:.0f}%" if b else "n/a"


for label, fname in ARMS:
    rows = load(fname)
    print("=" * 78)
    print(label)
    print("=" * 78)
    if not rows:
        print("  MISSING / not finished\n")
        continue
    n = len(rows)
    perr = sum(1 for r in rows if r.get("parse_error"))
    # PRIMARY vs §6.9
    exact = sum(1 for r in rows if r.get("correct"))
    under = over = sev_tot = sev_rec = worst = 0
    for r in rows:
        p, g = r.get("risk_level", ""), r.get("ground_truth", "")
        if p not in IDX or g not in IDX:
            under += 1
            if g in SEVERE:
                sev_tot += 1
            worst = max(worst, IDX.get(g, 0))
            continue
        d = IDX[p] - IDX[g]
        if d < 0:
            under += 1; worst = max(worst, -d)
        elif d > 0:
            over += 1
        if g in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_rec += 1
    print(f"  N={n}  parse_err={perr}")
    print(f"  PRIMARY (vs §6.9 lookup GT): exact={pct(exact,n)}  severe-recall={pct(sev_rec,sev_tot)}  "
          f"under-esc={pct(under,n)}  over-esc={pct(over,n)}  worst-miss={worst}L")
    # PROCESS-VAR DELTA
    parsed = [r for r in rows if not r.get("parse_error")]
    impl_correct = sum(1 for r in parsed if r.get("implied_correct"))
    diverge = [r for r in parsed if r.get("lookup_vs_process_diverges")]
    consistent = sum(1 for r in parsed if r.get("internal_consistent"))
    # direction of divergence (implied vs lookup GT)
    higher = sum(1 for r in diverge if IDX.get(r.get("implied_risk_LxI"), -1) > IDX.get(r.get("ground_truth"), -1))
    lower = sum(1 for r in diverge if IDX.get(r.get("implied_risk_LxI"), 99) < IDX.get(r.get("ground_truth"), 99))
    print(f"  PROCESS DELTA: implied(LxI) exact vs §6.9={pct(impl_correct,len(parsed))}  "
          f"internal-consistent(stated==implied)={pct(consistent,len(parsed))}")
    print(f"    lookup-vs-process divergence: {len(diverge)}/{len(parsed)} ({pct(len(diverge),len(parsed))})  "
          f"[process HIGHER than lookup: {higher}; process LOWER: {lower}]")
    # likelihood / impact distributions
    print(f"    likelihood dist: {dict(Counter(r.get('likelihood') for r in parsed))}")
    print(f"    impact dist:     {dict(Counter(r.get('impact') for r in parsed))}")
    # per-class exact
    cls = {}
    for r in rows:
        c = r["true_class"]; cls.setdefault(c, [0, 0]); cls[c][1] += 1; cls[c][0] += 1 if r.get("correct") else 0
    print("    per-class exact:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(cls.items())})
    # a few divergence examples
    if diverge:
        print("    divergence examples (true_class | L x I -> implied | lookup GT):")
        for r in diverge[:5]:
            print(f"      {r['true_class']:12} {r.get('likelihood')}x{r.get('impact')} -> "
                  f"{r.get('implied_risk_LxI')}  vs GT={r.get('ground_truth')}")
    print()
