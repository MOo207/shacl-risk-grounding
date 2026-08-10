"""Recompute all Haiku-only statistics for the Sonnet-removal restructuring.

Reads the canonical Haiku result files (plus the newly-run Pure-LLM Haiku and
H8 Haiku files) and recomputes:
  - CIC unified ablation Haiku numbers (N=60) + McNemar Haiku-tristage vs XGBoost
  - NSL-KDD ablation Haiku numbers (N=50) + McNemar
  - Five-paradigm Set B Haiku vs Rule (H6) McNemar
  - H8 Haiku replication Set B vs ML v2 McNemar
  - Fisher combined test (k=4) on the four Haiku p-values

All correctness = (risk_level == ground_truth) computed here so scoring is
uniform across files. Run with no args; missing files are reported, not fatal.
"""
from __future__ import annotations
import json
from math import comb, log, exp, factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def correct_of(rec) -> bool:
    """Uniform correctness: prefer explicit 'correct', else risk_level==GT."""
    if "correct" in rec and rec["correct"] is not None:
        return bool(rec["correct"])
    gt = rec.get("ground_truth") or rec.get("ground_truth_risk_level") or ""
    return rec.get("risk_level", "") == gt and gt != ""


def acc(rows, ids=None):
    if rows is None:
        return None
    r = [x for x in rows if (ids is None or x["case_id"] in ids)]
    if not r:
        return None
    c = sum(1 for x in r if correct_of(x))
    return c, len(r), c / len(r) * 100


def mcnemar_one_sided(b, c):
    """H1: b>c. p = P(X<=c) under Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(0, c + 1)) * (0.5 ** n)


def cohens_h(p1, p2):
    from math import asin, sqrt
    return abs(2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2)))


def discordant(A, B, ids):
    da = {x["case_id"]: correct_of(x) for x in A}
    db = {x["case_id"]: correct_of(x) for x in B}
    common = [i for i in ids if i in da and i in db]
    b = sum(1 for i in common if da[i] and not db[i])
    c = sum(1 for i in common if not da[i] and db[i])
    return b, c, len(common)


def chi2_sf(x, df):
    k = df // 2
    s = sum((x / 2) ** i / factorial(i) for i in range(k))
    return exp(-x / 2) * s


def setB(rows):
    return [x for x in rows if x.get("subset") == "B"]


print("=" * 70)
print("CIC UNIFIED ABLATION (Haiku, canonical N=60)")
print("=" * 70)
xgb60 = load("results/unified_ablation/ml_xgb_n60.jsonl")
n60 = set(x["case_id"] for x in xgb60) if xgb60 else set()
for name, rel in [
    ("Pure LLM (Haiku)", "results/unified_ablation/pure_llm_haiku.jsonl"),
    ("Tri-stage LLM (Haiku)", "results/unified_ablation/tristage_llm_haiku.jsonl"),
    ("Pure ML (XGBoost)", "results/unified_ablation/ml_xgb_n60.jsonl"),
]:
    rows = load(rel)
    a = acc(rows, n60)
    print(f"  {name:<26} {'MISSING' if a is None else f'{a[0]}/{a[1]} = {a[2]:.1f}%'}")

tri = load("results/unified_ablation/tristage_llm_haiku.jsonl")
p_cic = None
if tri and xgb60:
    b, c, n = discordant(tri, xgb60, n60)
    p_cic = mcnemar_one_sided(b, c)
    print(f"  McNemar Tri-Haiku vs XGB (N60): b={b} c={c} n={n} p={p_cic:.4g}")

print("=" * 70)
print("NSL-KDD ABLATION (Haiku, N=50)")
print("=" * 70)
nsl_xgb = load("results/nslkdd_ablation/ml_xgb.jsonl")
nsl_ids = set(x["case_id"] for x in nsl_xgb) if nsl_xgb else set()
for name, rel in [
    ("Pure LLM (Haiku)", "results/nslkdd_ablation/pure_llm_haiku.jsonl"),
    ("Tri-stage LLM (Haiku)", "results/nslkdd_ablation/tristage_llm_haiku.jsonl"),
    ("Pure ML (XGBoost)", "results/nslkdd_ablation/ml_xgb.jsonl"),
    ("Pure ML (RF)", "results/nslkdd_ablation/ml_rf.jsonl"),
]:
    rows = load(rel)
    a = acc(rows, nsl_ids)
    print(f"  {name:<26} {'MISSING' if a is None else f'{a[0]}/{a[1]} = {a[2]:.1f}%'}")

nsl_tri = load("results/nslkdd_ablation/tristage_llm_haiku.jsonl")
p_nsl = None
if nsl_tri and nsl_xgb:
    b, c, n = discordant(nsl_tri, nsl_xgb, nsl_ids)
    p_nsl = mcnemar_one_sided(b, c)
    print(f"  McNemar Tri-Haiku vs XGB: b={b} c={c} n={n} p={p_nsl:.4g}")

print("=" * 70)
print("FIVE-PARADIGM Set B (Haiku vs Rule = H6)")
print("=" * 70)
fp_haiku = load("results/llm_experiment/raw/run_1_arm_rule_llm_haiku.jsonl")
fp_rule = load("results/llm_experiment/raw/run_1_arm_rule.jsonl")
fp_ml = load("results/llm_experiment/raw/run_1_arm_ml.jsonl")
p_5par = None
if fp_haiku and fp_rule:
    Hb, Rb = setB(fp_haiku), setB(fp_rule)
    ha = acc(Hb); ra = acc(Rb)
    print(f"  Haiku Set B: {ha[0]}/{ha[1]}={ha[2]:.1f}%  Rule Set B: {ra[0]}/{ra[1]}={ra[2]:.1f}%")
    ids = set(x["case_id"] for x in Hb) & set(x["case_id"] for x in Rb)
    b, c, n = discordant(Hb, Rb, ids)
    p_5par = mcnemar_one_sided(b, c)
    print(f"  McNemar Haiku vs Rule (Set B): b={b} c={c} n={n} p={p_5par:.4g} h={cohens_h(ha[2]/100, ra[2]/100):.3f}")

print("=" * 70)
print("H8 REPLICATION (Haiku Set B vs ML v2 Set B)")
print("=" * 70)
h8_haiku = load("results/llm_experiment/raw/run_v2_arm_rule_llm_haiku.jsonl")
h8_ml = load("results/llm_experiment/raw/run_v2_arm_ml.jsonl")
p_h8 = None
if h8_haiku and h8_ml:
    Hb, Mb = setB(h8_haiku), setB(h8_ml)
    ha = acc(Hb); ma = acc(Mb)
    print(f"  Haiku Set B: {ha[0]}/{ha[1]}={ha[2]:.1f}%  ML Set B: {ma[0]}/{ma[1]}={ma[2]:.1f}%")
    ids = set(x["case_id"] for x in Hb) & set(x["case_id"] for x in Mb)
    b, c, n = discordant(Hb, Mb, ids)
    p_h8 = mcnemar_one_sided(b, c)
    print(f"  McNemar Haiku vs ML (Set B): b={b} c={c} n={n} p={p_h8:.4g} h={cohens_h(ha[2]/100, ma[2]/100):.3f}")
else:
    print("  H8 Haiku file not ready yet.")

print("=" * 70)
print("FISHER COMBINED (Haiku, k=4)")
print("=" * 70)
ps = {"CIC": p_cic, "NSL-KDD": p_nsl, "5-paradigm": p_5par, "H8": p_h8}
if all(v is not None for v in ps.values()):
    chi = -2 * sum(log(p) for p in ps.values())
    df = 2 * len(ps)
    pcomb = chi2_sf(chi, df)
    print(f"  p-values: {', '.join(f'{k}={v:.4g}' for k,v in ps.items())}")
    print(f"  Fisher chi2={chi:.2f} df={df} combined p={pcomb:.3g}")
    # robustness: drop CIC (weakest)
    ps2 = {k: v for k, v in ps.items() if k != "CIC"}
    chi2 = -2 * sum(log(p) for p in ps2.values())
    print(f"  drop-CIC robustness: chi2={chi2:.2f} df={2*len(ps2)} p={chi2_sf(chi2, 2*len(ps2)):.3g}")
else:
    avail = {k: v for k, v in ps.items() if v is not None}
    print(f"  Not all p-values ready. Have: {', '.join(f'{k}={v:.4g}' for k,v in avail.items())}")
