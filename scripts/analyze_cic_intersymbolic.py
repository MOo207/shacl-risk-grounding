"""Analyze the full intersymbolic §6.9 risk run (symbolic engine + LLM + SHACL).

Reports per arm: symbolic risk vs flat-lookup GT (exact + divergence direction),
SHACL conformance + retries, operational safety (severe-recall / under-/over-escalation),
and the comprehensive-vs-flat escalation pattern. No args.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results/cic_intersymbolic"
LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEV = {"High", "Catastrophic"}
ARMS = [("TRI-STAGE intersymbolic", "tristage_intersymbolic.jsonl"),
        ("PURE-LLM intersymbolic", "pure_intersymbolic.jsonl")]


def load(fn):
    p = D / fn
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else None


for label, fn in ARMS:
    r = load(fn)
    print("=" * 76)
    print(label)
    print("=" * 76)
    if not r:
        print("  MISSING\n"); continue
    n = len(r)
    exact = sum(1 for x in r if x["correct"])
    perr = sum(1 for x in r if x.get("parse_error"))
    shacl_ok = sum(1 for x in r if x.get("shacl_conform"))
    retried = sum(x.get("shacl_retried", 0) for x in r)
    overr = sum(1 for x in r if x.get("symbolic_override"))
    under = over = sevtot = sevrec = 0
    higher = lower = 0
    for x in r:
        p, g = x.get("risk_level", ""), x.get("ground_truth", "")
        if p in IDX and g in IDX:
            d = IDX[p] - IDX[g]
            if d < 0: under += 1; lower += 1
            elif d > 0: over += 1; higher += 1
            if g in SEV:
                sevtot += 1
                if p in SEV: sevrec += 1
    print(f"  N={n}  symbolic-override fired: {overr}")
    print(f"  PRIMARY symbolic risk vs flat §6.9 GT: exact={exact}/{n}={exact/n*100:.0f}%  "
          f"divergences={n-exact} (escalate↑={higher}, lower↓={lower})")
    print(f"  SHACL conformance: {shacl_ok}/{n}={shacl_ok/n*100:.0f}%  retries={retried}  parse_err={perr}")
    print(f"  safety (vs flat GT): severe-recall={sevrec}/{sevtot}  under-esc={under}  over-esc={over}")
    cls = {}
    for x in r:
        c = x["true_class"]; cls.setdefault(c, [0, 0]); cls[c][1] += 1; cls[c][0] += 1 if x["correct"] else 0
    print("  per-class exact:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(cls.items())})
    print(f"  mean narrative len: {sum(len((x.get('narrative') or '').split()) for x in r)//max(1,n)} words")
    print()
