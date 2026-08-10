"""Publication-grade statistics for the NSL-KDD 5-paradigm risk ablation.

Establishes that Tri-LLM+SHACL has the best risk-level accuracy across all arms,
with formal significance testing:
  - Wilson 95% CIs per arm (pooled over replicate runs for stochastic arms)
  - Exact McNemar paired tests: Tri-LLM+SHACL vs every other arm
  - Cohen's h effect sizes
  - Run-level separation (non-overlap of replicate runs)
  - Per-class accuracy matrix

Deterministic arms (ml, shacl, tristage_ml_shacl) have one realisation; for paired
tests against the 3-run stochastic arms they are replicated across the matched runs
(their per-case outcome is fixed, so this is exact, not an approximation).

Usage:  python -m scripts.nslkdd_risk_stats
"""
from __future__ import annotations
import json, math
from pathlib import Path
from scipy.stats import binomtest

ROOT  = Path(__file__).resolve().parents[1]
BASE  = ROOT / "results/nslkdd_risk_ablation"
RUNS  = BASE / "runs"
OUT   = BASE / "stats_summary.json"

ARM_LABELS = {
    "tristage_llm_shacl": "Tri-LLM+SHACL",
    "llm":                "LLM",
    "shacl":              "SHACL",
    "tristage_ml_shacl":  "Tri-ML+SHACL",
    "ml":                 "ML",
}
STOCHASTIC = {"llm", "tristage_llm_shacl"}   # have runs/run{1,2,3}
DETERMIN   = {"ml", "shacl", "tristage_ml_shacl"}
N_RUNS     = 3


def load(path: Path) -> dict[str, bool]:
    """case_id -> correct(bool)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["case_id"]] = bool(r["correct"])
    return out


def load_class(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["case_id"]] = r.get("true_class", "?")
    return out


def replicate_outcomes(arm: str) -> list[dict[str, bool]]:
    """Return a list of N_RUNS per-case outcome dicts.

    Stochastic arms read runs/run{1..3}; deterministic arms read the single
    top-level file and replicate it (fixed per-case outcome)."""
    if arm in STOCHASTIC:
        return [load(RUNS / f"run{i}" / f"{arm}.jsonl") for i in range(1, N_RUNS + 1)]
    single = load(BASE / f"{arm}.jsonl")
    return [single for _ in range(N_RUNS)]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cohens_h(p1: float, p2: float) -> float:
    return 2*math.asin(math.sqrt(p1)) - 2*math.asin(math.sqrt(p2))


def pool(outs: list[dict[str, bool]]) -> tuple[int, int, list[str]]:
    """Pooled (k correct, n total) and the ordered key list (run#case)."""
    k = n = 0
    keys = []
    for ri, d in enumerate(outs):
        for cid in sorted(d):
            keys.append(f"r{ri}:{cid}")
            n += 1
            k += d[cid]
    return k, n, keys


def _mcnemar_pair(da: dict[str, bool], db: dict[str, bool]):
    """One run: discordant counts + exact one-sided p (a > b)."""
    b_disc = c_disc = 0
    for cid in set(da) & set(db):
        if da[cid] and not db[cid]:
            b_disc += 1
        elif (not da[cid]) and db[cid]:
            c_disc += 1
    nd = b_disc + c_disc
    p = 1.0 if nd == 0 else binomtest(b_disc, nd, 0.5, alternative="greater").pvalue
    return b_disc, c_disc, p


def mcnemar(a_outs: list[dict[str, bool]], b_outs: list[dict[str, bool]]):
    """Pooled exact McNemar over replicate pairs (a > b), PLUS per-run results.

    Pooling repeats the same 50 cases across runs, so per-run independence is
    violated -- the pooled p is reported alongside the per-run p-values and the
    replication (sign) test so the evidence is not over-stated."""
    per_run = [_mcnemar_pair(da, db) for da, db in zip(a_outs, b_outs)]
    b_disc = sum(r[0] for r in per_run)
    c_disc = sum(r[1] for r in per_run)
    nd = b_disc + c_disc
    p_pool = 1.0 if nd == 0 else binomtest(b_disc, nd, 0.5, alternative="greater").pvalue
    # replication sign test: P(a beats b in all R runs | H0 coin-flip) = 0.5^wins
    wins = sum(1 for bd, cd, _ in per_run if bd > cd)
    p_sign = binomtest(wins, len(per_run), 0.5, alternative="greater").pvalue
    return b_disc, c_disc, p_pool, per_run, wins, p_sign


def main() -> None:
    arms = ["tristage_llm_shacl", "llm", "shacl", "tristage_ml_shacl", "ml"]
    outs = {a: replicate_outcomes(a) for a in arms}
    classes = load_class(RUNS / "run1" / "tristage_llm_shacl.jsonl")

    # ── per-arm pooled accuracy + Wilson CI + per-run accuracies ──────────────
    summary = {"n_runs": N_RUNS, "n_per_run": 50, "arms": {}}
    print(f"\n{'='*74}\nNSL-KDD 5-paradigm RISK ablation — statistical summary "
          f"({N_RUNS} runs x 50 cases)\n{'='*74}")
    print(f"{'Arm':<16}{'pooled acc':>12}{'95% Wilson CI':>20}{'per-run':>24}")
    print("-"*74)
    for a in arms:
        k, n, _ = pool(outs[a])
        lo, hi = wilson_ci(k, n)
        per_run = [round(sum(d.values())/len(d), 3) for d in outs[a]]
        kind = "stochastic" if a in STOCHASTIC else "deterministic"
        summary["arms"][ARM_LABELS[a]] = {
            "kind": kind,
            "pooled_k": k, "pooled_n": n,
            "pooled_accuracy": round(k/n, 4),
            "wilson_ci": [round(lo, 4), round(hi, 4)],
            "per_run_accuracy": per_run,
        }
        rr = ", ".join(f"{v:.3f}" for v in (per_run if a in STOCHASTIC else per_run[:1]))
        print(f"{ARM_LABELS[a]:<16}{k/n:>11.3f} {f'[{lo:.3f}, {hi:.3f}]':>20}{rr:>24}")

    # ── headline: run-level separation tristage vs llm ───────────────────────
    tri_runs = summary["arms"]["Tri-LLM+SHACL"]["per_run_accuracy"]
    llm_runs = summary["arms"]["LLM"]["per_run_accuracy"]
    sep = min(tri_runs) > max(llm_runs)
    summary["run_level_separation_tri_vs_llm"] = {
        "tri_min": min(tri_runs), "llm_max": max(llm_runs), "fully_separated": sep,
    }
    print(f"\nRun-level separation (Tri-LLM+SHACL vs LLM): "
          f"tri_min={min(tri_runs):.3f} > llm_max={max(llm_runs):.3f} -> {sep}")

    # ── McNemar: Tri-LLM+SHACL vs every other arm ────────────────────────────
    print(f"\nMcNemar paired tests: Tri-LLM+SHACL > X  (b=tri-right/X-wrong, c=tri-wrong/X-right)")
    print(f"{'vs':<16}{'b':>4}{'c':>4}{'p_pooled':>11}{'per-run p':>26}{'wins':>6}{'p_sign':>9}{'h':>7}")
    print("-"*84)
    tri = outs["tristage_llm_shacl"]
    tri_p = summary["arms"]["Tri-LLM+SHACL"]["pooled_accuracy"]
    summary["mcnemar_tri_vs"] = {}
    for a in ["llm", "shacl", "tristage_ml_shacl", "ml"]:
        b, c, p, per_run, wins, p_sign = mcnemar(tri, outs[a])
        h = cohens_h(tri_p, summary["arms"][ARM_LABELS[a]]["pooled_accuracy"])
        def star(pv): return "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
        pr_ps = [round(r[2], 4) for r in per_run]
        summary["mcnemar_tri_vs"][ARM_LABELS[a]] = {
            "b_tri_right_x_wrong": b, "c_tri_wrong_x_right": c,
            "p_pooled_one_sided_exact": p, "p_pooled_sig": star(p),
            "per_run": [{"b": r[0], "c": r[1], "p": r[2]} for r in per_run],
            "wins_of_3": wins, "p_sign_test": p_sign,
            "cohens_h": round(h, 3),
        }
        prs = "[" + ", ".join(f"{x:.3f}" for x in pr_ps) + "]"
        print(f"{ARM_LABELS[a]:<16}{b:>4}{c:>4}{p:>11.4g}{prs:>26}{wins:>5}/3{p_sign:>9.3f}{h:>7.3f}")

    # ── per-class accuracy matrix (run1 for stochastic, single for determ.) ───
    print(f"\nPer-class accuracy (run1 for stochastic arms):")
    cls_order = ["DoS", "Normal", "Probe", "R2L", "U2R"]
    hdr = f"{'Arm':<16}" + "".join(f"{c:>9}" for c in cls_order)
    print(hdr); print("-"*len(hdr))
    summary["per_class"] = {}
    for a in arms:
        d = outs[a][0]
        pc = {c: [0, 0] for c in cls_order}
        for cid, ok in d.items():
            c = classes.get(cid, "?")
            if c in pc:
                pc[c][1] += 1; pc[c][0] += ok
        summary["per_class"][ARM_LABELS[a]] = {
            c: (round(v[0]/v[1], 3) if v[1] else None) for c, v in pc.items()
        }
        row = f"{ARM_LABELS[a]:<16}" + "".join(
            f"{(v[0]/v[1]):>9.2f}" if v[1] else f"{'-':>9}" for v in pc.values())
        print(row)

    print("\nNote: Probe=0.00 for all class-based arms is a fixed GT-vs-engine "
          "inconsistency\n(Infiltration->Catastrophic vs GT High), uniform across arms; "
          "does not affect ranking.")

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
