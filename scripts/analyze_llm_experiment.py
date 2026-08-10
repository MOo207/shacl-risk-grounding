"""Statistical analysis of four-paradigm LLM experiment results.

Computes:
  - Per-arm accuracy with 95% Wilson CIs (overall / Set A / Set B)
  - Paired McNemar tests between every LLM arm and every deterministic arm on Set B
  - Cohen's h effect size on Set B proportions
  - Prints a formatted summary table + LaTeX table snippet

Usage:
    python scripts/analyze_llm_experiment.py
    python scripts/analyze_llm_experiment.py --latex
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """One-sided exact McNemar p-value: P(X >= b) under H0: X ~ Bin(b+c, 0.5).

    H0: acc_A <= acc_B  (testing that A > B, i.e. more b than c).
    Returns p-value for the one-sided test that A is better than B.
    If b <= c returns 1.0 (no evidence A > B).
    """
    n = b + c
    if n == 0:
        return 1.0
    if b <= c:
        return 1.0
    # P(X >= b) = sum_{i=b}^{n} C(n,i) * 0.5^n
    total = 0.0
    coeff = 1.0
    # Build binomial PMF iteratively
    pmf = [0.0] * (n + 1)
    pmf[0] = 0.5 ** n
    for i in range(1, n + 1):
        pmf[i] = pmf[i - 1] * (n - i + 1) / i
    p = sum(pmf[i] for i in range(b, n + 1))
    return p


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ARM_FILES = {
    "Rule":   "results/llm_experiment/raw/run_1_arm_rule.jsonl",
    "ML":     "results/llm_experiment/raw/run_1_arm_ml.jsonl",
    "Rule+ML":"results/llm_experiment/raw/run_1_arm_rule_ml.jsonl",
    "Rule+LLM (Haiku)":  "results/llm_experiment/raw/run_1_arm_rule_llm_haiku.jsonl",
    "Rule+LLM (Sonnet)": "results/llm_experiment/raw/run_1_arm_rule_llm_sonnet.jsonl",
    # v2 arms — 100-case test set (seed=43), Set B = 50 cases
    "ML (v2)":              "results/llm_experiment/raw/run_v2_arm_ml.jsonl",
    "Rule+LLM Sonnet (v2)": "results/llm_experiment/raw/run_v2_arm_rule_llm_sonnet.jsonl",
}


def load_arm(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def accuracy_stats(rows: list[dict], subset: str | None = None) -> dict:
    filtered = [r for r in rows if subset is None or r.get("subset") == subset]
    n = len(filtered)
    k = sum(1 for r in filtered if r.get("risk_level") == r.get("ground_truth_risk_level"))
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "pct": 100 * k / n if n else 0.0,
            "ci_lo": 100 * lo, "ci_hi": 100 * hi}


# ---------------------------------------------------------------------------
# McNemar paired comparison (requires matching on case_id)
# ---------------------------------------------------------------------------

def paired_mcnemar(arm_a: list[dict], arm_b: list[dict],
                   subset: str | None = None) -> dict:
    """One-sided McNemar: A better than B?"""
    a_map = {r["case_id"]: r for r in arm_a}
    b_map = {r["case_id"]: r for r in arm_b}
    common = set(a_map) & set(b_map)
    if subset:
        common = {cid for cid in common if a_map[cid].get("subset") == subset}
    b_wins = 0  # A correct, B wrong
    c_wins = 0  # B correct, A wrong
    for cid in common:
        a_ok = a_map[cid].get("risk_level") == a_map[cid].get("ground_truth_risk_level")
        b_ok = b_map[cid].get("risk_level") == b_map[cid].get("ground_truth_risk_level")
        if a_ok and not b_ok:
            b_wins += 1
        elif b_ok and not a_ok:
            c_wins += 1
    p = mcnemar_exact_pvalue(b_wins, c_wins)
    return {"n_paired": len(common), "b": b_wins, "c": c_wins, "p": p}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    arms = {name: load_arm(path) for name, path in ARM_FILES.items()}

    # Check completeness
    for name, rows in arms.items():
        print(f"  {name}: {len(rows)} rows loaded")

    # Accuracy table
    print("\n=== ACCURACY WITH 95% WILSON CIs ===")
    print(f"{'Arm':<24} {'N':>4}  {'Overall':>16}  {'Set A':>16}  {'Set B':>16}")
    print("-" * 82)
    stats: dict[str, dict] = {}
    for name, rows in arms.items():
        ov = accuracy_stats(rows)
        sa = accuracy_stats(rows, "A")
        sb = accuracy_stats(rows, "B")
        stats[name] = {"overall": ov, "set_a": sa, "set_b": sb}
        def fmt(s): return f"{s['pct']:5.1f}% [{s['ci_lo']:4.1f},{s['ci_hi']:4.1f}]"
        print(f"{name:<24} {ov['n']:>4}  {fmt(ov)}  {fmt(sa)}  {fmt(sb)}")

    # McNemar tests — LLM arms vs each deterministic arm on Set B
    llm_arms = ["Rule+LLM (Haiku)", "Rule+LLM (Sonnet)"]
    det_arms = ["Rule", "ML", "Rule+ML"]

    print("\n=== PAIRED McNEMAR (one-sided: LLM > deterministic) on Set B [v1, N=25] ===")
    print(f"{'LLM arm':<24} {'vs':<14} {'N paired':>8}  {'b':>4}  {'c':>4}  {'p-value':>10}  sig")
    print("-" * 72)
    for llm in llm_arms:
        for det in det_arms:
            res = paired_mcnemar(arms[llm], arms[det], subset="B")
            sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 else "*" if res["p"] < 0.05 else "ns"
            print(f"{llm:<24} {det:<14} {res['n_paired']:>8}  {res['b']:>4}  {res['c']:>4}  {res['p']:>10.4f}  {sig}")

    # v2 McNemar (50 Set-B pairs) — only runs if both v2 files exist
    v2_sonnet = arms.get("Rule+LLM Sonnet (v2)", [])
    v2_ml = arms.get("ML (v2)", [])
    if v2_sonnet and v2_ml:
        print("\n=== PAIRED McNEMAR (one-sided: Sonnet > ML) on Set B [v2, N=50] ===")
        print(f"{'LLM arm':<30} {'vs':<14} {'N paired':>8}  {'b':>4}  {'c':>4}  {'p-value':>10}  sig")
        print("-" * 78)
        res = paired_mcnemar(v2_sonnet, v2_ml, subset="B")
        sig = "***" if res["p"] < 0.001 else "**" if res["p"] < 0.01 else "*" if res["p"] < 0.05 else "ns"
        print(f"{'Rule+LLM Sonnet (v2)':<30} {'ML (v2)':<14} {res['n_paired']:>8}  {res['b']:>4}  {res['c']:>4}  {res['p']:>10.4f}  {sig}")
        sb_s = stats["Rule+LLM Sonnet (v2)"]["set_b"]["pct"] / 100
        sb_m = stats["ML (v2)"]["set_b"]["pct"] / 100
        h = cohens_h(sb_s, sb_m)
        magnitude = "large" if abs(h) >= 0.8 else "medium" if abs(h) >= 0.5 else "small"
        print(f"  Cohen's h (v2 Sonnet vs v2 ML, Set B): h = {h:.3f} ({magnitude})")

    # Cohen's h on Set B
    print("\n=== COHEN'S h EFFECT SIZE (Set B, LLM vs deterministic baselines) ===")
    for llm in llm_arms:
        sb_llm = stats[llm]["set_b"]["pct"] / 100
        for det in det_arms:
            sb_det = stats[det]["set_b"]["pct"] / 100
            h = cohens_h(sb_llm, sb_det)
            magnitude = "large" if abs(h) >= 0.8 else "medium" if abs(h) >= 0.5 else "small"
            print(f"  {llm} vs {det}: h = {h:.3f} ({magnitude})")

    # Save JSON summary
    out = {
        "arms": {
            name: {
                "n": stats[name]["overall"]["n"],
                "overall": stats[name]["overall"],
                "set_a": stats[name]["set_a"],
                "set_b": stats[name]["set_b"],
            }
            for name in stats
        }
    }
    Path("results/llm_experiment/stats_summary.json").write_text(json.dumps(out, indent=2))
    print("\nSaved results/llm_experiment/stats_summary.json")

    if args.latex:
        print("\n=== LaTeX TABLE SNIPPET ===")
        print(r"\begin{tabular}{lcccc}")
        print(r"\toprule")
        print(r"\textbf{Paradigm} & \textbf{N} & \textbf{Overall} & \textbf{Set A} & \textbf{Set B (masked)} \\")
        print(r"\midrule")
        for name, rows in arms.items():
            ov = stats[name]["overall"]
            sa = stats[name]["set_a"]
            sb = stats[name]["set_b"]
            def lfmt(s):
                return f"{s['pct']:.1f}\\% [{s['ci_lo']:.0f},{s['ci_hi']:.0f}]"
            bold_open = r"\textbf{" if "Sonnet" in name else ""
            bold_close = "}" if "Sonnet" in name else ""
            print(f"{name} & {ov['n']} & {bold_open}{lfmt(ov)}{bold_close} & {lfmt(sa)} & {bold_open}{lfmt(sb)}{bold_close} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
