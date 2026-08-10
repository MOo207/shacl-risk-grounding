"""Compare full NL features (W17 v3) vs stripped (no signature hints).

Reads:
    results/slm_nl_classification.json            (full features)
    results/slm_nl_classification_stripped.json   (stripped, ablation)

Computes:
    - Per-class accuracy + bootstrap 95% CI for each run
    - Overall accuracy + bootstrap 95% CI
    - McNemar test (sample-paired) for accuracy difference

Writes:
    results/slm_ablation_comparison.json
    results/slm_ablation_comparison.md
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FULL = RESULTS / "slm_nl_classification.json"
STRIPPED = RESULTS / "slm_nl_classification_stripped.json"
N_BOOT = 1000
SEED = 42


def load(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(r["true"], r["predicted"] if r["predicted"] else "<empty>")
            for r in data["results"]]


def acc(pairs):
    return sum(1 for t, p in pairs if t == p) / len(pairs) if pairs else 0.0


def bootstrap(pairs, n_boot, seed):
    rng = random.Random(seed)
    point = acc(pairs)
    n = len(pairs)
    accs = sorted(acc([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return point, accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot) - 1]


def per_class_acc(pairs):
    out = {}
    classes = sorted({t for t, _ in pairs})
    for c in classes:
        cp = [(t, p) for t, p in pairs if t == c]
        out[c] = acc(cp)
    return out


def mcnemar(pairs_a, pairs_b):
    """Sample-paired McNemar test. Both lists must align row-by-row.
    Returns (b, c, two_sided_p) where b = a_correct & b_wrong, c = a_wrong & b_correct.
    """
    if len(pairs_a) != len(pairs_b):
        return None
    b = sum(1 for (ta, pa), (tb, pb) in zip(pairs_a, pairs_b)
            if ta == pa and tb != pb)
    c = sum(1 for (ta, pa), (tb, pb) in zip(pairs_a, pairs_b)
            if ta != pa and tb == pb)
    if b + c == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0, "note": "no discordant pairs"}
    # Exact binomial test, two-sided. P(X >= max(b, c) | n=b+c, p=0.5) * 2
    n = b + c
    k = max(b, c)
    # Sum binomial PMF from k to n
    cum = sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))
    p_two = min(1.0, 2.0 * cum)
    return {"b": b, "c": c, "p_two_sided": p_two}


def render_md(report):
    lines = ["# SLM Ablation: Full NL features vs Stripped (no signature hints)", ""]
    lines.append("Held constant across runs: random_state=42 (same 60 samples), "
                 "model=glm-4.7, temperature=0, ZAI API.")
    lines.append("")
    lines.append("| Variant | Accuracy [95% CI] | DoS | DDoS | WebAttack | BruteForce | Benign | Infiltration |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, run in [("Full NL (W17 v3)", report["full"]), ("Stripped (no hints)", report["stripped"])]:
        ov = run["overall"]
        cells = [f"{run['per_class'].get(c, 0)*100:.0f}%" for c in
                 ["DoS", "DDoS", "WebAttack", "BruteForce", "Benign", "Infiltration"]]
        lines.append(
            f"| {name} | {ov['point']*100:.1f}% [{ov['lo']*100:.1f}, {ov['hi']*100:.1f}] | "
            + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append(f"**Delta (Full - Stripped):** {report['delta']*100:+.1f} percentage points")
    if report.get("mcnemar"):
        m = report["mcnemar"]
        lines.append(f"**McNemar test:** b={m['b']} (full-only correct), c={m['c']} (stripped-only correct), p={m['p_two_sided']:.4f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    delta_pp = report["delta"] * 100
    p = report.get("mcnemar", {}).get("p_two_sided", 1.0)
    sig = p < 0.05
    if not sig:
        lines.append(
            f"- Delta = {delta_pp:+.1f} pp, but the sample-paired McNemar test gives "
            f"p={p:.3f} (NOT significant at α=0.05). At N={report['full']['n']}, the data "
            "are consistent with BOTH ``the LLM relies on the engineered signature hints'' "
            "AND ``the LLM does real reasoning over base features''. We cannot distinguish "
            "these hypotheses at this sample size. Report this as an **honest null** on "
            "the contribution of NL feature engineering."
        )
    elif delta_pp >= 5:
        lines.append(
            f"- Full beats stripped by {delta_pp:.1f} pp (McNemar p={p:.3f}, significant). "
            "Evidence that the engineered NL hints carry weight; the LLM relies on those cues."
        )
    else:
        lines.append(
            f"- Stripped beats full by {abs(delta_pp):.1f} pp (McNemar p={p:.3f}). "
            "Engineered hints may be misleading the LLM. Investigate."
        )
    # Also note where the delta concentrates per-class
    pc_full = report["full"]["per_class"]
    pc_strip = report["stripped"]["per_class"]
    diffs = sorted(((pc_full[c] - pc_strip[c]) * 100, c) for c in pc_full)
    biggest = diffs[-1] if diffs else (0, None)
    if biggest[1] and abs(biggest[0]) >= 10:
        lines.append(
            f"- Per-class: the largest gap is **{biggest[1]}** "
            f"(full {pc_full[biggest[1]]*100:.0f}%, stripped {pc_strip[biggest[1]]*100:.0f}%, "
            f"delta {biggest[0]:+.0f} pp). On n=10 per class the per-class CIs are wide "
            "(roughly ±30 pp), so even this gap is not individually significant."
        )
    return "\n".join(lines)


def main():
    if not STRIPPED.exists():
        print(f"ERROR: stripped run not found at {STRIPPED}")
        return 1
    full = load(FULL)
    stripped = load(STRIPPED)

    full_overall = bootstrap(full, N_BOOT, SEED)
    stripped_overall = bootstrap(stripped, N_BOOT, SEED)

    report = {
        "full": {
            "n": len(full),
            "overall": {"point": full_overall[0], "lo": full_overall[1], "hi": full_overall[2]},
            "per_class": per_class_acc(full),
        },
        "stripped": {
            "n": len(stripped),
            "overall": {"point": stripped_overall[0], "lo": stripped_overall[1], "hi": stripped_overall[2]},
            "per_class": per_class_acc(stripped),
        },
        "delta": full_overall[0] - stripped_overall[0],
        "mcnemar": mcnemar(full, stripped),
        "n_boot": N_BOOT,
        "seed": SEED,
    }

    json_path = RESULTS / "slm_ablation_comparison.json"
    md_path = RESULTS / "slm_ablation_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")

    print(f"Full:     {full_overall[0]*100:.1f}% [{full_overall[1]*100:.1f}, {full_overall[2]*100:.1f}]  (N={len(full)})")
    print(f"Stripped: {stripped_overall[0]*100:.1f}% [{stripped_overall[1]*100:.1f}, {stripped_overall[2]*100:.1f}]  (N={len(stripped)})")
    print(f"Delta:    {report['delta']*100:+.1f} pp")
    if report["mcnemar"]:
        m = report["mcnemar"]
        print(f"McNemar:  b={m['b']}, c={m['c']}, p={m['p_two_sided']:.4f}")
    print(f"  -> {json_path.relative_to(REPO)}")
    print(f"  -> {md_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
