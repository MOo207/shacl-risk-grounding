"""Bootstrap CIs and paired McNemar for tri-stage ablation (N=2,122).

Reads:
    results/ablation_study_v2.json    (baseline + tri_stage predictions, test labels)

Computes:
    - Overall accuracy + 95% bootstrap CI for baseline and tri_stage
    - Macro-F1 + bootstrap CI for each
    - Per-class precision/recall/F1 with CIs
    - Sample-paired McNemar test (H1: tri-stage accuracy != baseline accuracy)
    - Per-class McNemar tests

Writes:
    results/tristage_bootstrap_metrics.json
    results/tristage_bootstrap_metrics.md

Stdlib only. Uses 1,000 bootstrap resamples, seed=42.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
ABLATION = RESULTS / "ablation_study_v2.json"
N_BOOT = 1000
SEED = 42


def acc(true, pred):
    return sum(1 for t, p in zip(true, pred) if t == p) / len(true) if true else 0.0


def bootstrap_acc(true, pred, n_boot, seed):
    rng = random.Random(seed)
    n = len(true)
    point = acc(true, pred)
    accs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        accs.append(acc([true[i] for i in idx], [pred[i] for i in idx]))
    accs.sort()
    return point, accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot) - 1]


def per_class_metrics(true, pred):
    classes = sorted(set(true) | set(pred))
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(true, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": sum(1 for t in true if t == c),
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def macro_f1(per_class):
    return sum(m["f1"] for m in per_class.values()) / len(per_class) if per_class else 0.0


def bootstrap_macro_f1(true, pred, n_boot, seed):
    rng = random.Random(seed)
    n = len(true)
    point = macro_f1(per_class_metrics(true, pred))
    f1s = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bt = [true[i] for i in idx]
        bp = [pred[i] for i in idx]
        f1s.append(macro_f1(per_class_metrics(bt, bp)))
    f1s.sort()
    return point, f1s[int(0.025 * n_boot)], f1s[int(0.975 * n_boot) - 1]


def mcnemar_paired(true, pred_a, pred_b):
    """Sample-paired McNemar: did pred_a vs pred_b classify the same items differently?"""
    b = sum(1 for t, a, bb in zip(true, pred_a, pred_b) if a == t and bb != t)
    c = sum(1 for t, a, bb in zip(true, pred_a, pred_b) if a != t and bb == t)
    if b + c == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0, "note": "no discordant pairs"}
    n = b + c
    k = max(b, c)
    cum = sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))
    p_two = min(1.0, 2.0 * cum)
    return {"b": b, "c": c, "n_discordant": n, "p_two_sided": p_two}


def mcnemar_per_class(true, pred_a, pred_b):
    """Per-class one-vs-rest McNemar."""
    classes = sorted(set(true))
    out = {}
    for c in classes:
        true_bin = [1 if t == c else 0 for t in true]
        a_bin = [1 if p == c else 0 for p in pred_a]
        b_bin = [1 if p == c else 0 for p in pred_b]
        # Treat per-class as a binary problem; correct = label match
        a_corr = [1 if t == a else 0 for t, a in zip(true_bin, a_bin)]
        b_corr = [1 if t == b else 0 for t, b in zip(true_bin, b_bin)]
        bb = sum(1 for ac, bc in zip(a_corr, b_corr) if ac == 1 and bc == 0)
        cc = sum(1 for ac, bc in zip(a_corr, b_corr) if ac == 0 and bc == 1)
        if bb + cc == 0:
            out[c] = {"b": bb, "c": cc, "p_two_sided": 1.0}
            continue
        n = bb + cc
        k = max(bb, cc)
        cum = sum(math.comb(n, i) * 0.5 ** n for i in range(k, n + 1))
        out[c] = {"b": bb, "c": cc, "n_discordant": n, "p_two_sided": min(1.0, 2.0 * cum)}
    return out


def render_md(report):
    lines = ["# Tri-stage Pipeline Bootstrap Metrics (N=2,122)", ""]
    lines.append(f"_Bootstrap N={report['n_boot']}, seed={report['seed']}._  ")
    lines.append(f"_Source: \\texttt{{results/ablation\\_study\\_v2.json}}_")
    lines.append("")
    lines.append("## H1 — Pre-inference symbolic enrichment changes classifier accuracy")
    lines.append("")
    lines.append("| Variant | Accuracy [95% CI] | Macro-F1 [95% CI] |")
    lines.append("|---|---|---|")
    for name, key in [("RF Baseline", "baseline"), ("Tri-stage (full)", "tri_stage")]:
        r = report[key]
        a = r["accuracy_ci"]
        f = r["macro_f1_ci"]
        lines.append(
            f"| {name} | {a['point']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}] | "
            f"{f['point']:.4f} [{f['lo']:.4f}, {f['hi']:.4f}] |"
        )
    lines.append("")
    delta_acc = report["tri_stage"]["accuracy_ci"]["point"] - report["baseline"]["accuracy_ci"]["point"]
    delta_f1 = report["tri_stage"]["macro_f1_ci"]["point"] - report["baseline"]["macro_f1_ci"]["point"]
    lines.append(f"**Accuracy delta (tri-stage − baseline):** {delta_acc:+.4f}")
    lines.append(f"**Macro-F1 delta (tri-stage − baseline):** {delta_f1:+.4f}")
    lines.append("")
    m = report["mcnemar_global"]
    lines.append(f"**Sample-paired McNemar test:** b={m['b']} (baseline-only correct), "
                 f"c={m['c']} (tri-stage-only correct), n_discordant={m.get('n_discordant', m['b']+m['c'])}, "
                 f"p={m['p_two_sided']:.4f} (two-sided exact)")
    lines.append("")
    if m["p_two_sided"] >= 0.05:
        lines.append("**Verdict on H1: Fail to reject null.** At α=0.05, the tri-stage pipeline does "
                     f"not produce a statistically significant change in classification accuracy "
                     f"versus the RF baseline. Confidence intervals overlap substantially (delta is "
                     f"~{delta_acc*100:+.2f} pp; CI half-widths are ~{(report['baseline']['accuracy_ci']['hi']-report['baseline']['accuracy_ci']['lo'])/2*100:.2f} pp). The pipeline neither helps nor hurts accuracy at this scale.")
    else:
        lines.append(f"**Verdict on H1: Reject null** (p={m['p_two_sided']:.4f}).")
    lines.append("")
    lines.append("## Per-class significance (one-vs-rest McNemar)")
    lines.append("")
    lines.append("| Class | Support | b (base-only) | c (tri-only) | p (two-sided) | Significant at α=0.05? |")
    lines.append("|---|---|---|---|---|---|")
    for cls, m in report["mcnemar_per_class"].items():
        sup = report["baseline"]["per_class"][cls]["support"]
        sig = "yes" if m["p_two_sided"] < 0.05 else "no"
        lines.append(f"| {cls} | {sup} | {m['b']} | {m['c']} | {m['p_two_sided']:.4f} | {sig} |")
    lines.append("")
    lines.append("## Per-class precision/recall/F1 (point estimates)")
    lines.append("")
    lines.append("| Class | RF Baseline (P/R/F1) | Tri-stage (P/R/F1) |")
    lines.append("|---|---|---|")
    for cls in sorted(report["baseline"]["per_class"]):
        b = report["baseline"]["per_class"][cls]
        t = report["tri_stage"]["per_class"][cls]
        lines.append(
            f"| {cls} | {b['precision']:.3f} / {b['recall']:.3f} / {b['f1']:.3f} | "
            f"{t['precision']:.3f} / {t['recall']:.3f} / {t['f1']:.3f} |"
        )
    return "\n".join(lines)


def main():
    d = json.loads(ABLATION.read_text(encoding="utf-8"))
    true = d["test_labels"]
    base_pred = d["baseline"]["predictions"]
    tri_pred = d["tri_stage"]["predictions"]

    base_acc = bootstrap_acc(true, base_pred, N_BOOT, SEED)
    tri_acc = bootstrap_acc(true, tri_pred, N_BOOT, SEED)
    base_f1 = bootstrap_macro_f1(true, base_pred, N_BOOT, SEED)
    tri_f1 = bootstrap_macro_f1(true, tri_pred, N_BOOT, SEED)

    base_pc = per_class_metrics(true, base_pred)
    tri_pc = per_class_metrics(true, tri_pred)

    mc_global = mcnemar_paired(true, base_pred, tri_pred)
    mc_pc = mcnemar_per_class(true, base_pred, tri_pred)

    report = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "n_test": len(true),
        "baseline": {
            "accuracy_ci": {"point": base_acc[0], "lo": base_acc[1], "hi": base_acc[2]},
            "macro_f1_ci": {"point": base_f1[0], "lo": base_f1[1], "hi": base_f1[2]},
            "per_class": base_pc,
        },
        "tri_stage": {
            "accuracy_ci": {"point": tri_acc[0], "lo": tri_acc[1], "hi": tri_acc[2]},
            "macro_f1_ci": {"point": tri_f1[0], "lo": tri_f1[1], "hi": tri_f1[2]},
            "per_class": tri_pc,
        },
        "mcnemar_global": mc_global,
        "mcnemar_per_class": mc_pc,
    }
    (RESULTS / "tristage_bootstrap_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (RESULTS / "tristage_bootstrap_metrics.md").write_text(
        render_md(report), encoding="utf-8"
    )

    print(f"Baseline:  acc={base_acc[0]:.4f} [{base_acc[1]:.4f}, {base_acc[2]:.4f}]   "
          f"f1={base_f1[0]:.4f} [{base_f1[1]:.4f}, {base_f1[2]:.4f}]")
    print(f"Tri-stage: acc={tri_acc[0]:.4f} [{tri_acc[1]:.4f}, {tri_acc[2]:.4f}]   "
          f"f1={tri_f1[0]:.4f} [{tri_f1[1]:.4f}, {tri_f1[2]:.4f}]")
    print(f"McNemar global: b={mc_global['b']}, c={mc_global['c']}, p={mc_global['p_two_sided']:.4f}")
    print("Per-class McNemar:")
    for cls, m in mc_pc.items():
        sig = "*" if m["p_two_sided"] < 0.05 else " "
        print(f"  {sig} {cls:15s}  b={m['b']:3d}  c={m['c']:3d}  p={m['p_two_sided']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
