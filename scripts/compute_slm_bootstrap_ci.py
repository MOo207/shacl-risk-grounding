"""Bootstrap CIs + per-class precision/recall/F1 for SLM evaluation runs.

Reads:
    results/slm_nl_classification.json        (CIC-IDS2018, N=60)
    results/slm_nslkdd_nl_classification.json  (NSL-KDD, N=50)

Writes:
    results/slm_bootstrap_metrics.json         (machine-readable)
    results/slm_bootstrap_metrics.md           (human-readable)

Stdlib only.  1000 bootstrap resamples per dataset.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
INPUTS = [
    ("CIC-IDS2018", "slm_nl_classification.json"),
    ("NSL-KDD", "slm_nslkdd_nl_classification.json"),
]
N_BOOT = 1000
SEED = 42


def load_predictions(path: Path) -> list[tuple[str, str]]:
    """Return list of (true_label, predicted_label) tuples; drop empty preds."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in data["results"]:
        true = r["true"]
        pred = r["predicted"] if r["predicted"] else "<empty>"
        out.append((true, pred))
    return out


def accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for t, p in pairs if t == p) / len(pairs)


def bootstrap_ci(pairs: list[tuple[str, str]], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Return (point_estimate, lo95, hi95) for accuracy via bootstrap resampling."""
    rng = random.Random(seed)
    point = accuracy(pairs)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    accs = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        accs.append(accuracy(sample))
    accs.sort()
    lo = accs[int(0.025 * n_boot)]
    hi = accs[int(0.975 * n_boot) - 1]
    return (point, lo, hi)


def per_class_metrics(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Compute precision/recall/F1 per class."""
    classes = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    out: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        support = sum(1 for t, _ in pairs if t == c)
        out[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def per_class_accuracy_ci(pairs: list[tuple[str, str]], cls: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Class-conditional accuracy (recall) with bootstrap CI on resamples of that class."""
    class_pairs = [(t, p) for t, p in pairs if t == cls]
    if not class_pairs:
        return (0.0, 0.0, 0.0)
    return bootstrap_ci(class_pairs, n_boot, seed)


def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    if not per_class:
        return 0.0
    return sum(m["f1"] for m in per_class.values()) / len(per_class)


def render_md(report: dict) -> str:
    lines = ["# SLM Bootstrap Metrics", ""]
    lines.append(f"_Bootstrap N={report['n_boot']}, seed={report['seed']}._")
    lines.append("")
    for ds in report["datasets"]:
        lines.append(f"## {ds['name']} (N={ds['n_samples']})")
        lines.append("")
        ov = ds["overall"]
        lines.append(
            f"**Overall accuracy:** {ov['point']:.3f}  [95% CI: {ov['lo']:.3f}, {ov['hi']:.3f}]"
        )
        lines.append(f"**Macro-F1:** {ds['macro_f1']:.3f}")
        lines.append("")
        lines.append("| Class | Support | Recall (Acc) [95% CI] | Precision | F1 |")
        lines.append("|---|---|---|---|---|")
        for cls in sorted(ds["per_class"].keys()):
            m = ds["per_class"][cls]
            ci = ds["per_class_ci"][cls]
            lines.append(
                f"| {cls} | {m['support']} | "
                f"{m['recall']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] | "
                f"{m['precision']:.3f} | {m['f1']:.3f} |"
            )
        lines.append("")
        lines.append("### Confusion matrix")
        lines.append("")
        cm = ds["confusion_matrix"]
        classes = sorted(cm.keys())
        header = "| true \\\\ pred | " + " | ".join(classes) + " |"
        sep = "|" + "---|" * (len(classes) + 1)
        lines.append(header)
        lines.append(sep)
        for t in classes:
            row = [str(cm[t].get(p, 0)) for p in classes]
            lines.append(f"| **{t}** | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    classes = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    cm: dict[str, dict[str, int]] = {t: {p: 0 for p in classes} for t in classes}
    for t, p in pairs:
        cm[t][p] += 1
    return cm


def analyze_dataset(name: str, fname: str) -> dict:
    pairs = load_predictions(RESULTS / fname)
    overall_point, overall_lo, overall_hi = bootstrap_ci(pairs, N_BOOT, SEED)
    per_class = per_class_metrics(pairs)
    per_class_ci = {}
    for cls in per_class:
        p, lo, hi = per_class_accuracy_ci(pairs, cls, N_BOOT, SEED)
        per_class_ci[cls] = {"point": p, "lo": lo, "hi": hi}
    return {
        "name": name,
        "source_file": fname,
        "n_samples": len(pairs),
        "overall": {"point": overall_point, "lo": overall_lo, "hi": overall_hi},
        "per_class": per_class,
        "per_class_ci": per_class_ci,
        "macro_f1": macro_f1(per_class),
        "confusion_matrix": confusion_matrix(pairs),
    }


def main() -> int:
    report = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "datasets": [analyze_dataset(name, fname) for name, fname in INPUTS],
    }

    json_path = RESULTS / "slm_bootstrap_metrics.json"
    md_path = RESULTS / "slm_bootstrap_metrics.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")

    # Print summary to stdout.
    for ds in report["datasets"]:
        ov = ds["overall"]
        print(f"{ds['name']:12s} N={ds['n_samples']:3d}  "
              f"acc={ov['point']:.3f} [{ov['lo']:.3f}, {ov['hi']:.3f}]  "
              f"macro-F1={ds['macro_f1']:.3f}")
    print(f"  -> {json_path.relative_to(REPO)}")
    print(f"  -> {md_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
