"""
Generate fresh thesis figures at 300 DPI from results JSON files.
Saves to thesis/figures/ with names matching includegraphics calls.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "thesis" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.constrained_layout.use": True,
})

PALETTE = {
    "rule":   "#4e79a7",
    "ml":     "#f28e2b",
    "ruleml": "#59a14f",
    "haiku":  "#e15759",
    "sonnet": "#76b7b2",
    "base":   "#4e79a7",
    "tri":    "#e15759",
    "xgb":   "#59a14f",
}


# ── 1. Five-paradigm Set A / Set B grouped bar chart ──────────────────────

def fig_five_paradigm():
    with open(RESULTS / "llm_experiment" / "stats_summary.json") as f:
        data = json.load(f)["arms"]

    arms_order = ["Rule", "ML", "Rule+ML", "Rule+LLM (Haiku)", "Rule+LLM (Sonnet)"]
    short_labels = ["Pure\nRule", "Pure\nML", "Rule\n+ML", "Rule+LLM\n(Haiku 4.5)", "Rule+LLM\n(Sonnet 4.6)"]

    set_a = [data[a]["set_a"]["pct"] for a in arms_order]
    set_b = [data[a]["set_b"]["pct"] for a in arms_order]
    set_b_lo = [data[a]["set_b"]["pct"] - data[a]["set_b"]["ci_lo"] for a in arms_order]
    set_b_hi = [data[a]["set_b"]["ci_hi"] - data[a]["set_b"]["pct"] for a in arms_order]

    x = np.arange(len(arms_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_a = ax.bar(x - width / 2, set_a, width, label="Set A (known class)", color="#4e79a7", alpha=0.85)
    bars_b = ax.bar(x + width / 2, set_b, width, label="Set B (masked class)", color="#e15759", alpha=0.85,
                    yerr=[set_b_lo, set_b_hi], capsize=4, error_kw={"linewidth": 1.2})

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Five-Paradigm Experiment: Set A vs Set B Accuracy\n"
                 "NFCRM §6.9 Risk-Level Reasoning (Task ii)", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left")
    ax.axhline(100, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    # Annotate Set B bars
    for bar, val in zip(bars_b, set_b):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    # H6/H7 significance markers
    sig_x = [x[3] + width / 2, x[4] + width / 2]
    sig_vals = [set_b[3], set_b[4]]
    for sx, sv, label in zip(sig_x, sig_vals, ["**", "***"]):
        ax.text(sx, sv + 8, label, ha="center", va="bottom", fontsize=11, color="#333")

    ax.text(0.98, 0.02, "** p=0.0020  *** p=0.0005\nError bars: 95% CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555")

    out = FIGURES / "fig_five_paradigm_setb.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")


# ── 2. CIC-IDS2018 class distribution (from cleaned_rf_baseline.json test split) ──

def fig_cicids_class_dist():
    with open(RESULTS / "cleaned_rf_baseline.json") as f:
        data = json.load(f)

    report = data["test"]["per_class_report"]
    classes = ["Benign", "BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"]
    supports = [report[c]["support"] for c in classes]
    total = sum(supports)
    pcts = [s / total * 100 for s in supports]

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#ff9da7"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Bar chart
    ax = axes[0]
    bars = ax.bar(classes, supports, color=colors, alpha=0.88)
    ax.set_title("Class Distribution — CSE-CIC-IDS2018 (Test Split)")
    ax.set_ylabel("Sample Count")
    ax.set_xticklabels(classes, rotation=30, ha="right")
    for bar, s in zip(bars, supports):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{int(s)}", ha="center", va="bottom", fontsize=8)

    # Pie chart
    ax2 = axes[1]
    wedge_props = {"linewidth": 0.8, "edgecolor": "white"}
    ax2.pie(pcts, labels=classes, autopct="%1.1f%%", colors=colors,
            startangle=140, wedgeprops=wedge_props, textprops={"fontsize": 8})
    ax2.set_title("Class Proportions")

    out = FIGURES / "fig_cicids_class_dist.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")


# ── 3. NSL-KDD class distribution (from rf_baseline.json test split) ──────

def fig_nslkdd_class_dist():
    with open(RESULTS / "rf_baseline.json") as f:
        data = json.load(f)

    report = data["test"]["per_class_report"]
    classes = ["DoS", "Normal", "Probe", "R2L", "U2R"]
    supports = [report[c]["support"] for c in classes]
    total = sum(supports)
    pcts = [s / total * 100 for s in supports]

    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    ax = axes[0]
    bars = ax.bar(classes, supports, color=colors, alpha=0.88)
    ax.set_title("Class Distribution — NSL-KDD (Test Split)")
    ax.set_ylabel("Sample Count")
    for bar, s in zip(bars, supports):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{int(s)}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    wedge_props = {"linewidth": 0.8, "edgecolor": "white"}
    ax2.pie(pcts, labels=classes, autopct="%1.1f%%", colors=colors,
            startangle=140, wedgeprops=wedge_props, textprops={"fontsize": 8})
    ax2.set_title("Class Proportions")

    out = FIGURES / "fig_nslkdd_class_dist.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")


# ── 4. Per-class F1: baseline vs tri-stage (CIC-IDS2018) ──────────────────

def fig_perclass_f1():
    with open(RESULTS / "ablation_study_v2.json") as f:
        data = json.load(f)

    classes = ["Benign", "BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"]
    base_f1 = [data["baseline"]["per_class_report"][c]["f1-score"] for c in classes]
    tri_f1  = [data["tri_stage"]["per_class_report"][c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, [v * 100 for v in base_f1], width,
           label="RF Baseline", color="#4e79a7", alpha=0.88)
    ax.bar(x + width / 2, [v * 100 for v in tri_f1], width,
           label="Tri-Stage Pipeline", color="#e15759", alpha=0.88)

    ax.set_title("Per-Class F1-Score: RF Baseline vs Tri-Stage Pipeline\n(CSE-CIC-IDS2018 Test Split)")
    ax.set_ylabel("F1-Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend()
    ax.axhline(100, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    # Annotate tri-stage bars
    for xi, (bf, tf) in enumerate(zip(base_f1, tri_f1)):
        delta = (tf - bf) * 100
        color = "#2ca02c" if delta >= 0 else "#d62728"
        ax.text(xi + width / 2, tf * 100 + 1.5,
                f"{delta:+.1f}", ha="center", va="bottom", fontsize=7.5, color=color)

    ax.text(0.98, 0.02, "Values above bars: Δ = Tri-Stage − Baseline",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555")

    out = FIGURES / "fig_perclass_f1.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")


# ── 5. Ablation study summary bar chart ───────────────────────────────────

def fig_ablation_bars():
    with open(RESULTS / "ablation_study_v2.json") as f:
        raw = json.load(f)

    conditions = ["baseline", "pre_only", "post_annotate", "post_override", "tri_stage"]
    labels = ["RF Baseline\n(No Symbolic)", "Pre-Symbolic\nOnly", "Post-Annotate\nOnly",
              "Post-Override\nOnly", "Tri-Stage\n(Full Pipeline)"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#9c755f", "#e15759"]

    # Pull accuracy from summary_metrics if available, else from condition dict
    acc_vals = []
    f1_vals  = []
    for cond in conditions:
        if cond in raw:
            acc_vals.append(raw[cond]["accuracy"] * 100)
            f1_vals.append(raw[cond]["f1_macro"] * 100)
        else:
            # Some conditions stored in summary_metrics
            sm = raw.get("summary_metrics", {})
            acc_vals.append(sm.get("accuracy", {}).get(cond, 0) * 100)
            f1_vals.append(sm.get("f1_macro", {}).get(cond, 0) * 100)

    x = np.arange(len(conditions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars_acc = ax.bar(x - width / 2, acc_vals, width, label="Accuracy", color=colors, alpha=0.88)
    bars_f1  = ax.bar(x + width / 2, f1_vals,  width, label="F1-Macro", color=colors, alpha=0.55,
                      hatch="///", edgecolor="white")

    ax.set_title("Ablation Study: Accuracy and F1-Macro by Pipeline Condition\n(CSE-CIC-IDS2018 Test Split)")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(82, 96)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    acc_patch = mpatches.Patch(color="grey", alpha=0.88, label="Accuracy (solid)")
    f1_patch  = mpatches.Patch(color="grey", alpha=0.55, hatch="///", label="F1-Macro (hatched)")
    ax.legend(handles=[acc_patch, f1_patch], loc="lower right")

    for bar, val in zip(bars_acc, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=7.5)

    out = FIGURES / "fig_ablation_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")


# ── 6. ML model comparison: RF vs XGBoost vs Tri-Stage ───────────────────

def fig_ml_comparison():
    metrics = {
        "RF Baseline\n(CIC-IDS2018)": {"acc": 87.70, "f1": 89.60},
        "XGBoost\n(CIC-IDS2018)":     {"acc": 95.90, "f1": 69.20},  # weighted F1 N/A; macro low
        "Tri-Stage\n(CIC-IDS2018)":   {"acc": 87.98, "f1": 88.79},
        "RF Baseline\n(NSL-KDD)":     {"acc": 74.56, "f1": 47.19},
    }

    labels = list(metrics.keys())
    accs   = [v["acc"] for v in metrics.values()]
    f1s    = [v["f1"]  for v in metrics.values()]
    colors = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b"]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(x - width / 2, accs, width, label="Accuracy (%)", color=colors, alpha=0.88)
    ax.bar(x + width / 2, f1s,  width, label="F1-Macro (%)", color=colors, alpha=0.55,
           hatch="///", edgecolor="white")

    ax.set_title("ML Model Comparison: Accuracy and F1-Macro\n(CIC-IDS2018 and NSL-KDD Test Splits)")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(40, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    acc_patch = mpatches.Patch(color="grey", alpha=0.88, label="Accuracy (solid)")
    f1_patch  = mpatches.Patch(color="grey", alpha=0.55, hatch="///", label="F1-Macro (hatched)")
    ax.legend(handles=[acc_patch, f1_patch])

    for xi, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(xi - width / 2, a + 0.5, f"{a:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, f  + 0.5, f"{f:.2f}%", ha="center", va="bottom", fontsize=8)

    ax.text(0.98, 0.02, "XGBoost F1-Macro low due to class imbalance (5 classes, WebAttack excluded)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555")

    out = FIGURES / "fig_ml_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating thesis figures...")
    fig_five_paradigm()
    fig_cicids_class_dist()
    fig_nslkdd_class_dist()
    fig_perclass_f1()
    fig_ablation_bars()
    fig_ml_comparison()
    print(f"\nAll figures saved to {FIGURES}")
