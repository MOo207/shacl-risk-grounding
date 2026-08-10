#!/usr/bin/env python3
"""Generate the two empirical figures for the NeSy cybersecurity-risk paper.

Outputs
-------
fig_semantic_validation.pdf / .png
    Two-panel summary of primary GRC artefact conformance and
    JSON-Schema-versus-SHACL validation outcomes.

fig_inference_tradeoffs.pdf / .png
    Four-panel summary of CIC-IDS2018 and NSL-KDD safety/accuracy
    trade-offs and the NSL-KDD CVE-channel decomposition.

The script is self-contained and uses the verified values reported in the
current manuscript. To decouple plotting from manuscript revisions, pass a
JSON override with --metrics. Use --write-template to create the expected
JSON structure in the repository.

Examples
--------
python scripts/generate_nesy_figures.py
python scripts/generate_nesy_figures.py --out-dir figures
python scripts/generate_nesy_figures.py --metrics results/figure_metrics.json
python scripts/generate_nesy_figures.py --write-template results/figure_metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# ---------------------------------------------------------------------------
# Verified manuscript values
# ---------------------------------------------------------------------------
# These defaults deliberately mirror the current paper. A repository-side
# results/figure_metrics.json can override them without editing this script.
DEFAULT_METRICS: dict[str, Any] = {
    "figure3": {
        "grc_corpus": {
            "RiskCase": {"conforming": 150, "non_conforming": 30},
            "AuditLog": {"conforming": 180, "non_conforming": 0},
            "Control": {"conforming": 4, "non_conforming": 0},
        },
        "control_recommendation": {
            "total": 99,
            "JSON Schema": {"accepted": 99, "rejected": 0},
            "SHACL": {"accepted": 71, "rejected": 28},
        },
    },
    "figure4": {
        "cic_tradeoff": [
            {
                "label": "XGBoost",
                "exact_accuracy": 95.0,
                "under_escalation": 3.33,
                "marker": "o",
                "annotation_offset": [-25, -12],
            },
            {
                "label": "Haiku + CVE",
                "exact_accuracy": 85.0,
                "under_escalation": 0.83,
                "marker": "s",
                "annotation_offset": [0.5, 9.5],
            },
            {
                "label": "Sonnet + CVE",
                "exact_accuracy": 89.1667,
                "under_escalation": 1.67,
                "marker": "^",
                "annotation_offset": [5, 7],
            },
            {
                "label": "Haiku, CVE removed",
                "exact_accuracy": 82.5,
                "under_escalation": 3.33,
                "marker": "D",
                "annotation_offset": [1, 8],
            },
        ],
        "nsl_tradeoff": [
            {
                "label": "XGBoost",
                "exact_accuracy": 49.0,
                "under_escalation": 48.0,
                "marker": "o",
                "annotation_offset": [10, -1],
            },
            {
                "label": "Haiku + CVE",
                "exact_accuracy": 65.0,
                "under_escalation": 10.0,
                "marker": "s",
                "annotation_offset": [5, -13],
            },
            {
                "label": "Sonnet + CVE",
                "exact_accuracy": 63.0,
                "under_escalation": 28.0,
                "marker": "^",
                "annotation_offset": [-55, 5],
            },
            {
                "label": "Haiku, CVE removed",
                "exact_accuracy": 49.0,
                "under_escalation": 36.0,
                "marker": "D",
                "annotation_offset": [5, 7],
            },
            {
                "label": "XGBoost + CVE-presence",
                "exact_accuracy": 70.0,
                "under_escalation": 27.0,
                "marker": "P",
                "annotation_offset": [-56, -15],
            },
        ],
        "nsl_cve_decomposition": [
            {
                "label": "XGBoost",
                "under_escalation": 48.0,
                "severe_recall": 42.9,
            },
            {
                "label": "Haiku\nCVE removed",
                "under_escalation": 36.0,
                "severe_recall": 73.5,
            },
            {
                "label": "XGBoost +\nCVE-presence",
                "under_escalation": 27.0,
                "severe_recall": 77.6,
            },
            {
                "label": "Haiku +\nclass-paired CVE",
                "under_escalation": 10.0,
                "severe_recall": 93.9,
            },
            {
                "label": "Haiku +\npermuted CVE",
                "under_escalation": 4.0,
                "severe_recall": 98.0,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Input, validation, and formatting helpers
# ---------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge an override mapping into a copied base mapping."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_metrics(path: Path | None) -> tuple[dict[str, Any], str]:
    """Load optional metrics JSON and merge it over the verified defaults."""
    if path is None:
        return copy.deepcopy(DEFAULT_METRICS), "verified manuscript defaults"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    try:
        override = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(override, Mapping):
        raise TypeError("The metrics JSON must contain an object at the top level.")
    return deep_merge(DEFAULT_METRICS, override), str(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_percentage(value: Any, name: str) -> None:
    require(isinstance(value, (int, float)), f"{name} must be numeric.")
    require(math.isfinite(float(value)), f"{name} must be finite.")
    require(0.0 <= float(value) <= 100.0, f"{name} must lie in [0, 100].")


def validate_metrics(metrics: Mapping[str, Any]) -> None:
    """Fail early if totals or percentages do not reproduce the paper."""
    fig3 = metrics["figure3"]

    corpus = fig3["grc_corpus"]
    conforming = sum(v["conforming"] for v in corpus.values())
    non_conforming = sum(v["non_conforming"] for v in corpus.values())
    require(conforming == 334, "Primary GRC conforming count must be 334.")
    require(non_conforming == 30, "Primary GRC non-conforming count must be 30.")
    require(conforming + non_conforming == 364, "Primary GRC corpus must total 364.")

    recommendation = fig3["control_recommendation"]
    for validator in ("JSON Schema", "SHACL"):
        counts = recommendation[validator]
        require(
            counts["accepted"] + counts["rejected"] == recommendation["total"],
            f"{validator} accepted/rejected counts must sum to the total.",
        )
    require(recommendation["total"] == 99, "ControlRecommendation total must be 99.")
    require(
        recommendation["SHACL"]["rejected"] == 28,
        "Current manuscript reports 28 SHACL rejections.",
    )

    fig4 = metrics["figure4"]
    for group_name in ("cic_tradeoff", "nsl_tradeoff"):
        labels: set[str] = set()
        for row in fig4[group_name]:
            require(
                row["label"] not in labels,
                f"Duplicate label in {group_name}: {row['label']}",
            )
            labels.add(row["label"])
            validate_percentage(
                row["exact_accuracy"],
                f"{group_name}.{row['label']}.exact_accuracy",
            )
            validate_percentage(
                row["under_escalation"],
                f"{group_name}.{row['label']}.under_escalation",
            )

    for row in fig4["nsl_cve_decomposition"]:
        validate_percentage(
            row["under_escalation"],
            f"{row['label']}.under_escalation",
        )
        validate_percentage(
            row["severe_recall"],
            f"{row['label']}.severe_recall",
        )

def configure_matplotlib() -> None:
    """Set publication-friendly dimensions without requiring seaborn."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="center", fontweight="bold", pad=12)


def add_bar_labels(
    ax: plt.Axes,
    bars: Sequence[Any],
    *,
    suffix: str = "",
    decimals: int = 0,
    horizontal: bool = False,
) -> None:
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        if value == 0:
            continue
        label = f"{value:.{decimals}f}{suffix}"
        if horizontal:
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
            )
        else:
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


# ---------------------------------------------------------------------------
# Figure 3: Semantic-validation outcomes
# ---------------------------------------------------------------------------
def generate_semantic_validation(metrics: Mapping[str, Any], out_dir: Path) -> None:
    """Generate the compact two-panel knowledge-layer validation figure."""
    data = metrics["figure3"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)

    # (a) Primary GRC artefact conformance
    ax = axes[0]
    artifact_types = list(data["grc_corpus"].keys())
    conforming = np.array(
        [data["grc_corpus"][name]["conforming"] for name in artifact_types]
    )
    non_conforming = np.array(
        [data["grc_corpus"][name]["non_conforming"] for name in artifact_types]
    )
    y = np.arange(len(artifact_types))

    Set2 = cm.Set2.colors
    ax.barh(y, conforming, label="Conforming", color=Set2[0])
    ax.barh(
        y,
        non_conforming,
        left=conforming,
        label="Non-conforming",
        color=Set2[1],
    )

    ax.set_yticks(y, artifact_types)
    ax.invert_yaxis()
    ax.set_xlabel("Artefact count",labelpad=8)
    ax.set_xlim(0, max(conforming + non_conforming) * 1.10)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    panel_title(ax, "a", "Primary GRC artefact conformance")

    for idx, (ok, bad) in enumerate(zip(conforming, non_conforming)):
        if ok:
            ax.text(ok / 2, idx, str(int(ok)), ha="center", va="center", fontsize=7)
        if bad:
            ax.text(
                ok + bad / 2,
                idx,
                str(int(bad)),
                ha="center",
                va="center",
                fontsize=7,
            )

    # (b) JSON Schema versus SHACL
    ax = axes[1]
    recommendation = data["control_recommendation"]
    validators = ["JSON Schema", "SHACL"]
    accepted = np.array(
        [recommendation[name]["accepted"] for name in validators]
    )
    rejected = np.array(
        [recommendation[name]["rejected"] for name in validators]
    )
    y = np.arange(len(validators))

    
    Set2 = cm.Set2.colors
    ax.barh(y, accepted, label="Accepted", color=Set2[0])
    ax.barh(
        y,
        rejected,
        left=accepted,
        label="Rejected",
        color=Set2[1],
    )

    ax.set_yticks(y, validators)
    ax.invert_yaxis()
    ax.set_xlim(0, recommendation["total"])
    ax.set_xlabel("ControlRecommendation outputs",labelpad=8)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    panel_title(ax, "b", "Structural and semantic validation")

    for idx, (ok, bad) in enumerate(zip(accepted, rejected)):
        if ok:
            ax.text(ok / 2, idx, str(int(ok)), ha="center", va="center", fontsize=7)
        if bad:
            ax.text(
                ok + bad / 2,
                idx,
                str(int(bad)),
                ha="center",
                va="center",
                fontsize=7,
            )

    save_figure(fig, out_dir, "fig_semantic_validation")


# ---------------------------------------------------------------------------
# Figure 4: Predictive trade-offs and CVE-context effects
# ---------------------------------------------------------------------------
def plot_tradeoff_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    letter: str,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for row in rows:
        ax.scatter(
            row["exact_accuracy"],
            row["under_escalation"],
            marker=row.get("marker", "o"),
            s=48,
            linewidths=1.0,
        )
        offset = row.get("annotation_offset", [4, 4])
        ax.annotate(
            row["label"],
            (row["exact_accuracy"], row["under_escalation"]),
            xytext=tuple(offset),
            textcoords="offset points",
            fontsize=6.8,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Exact accuracy (%)")
    ax.set_ylabel("Under-escalation (%)")
    ax.grid(alpha=0.25)
    panel_title(ax, letter, title)


def generate_inference_tradeoffs(metrics: Mapping[str, Any], out_dir: Path) -> None:
    data = metrics["figure4"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.35), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.2, w_pad=0.2)

    plot_tradeoff_panel(
        axes[0, 0],
        data["cic_tradeoff"],
        "a",
        "CIC-IDS2018 safety–accuracy trade-off",
        (80, 97),
        (0, 4.6),
    )
    axes[0, 0].text(
        0.02,
        0.04,
        "Primary N=120 risk-level evaluation",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
    )

    plot_tradeoff_panel(
        axes[0, 1],
        data["nsl_tradeoff"],
        "b",
        "NSL-KDD safety–accuracy trade-off",
        (45, 73),
        (0, 53),
    )
    axes[0, 1].text(
        0.02,
        0.04,
        "Balanced N=100 stress sample",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
    )

    decomposition = data["nsl_cve_decomposition"]
    labels = [row["label"].replace("\n", " ") for row in decomposition]
    y = np.arange(len(labels))
    hatch_patterns = ["", "//", "..", "xx", "\\\\"]

    # (c) Under-escalation decomposition
    ax = axes[1, 0]
    under = [row["under_escalation"] for row in decomposition]
    colors = plt.cm.tab20(np.linspace(0, 1, len(under)))
    bars = ax.barh(y, under, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Under-escalation (%)")
    ax.set_xlim(0, 55)
    ax.grid(axis="x", alpha=0.25)
    add_bar_labels(ax, bars, suffix="%", decimals=1, horizontal=True)
    panel_title(ax, "c", "NSL-KDD CVE-channel decomposition")

    # (d) Severe-case recall under the same conditions
    ax = axes[1, 1]
    recall = [row["severe_recall"] for row in decomposition]
    bars = ax.barh(y, recall, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Severe-case recall (%)")
    ax.set_xlim(0, 108)
    ax.grid(axis="x", alpha=0.25)
    add_bar_labels(ax, bars, suffix="%", decimals=1, horizontal=True)
    panel_title(ax, "d", "Severe-case recall by CVE condition")

    save_figure(fig, out_dir, "fig_inference_tradeoffs")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figures 3 and 4 for the knowledge-grounded cybersecurity-risk paper."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory (default: figures).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional JSON file whose values override the verified manuscript defaults.",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the current metrics structure to PATH and exit.",
    )
    parser.add_argument(
        "--only",
        choices=("figure3", "figure4", "all"),
        default="all",
        help="Generate one figure or both (default: all).",
    )
    return parser.parse_args(argv)


def print_metric_summary(metrics: Mapping[str, Any], source: str) -> None:
    fig3 = metrics["figure3"]
    fig4 = metrics["figure4"]
    corpus = fig3["grc_corpus"]
    total_ok = sum(v["conforming"] for v in corpus.values())
    total_bad = sum(v["non_conforming"] for v in corpus.values())

    print("=" * 72)
    print("FIGURE METRICS")
    print("=" * 72)
    print(f"Source: {source}")
    print(f"Primary GRC corpus: {total_ok} conforming, {total_bad} non-conforming")
    print(
        "ControlRecommendation: JSON Schema "
        f"{fig3['control_recommendation']['JSON Schema']['accepted']}/99 accepted; "
        "SHACL "
        f"{fig3['control_recommendation']['SHACL']['accepted']}/99 accepted"
    )
    print("CIC points:")
    for row in fig4["cic_tradeoff"]:
        print(
            f"  {row['label']}: exact={row['exact_accuracy']:.2f}%, "
            f"under={row['under_escalation']:.2f}%"
        )
    print("NSL-KDD decomposition:")
    for row in fig4["nsl_cve_decomposition"]:
        print(
            f"  {row['label'].replace(chr(10), ' ')}: "
            f"under={row['under_escalation']:.1f}%, "
            f"severe recall={row['severe_recall']:.1f}%"
        )
    print("=" * 72)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.write_template is not None:
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(
            json.dumps(DEFAULT_METRICS, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote metrics template: {args.write_template}")
        return 0

    try:
        metrics, source = load_metrics(args.metrics)
        validate_metrics(metrics)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    configure_matplotlib()
    print_metric_summary(metrics, source)

    if args.only in ("figure3", "all"):
        generate_semantic_validation(metrics, args.out_dir)
    if args.only in ("figure4", "all"):
        generate_inference_tradeoffs(metrics, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
