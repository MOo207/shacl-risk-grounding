"""Generate the two thesis confusion-matrix figures from persisted, real
per-sample predictions (not the earlier illustrative reconstructions).

Sources
-------
- results/cleaned_rf_baseline.json   -> CIC-IDS2018 test confusion matrix
  (RF run, N=2,122, accuracy 86.90%)
- results/nslkdd_xgb_baseline.json   -> NSL-KDD KDDTest+ confusion matrix
  (XGBoost run, N=22,184, accuracy 75.43%)

Both JSON files store a `test.confusion_matrix` field with exact per-sample
counts (rows = true label, columns = predicted label, in the order given by
the model's class list / `label_classes`). This script renders each matrix
as a colorblind-safe sequential heatmap (single blue hue, light->dark, per
the dataviz skill's palette) with annotated cell counts, row-normalised
shading so that the diagonal dominance and confusion patterns are visible
independent of per-class support imbalance, and saves the output directly
over the existing figure filenames referenced by the thesis
(`thesis/figures/confusion_cicids.png`, `thesis/figures/confusion_nslkdd.png`).

Usage:
    python scripts/generate_real_confusion_matrices.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "thesis" / "figures"

# --- dataviz skill palette: sequential blue ramp, light -> dark, plus ink/chrome ---
SEQUENTIAL_BLUE = [
    "#fcfcfb",  # surface (0 count)
    "#cde2fb",  # step 100
    "#9ec5f4",  # step 200
    "#5598e7",  # step 350
    "#2a78d6",  # step 450
    "#1c5cab",  # step 550
    "#104281",  # step 650
    "#0d366b",  # step 700
]
CMAP = LinearSegmentedColormap.from_list("dataviz_sequential_blue", SEQUENTIAL_BLUE, N=256)

PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "axes.edgecolor": GRIDLINE,
        "text.color": PRIMARY_INK,
        "axes.labelcolor": PRIMARY_INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def render_confusion_matrix(matrix, labels, title, out_path, accuracy, n_samples):
    """Render a single confusion-matrix heatmap in the dataviz house style.

    Cell shading is row-normalised (recall per true class) so minority
    classes with small raw counts are still visually legible; the raw
    integer count is always the annotation shown in each cell.
    """
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalised = matrix / row_sums

    n = len(labels)
    fig_size = max(5.5, 1.15 * n + 2.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.92))

    im = ax.imshow(normalised, cmap=CMAP, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted class", fontsize=11, labelpad=10)
    ax.set_ylabel("True class", fontsize=11, labelpad=10)

    # gridlines between cells (hairline, recessive)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=GRIDLINE, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # annotate raw counts; ink color flips for readability on dark cells
    for i in range(n):
        for j in range(n):
            count = int(matrix[i, j])
            shade = normalised[i, j]
            text_color = "#ffffff" if shade > 0.55 else PRIMARY_INK
            ax.text(
                j,
                i,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=9.5,
                color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised proportion (recall)", fontsize=9.5, color=SECONDARY_INK)
    cbar.ax.tick_params(labelsize=8.5, colors=MUTED_INK)
    cbar.outline.set_visible(False)

    ax.set_title(
        f"{title}\naccuracy {accuracy * 100:.2f}% (N={n_samples:,})",
        fontsize=12,
        color=PRIMARY_INK,
        pad=14,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} (accuracy {accuracy * 100:.2f}%, N={n_samples})")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- CIC-IDS2018: RF baseline ---
    cic_path = RESULTS_DIR / "cleaned_rf_baseline.json"
    cic = json.loads(cic_path.read_text())
    cic_test = cic["test"]
    cic_labels = list(cic_test["per_class_report"].keys())
    cic_labels = [c for c in cic_labels if c not in ("accuracy", "macro avg", "weighted avg")]
    render_confusion_matrix(
        matrix=cic_test["confusion_matrix"],
        labels=cic_labels,
        title="CIC-IDS2018 test set confusion matrix (Random Forest baseline)",
        out_path=FIGURES_DIR / "confusion_cicids.png",
        accuracy=cic_test["accuracy"],
        n_samples=cic_test["samples"],
    )

    # --- NSL-KDD: XGBoost baseline ---
    nsl_path = RESULTS_DIR / "nslkdd_xgb_baseline.json"
    nsl = json.loads(nsl_path.read_text())
    nsl_test = nsl["test"]
    nsl_labels = nsl.get("label_classes") or list(nsl_test["per_class_report"].keys())
    render_confusion_matrix(
        matrix=nsl_test["confusion_matrix"],
        labels=nsl_labels,
        title="NSL-KDD KDDTest+ confusion matrix (XGBoost baseline)",
        out_path=FIGURES_DIR / "confusion_nslkdd.png",
        accuracy=nsl_test["accuracy"],
        n_samples=nsl_test["samples"],
    )

    print(
        "\nSource verification:\n"
        f"  CIC RF accuracy   = {cic_test['accuracy']:.6f} (expect ~0.8690 / 86.90%)\n"
        f"  NSL-KDD XGB accuracy = {nsl_test['accuracy']:.6f} (expect ~0.7543 / 75.43%)"
    )


if __name__ == "__main__":
    main()
