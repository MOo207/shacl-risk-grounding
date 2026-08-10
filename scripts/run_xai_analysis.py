"""
XAI Analysis — SHAP on RF Model (T-A6, Layer 2)
================================================
Loads the fitted RF model saved by scripts/run_rf_baseline.py,
computes SHAP TreeExplainer values on a test sample,
and saves:
  results/shap_values.npy         — raw SHAP values array
  results/shap_top20_features.json — top-20 feature names + mean |SHAP|
  thesis/figures/fig-shap-beeswarm.png — beeswarm plot

Requires: pip install shap matplotlib

Usage:
    python scripts/run_xai_analysis.py
    python scripts/run_xai_analysis.py --model results/rf_model.joblib
    python scripts/run_xai_analysis.py --max-shap-rows 2000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Run: pip install joblib")
    sys.exit(1)


def run_xai_analysis(model_path: Path, data_path: Path,
                     max_shap_rows: int, output_dir: Path):
    # ---- Load model --------------------------------------------------------
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("  Run scripts/run_rf_baseline.py first (T-A2)")
        sys.exit(1)

    print(f"[T-A6] Loading RF model from {model_path} ...")
    artifact = joblib.load(model_path)
    clf = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # ---- Load data ---------------------------------------------------------
    print(f"[T-A6] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP", "Destination IP",
               "Source Port", "Destination Port", "Protocol"}
    X = df[[c for c in df.columns if c in feature_cols]].fillna(0).replace([np.inf, -np.inf], 0)
    # Align columns to training order
    missing = [c for c in feature_cols if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[feature_cols]

    # Use test split (last 15%)
    n = len(X)
    X_test = X.iloc[int(n * 0.85):]
    y_test = df["Label"].iloc[int(n * 0.85):]

    # Cap rows for SHAP speed
    if len(X_test) > max_shap_rows:
        X_sample = X_test.sample(max_shap_rows, random_state=42)
        y_sample = y_test.loc[X_sample.index]
        print(f"[T-A6] Sampling {max_shap_rows:,} rows for SHAP (full test: {len(X_test):,})")
    else:
        X_sample = X_test
        y_sample = y_test

    # ---- SHAP TreeExplainer ------------------------------------------------
    print("[T-A6] Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # shap_values is list[n_classes] (old shap), 3-D ndarray (rows, features,
    # classes) (shap >= 0.45), or 2-D ndarray (binary/regression)
    if isinstance(shap_values, list):
        # Multi-class: average abs SHAP across classes
        shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif getattr(shap_values, "ndim", 2) == 3:
        shap_abs_mean = np.abs(shap_values).mean(axis=2)
    else:
        shap_abs_mean = np.abs(shap_values)

    # ---- Save raw SHAP values ----------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "shap_values.npy"
    np.save(npy_path, shap_abs_mean)
    print(f"[T-A6] Saved shap_values.npy ({shap_abs_mean.shape})")

    # ---- Top-20 features ---------------------------------------------------
    mean_shap_per_feature = shap_abs_mean.mean(axis=0)
    top20_idx = np.argsort(mean_shap_per_feature)[::-1][:20]
    top20 = [
        {"rank": int(i + 1), "feature": feature_cols[idx],
         "mean_abs_shap": float(mean_shap_per_feature[idx])}
        for i, idx in enumerate(top20_idx)
    ]
    top20_path = output_dir / "shap_top20_features.json"
    with open(top20_path, "w") as f:
        json.dump(top20, f, indent=2)
    print(f"[T-A6] Top-3 features: {[t['feature'] for t in top20[:3]]}")
    print(f"[T-A6] Saved {top20_path}")

    # ---- Beeswarm plot -----------------------------------------------------
    figures_dir = ROOT / "thesis" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "fig-shap-beeswarm.png"

    print("[T-A6] Generating beeswarm plot ...")
    # Use the class with most support for readable beeswarm
    if isinstance(shap_values, list):
        # Pick class 0 (Benign) SHAP values — most rows, clearest plot
        sv_plot = shap_values[0]
    elif getattr(shap_values, "ndim", 2) == 3:
        sv_plot = shap_values[:, :, 0]
    else:
        sv_plot = shap_values

    # Only top 20 features for plot
    sv_top20 = sv_plot[:, top20_idx]
    X_top20  = X_sample.iloc[:, top20_idx]
    X_top20.columns = [feature_cols[i] for i in top20_idx]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv_top20, X_top20, plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Feature Importance — Random Forest on CIC-IDS2018\n"
              f"(n={len(X_sample):,} flows, top-20 features)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[T-A6] Saved {fig_path}")

    print("\n[T-A6] XAI analysis complete.")
    print(f"  shap_values.npy:         {npy_path}")
    print(f"  shap_top20_features.json: {top20_path}")
    print(f"  fig-shap-beeswarm.png:    {fig_path}")

    return top20


def main():
    parser = argparse.ArgumentParser(description="SHAP XAI analysis on RF model (T-A6)")
    parser.add_argument("--model", default="results/rf_model.joblib",
                        help="Path to fitted RF model (from run_rf_baseline.py)")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--max-shap-rows", type=int, default=3000,
                        help="Max rows for SHAP computation (default: 3000)")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_xai_analysis(
        model_path=Path(args.model),
        data_path=Path(args.data),
        max_shap_rows=args.max_shap_rows,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
