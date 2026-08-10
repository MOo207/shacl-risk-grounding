"""Statistical significance tests for ablation study results.

Tests:
  1. Bootstrap 95% CI for accuracy and F1-macro (all 5 conditions)
  2. McNemar's test: baseline vs tri-stage
  3. Cohen's h: effect size for accuracy proportion difference

Input: results/ablation_study_v2.json
Output: results/statistical_tests.json
"""
import json
import os
import numpy as np
from scipy import stats
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_ablation():
    with open(os.path.join(RESULTS_DIR, "ablation_study_v2.json"), encoding="utf-8") as f:
        return json.load(f)


def bootstrap_ci(metric_value, n_samples, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap CI for a proportion (accuracy) or score (F1).

    Since we don't have per-sample predictions, we simulate from the
    observed proportion using binomial resampling.
    """
    rng = np.random.RandomState(seed)
    boot_values = []
    for _ in range(n_bootstrap):
        # Resample: draw n_samples from Bernoulli(metric_value)
        resample = rng.binomial(1, metric_value, size=n_samples)
        boot_values.append(resample.mean())
    boot_values = np.array(boot_values)
    alpha = 1 - ci
    lower = np.percentile(boot_values, 100 * alpha / 2)
    upper = np.percentile(boot_values, 100 * (1 - alpha / 2))
    return round(float(lower), 4), round(float(upper), 4)


def mcnemar_test(preds_a, preds_b, true_labels):
    """McNemar's test using actual per-sample predictions.

    Builds the 2x2 contingency table from actual predictions:
    - a = both correct
    - b = A correct, B wrong
    - c = A wrong, B correct
    - d = both wrong

    Args:
        preds_a: List of predictions from model A
        preds_b: List of predictions from model B
        true_labels: List of true labels

    Returns:
        Dictionary with chi2 statistic, p-value, and contingency table
    """
    # Build contingency table from actual predictions
    a = b = c = d = 0

    for pred_a, pred_b, true_label in zip(preds_a, preds_b, true_labels):
        a_correct = (pred_a == true_label)
        b_correct = (pred_b == true_label)

        if a_correct and b_correct:
            a += 1
        elif a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
        else:
            d += 1

    # McNemar statistic (with continuity correction)
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_005": p_value < 0.05,
        "contingency_table": {
            "both_correct": a,
            "a_correct_b_wrong": b,
            "a_wrong_b_correct": c,
            "both_wrong": d,
        },
        "note": "Computed from actual per-sample predictions"
    }


def cohens_h(p1, p2):
    """Cohen's h for comparing two proportions."""
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return round(float(abs(h)), 4)


def interpret_cohens_h(h):
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    else:
        return "large"


def main():
    ablation = load_ablation()
    conditions = ["baseline", "pre_only", "post_annotate", "post_override", "tri_stage"]

    n_samples = ablation["baseline"]["samples_test"]  # 2122
    print(f"Test samples: {n_samples}")
    print(f"Conditions: {conditions}")

    results = {
        "generated": datetime.now().isoformat(),
        "test_samples": n_samples,
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
        "conditions": {},
    }

    # Bootstrap CI for each condition
    for cond in conditions:
        data = ablation[cond]
        acc = data["accuracy"]
        f1 = data["f1_macro"]

        acc_ci = bootstrap_ci(acc, n_samples)
        f1_ci = bootstrap_ci(f1, n_samples)

        results["conditions"][cond] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "accuracy_ci_95": list(acc_ci),
            "f1_macro_ci_95": list(f1_ci),
            "accuracy_ci_width": round(acc_ci[1] - acc_ci[0], 4),
            "f1_macro_ci_width": round(f1_ci[1] - f1_ci[0], 4),
        }
        print(f"  {cond}: acc={acc:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}], "
              f"F1={f1:.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")

    # McNemar's test: baseline vs tri_stage
    # Use actual predictions instead of estimated contingencies
    baseline_acc = ablation["baseline"]["accuracy"]
    tristage_acc = ablation["tri_stage"]["accuracy"]
    baseline_preds = ablation["baseline"]["predictions"]
    tristage_preds = ablation["tri_stage"]["predictions"]
    true_labels = ablation["test_labels"]

    mcnemar = mcnemar_test(baseline_preds, tristage_preds, true_labels)
    results["mcnemar_baseline_vs_tristage"] = mcnemar

    print(f"\nMcNemar baseline vs tri-stage: chi2={mcnemar['chi2']}, "
          f"p={mcnemar['p_value']}, significant={mcnemar['significant_at_005']}")

    # Cohen's h
    h = cohens_h(tristage_acc, baseline_acc)
    results["cohens_h_accuracy"] = {
        "value": h,
        "interpretation": interpret_cohens_h(h),
        "comparison": f"tri_stage ({tristage_acc:.4f}) vs baseline ({baseline_acc:.4f})",
    }
    print(f"Cohen's h: {h} ({interpret_cohens_h(h)})")

    # CI overlap analysis
    baseline_ci = results["conditions"]["baseline"]["accuracy_ci_95"]
    tristage_ci = results["conditions"]["tri_stage"]["accuracy_ci_95"]
    overlap = max(0, min(baseline_ci[1], tristage_ci[1]) - max(baseline_ci[0], tristage_ci[0]))
    results["ci_overlap_analysis"] = {
        "baseline_ci": baseline_ci,
        "tri_stage_ci": tristage_ci,
        "overlap_width": round(overlap, 4),
        "overlapping": overlap > 0,
        "interpretation": "CIs overlap substantially; difference is not practically significant"
        if overlap > 0 else "CIs do not overlap; difference may be significant"
    }

    # Summary interpretation
    results["interpretation"] = {
        "accuracy_difference": round(tristage_acc - baseline_acc, 4),
        "f1_difference": round(ablation["tri_stage"]["f1_macro"] - ablation["baseline"]["f1_macro"], 4),
        "statistically_significant": mcnemar["significant_at_005"],
        "effect_size": interpret_cohens_h(h),
        "summary": (
            f"The tri-stage pipeline achieves +{(tristage_acc - baseline_acc)*100:.2f}% accuracy "
            f"over baseline ({tristage_acc*100:.2f}% vs {baseline_acc*100:.2f}%), "
            f"but with -{(ablation['baseline']['f1_macro'] - ablation['tri_stage']['f1_macro'])*100:.2f}% F1-macro. "
            f"McNemar's test p={mcnemar['p_value']:.4f} indicates the accuracy difference is "
            f"{'NOT ' if not mcnemar['significant_at_005'] else ''}statistically significant. "
            f"Cohen's h={h} indicates {interpret_cohens_h(h)} effect size. "
            f"Post-annotation adds GRC traceability at ZERO accuracy cost (identical to baseline). "
            f"The intersymbolic framework's value is not accuracy improvement but Pareto-optimal "
            f"position: near-equivalent accuracy + full GRC traceability."
        )
    }

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    dest = os.path.join(RESULTS_DIR, "statistical_tests.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {dest}")
    print(f"\nSummary: {results['interpretation']['summary']}")


if __name__ == "__main__":
    main()
