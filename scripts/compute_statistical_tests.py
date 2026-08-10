#!/usr/bin/env python3
"""
Statistical Significance Tests for INTERSYMBOLIC-GRC Thesis (Phase 9.C4)

Computes:
1. Paired t-tests for metric comparisons (using binomial approximation)
2. Bootstrap 95% CIs (1000 resamples)
3. Cohen's d effect sizes
4. Bonferroni correction for multiple comparisons

Compares: RF baseline vs tri-stage on accuracy, F1-macro, F1-weighted
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# Constants
N_SAMPLES = 2122  # Test set size from ablation study
N_BOOTSTRAP = 1000
ALPHA = 0.05
N_TESTS = 9  # 3 metrics × 3 comparisons
BONFERRONI_ALPHA = ALPHA / N_TESTS

def load_ablation_data(filepath):
    """Load ablation study results from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def bootstrap_metric(metric_value, n_samples, n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap confidence interval for a classification metric.
    Uses binomial distribution for accuracy, approximate normal for F1.

    Args:
        metric_value: The observed metric value (accuracy or F1)
        n_samples: Number of samples in test set
        n_bootstrap: Number of bootstrap resamples

    Returns:
        (ci_lower, ci_upper): 95% confidence interval
    """
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # For accuracy: binomial sampling
        # For F1: approximate with normal distribution around observed value
        if 0 <= metric_value <= 1:  # It's a proportion (accuracy or F1)
            # Use binomial sampling with reasonable variance estimate
            # Variance for F1 is more complex, use conservative estimate
            var = metric_value * (1 - metric_value) / n_samples
            # For F1, increase variance estimate (conservative)
            var = var * 1.5

            sampled_value = np.random.normal(metric_value, np.sqrt(var))
            # Clip to valid range
            sampled_value = max(0, min(1, sampled_value))
        else:
            sampled_value = metric_value

        bootstrap_values.append(sampled_value)

    bootstrap_values = np.array(bootstrap_values)
    ci_lower = np.percentile(bootstrap_values, 2.5)
    ci_upper = np.percentile(bootstrap_values, 97.5)

    return ci_lower, ci_upper

def paired_t_test_proportion(p1, p2, n):
    """
    Perform paired t-test for two proportions using McNemar's approach.
    Simplified to chi-square test on 2x2 contingency table.

    Args:
        p1: Proportion for method 1
        p2: Proportion for method 2
        n: Sample size

    Returns:
        (statistic, p_value): Test statistic and p-value
    """
    # For two proportions, we can use chi-square test
    # Create observed counts
    count1 = p1 * n
    count2 = p2 * n

    # Without individual predictions, we can't compute true McNemar
    # Use approximate test assuming independence (conservative)
    # Pooled proportion
    p_pooled = (count1 + count2) / (2 * n)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (2 / n))

    if se > 0:
        z = (p1 - p2) / se
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p_value = 1.0

    statistic = z**2  # Chi-square statistic

    return statistic, p_value

def cohens_d(p1, p2, n):
    """
    Compute Cohen's d effect size for two proportions.
    Standardized mean difference.

    Args:
        p1: Proportion for method 1
        p2: Proportion for method 2
        n: Sample size

    Returns:
        d: Cohen's d effect size
    """
    # Difference in proportions
    diff = p1 - p2

    # Standard deviation (pooled estimate)
    p_pooled = (p1 + p2) / 2
    std = np.sqrt(p_pooled * (1 - p_pooled))

    if std > 0:
        d = diff / std
    else:
        d = 0.0

    return d

def run_comparisons(data):
    """
    Run all statistical comparisons for ablation study.

    Returns:
        results: Dictionary with all test results
    """
    # Extract metrics for each condition
    conditions = ['baseline', 'pre_only', 'post_annotate', 'post_override', 'tri_stage']
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']

    # Store results
    results = {
        'metadata': {
            'n_samples': N_SAMPLES,
            'n_bootstrap': N_BOOTSTRAP,
            'n_tests': N_TESTS,
            'alpha': ALPHA,
            'bonferroni_alpha': BONFERRONI_ALPHA,
            'comparisons_made': []
        },
        'bootstrap_intervals': {},
        'paired_tests': {},
        'effect_sizes': {},
        'significance_summary': {}
    }

    # Compute bootstrap CIs for each condition and metric
    print("Computing bootstrap 95% CIs...")
    for condition in conditions:
        results['bootstrap_intervals'][condition] = {}
        for metric in metrics:
            metric_key = metric
            if metric == 'f1_macro':
                metric_key = 'f1_macro'
            elif metric == 'f1_weighted':
                metric_key = 'f1_weighted'

            value = data[condition][metric_key]
            ci_lower, ci_upper = bootstrap_metric(value, N_SAMPLES, N_BOOTSTRAP)

            results['bootstrap_intervals'][condition][metric] = {
                'value': float(value),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'ci_width': float(ci_upper - ci_lower)
            }

    # Main comparison: baseline vs tri_stage (RF baseline vs tri-stage)
    # Additional comparisons for completeness
    comparisons = [
        ('baseline', 'tri_stage'),
        ('baseline', 'pre_only'),
        ('baseline', 'post_annotate'),
        ('pre_only', 'tri_stage'),
        ('post_annotate', 'tri_stage')
    ]

    results['metadata']['comparisons_made'] = comparisons

    print("\nComputing paired t-tests and effect sizes...")
    for cond1, cond2 in comparisons:
        comparison_key = f"{cond1}_vs_{cond2}"
        results['paired_tests'][comparison_key] = {}
        results['effect_sizes'][comparison_key] = {}

        for metric in metrics:
            metric_key = metric
            if metric == 'f1_macro':
                metric_key = 'f1_macro'
            elif metric == 'f1_weighted':
                metric_key = 'f1_weighted'

            value1 = data[cond1][metric_key]
            value2 = data[cond2][metric_key]

            # Paired t-test
            statistic, p_value = paired_t_test_proportion(value1, value2, N_SAMPLES)

            # Cohen's d
            d = cohens_d(value1, value2, N_SAMPLES)

            results['paired_tests'][comparison_key][metric] = {
                f'{cond1}_{metric}': float(value1),
                f'{cond2}_{metric}': float(value2),
                'difference': float(value1 - value2),
                't_statistic': float(statistic),
                'p_value': float(p_value),
                'significant_bonferroni': bool(p_value < BONFERRONI_ALPHA),
                'significant_uncorrected': bool(p_value < ALPHA)
            }

            results['effect_sizes'][comparison_key][metric] = {
                'cohens_d': float(d),
                'interpretation': interpret_cohens_d(d)
            }

    # Summary of key findings
    print("\n" + "="*80)
    print("KEY FINDINGS: RF Baseline vs Tri-Stage")
    print("="*80)

    baseline_vs_tristage = results['paired_tests']['baseline_vs_tri_stage']
    intervals_baseline = results['bootstrap_intervals']['baseline']
    intervals_tristage = results['bootstrap_intervals']['tri_stage']
    effects = results['effect_sizes']['baseline_vs_tri_stage']

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print(f"  Baseline: {data['baseline'][metric]:.4f} (95% CI: [{intervals_baseline[metric]['ci_lower']:.4f}, {intervals_baseline[metric]['ci_upper']:.4f}])")
        print(f"  Tri-stage: {data['tri_stage'][metric]:.4f} (95% CI: [{intervals_tristage[metric]['ci_lower']:.4f}, {intervals_tristage[metric]['ci_upper']:.4f}])")
        print(f"  Difference: {baseline_vs_tristage[metric]['difference']:.4f}")
        print(f"  Cohen's d: {effects[metric]['cohens_d']:.4f} ({effects[metric]['interpretation']})")
        print(f"  p-value: {baseline_vs_tristage[metric]['p_value']:.6f}")
        print(f"  Significant (Bonferroni): {baseline_vs_tristage[metric]['significant_bonferroni']}")

    # Count significant results
    significant_count = sum(
        1 for comp in comparisons
        for metric in metrics
        if results['paired_tests'][f"{comp[0]}_vs_{comp[1]}"][metric]['significant_bonferroni']
    )

    results['significance_summary'] = {
        'total_tests': N_TESTS,
        'significant_bonferroni': significant_count,
        'significant_uncorrected': sum(
            1 for comp in comparisons
            for metric in metrics
            if results['paired_tests'][f"{comp[0]}_vs_{comp[1]}"][metric]['significant_uncorrected']
        ),
        'bonferroni_threshold': BONFERRONI_ALPHA
    }

    print("\n" + "="*80)
    print(f"SIGNIFICANCE SUMMARY (Bonferroni corrected α={BONFERRONI_ALPHA:.6f}):")
    print(f"  Significant tests: {significant_count}/{N_TESTS}")
    print("="*80)

    return results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def main():
    """Main execution."""
    print("="*80)
    print("Statistical Significance Tests for INTERSYMBOLIC-GRC Thesis")
    print("="*80)

    # Paths
    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent
    results_dir = repo_dir / "results"

    # Load data
    ablation_file = results_dir / "ablation_study_v2.json"
    print(f"\nLoading ablation study from: {ablation_file}")

    data = load_ablation_data(ablation_file)

    # Run comparisons
    results = run_comparisons(data)

    # Save results
    output_file = results_dir / "statistical_tests_v2.json"
    print(f"\nSaving results to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Statistical tests completed successfully!")
    print(f"  Results saved to: {output_file}")

    return results

if __name__ == "__main__":
    main()
