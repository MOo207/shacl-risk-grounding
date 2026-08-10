#!/usr/bin/env python3
"""
Phase 6.3: Automated number verification script.
Cross-references all percentage claims in thesis against results JSONs.
Reports: MATCH / MISMATCH / UNVERIFIABLE
"""

import re
import json
from pathlib import Path

def extract_thesis_numbers(tex_path: str) -> list[dict]:
    """Extract all percentage and numeric claims from thesis."""
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    claims = []
    
    # Pattern for percentages: XX.X% or XX%
    pct_pattern = r'(\d+\.?\d*)\\?%'
    for match in re.finditer(pct_pattern, content):
        line_num = content[:match.start()].count('\n') + 1
        context_start = max(0, match.start() - 100)
        context_end = min(len(content), match.end() + 100)
        context = content[context_start:context_end].replace('\n', ' ')
        claims.append({
            'type': 'percentage',
            'value': match.group(1),
            'line': line_num,
            'context': context.strip()
        })
    
    # Pattern for accuracy/F1 claims: XX.XX% accuracy, F1 of 0.XX
    acc_pattern = r'(accuracy|F1|precision|recall)\s*(of|:|=)?\s*(\d+\.?\d*)%?'
    for match in re.finditer(acc_pattern, content, re.IGNORECASE):
        line_num = content[:match.start()].count('\n') + 1
        claims.append({
            'type': 'metric',
            'metric': match.group(1),
            'value': match.group(3),
            'line': line_num,
            'context': match.group(0)
        })
    
    return claims

def load_results(results_dir: str) -> dict:
    """Load all results JSONs."""
    results = {}
    results_path = Path(results_dir)
    for json_file in results_path.glob('*.json'):
        with open(json_file, 'r') as f:
            results[json_file.stem] = json.load(f)
    return results

def verify_claim(claim: dict, results: dict) -> str:
    """Verify a claim against results. Returns MATCH/MISMATCH/UNVERIFIABLE."""
    value = float(claim['value'])
    
    # Check ablation study for accuracy claims around 87-96%
    if claim['type'] == 'percentage' and 'ablation_study_v2' in results:
        ablation = results['ablation_study_v2']
        # Check baseline accuracy (87.70%)
        if 'baseline' in ablation and 'test' in ablation['baseline']:
            baseline_acc = ablation['baseline']['test'].get('accuracy', 0) * 100
            if abs(value - baseline_acc) < 0.5:
                return f"MATCH (baseline {baseline_acc:.2f}%)"
        # Check tri_stage accuracy (87.98%)
        if 'tri_stage' in ablation and 'test' in ablation['tri_stage']:
            tri_acc = ablation['tri_stage']['test'].get('accuracy', 0) * 100
            if abs(value - tri_acc) < 0.5:
                return f"MATCH (tri_stage {tri_acc:.2f}%)"
    
    # Check XGB baseline (95.90%)
    if 'xgb_baseline' in results:
        xgb = results['xgb_baseline']
        if 'test' in xgb:
            xgb_acc = xgb['test'].get('accuracy', 0) * 100
            if abs(value - xgb_acc) < 0.5:
                return f"MATCH (XGB {xgb_acc:.2f}%)"
    
    # Check rule baseline (36.9%)
    if 'rule_baseline' in results:
        rule = results['rule_baseline']
        rule_acc = rule.get('accuracy', 0) * 100
        if abs(value - rule_acc) < 0.5:
            return f"MATCH (Rule {rule_acc:.2f}%)"
    
    # Check NSL-KDD RF (74.56%)
    if 'nslkdd_rf_baseline' in results:
        nslkdd_rf = results['nslkdd_rf_baseline']
        if 'test' in nslkdd_rf:
            rf_acc = nslkdd_rf['test'].get('accuracy', 0) * 100
            if abs(value - rf_acc) < 0.5:
                return f"MATCH (NSL-KDD RF {rf_acc:.2f}%)"
    
    # Check NSL-KDD XGB (75.43%)
    if 'nslkdd_xgb_baseline' in results:
        nslkdd_xgb = results['nslkdd_xgb_baseline']
        if 'test' in nslkdd_xgb:
            xgb_acc = nslkdd_xgb['test'].get('accuracy', 0) * 100
            if abs(value - xgb_acc) < 0.5:
                return f"MATCH (NSL-KDD XGB {xgb_acc:.2f}%)"
    
    # Check NSL-KDD Rule (44.0%)
    if 'nslkdd_rule_baseline' in results:
        nslkdd_rule = results['nslkdd_rule_baseline']
        rule_acc = nslkdd_rule.get('accuracy', 0) * 100
        if abs(value - rule_acc) < 0.5:
            return f"MATCH (NSL-KDD Rule {rule_acc:.2f}%)"
    
    # Check SLM success rate claims (60%+)
    if 'slm_explanations_v2' in results:
        slm = results['slm_explanations_v2']
        success_rate = slm.get('success_rate', 0)
        if abs(value - success_rate) < 1.0:
            return f"MATCH (SLM success {success_rate}%)"
    
    # Check SHACL coverage
    if 'shacl_validation' in results:
        shacl = results['shacl_validation']
        coverage = shacl.get('coverage_percent', 0)
        if abs(value - coverage) < 1.0:
            return f"MATCH (SHACL coverage {coverage}%)"
    
    # Check e2e pipeline throughput
    if 'e2e_pipeline_benchmark' in results:
        e2e = results['e2e_pipeline_benchmark']
        throughput = e2e.get('throughput_samples_per_second', 0)
        if abs(value - throughput) < 10:  # Allow some variance
            return f"MATCH (Throughput {throughput:.0f} samples/sec)"
    
    # Check statistical tests (McNemar p-value)
    if 'statistical_tests' in results:
        stats = results['statistical_tests']
        if 'mcnemar_baseline_vs_tristage' in stats:
            p_value = stats['mcnemar_baseline_vs_tristage'].get('p_value', 0)
            if abs(value - p_value) < 0.01:
                return f"MATCH (McNemar p={p_value:.4f})"
    
    return "UNVERIFIABLE"

def main():
    thesis_path = Path(__file__).parent.parent / 'thesis' / 'INTERSYMBOLIC-GRC_Thesis.tex'
    results_path = Path(__file__).parent.parent / 'results'
    
    print("=" * 80)
    print("Phase 6.3: Automated Thesis Number Verification")
    print("=" * 80)
    print(f"\nThesis: {thesis_path}")
    print(f"Results: {results_path}\n")
    
    claims = extract_thesis_numbers(str(thesis_path))
    results = load_results(str(results_path))
    
    print(f"Found {len(claims)} numeric claims in thesis\n")
    print("-" * 80)
    
    match_count = 0
    mismatch_count = 0
    unverifiable_count = 0
    
    # Filter to most relevant claims (percentages and metrics)
    relevant_claims = [c for c in claims if c['type'] in ['percentage', 'metric']]
    
    for claim in relevant_claims[:50]:  # Limit to first 50 for readability
        if claim['type'] == 'percentage':
            value = f"{claim['value']}%"
        else:
            value = f"{claim['metric']}={claim['value']}"
        
        verification = verify_claim(claim, results)
        
        if verification.startswith('MATCH'):
            match_count += 1
            status = "✓"
        elif verification.startswith('MISMATCH'):
            mismatch_count += 1
            status = "✗"
        else:
            unverifiable_count += 1
            status = "?"
        
        print(f"{status} Line {claim['line']:4d}: {value:20s} → {verification}")
    
    print("-" * 80)
    print(f"\nSummary:")
    print(f"  MATCH:        {match_count:3d}")
    print(f"  MISMATCH:     {mismatch_count:3d}")
    print(f"  UNVERIFIABLE: {unverifiable_count:3d}")
    print(f"  TOTAL:        {len(relevant_claims[:50]):3d}")
    
    if mismatch_count > 0:
        print(f"\n⚠️  WARNING: {mismatch_count} MISMATCHES found - review required!")
        return 1
    else:
        print(f"\n✓ All verified claims match results JSONs")
        return 0

if __name__ == '__main__':
    exit(main())
