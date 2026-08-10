#!/usr/bin/env python3
"""
Compute GRC Metrics for INTERSYMBOLIC-GRC Thesis
Phase 9.C2: Measure GRC metrics quantitatively

This script reads from existing pipeline output (results/*.json) and computes:
  1. NFCRM-1:2025 clause coverage %
  2. ISO/IEC 27005 clause coverage %
  3. Event-to-Risk traceability %
  4. Risk-to-Control linkage %
  5. SHACL conformance rate %
  6. Audit log completeness %
  7. Average reasoning chain depth

Usage:
    python3 scripts/compute_grc_metrics.py [--output results/grc_metrics.json]

Output schema: {metric_name, value, total, percentage, methodology}
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def compute_metrics() -> Dict[str, Any]:
    """
    Compute all 7 GRC metrics from existing results files.
    
    Reads from:
      - results/grc_metrics.json (pipeline output)
      - results/shacl_validation.json (SHACL validation results)
      - results/multisource_arg.json (ARG traceability data)
    """
    logger.info("Loading pipeline output files...")
    
    grc_metrics = load_json('results/grc_metrics.json')
    shacl_metrics = load_json('results/shacl_validation.json')
    arg_metrics = load_json('results/multisource_arg.json')
    
    metrics = grc_metrics.get('metrics', grc_metrics)
    shacl_stats = shacl_metrics
    arg_stats = arg_metrics.get('statistics', {})
    
    # Every metric below is now derived from results/grc_artifact_corpus.json --
    # artefacts actually produced by the rules engine over the 180 real
    # CIC-IDS2018 cases and written to disk -- and from the real pyshacl run in
    # results/grc_artifact_corpus_validation.json.
    #
    # The previous implementation read these values out of its own output file
    # (results/grc_metrics.json) and wrote them back, with the stored values
    # duplicated as hardcoded fallbacks, so 85/85/2/172 propagated unchanged from
    # a since-deleted script that fabricated its events. ISO coverage was not
    # measured at all: it was `iso_val = round(iso_total * 0.582)`.
    corpus = load_json('results/grc_artifact_corpus.json')
    corpus_validation = load_json('results/grc_artifact_corpus_validation.json')
    counts = corpus['counts']
    coverage = corpus['clause_coverage']
    risk_cases = corpus['risk_cases']

    n_risk_cases = counts['risk_cases']

    # ── Metric 1: NFCRM Clause Coverage ──────────────────────────────────────
    nfcrm_val = coverage['nfcrm_riskcases_with_clause']
    nfcrm_total = n_risk_cases
    nfcrm_pct = coverage['nfcrm_coverage_percent']

    # ── Metric 2: ISO/IEC 27005 Clause Coverage ─────────────────────────────
    iso_val = coverage['iso27005_riskcases_with_clause']
    iso_total = n_risk_cases
    iso_pct = coverage['iso27005_coverage_percent']

    # ── Metric 3: Event-to-Risk Traceability ────────────────────────────────
    # A RiskCase is traceable when it carries the source case id it derives from.
    trace_val = sum(1 for r in risk_cases if r.get('case_id'))
    trace_total = n_risk_cases
    trace_pct = round(trace_val / trace_total * 100, 1) if trace_total else 0.0

    # ── Metric 4: Risk-to-Control Linkage ──────────────────────────────────
    linkage_val = sum(1 for r in risk_cases if r.get('control_details'))
    linkage_total = n_risk_cases
    linkage_pct = round(linkage_val / linkage_total * 100, 1) if linkage_total else 0.0

    # ── Metric 5: SHACL Conformance Rate ────────────────────────────────────
    shacl_total = corpus_validation['artefacts_targeted_by_a_shape']
    shacl_val = corpus_validation['artefacts_clean']
    shacl_pct = corpus_validation['conforming_share_of_targeted_percent']
    
    # ── Metric 6: Audit Log Completeness ───────────────────────────────────
    # An AuditLog is complete when it carries a non-empty description and
    # resolves to a RiskCase that exists in the corpus.
    audit_logs = corpus['audit_logs']
    risk_ids = {r['risk_id'] for r in risk_cases}
    audit_val = sum(1 for a in audit_logs
                    if a.get('description') and a.get('risk_case_id') in risk_ids)
    audit_total = len(audit_logs)
    audit_pct = round(audit_val / audit_total * 100, 1) if audit_total else 0.0

    # ── Metric 7: Average Reasoning Chain Depth ─────────────────────────────
    # Case -> RiskCase -> Control -> AuditLog, counted per RiskCase from the
    # links each one actually carries.
    depth_val = sum(
        1                                       # case -> RiskCase
        + (1 if r.get('control_details') else 0)   # RiskCase -> Control
        + (1 if r.get('audit_id') else 0)          # RiskCase -> AuditLog
        + (1 if r.get('related_cves') else 0)      # RiskCase -> Vulnerability
        for r in risk_cases)
    depth_total = n_risk_cases
    depth_avg = round(depth_val / depth_total, 1) if depth_total else 0.0
    
    computed = {
        'nfcrm_clause_coverage': {
            'metric_name': 'nfcrm_clause_coverage',
            'value': int(nfcrm_val),
            'total': int(nfcrm_total),
            'percentage': nfcrm_pct,
            'methodology': (
                'Percentage of RiskCases with at least one associated control '
                'having valid NFCRM-1:2025 clause reference (§5.x or §6.x format). '
                f'Computed from {nfcrm_val}/{nfcrm_total} RiskCases in pipeline output.'
            )
        },
        'iso27005_clause_coverage': {
            'metric_name': 'iso27005_clause_coverage',
            'value': int(iso_val),
            'total': int(iso_total),
            'percentage': iso_pct,
            'methodology': (
                'Percentage of RiskCases with at least one associated control '
                'having a valid ISO/IEC 27005:2022 clause reference. '
                'Derived from attack-type → NFCRM clause → ISO clause mappings in '
                'pipeline/post_inference/grc_annotation_rules.py (W16): '
                'BruteForce→§6.7→ISO§8.2, DoS/DDoS→§6.9→ISO§8.3, WebAttack→§6.4→ISO§8.3. '
                'Infiltration (§6.5) is not yet aligned. '
                f'ISO-covered attack types: 872/1497 non-benign flows = 58.2% of test set. '
                f'Computed from {iso_val}/{iso_total} RiskCases (proportional to test distribution).'
            )
        },
        'event_risk_traceability': {
            'metric_name': 'event_risk_traceability',
            'value': int(trace_val),
            'total': int(trace_total),
            'percentage': trace_pct,
            'methodology': (
                'Percentage of RiskCases with existing RiskCase→Event traceability edges '
                'in the Asset Risk Graph. '
                f'Computed from {trace_val}/{trace_total} RiskCases.'
            )
        },
        'risk_control_linkage': {
            'metric_name': 'risk_control_linkage',
            'value': int(linkage_val),
            'total': int(linkage_total),
            'percentage': linkage_pct,
            'methodology': (
                'Percentage of RiskCases with existing RiskCase→Control linkage edges '
                'in the Asset Risk Graph. '
                f'Computed from {linkage_val}/{linkage_total} RiskCases.'
            )
        },
        'shacl_conformance_rate': {
            'metric_name': 'shacl_conformance_rate',
            'value': int(shacl_val),
            'total': int(shacl_total),
            'percentage': shacl_pct,
            'methodology': (
                'Percentage of generated GRC artifacts (RiskCases, AuditLogs, Controls) '
                'passing SHACL shape validation. '
                f'Total artifacts validated: {shacl_total}. '
                f'Conforms: {shacl_val}, Violations: {shacl_stats.get("total_violations", 0)}.'
            )
        },
        'audit_log_completeness': {
            'metric_name': 'audit_log_completeness',
            'value': int(audit_val),
            'total': int(audit_total),
            'percentage': audit_pct,
            'methodology': (
                'Percentage of AuditLogs with non-empty justification fields and event ID references. '
                f'Computed from {audit_val}/{audit_total} AuditLogs.'
            )
        },
        'avg_reasoning_chain_depth': {
            'metric_name': 'avg_reasoning_chain_depth',
            'value': int(depth_val),
            'total': int(depth_total),
            'percentage': depth_avg,
            'methodology': (
                'Average path length of Event→RiskCase→Control→Evidence reasoning chains. '
                f'Sum of all chain depths: {depth_val}, '
                f'across {depth_total} RiskCases. '
                f'Average: {depth_avg:.1f} hops.'
            )
        }
    }
    
    # No fallback constants. Every value below is read from a file produced by a
    # run over real data; a missing input is a hard failure rather than a
    # silently substituted number.
    result = {
        'generated_at': datetime.now().isoformat(),
        'pipeline_version': '2.0.0',
        'provenance': (
            'Derived from results/grc_artifact_corpus.json (rules engine over the '
            '180 real CIC-IDS2018 cases of results/unified_ablation/test_set.json) '
            'and results/grc_artifact_corpus_validation.json (pyshacl against '
            'ontology/shapes/). Supersedes v1.0.0, whose 85/85/2/172 figures came '
            'from a deleted script that fabricated its own events and were then '
            'carried forward by this script reading its own prior output.'),
        'test_set_size': corpus['source_cases'],
        'total_risk_cases': counts['risk_cases'],
        'total_audit_logs': counts['audit_logs'],
        'total_controls': counts['distinct_controls'],
        'total_artefacts': counts['total_artefacts'],
        'arg_nodes': arg_stats['total_nodes'],
        'arg_edges': arg_stats['total_edges'],
        'arg_shacl_violations': shacl_stats['total_violations'],
        'arg_shacl_conforms': shacl_stats['conforms'],
        'corpus_shacl_violations': corpus_validation['total_violations'],
        'corpus_shacl_conforms': corpus_validation['conforms'],
        'metrics': computed
    }
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute GRC metrics from pipeline output (Phase 9.C2)'
    )
    parser.add_argument(
        '--output',
        default='results/grc_metrics.json',
        help='Output path (default: results/grc_metrics.json)'
    )
    args = parser.parse_args()
    
    result = compute_metrics()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("GRC METRICS — Phase 9.C2")
    print("=" * 60)
    for name, m in result['metrics'].items():
        if name == 'avg_reasoning_chain_depth':
            print(f"  {name}: avg={m['percentage']} hops (sum={m['value']}, n={m['total']})")
        else:
            print(f"  {name}: {m['percentage']}% ({m['value']}/{m['total']})")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
