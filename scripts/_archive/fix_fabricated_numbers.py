#!/usr/bin/env python3
"""
Fix fabricated numbers in thesis (Phase 4.9 / Phase 5.4).

Replaces fabricated claims with actual results from JSON files.

Fabricated Content Registry (F1-F10):
- F4: "91.5% accuracy" → 87.98% (tri_stage)
- F5: "99.71% accuracy" → 74.56% (NSL-KDD RF) or 75.43% (NSL-KDD XGB)
- F6: Human eval 4.5/5, 4.7/5 → REMOVE (no human eval conducted)
- F8: "7,874 events/sec" → 424 samples/sec
- F9: "127ms latency" → 2.36ms
"""

import re

THESIS_PATH = 'thesis/INTERSYMBOLIC-GRC_Thesis.tex'

# Read thesis
with open(THESIS_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# Fix F8, F9: Performance numbers (line 2497)
# Old: "127 ms vs. 98 ms pure-ML" and "7,874 events/sec vs. 10,204 pure-ML"
# New: "2.36 ms vs. ~1 ms pure-ML" and "424 events/sec"
content = content.replace(
    'While the intersymbolic framework has higher latency (127 ms vs. 98 ms pure-ML, vs. 45 ms pure-rule) and lower throughput (7,874 events/sec vs. 10,204 pure-ML, vs. 22,222 pure-rule)',
    'While the intersymbolic framework has modest overhead (2.36 ms average latency) and throughput of 424 events/sec in the e2e benchmark'
)

# Fix F4: "91.5%" accuracy (line 2775)
content = content.replace(
    'Tri-stage pipeline} achieves the best balance: highest accuracy (91.5\\%)',
    'Tri-stage pipeline} achieves equivalent accuracy (87.98\\%) to pre-inference alone'
)

# Fix F6: Human evaluation scores (lines 2800-2801) - REMOVE these claims
# Replace with honest statement
content = content.replace(
    '\\item \\textbf{Overall Trust}: 4.5/5 (intersymbolic) vs. 2.3/5 (pure-ML) vs. 4.0/5 (pure-rule) (Chapter 4, Table \\ref{tab:human-evaluation}).',
    '\\item \\textbf{Human Evaluation}: Not conducted in this study. Future work should include expert evaluation of GRC artifact quality.'
)
content = content.replace(
    '\\item \\textbf{Standards Mapping Accuracy}: 4.7/5 (intersymbolic) vs. 1.0/5 (pure-ML) vs. 4.2/5 (pure-rule) (Chapter 4, Table \\ref{tab:human-evaluation}).',
    '\\item \\textbf{Standards Mapping}: INTERSYMBOLIC-GRC provides 100\\% NFCRM-1:2025 clause traceability by design (Chapter 3).'
)

# Fix F8: "7,874 events/sec" (line 2832)
content = content.replace(
    'Current implementation achieves 7,874 events/sec, which may be insufficient',
    'Current implementation achieves 424 events/sec in e2e benchmark, which is sufficient for batch processing but may require optimization for high-throughput streaming'
)

# Fix F9: "127 ms" latency (line 2845)
content = content.replace(
    'Average latency of 127 ms may be too slow',
    'Average latency of 2.36 ms is suitable for near-real-time processing'
)

# Fix F5: "99.71% accuracy" (line 2957) - this is NSL-KDD RF result
content = content.replace(
    'Baseline Random Forest classifier achieves superior technical performance (99.71\\% accuracy)',
    'Baseline Random Forest classifier on NSL-KDD achieves 74.56\\% accuracy (F1-macro: 0.472); XGBoost achieves 75.43\\% (F1-macro: 0.519)'
)

# Write back
if content != original:
    with open(THESIS_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ Fixed fabricated numbers in thesis')
    print('  - F4: 91.5% → 87.98% (tri-stage accuracy)')
    print('  - F5: 99.71% → 74.56% (NSL-KDD RF accuracy)')
    print('  - F6: Removed fabricated human eval scores')
    print('  - F8: 7,874 events/sec → 424 samples/sec')
    print('  - F9: 127ms → 2.36ms latency')
else:
    print('No changes needed (already fixed or patterns not found)')
