# Tri-stage Pipeline Bootstrap Metrics (N=2,122)

_Bootstrap N=1000, seed=42._  
_Source: \texttt{results/ablation\_study\_v2.json}_

## H1 — Pre-inference symbolic enrichment changes classifier accuracy

| Variant | Accuracy [95% CI] | Macro-F1 [95% CI] |
|---|---|---|
| RF Baseline | 0.8770 [0.8629, 0.8907] | 0.8960 [0.8593, 0.9229] |
| Tri-stage (full) | 0.8798 [0.8662, 0.8935] | 0.8879 [0.8473, 0.9177] |

**Accuracy delta (tri-stage − baseline):** +0.0028
**Macro-F1 delta (tri-stage − baseline):** -0.0081

**Sample-paired McNemar test:** b=21 (baseline-only correct), c=27 (tri-stage-only correct), n_discordant=48, p=0.4709 (two-sided exact)

**Verdict on H1: Fail to reject null.** At α=0.05, the tri-stage pipeline does not produce a statistically significant change in classification accuracy versus the RF baseline. Confidence intervals overlap substantially (delta is ~+0.28 pp; CI half-widths are ~1.39 pp). The pipeline neither helps nor hurts accuracy at this scale.

## Per-class significance (one-vs-rest McNemar)

| Class | Support | b (base-only) | c (tri-only) | p (two-sided) | Significant at α=0.05? |
|---|---|---|---|---|---|
| Benign | 625 | 20 | 27 | 0.3817 | no |
| BruteForce | 118 | 1 | 2 | 1.0000 | no |
| DDoS | 375 | 0 | 1 | 1.0000 | no |
| DoS | 367 | 0 | 0 | 1.0000 | no |
| Infiltration | 625 | 21 | 26 | 0.5601 | no |
| WebAttack | 12 | 2 | 0 | 0.5000 | no |

## Per-class precision/recall/F1 (point estimates)

| Class | RF Baseline (P/R/F1) | Tri-stage (P/R/F1) |
|---|---|---|
| Benign | 0.823 / 0.757 / 0.788 | 0.820 / 0.774 / 0.797 |
| BruteForce | 0.957 / 0.941 / 0.949 | 0.965 / 0.941 / 0.953 |
| DDoS | 0.997 / 1.000 / 0.999 | 1.000 / 1.000 / 1.000 |
| DoS | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Infiltration | 0.777 / 0.840 / 0.807 | 0.787 / 0.832 / 0.809 |
| WebAttack | 0.833 / 0.833 / 0.833 | 0.714 / 0.833 / 0.769 |