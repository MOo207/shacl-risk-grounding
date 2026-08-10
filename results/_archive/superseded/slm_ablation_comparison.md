# SLM Ablation: Full NL features vs Stripped (no signature hints)

Held constant across runs: random_state=42 (same 60 samples), model=glm-4.7, temperature=0, ZAI API.

| Variant | Accuracy [95% CI] | DoS | DDoS | WebAttack | BruteForce | Benign | Infiltration |
|---|---|---|---|---|---|---|---|
| Full NL (W17 v3) | 55.0% [41.7, 68.3] | 70% | 70% | 70% | 40% | 40% | 40% |
| Stripped (no hints) | 50.0% [38.3, 61.7] | 70% | 40% | 70% | 40% | 40% | 40% |

**Delta (Full - Stripped):** +5.0 percentage points
**McNemar test:** b=4 (full-only correct), c=1 (stripped-only correct), p=0.3750

## Interpretation

- Delta = +5.0 pp, but the sample-paired McNemar test gives p=0.375 (NOT significant at α=0.05). At N=60, the data are consistent with BOTH ``the LLM relies on the engineered signature hints'' AND ``the LLM does real reasoning over base features''. We cannot distinguish these hypotheses at this sample size. Report this as an **honest null** on the contribution of NL feature engineering.
- Per-class: the largest gap is **DDoS** (full 70%, stripped 40%, delta +30 pp). On n=10 per class the per-class CIs are wide (roughly ±30 pp), so even this gap is not individually significant.