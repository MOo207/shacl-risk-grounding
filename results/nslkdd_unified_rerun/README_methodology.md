# NSL-KDD Unified Rerun — Ground Truth (Non-LLM Stage)

Built by `scripts/build_nslkdd_unified_rerun.py`. This is the **preparation
stage only** — no LLM/Claude calls were made. It fixes two problems found in
the existing NSL-KDD safety evaluation (`results/nslkdd_ablation/`) and
produces a clean, non-leaking, reproducible ground truth ready for an LLM
rerun in a separate, budgeted task.

## Problems being fixed

1. **NL-description leakage.** The original generator
   (`scripts/_archive/run_slm_nslkdd_classification.py::features_to_nl_nslkdd`)
   inserts the true attack-class name or category tag directly into the prose
   shown to the LLM — e.g. literal `(Probe)`/`(U2R)`/`(R2L)`/`(DoS)` tags, the
   phrases `denial-of-service` / `network reconnaissance` / `user-to-root` /
   `remote-to-local`, and named-attack substrings (`httptunnel`, `mailbomb`,
   `warezmaster`, `snmpguess`/`snmpgetattack`, `rootkit installation`, the
   `'back'` denial-of-service phrase). An audit
   (`results/nslkdd_ablation/leakage_tagging_clean_subset.json`) found 30/50
   existing test cases leak the answer this way, and the safety finding's
   McNemar significance collapses (p=6.1e-5 → p≈0.5) once leaky cases are
   excluded.

2. **XGBoost non-reproducibility.** A prior retrain with the same
   seed/data/features only matched 24/50 (48%) of the historical frozen
   predictions in `results/nslkdd_ablation/test_set.json`
   (`results/nslkdd_ablation/xgb_cost_sensitive_baseline.json` `meta.reproducibility`).

## Fix #1 — reproducible XGBoost

`build_nslkdd_unified_rerun.py::train_xgb()` pins:

```python
XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss",
              tree_method="hist",   # explicit, not "auto"
              n_jobs=1)             # single-threaded
```

**Rationale:** the original `build_nslkdd_ablation_testset.py::train_ml_models()`
left `tree_method`/`n_jobs` unset. XGBoost's histogram-based tree builder can
run multi-threaded even with `random_state` fixed, and parallel
floating-point summation order for histogram bins is not guaranteed
bit-identical run-to-run. Pinning `tree_method="hist"` explicitly and
`n_jobs=1` removes that source of nondeterminism while staying fast
(~10–30s per 126k-row train on this ~123-column one-hot-encoded feature
matrix, vs. `tree_method="exact"`, which is also deterministic but far
slower on data this wide).

**Verification performed:** trained twice from scratch (identical seed/data/
params) and diffed predictions + predicted probabilities on the held-out test
sample.

- **Result: 100/100 = 100.0% match, max predicted-probability abs diff = 0.0**
  (`xgboost_baseline_results.json` → `reproducibility_verification`).
- A separate exploratory check (before finalizing settings) also found
  default settings (no `tree_method`/`n_jobs` pinned), `tree_method="exact"`
  +`n_jobs=1`, and `tree_method="hist"`+`n_jobs=1` **all** independently
  reproduced 50/50 identical predictions across two from-scratch retrains on
  the original N=50 eval indices, in this session (xgboost 3.2.0).
  The originally reported 24/50 (48%) mismatch could **not** be reproduced
  here; it is most likely attributable to an xgboost/sklearn version
  difference between when `results/nslkdd_ablation/test_set.json` was first
  generated and the later verification retrain (versions were not pinned/
  recorded at generation time), not to genuine algorithmic nondeterminism at
  a fixed version. `tree_method="hist"` + `n_jobs=1` is still pinned here as
  the safer, explicit, single-threaded choice going forward, removing any
  residual risk from parallel execution.

## Fix #2 — neutral (non-leaking) NL generator

`pipeline/nfcrm/nslkdd_neutral_nl.py::features_to_nl_neutral()` converts the
same underlying NSL-KDD signal families the original generator reads
(protocol/service/flag, byte counts, same-host/same-service connection
counts, SYN/REJ error rates, service-diversity rates, destination-host
2-second-window statistics, login/privilege/session indicators) into
purely observational prose. It reports **only measured field values** —
no attack-class name, category tag, named-attack substring, or interpretive
verdict language (e.g. "flood", "scan", "brute-force", "exfiltration",
"privilege escalation", "reconnaissance", "attack", "exploit", "malicious",
"compromise" are all absent from the templates).

**Validation:** every generated description in the calibration + test samples
(150 cases, all 5 classes) was checked against:

- the exact named-attack + class-name regex patterns from
  `scripts/tag_nslkdd_leakage_and_clean_subset.py` (the same patterns used to
  indict the original generator), **plus**
- a broader additional ban list (flood/scan/brute-force/exfiltrate/
  privilege escalation/reconnaissance/attack/exploit/malicious/compromise),
  since this is validating a brand-new generator, not re-scoring the old one.

**Result: 0/150 cases flagged, 0 total regex hits** (`xgboost_baseline_results.json`
→ `leakage_validation`). A larger ad-hoc sample of 1,800 rows (up to 400/class
from the full KDDTest+ pool, independent random seed) also confirmed zero
hits with the same pattern set.

## Sample construction

Both samples are drawn from `data/raw/NSL-KDD/KDDTest+.txt`, **excluding**
every case in the compromised `results/nslkdd_ablation/test_set.json`
(N=50). Exclusion is exact: the old set's sampling code
(`build_nslkdd_ablation_testset.py`, seed=42, 10/class) is re-run to recover
the precise KDDTest+ row indices it drew, and those 50 indices are dropped
from the pool before any new sampling — not a fuzzy feature-match, so there
is zero risk of silent overlap.

| Sample | File | N | Per class | Purpose |
|---|---|---|---|---|
| Calibration | `calibration_sample.json` | 50 | 10 | Future LLM-override-threshold tuning. **Not** part of the final scored comparison. |
| Held-out test | `test_sample.json` | 100 | 20 | The actual evaluation set for the eventual LLM rerun. |

Sampling order: calibration (10/class) is drawn first from the post-exclusion
pool (seed=42), then test (20/class) is drawn from what remains after
removing the calibration rows (seed=42) — guaranteeing calibration and test
never overlap each other either. Remaining-pool sizes after excluding the old
50 (smallest class, U2R, still had 190 available): Normal 9,701 / DoS 7,448 /
Probe 2,411 / R2L 2,744 / U2R 190 — comfortably enough for 30/class total
(10 calibration + 20 test).

Each case record contains: `case_id`, `true_attack_class` / `true_nfcrm_class`,
a synthetic `asset` (hostname/criticality/type/OS) and `paired_cve` (drawn the
same way as the original ARG-backed CVE pool), `nl_description` (the neutral
generator's output — this is what an LLM stage would read), `raw_features`
(the full underlying 41-feature row, for auditability), the fixed XGBoost's
prediction (`xgb_predicted_class`/`xgb_nfcrm_class`/`xgb_confidence`/
`xgb_risk_level`), and `ground_truth_risk_level` (NFCRM §6.9
Likelihood×Impact from the true class + synthetic criticality, no CVE
override — identical scheme to the original `test_set.json`).

## Held-out test XGBoost baseline (N=100)

Computed with the identical NFCRM §6.9 mapping and under/over-escalation
metric code as `scripts/safety_table_haiku.py::metrics()`.

| Metric | Value |
|---|---|
| Exact risk-level accuracy | 49.0% |
| Within-1-level accuracy | 53.0% |
| **Under-escalation rate** | **48.0%** |
| Over-escalation rate | 3.0% |
| **Severe-risk recall** | **42.9%** (21/49 severe cases) |
| Worst miss | 4 risk-level bands |

These numbers are **directionally consistent** with the existing default-loss
XGBoost baseline on the old N=50 set (44% under-escalation, 45% severe
recall) — i.e. the fixed, reproducible retrain does not change the
qualitative picture of XGBoost's severe under-escalation problem on this
class-imbalanced dataset. They are not directly comparable to the old
numbers (different, non-overlapping 100-case sample), but they establish the
XGBoost half of the safety comparison for the upcoming clean LLM rerun.

## Files in this directory

- `calibration_sample.json` — N=50 (10/class), raw features + neutral NL description per case.
- `test_sample.json` — N=100 (20/class), raw features + neutral NL description per case. **This is the evaluation set for the LLM rerun.**
- `xgboost_baseline_results.json` — reproducibility verification, leakage validation, and the held-out test XGBoost baseline metrics.
- `README_methodology.md` — this file.

## Code

- `scripts/build_nslkdd_unified_rerun.py` — end-to-end builder (reproduces all of the above from raw data).
- `pipeline/nfcrm/nslkdd_neutral_nl.py` — the neutral NL generator (`features_to_nl_neutral`).

## Explicitly out of scope for this task

No LLM/Claude API or CLI wrapper calls were made anywhere in this pipeline.
The tri-stage / pure-LLM arms on `test_sample.json`, and the
override-threshold calibration pass on `calibration_sample.json`, are left
for the separate, budgeted follow-up task.
