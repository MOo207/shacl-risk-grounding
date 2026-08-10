# Tri-stage LLM override-mechanism investigation (NSL-KDD + CIC clean reruns)

Date: 2026-07-11. Pure code/data investigation — no LLM calls made.

## 1. Exact instruction text

**NSL-KDD tri-stage system prompt** (`scripts/run_nslkdd_ablation.py` lines 179-194,
copied verbatim into `scripts/run_nslkdd_tristage_haiku_clean.py` lines 77-92):

```
MANDATORY RULE — §6.9 Risk Level Lookup Table for NSL-KDD traffic (you MUST follow this exactly):
  [... table ...]
DO NOT override the table. The ML classifier's attack class is authoritative — use it as the row key.
```

There is **no verification/override clause at all** in the NSL-KDD tri-stage prompt — not a
loosened one, an absent one. The LLM is told to copy `xgb_predicted_class` in as the row key and
apply a fixed lookup table. Nothing in the prompt or the surrounding Python
(`run_tristage_llm_haiku`) gives the model permission to deviate.

**CIC tri-stage system prompt** (`scripts/run_unified_ablation.py` lines 167-191, reused by
`scripts/run_cic_tristage_haiku_clean.py`) is different and does contain a real, narrow
override clause:

```
VERIFICATION RULE — override only when ALL three conditions hold:
  (a) ML predicts Benign with confidence < 80%
  (b) The paired CVE has CVSS ≥ 9.5 (critical severity exploit)
  (c) Asset criticality is High or Critical
  → In this case classify as Infiltration (stealthy C2 or data exfiltration missed by ML).
In all other cases trust the ML class exactly as given.
```

## 2. Is the ML class given to the LLM, and what are the firing conditions?

(a) Yes in both pipelines — `attack_class`/`ml_pred_class` = `case["xgb_predicted_class"]` is
injected directly into the user prompt as "Attack class (from ML classifier): {attack_class}"
and the LLM is told to use it as the authoritative row key (NSL-KDD) or verify-then-trust it
(CIC). The LLM never independently infers the class from the NL description in the tri-stage
paradigm (that only happens in the separate `SYSTEM_PROMPT_PURE_LLM` paradigm, a different arm).

(b) CIC override firing conditions, quoted exactly from `run_unified_ablation.py` (mirrored in
code by `scripts/run_confidence_gated_override.py` / `symbolic_override()` in
`run_nslkdd_risk_ablation.py`, a *different*, non-LLM paradigm not wired into tri-stage):
`ml_class == "Benign" AND confidence < 0.80 AND cvss_v3 >= 9.5 AND criticality in {High, Critical}`.
NSL-KDD tri-stage has **no such rule at all** — "DO NOT override" is unconditional.

(c) Checked directly against the raw per-case data:
- **NSL-KDD clean, N=100** (`tristage_haiku_clean_test.jsonl` / `tristage_haiku_clean_results.json`):
  the override mechanism doesn't exist in this prompt, so it was never evaluated — not "evaluated
  and suppressed," genuinely absent. Result: 0/100 `risk_level != xgb_risk_level`, 0 parse errors,
  McNemar b=0, c=0, p=1.0.
- **CIC clean, N=120 each model** (`tristage_haiku_clean_results.json`,
  `tristage_sonnet_clean_results.json`): checked every case's `xgb_predicted_class`/`xgb_confidence`
  /paired-CVE-CVSS/criticality against the three-condition rule — **0/120 cases meet the
  preconditions** (no case has `Benign` + `confidence<0.80` + `CVSS≥9.5` + `High/Critical`
  criticality simultaneously). The override clause was present in the prompt but structurally
  had nothing to fire on. `llm_verified_cls != ml_pred_class` count: 0/120 for both models — the
  LLM never invoked it either. The single Haiku discordant case (`test_033`,
  `risk_level=High` vs `xgb_risk_level=Medium`, `Infiltration`→`Infiltration`, no class change) is
  a table-lookup slip, not an override — Sonnet has zero discordant cases (b=0, c=0 on Sonnet;
  Haiku shows one one-sided miss, not a genuine safety split).

## 3. Where did the old b=14 result actually come from?

Direct inspection of `results/nslkdd_ablation/tristage_llm_haiku.jsonl` (the file behind the
thesis's b=14,c=0,p=6.1e-5 spine test) shows **19/50 cases where `llm_verified_cls != ml_pred`**,
despite the system prompt saying "DO NOT override... authoritative." In 18 of those 19, the ML
class was `Normal` and the LLM "verified" it to `U2R`/`R2L`/`DoS`/`Probe` — i.e., the model
silently disobeyed the explicit no-override instruction. This lines up exactly with the
leaky-NL-description hypothesis: the old NL generator named the attack in the flow description,
and the LLM used that leaked token to overrule the ML row key rather than following the mandatory
rule. This is instruction non-compliance driven by label leakage, not a designed reasoning
mechanism — confirming the byte-identical clean-data result is not an artifact of "the override
became too strict," but of removing the leak that was silently causing rule violations.

## 4. Structural vs. fixable — and what a legitimate redesign looks like

**The current byte-identical result is structurally inevitable, by design, for both datasets on
clean data:**
- NSL-KDD tri-stage: no override clause exists; the LLM is a verbatim relay of
  `xgb_predicted_class` through a fixed lookup table. There is no code path for divergence at
  all (short of the model refusing/mis-parsing the instruction, which happened 0/100 times).
- CIC tri-stage: an override clause exists but its preconditions are so narrow that 0/120 clean
  cases ever satisfy them; whether the threshold is "well-calibrated" or "too conservative" is
  moot when zero cases reach it either way. It is not proven too tight or too loose — it simply
  never fires in this sample, so no adjustment to the CVSS/confidence thresholds is defensible
  from this data (there's no observed near-miss to recalibrate against).

**This is not a fundamental limit of "can an LLM ever add safety value here" — it is a specific,
nameable design choice**: the LLM is given the ML class as an anchor and instructed (NSL-KDD) or
default-biased (CIC) to copy it, with reasoning invoked only as a narrow verification gate rather
than as the primary path to the answer. A legitimate, non-p-hacking alternative:

**Proposed redesign ("independent-inference-then-reconcile"):**
1. Stage 1 (LLM): give the LLM ONLY the neutral NL description, asset, and CVE context — NOT
   `xgb_predicted_class` — and ask it to independently infer the attack class and risk level
   using the same §6.9 table. This is structurally identical to the existing
   `SYSTEM_PROMPT_PURE_LLM` arm (`run_unified_ablation.py` line 193, `run_nslkdd_ablation.py`
   line 196) — that code already exists and has already been run (Pure LLM paradigm), just not
   as part of the tri-stage safety spine.
2. Stage 2 (symbolic reconciliation, new): a deterministic function compares the LLM's
   independent class/risk_level to the ML's, and only when they *disagree* invokes a symbolic
   tie-break rule (e.g., prefer the higher risk_level, or apply a documented arbitration rule
   grounded in confidence/CVSS) — instead of LLM "verifying" ML with permission to rarely deviate.
3. The tri-stage "GRC completeness" value-add (narrative, control recommendation, NFCRM clauses)
   is unaffected — those stages already run over whichever class is finally selected.

**Size of the change:** small-to-medium, not a large pipeline redesign. The independent-inference
LLM call is a straight prompt swap (drop `attack_class`/`xgb_predicted_class` from
`PROMPT_TRISTAGE`, reuse `SYSTEM_PROMPT_PURE_LLM`/`PROMPT_PURE_LLM` already in the codebase) — no
new LLM-calling infrastructure needed. The new piece is the reconciliation function (Stage 2),
which is a small, pure Python addition (a dozen lines, analogous in shape to the already-existing
`symbolic_override()` in `run_nslkdd_risk_ablation.py`) that takes two independently-derived
(class, risk_level) pairs and applies a documented, pre-registered arbitration rule — not tuned
post hoc to the test set. Re-running both the NSL-KDD (100) and CIC (120) clean held-out sets
under this design is the same LLM-call volume already paid for in the two clean reruns; the work
is a new script (`run_nslkdd_tristage_independent.py` / CIC equivalent), on the order of the two
existing `*_clean.py` scripts, not a framework rewrite.
