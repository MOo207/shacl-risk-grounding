"""Task 15 — NSL-KDD NL-description leakage audit + clean-subset safety sensitivity.

Context: thesis EVT-5 (Threats to Validity) and the NSL-KDD ablation
construct-validity note (Section~nslkdd-ablation) disclose that the rule-based
NL-description generator (``features_to_nl_nslkdd()`` in
``scripts/_archive/run_slm_nslkdd_classification.py``) sometimes names the
attack pattern directly in the generated prose (e.g. an explicit "httptunnel
attack" mention, or "mailbomb denial-of-service attack", or a literal
parenthetical class tag such as "(Probe)"/"(U2R)"/"(R2L)"/"(DoS)"). For those
cases the LLM's job is closer to *keyword extraction* than *inference from
ambiguous connection statistics*.

This script:
  1. Loads the canonical N=50 NSL-KDD safety-synthesis test set
     (results/nslkdd_ablation/test_set.json), which already stores the
     generated nl_description actually shown to the LLM for every case.
  2. Programmatically tags each case for lexical leakage: does the
     nl_description contain (a) a distinctive named-attack substring that
     maps 1:1 to the case's TRUE class, or (b) a literal class-name /
     class-category phrase that matches the TRUE class? Tagging is evidence
     based (regex over the actual generated text), not a guess.
  3. Recomputes the flagship safety spine (Tri-stage LLM Haiku vs Pure
     ML/XGBoost: under-escalation rate, severe-risk recall, per-case
     one-sided McNemar on "dangerous under-escalation") restricted to the
     clean (non-leaky) subset, and compares to the full-N=50 numbers from
     results/safety_table_haiku.json.
  4. Saves results/nslkdd_ablation/leakage_tagging_clean_subset.json.

No randomness is introduced (case set is fixed, seed=42 upstream); this
script is a deterministic re-scoring of already-generated artifacts.
"""
from __future__ import annotations

import json
import re
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTSET = ROOT / "results/nslkdd_ablation/test_set.json"
OUT = ROOT / "results/nslkdd_ablation/leakage_tagging_clean_subset.json"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

# ---------------------------------------------------------------------------
# Leakage trigger patterns, sourced directly from
# scripts/_archive/run_slm_nslkdd_classification.py :: features_to_nl_nslkdd()
#
# Each pattern is (regex, class_it_names, trigger_description). A case is
# tagged leaky only if the pattern fires AND the class it names equals the
# case's TRUE attack class (a mismatched mention, e.g. a Normal case whose
# heuristic hedge-text says "(Probe)", does not leak the correct answer and
# is therefore NOT tagged leaky here).
# ---------------------------------------------------------------------------
NAMED_ATTACK_PATTERNS = [
    (re.compile(r"httptunnel", re.I), "U2R",
     "distinctive named-attack substring 'httptunnel'"),
    (re.compile(r"mailbomb", re.I), "DoS",
     "distinctive named-attack substring 'mailbomb'"),
    (re.compile(r"'back' denial-of-service|characteristic of a 'back'", re.I), "DoS",
     "distinctive named-attack substring 'back'"),
    (re.compile(r"warezmaster", re.I), "R2L",
     "distinctive named-attack substring 'warezmaster'"),
    (re.compile(r"snmpguess|snmpgetattack|SNMP community string guessing", re.I), "R2L",
     "distinctive named-attack phrase 'snmpguess/snmpgetattack (SNMP community string guessing)'"),
    (re.compile(r"rootkit installation", re.I), "U2R",
     "distinctive named-attack phrase 'rootkit installation'"),
]

CLASS_NAME_PATTERNS = [
    (re.compile(r"\(Probe\)"), "Probe", "literal class-name token '(Probe)'"),
    (re.compile(r"\(U2R\)"), "U2R", "literal class-name token '(U2R)'"),
    (re.compile(r"\(R2L\)"), "R2L", "literal class-name token '(R2L)'"),
    (re.compile(r"\(DoS\)"), "DoS", "literal class-name token '(DoS)'"),
    (re.compile(r"denial-of-service", re.I), "DoS",
     "literal class-category phrase 'denial-of-service'"),
    (re.compile(r"network reconnaissance", re.I), "Probe",
     "literal class-category phrase 'network reconnaissance'"),
    (re.compile(r"user-to-root", re.I), "U2R",
     "literal class-category phrase 'user-to-root'"),
    (re.compile(r"remote-to-local", re.I), "R2L",
     "literal class-category phrase 'remote-to-local'"),
]

ALL_PATTERNS = NAMED_ATTACK_PATTERNS + CLASS_NAME_PATTERNS


def tag_case(case: dict) -> dict:
    text = case["nl_description"]
    true_cls = case["true_attack_class"]
    triggers = []
    for pat, names_class, desc in ALL_PATTERNS:
        m = pat.search(text)
        if m and names_class == true_cls:
            triggers.append({
                "matched_text": m.group(0),
                "names_class": names_class,
                "trigger": desc,
                "category": "named-attack" if (pat, names_class, desc) in NAMED_ATTACK_PATTERNS else "class-name",
            })
    return {
        "case_id": case["case_id"],
        "true_attack_class": true_cls,
        "leaky": len(triggers) > 0,
        "triggers": triggers,
    }


# ---------------------------------------------------------------------------
# Safety metrics (mirrors scripts/safety_table_haiku.py, restricted to an
# arbitrary subset of case ids so it can be re-run on the clean subset).
# ---------------------------------------------------------------------------

def load_jsonl(rel):
    p = ROOT / rel
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return {r["case_id"]: r for r in out}


def gt_of(r):
    return r.get("ground_truth") or r.get("ground_truth_risk_level") or ""


def metrics(records: dict, ids) -> dict:
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for cid in ids:
        r = records.get(cid)
        if r is None:
            continue
        p, g = r.get("risk_level", ""), gt_of(r)
        if p not in IDX or g not in IDX:
            n += 1
            if g in SEVERE:
                sev_tot += 1
            under += 1
            worst = max(worst, IDX.get(g, 0))
            continue
        n += 1
        d = IDX[p] - IDX[g]
        if d == 0:
            exact += 1
        if abs(d) <= 1:
            within1 += 1
        if d < 0:
            under += 1
            worst = max(worst, -d)
        elif d > 0:
            over += 1
        if g in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_rec += 1
    pct = lambda a, b: (a / b * 100) if b else 0.0
    return dict(n=n, exact=pct(exact, n), within1=pct(within1, n),
                under=pct(under, n), over=pct(over, n),
                severe_recall=pct(sev_rec, sev_tot), sev_tot=sev_tot,
                worst_miss=worst)


def mcnemar_1s(b, c):
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) * (0.5 ** n) if n else 1.0


def unsafe(r):
    p, g = r.get("risk_level", ""), gt_of(r)
    if p not in IDX or g not in IDX:
        return True
    return IDX[p] < IDX[g]


def unsafe_mcnemar(llm: dict, ml: dict, ids):
    b = c = 0
    pairs = []
    for cid in ids:
        if cid in llm and cid in ml:
            lu, mu = unsafe(llm[cid]), unsafe(ml[cid])
            if mu and not lu:
                b += 1
            elif lu and not mu:
                c += 1
            pairs.append(cid)
    return b, c, mcnemar_1s(b, c), len(pairs)


def main():
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    cases = testset["cases"]

    tags = [tag_case(c) for c in cases]
    leaky_ids = [t["case_id"] for t in tags if t["leaky"]]
    clean_ids = [t["case_id"] for t in tags if not t["leaky"]]

    # breakdown by true class
    by_class = {}
    for t in tags:
        cls = t["true_attack_class"]
        by_class.setdefault(cls, {"n": 0, "leaky": 0})
        by_class[cls]["n"] += 1
        if t["leaky"]:
            by_class[cls]["leaky"] += 1

    # trigger-category breakdown
    named_attack_leaky = sum(
        1 for t in tags if any(tr["category"] == "named-attack" for tr in t["triggers"])
    )
    class_name_only_leaky = sum(
        1 for t in tags
        if t["leaky"] and not any(tr["category"] == "named-attack" for tr in t["triggers"])
    )

    llm = load_jsonl("results/nslkdd_ablation/tristage_llm_haiku.jsonl")
    ml = load_jsonl("results/nslkdd_ablation/ml_xgb.jsonl")
    llm_sonnet = load_jsonl("results/nslkdd_ablation/tristage_llm_sonnet.jsonl")

    all_ids = [c["case_id"] for c in cases]

    full_llm_m = metrics(llm, all_ids)
    full_ml_m = metrics(ml, all_ids)
    full_b, full_c, full_p, full_npairs = unsafe_mcnemar(llm, ml, all_ids)

    clean_llm_m = metrics(llm, clean_ids)
    clean_ml_m = metrics(ml, clean_ids)
    clean_b, clean_c, clean_p, clean_npairs = unsafe_mcnemar(llm, ml, clean_ids)

    # also report Sonnet (thesis-cited [ref] arm) on clean subset for context
    clean_llm_sonnet_m = metrics(llm_sonnet, clean_ids)
    clean_sonnet_b, clean_sonnet_c, clean_sonnet_p, _ = unsafe_mcnemar(llm_sonnet, ml, clean_ids)

    out = {
        "meta": {
            "description": (
                "Task 15 sensitivity analysis: bound how much of the NSL-KDD "
                "flagship safety finding (Tri-stage LLM Haiku vs Pure ML/XGBoost "
                "under-escalation McNemar, thesis Table safety-synthesis) survives "
                "once cases whose generated NL description names the true attack "
                "class or a distinctive named attack are excluded."
            ),
            "source_test_set": "results/nslkdd_ablation/test_set.json",
            "source_nl_generator": "scripts/_archive/run_slm_nslkdd_classification.py :: features_to_nl_nslkdd()",
            "full_set_safety_source": "results/safety_table_haiku.json (NSL-KDD (N=50))",
            "n_total": len(cases),
            "n_leaky": len(leaky_ids),
            "n_clean": len(clean_ids),
            "n_leaky_named_attack": named_attack_leaky,
            "n_leaky_class_name_only": class_name_only_leaky,
            "tagging_method": (
                "Regex over the actual generated nl_description text stored in "
                "test_set.json (the exact string shown to the LLM). A case is "
                "tagged leaky only if a matched pattern names the case's TRUE "
                "attack class (mismatched/hedge mentions that name the WRONG "
                "class are not counted as leaky)."
            ),
        },
        "leaky_by_true_class": by_class,
        "per_case_tags": tags,
        "safety_comparison": {
            "full_set_N50": {
                "tri_stage_llm_haiku": full_llm_m,
                "pure_ml_xgboost": full_ml_m,
                "unsafe_mcnemar_llm_vs_ml": {"b": full_b, "c": full_c, "p": full_p, "n_pairs": full_npairs},
            },
            "clean_subset": {
                "n_clean": len(clean_ids),
                "clean_case_ids": clean_ids,
                "tri_stage_llm_haiku": clean_llm_m,
                "pure_ml_xgboost": clean_ml_m,
                "unsafe_mcnemar_llm_vs_ml": {"b": clean_b, "c": clean_c, "p": clean_p, "n_pairs": clean_npairs},
                "ref_tri_stage_llm_sonnet": clean_llm_sonnet_m,
                "ref_unsafe_mcnemar_sonnet_vs_ml": {
                    "b": clean_sonnet_b, "c": clean_sonnet_c, "p": clean_sonnet_p,
                },
            },
            "leaky_subset_excluded_case_ids": leaky_ids,
        },
        "interpretation": (
            "See console output / assistant report for narrative interpretation; "
            "this file holds only the computed numbers and per-case tags."
        ),
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"n_total={len(cases)}  n_leaky={len(leaky_ids)}  n_clean={len(clean_ids)}")
    print(f"  named-attack leaky: {named_attack_leaky}   class-name-only leaky: {class_name_only_leaky}")
    print("Leaky by true class:", by_class)
    print()
    print("FULL N=50:  LLM under=%.1f%% sev.recall=%.1f%%  | ML under=%.1f%% sev.recall=%.1f%%  | McNemar b=%d c=%d p=%.4g" % (
        full_llm_m["under"], full_llm_m["severe_recall"], full_ml_m["under"], full_ml_m["severe_recall"], full_b, full_c, full_p))
    print("CLEAN N=%d: LLM under=%.1f%% sev.recall=%.1f%%  | ML under=%.1f%% sev.recall=%.1f%%  | McNemar b=%d c=%d p=%.4g" % (
        len(clean_ids), clean_llm_m["under"], clean_llm_m["severe_recall"], clean_ml_m["under"], clean_ml_m["severe_recall"], clean_b, clean_c, clean_p))
    print(f"Saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
