"""Pure Rule arm — NFCRM §6.9 deterministic risk scoring.

For each test case, calls compute_risk_score(attack_type) and maps the result
to a canonical risk-artifact dict.  No LLM or ML inference is used.

Output: results/llm_experiment/raw/run_1_arm_rule.jsonl (one JSON object per line).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a top-level script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.nfcrm.risk_score import compute_risk_score
from scripts.llm_artifact_lib import RISK_LEVELS, stratified_slice

# ── Static attack → recommended control mapping ───────────────────────────────
ATTACK_TO_CONTROL: dict[str, str] = {
    "Benign": "SS6.1-MON-1",
    "BruteForce": "SS6.7-AC-1",
    "DoS": "SS6.7-NET-1",
    "DDoS": "SS6.7-NET-2",
    "Infiltration": "SS6.5-NET-1",
    "WebAttack": "SS6.7-APP-1",
    "Bot": "SS6.7-NET-3",
}

# Fallback ordering used when level_en is outside RISK_LEVELS.
_LEVEL_FALLBACK: dict[str, str] = {
    "N/A (non-attack)": "Very Low",
}


def _normalise_level(raw: str) -> str:
    """Map compute_risk_score level_en to a canonical RISK_LEVELS entry.

    - If already valid, return as-is.
    - "N/A (non-attack)" → "Very Low" (Benign traffic).
    - Any other unknown string → "Medium" (conservative default).
    """
    if raw in RISK_LEVELS:
        return raw
    return _LEVEL_FALLBACK.get(raw, "Medium")


def run_rule_on_case(case: dict) -> dict:
    """Produce a risk artifact for *case* using NFCRM §6.9 rule engine only.

    Args:
        case: A test-set case dict (see results/llm_experiment/test_set.json).

    Returns:
        Risk artifact dict containing the 5 required fields plus ``case_id``
        and ``arm`` metadata for downstream aggregation.
        When ``true_attack_class`` is masked (Set B), returns a sentinel
        dict with ``masked_arm_fail=True`` and ``risk_level="Cannot_Determine"``
        so the aggregator can count this as an incorrect prediction without
        treating it as data leakage.
    """
    attack_type: str = case["true_attack_class"]
    masked: list[str] = case.get("masked_fields", [])

    # Cannot apply rule without attack class — return explicit failure marker.
    if "true_attack_class" in masked:
        return {
            "case_id": case["case_id"],
            "arm": "rule",
            "risk_level": "Cannot_Determine",
            "recommended_control_id": "",
            "nfcrm_clauses": [],
            "narrative": "",
            "evidence_refs": [],
            "masked_arm_fail": True,
        }

    # Build keyword overrides — skip criticality when masked.
    kwargs: dict = {}
    if "asset_criticality" not in masked:
        # Map asset criticality label → CIA impact numeric override.
        crit = case.get("asset", {}).get("criticality", "").lower()
        crit_map = {"low": 2, "medium": 3, "high": 4, "critical": 5}
        c_val = crit_map.get(crit)
        if c_val is not None:
            kwargs["c_override"] = c_val
            kwargs["i_override"] = c_val
            kwargs["a_override"] = c_val

    # Compute risk score via NFCRM §6.9.
    score_result = compute_risk_score(attack_type, **kwargs)
    risk_level = _normalise_level(score_result.level_en)

    # Build evidence_refs from paired_cve and ARG neighbourhood.
    evidence_refs: list[str] = [score_result.clause_reference]
    paired = case.get("paired_cve")
    if paired and "cve_id" in paired:
        evidence_refs.append(paired["cve_id"])
    arg = case.get("arg_neighborhood", {})
    for cve_entry in arg.get("related_cves", [])[:3]:
        cve_id = cve_entry.get("id") or cve_entry.get("cve_id")
        if cve_id and cve_id not in evidence_refs:
            evidence_refs.append(cve_id)

    return {
        # ── Metadata (not schema-required but needed for aggregation) ──────
        "case_id": case["case_id"],
        "arm": "rule",
        # ── Required artifact fields ──────────────────────────────────────
        "risk_level": risk_level,
        "recommended_control_id": ATTACK_TO_CONTROL.get(attack_type, "SS6.1-MON-1"),
        "nfcrm_clauses": [score_result.clause_reference],
        "narrative": "",
        "evidence_refs": evidence_refs,
        "masked_arm_fail": False,
    }


def main() -> None:
    """Run the Pure Rule arm and write JSONL output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="results/llm_experiment/test_set.json")
    ap.add_argument("--out", default="results/llm_experiment/raw/run_1_arm_rule.jsonl")
    ap.add_argument("--n-per-class", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.test_set).open(encoding="utf-8") as fh:
        all_cases = json.load(fh)["cases"]
    cases = stratified_slice(all_cases, args.n_per_class)
    print(f"Rule arm | Cases: {len(cases)}")

    artifacts: list[dict] = []
    for case in cases:
        artifact = run_rule_on_case(case)
        artifact["ground_truth_risk_level"] = case.get("ground_truth_risk_level", "")
        artifact["true_attack_class"] = case.get("true_attack_class", "")
        artifact["subset"] = case.get("subset", "")
        artifacts.append(artifact)

    with out_path.open("w", encoding="utf-8") as fh:
        for artifact in artifacts:
            fh.write(json.dumps(artifact, ensure_ascii=False) + "\n")

    from collections import Counter
    correct = sum(1 for a in artifacts if a["risk_level"] == a["ground_truth_risk_level"])
    level_counts = Counter(a["risk_level"] for a in artifacts)
    print(f"Wrote {len(artifacts)} rule-arm artifacts to {out_path} | acc={correct/len(artifacts)*100:.1f}%")
    for level in ["Very Low", "Low", "Medium", "High", "Catastrophic"]:
        count = level_counts.get(level, 0)
        if count:
            print(f"  {level}: {count}")


if __name__ == "__main__":
    main()
