from __future__ import annotations
from collections import defaultdict
from typing import Any, Optional

RISK_LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
ARM_NAMES = ["rule", "ml", "rule_ml", "rule_llm"]

REQUIRED_FIELDS = [
    "risk_level",
    "recommended_control_id",
    "nfcrm_clauses",
    "narrative",
    "evidence_refs",
]


def stratified_slice(cases: list[dict], n_per_class: Optional[int]) -> list[dict]:
    """Return first n_per_class cases from each (subset, true_attack_class) bucket."""
    if n_per_class is None:
        return cases
    buckets: dict[tuple, list] = defaultdict(list)
    for c in cases:
        key = (c.get("subset", ""), c.get("true_attack_class", ""))
        buckets[key].append(c)
    result = []
    for key in sorted(buckets):
        result.extend(buckets[key][:n_per_class])
    return result


def validate_artifact_schema(artifact: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if "risk_level" in artifact and artifact["risk_level"] not in RISK_LEVELS:
        errors.append(
            f"risk_level={artifact['risk_level']!r} not in {RISK_LEVELS}"
        )
    if "nfcrm_clauses" in artifact and not isinstance(artifact["nfcrm_clauses"], list):
        errors.append("nfcrm_clauses must be a list")
    if "evidence_refs" in artifact and not isinstance(artifact["evidence_refs"], list):
        errors.append("evidence_refs must be a list")
    return len(errors) == 0, errors
