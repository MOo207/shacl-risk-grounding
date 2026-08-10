"""NFCRM-1:2025 §6.17–§6.20 monitoring, reports, and trigger detection.

  §6.17  Develop risk-management reports
  §6.18  Statistical-data collection criteria
  §6.19  Periodic follow-up on treatment execution
  §6.20  Trigger-based re-execution of methodology phases on changes

The framework here is deliberately simple: a register snapshot in JSON
form produces a Report with the §6.18 statistics. A pair of snapshots
yields a TriggerSet describing what has changed per the §6.20 trigger
classes (treatment plans, threats, scenarios, controls, infrastructure).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .acceptable_risk import RISK_LEVEL_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# §6.18 — Statistical-data criteria
# ─────────────────────────────────────────────────────────────────────────────
#
# Per the standard, entities must define what statistics they collect on
# risk monitoring. We define a minimal default set that the framework can
# always compute from a register snapshot.

STATISTICAL_DATA_CRITERIA: list[dict[str, str]] = [
    {
        "id": "STAT-01",
        "name": "Risk-level distribution",
        "computed_from": "register entries grouped by inherent_risk_level_en",
        "unit": "count per level",
    },
    {
        "id": "STAT-02",
        "name": "Treatment-status distribution",
        "computed_from": "register entries grouped by treatment_status",
        "unit": "count per status",
    },
    {
        "id": "STAT-03",
        "name": "Residual-risk-level distribution",
        "computed_from": "register entries grouped by residual_risk_level_en (post-treatment)",
        "unit": "count per level",
    },
    {
        "id": "STAT-04",
        "name": "Treatment-decision mix",
        "computed_from": "register entries grouped by treatment_decision",
        "unit": "count per decision",
    },
    {
        "id": "STAT-05",
        "name": "Above-acceptable count",
        "computed_from": "register entries where residual_risk exceeds acceptable_level",
        "unit": "count",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# §6.17 — Risk-management report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Report:
    """A periodic NFCRM §6.17 report computed from a register snapshot."""
    generated_at: str
    n_entries: int
    inherent_distribution: dict[str, int]
    residual_distribution: dict[str, int]
    treatment_status_distribution: dict[str, int]
    treatment_decision_distribution: dict[str, int]
    above_acceptable_count: int
    above_acceptable_rationale: str
    clause_reference: str = "NFCRM-1:2025 §6.17 + §6.18"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "clause_reference": self.clause_reference,
            "n_entries": self.n_entries,
            "inherent_distribution": self.inherent_distribution,
            "residual_distribution": self.residual_distribution,
            "treatment_status_distribution": self.treatment_status_distribution,
            "treatment_decision_distribution": self.treatment_decision_distribution,
            "above_acceptable_count": self.above_acceptable_count,
            "above_acceptable_rationale": self.above_acceptable_rationale,
        }


def _level_rank(level: str | None) -> int:
    if not level:
        return -2
    if level == "N/A (non-attack)" or level == "Eliminated":
        return -1
    if level not in RISK_LEVEL_ORDER:
        return -2  # unknown — won't compare against acceptable
    return RISK_LEVEL_ORDER.index(level)


def generate_report(
    register_entries: list[dict],
    *,
    acceptable_level_en: str = "Low",
) -> Report:
    """Produce a §6.17 report from a list of register entry dicts.

    `register_entries` is the dict form (e.g., from a register JSON file).
    """
    def bump(d: dict, key: str | None) -> None:
        if key is None:
            key = "(unset)"
        d[key] = d.get(key, 0) + 1

    inherent: dict[str, int] = {}
    residual: dict[str, int] = {}
    statuses: dict[str, int] = {}
    decisions: dict[str, int] = {}
    above_count = 0
    acceptable_rank = _level_rank(acceptable_level_en)

    for e in register_entries:
        bump(inherent, e.get("inherent_risk_level_en"))
        bump(residual, e.get("residual_risk_level_en") or "(no residual)")
        bump(statuses, e.get("treatment_status") or "(unset)")
        bump(decisions, e.get("treatment_decision") or "(unset)")
        residual_level = e.get("residual_risk_level_en")
        if residual_level and _level_rank(residual_level) > acceptable_rank:
            above_count += 1

    return Report(
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_entries=len(register_entries),
        inherent_distribution=inherent,
        residual_distribution=residual,
        treatment_status_distribution=statuses,
        treatment_decision_distribution=decisions,
        above_acceptable_count=above_count,
        above_acceptable_rationale=(
            f"Counts entries whose residual risk strictly exceeds the acceptable "
            f"level ({acceptable_level_en}). These warrant escalation per §6.5 "
            f"(immediate notification of catastrophic-level risks to NCA)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §6.20 — Trigger detection
# ─────────────────────────────────────────────────────────────────────────────
#
# The standard requires re-execution of methodology phases on change to
# any of: treatment plans, threats, scenarios, controls, infrastructure.
# We compute a diff between two register snapshots.

@dataclass
class TriggerSet:
    """Detected §6.20 triggers between two register snapshots."""
    treatment_plan_changes: list[str] = field(default_factory=list)
    threat_changes: list[str] = field(default_factory=list)
    scenario_changes: list[str] = field(default_factory=list)
    control_changes: list[str] = field(default_factory=list)
    infrastructure_changes: list[str] = field(default_factory=list)
    clause_reference: str = "NFCRM-1:2025 §6.20"

    @property
    def any_triggered(self) -> bool:
        return any([
            self.treatment_plan_changes, self.threat_changes,
            self.scenario_changes, self.control_changes,
            self.infrastructure_changes,
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_reference": self.clause_reference,
            "any_triggered": self.any_triggered,
            "treatment_plan_changes": self.treatment_plan_changes,
            "threat_changes": self.threat_changes,
            "scenario_changes": self.scenario_changes,
            "control_changes": self.control_changes,
            "infrastructure_changes": self.infrastructure_changes,
        }


def _index(entries: list[dict]) -> dict[str, dict]:
    return {str(e.get("entry_id")): e for e in entries}


def detect_triggers(
    previous: list[dict] | None,
    current: list[dict],
) -> TriggerSet:
    """Detect §6.20 triggers by diffing two register snapshots.

    `previous` may be None (no prior snapshot) — in which case all entries
    in `current` are reported as scenario_changes (they are new).
    """
    triggers = TriggerSet()
    cur_by_id = _index(current)
    if previous is None:
        triggers.scenario_changes = [f"new entry: {eid}" for eid in cur_by_id]
        return triggers
    prev_by_id = _index(previous)

    new_ids = set(cur_by_id) - set(prev_by_id)
    removed_ids = set(prev_by_id) - set(cur_by_id)
    common_ids = set(cur_by_id) & set(prev_by_id)

    for eid in sorted(new_ids):
        triggers.scenario_changes.append(f"new entry: {eid}")
    for eid in sorted(removed_ids):
        triggers.scenario_changes.append(f"removed entry: {eid}")

    for eid in sorted(common_ids):
        prev_e = prev_by_id[eid]
        cur_e = cur_by_id[eid]

        if prev_e.get("treatment_plan") != cur_e.get("treatment_plan") or \
                prev_e.get("treatment_decision") != cur_e.get("treatment_decision") or \
                prev_e.get("treatment_status") != cur_e.get("treatment_status"):
            triggers.treatment_plan_changes.append(
                f"{eid}: plan/decision/status changed"
            )

        prev_threat = (prev_e.get("scenario") or {}).get("threat_id")
        cur_threat = (cur_e.get("scenario") or {}).get("threat_id")
        if prev_threat != cur_threat:
            triggers.threat_changes.append(f"{eid}: threat {prev_threat} -> {cur_threat}")

        prev_ctrl = sorted(prev_e.get("currently_applied_controls") or [])
        cur_ctrl = sorted(cur_e.get("currently_applied_controls") or [])
        if prev_ctrl != cur_ctrl:
            triggers.control_changes.append(
                f"{eid}: controls changed ({len(prev_ctrl)} -> {len(cur_ctrl)})"
            )

        prev_asset = (prev_e.get("scenario") or {}).get("asset_id")
        cur_asset = (cur_e.get("scenario") or {}).get("asset_id")
        if prev_asset != cur_asset:
            triggers.infrastructure_changes.append(
                f"{eid}: asset {prev_asset} -> {cur_asset}"
            )
    return triggers
