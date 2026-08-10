"""NFCRM-1:2025 §6.10 Risk Register schema and entry construction.

Per the standard, the cybersecurity risk register documents the results of
inherent-risk analysis (§6.9) and is updated with treatment plans and
residual-risk values per §6.16. This module defines the entry schema and
provides constructors that compose the §6.6 scenario fields with §6.9
score fields.

The register is a list of `RiskRegisterEntry` records. Treatment fields
(decision, plan, residual) are populated by sub-project E and are
absent (None) until then.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .risk_score import RiskScore, compute_risk_score


@dataclass
class RiskScenario:
    """A §6.6 cybersecurity risk scenario: asset × threat × vulnerability."""
    scenario_id: str                      # stable identifier
    asset_id: str | None                  # FK to §6.3 asset inventory; None when unknown
    asset_name: str | None
    asset_classification: str | None      # per §6.3 (e.g., "sensitive system")
    threat_id: str | None                 # FK to §6.5 (e.g., MITRE ATT&CK technique ID)
    threat_description: str
    vulnerability_id: str | None          # FK to §6.4 (e.g., CVE ID)
    vulnerability_description: str | None
    attack_type: str                      # CIC-IDS2018 category (DoS, DDoS, ...)
    rationale: str = ""                   # why these constituents form this scenario


@dataclass
class RiskRegisterEntry:
    """A single row in the §6.10 cybersecurity risk register."""
    entry_id: str
    scenario: RiskScenario
    inherent_risk_score: int              # §6.9 score in [1, 25]
    inherent_risk_level_en: str
    inherent_risk_level_ar: str
    likelihood_value: int                 # §6.9 likelihood max(time, exp)
    likelihood_time_period: int
    likelihood_exploitability: int
    impact_value: int                     # §6.9 impact max(C, I, A)
    impact_confidentiality: int
    impact_integrity: int
    impact_availability: int
    likelihood_rationale: str
    impact_rationale: str
    currently_applied_controls: list[str] = field(default_factory=list)  # §6.7 inventory
    # §6.11–§6.16 fields (populated by sub-project E):
    treatment_decision: str | None = None       # Accept / Share / Mitigate / Avoid
    treatment_plan: str | None = None
    treatment_priority: int | None = None
    residual_likelihood: int | None = None
    residual_impact: int | None = None
    residual_risk_score: int | None = None
    residual_risk_level_en: str | None = None
    treatment_status: str | None = None         # planned / in_progress / implemented
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    clause_reference: str = "NFCRM-1:2025 §6.10"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_register_entry(
    scenario: RiskScenario,
    *,
    risk_score: RiskScore | None = None,
    currently_applied_controls: list[str] | None = None,
    entry_id: str | None = None,
) -> RiskRegisterEntry:
    """Construct a §6.10 register entry for a §6.6 scenario.

    If `risk_score` is not provided, it is computed from the scenario's
    attack_type using attack-type defaults per §6.9.

    Args:
        scenario: The §6.6 risk scenario.
        risk_score: Pre-computed §6.9 RiskScore (optional). If None,
            `compute_risk_score(scenario.attack_type)` is used.
        currently_applied_controls: §6.7 inventory of controls already in
            place against this scenario. Defaults to empty list.
        entry_id: Optional override; defaults to "RR-{scenario_id}".

    Returns:
        A populated RiskRegisterEntry. Treatment fields are None.
    """
    rs = risk_score if risk_score is not None else compute_risk_score(scenario.attack_type)
    return RiskRegisterEntry(
        entry_id=entry_id or f"RR-{scenario.scenario_id}",
        scenario=scenario,
        inherent_risk_score=rs.score,
        inherent_risk_level_en=rs.level_en,
        inherent_risk_level_ar=rs.level_ar,
        likelihood_value=rs.likelihood.value,
        likelihood_time_period=rs.likelihood.time_period,
        likelihood_exploitability=rs.likelihood.exploitability,
        impact_value=rs.impact.value,
        impact_confidentiality=rs.impact.confidentiality,
        impact_integrity=rs.impact.integrity,
        impact_availability=rs.impact.availability,
        likelihood_rationale=rs.likelihood.rationale,
        impact_rationale=rs.impact.rationale,
        currently_applied_controls=list(currently_applied_controls or []),
    )


def required_fields() -> list[str]:
    """The fields a §6.10-conformant register entry must contain.

    Used by validators to check that an entry is complete before being
    written to the register.
    """
    return [
        "entry_id",
        "scenario",
        "inherent_risk_score",
        "inherent_risk_level_en",
        "likelihood_value",
        "impact_value",
        "currently_applied_controls",
        "clause_reference",
        "last_updated",
    ]


def validate_entry(entry: RiskRegisterEntry) -> list[str]:
    """Return a list of validation problems; empty list = valid."""
    problems = []
    if not entry.entry_id:
        problems.append("entry_id is empty")
    if entry.scenario is None:
        problems.append("scenario is None")
    elif not entry.scenario.scenario_id:
        problems.append("scenario.scenario_id is empty")
    if entry.inherent_risk_score == 0 and entry.scenario and entry.scenario.attack_type != "Benign":
        problems.append(
            f"inherent_risk_score is 0 for non-Benign attack_type "
            f"{entry.scenario.attack_type!r}"
        )
    if entry.inherent_risk_score not in range(0, 26):
        problems.append(f"inherent_risk_score {entry.inherent_risk_score} out of [0, 25]")
    if entry.likelihood_value not in range(0, 6):
        problems.append(f"likelihood_value {entry.likelihood_value} out of [0, 5]")
    if entry.impact_value not in range(0, 6):
        problems.append(f"impact_value {entry.impact_value} out of [0, 5]")
    if entry.clause_reference != "NFCRM-1:2025 §6.10":
        problems.append(f"clause_reference is {entry.clause_reference!r}")
    return problems
