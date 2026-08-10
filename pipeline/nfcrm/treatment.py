"""NFCRM-1:2025 §6.11–§6.16 Treatment workflow.

Implements the Treatment phase of the NFCRM methodology:

    §6.11  Develop treatment plans (decision: Accept/Share/Mitigate/Avoid)
    §6.12  Assess residual risk after treatment
    §6.13  Make treatment decisions
    §6.14  Prioritise treatment plans
    §6.15  Implement treatment plans (status tracking)
    §6.16  Update the risk register with treatment plans + residual risks

Residual-risk computation: residual likelihood/impact each reduce by an
integer number of Figure 4/5 bands per the treatment's stated reduction.
A clamped formula keeps results in [1, 5] for the active components and
in [0, 25] for the score (0 only for Avoid decisions, where the risk
source is eliminated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .acceptable_risk import (
    AcceptableRiskConfig,
    is_treatment_required,
)
from .constants import risk_level_for_score
from .risk_register import RiskRegisterEntry

TreatmentDecisionType = Literal["Accept", "Share", "Mitigate", "Avoid"]
TreatmentStatus = Literal["planned", "in_progress", "implemented", "n/a"]


@dataclass
class Treatment:
    """A treatment plan for a single risk register entry per §6.11–§6.15."""
    decision: TreatmentDecisionType
    plan_description: str
    likelihood_reduction: int = 0       # bands by which likelihood drops (0..4)
    impact_reduction: int = 0           # bands by which impact drops (0..4)
    priority: int = 0                   # §6.14 priority (1 = highest)
    status: TreatmentStatus = "planned"
    rationale: str = ""
    clause_reference: str = "NFCRM-1:2025 §6.11"

    def __post_init__(self) -> None:
        if self.decision not in {"Accept", "Share", "Mitigate", "Avoid"}:
            raise ValueError(
                f"decision must be Accept/Share/Mitigate/Avoid, got {self.decision!r}"
            )
        if not (0 <= self.likelihood_reduction <= 4):
            raise ValueError("likelihood_reduction must be in [0, 4]")
        if not (0 <= self.impact_reduction <= 4):
            raise ValueError("impact_reduction must be in [0, 4]")
        if self.status not in {"planned", "in_progress", "implemented", "n/a"}:
            raise ValueError(f"unknown status {self.status!r}")


@dataclass
class ResidualRisk:
    """§6.12 residual-risk computation result."""
    residual_likelihood: int
    residual_impact: int
    residual_score: int                 # 0 for Avoid; 1..25 otherwise
    residual_level_en: str              # Figure 3 band, or "Eliminated" for Avoid
    rationale: str
    clause_reference: str = "NFCRM-1:2025 §6.12"


def compute_residual_risk(
    entry: RiskRegisterEntry,
    treatment: Treatment,
) -> ResidualRisk:
    """Compute residual risk per §6.12 from inherent risk + treatment.

    Logic by decision:
      Accept   — no reduction; residual == inherent.
      Avoid    — risk source eliminated; residual_score = 0, level "Eliminated".
      Mitigate — reduce likelihood and/or impact by the treatment's stated
                 band reductions (clamped to floor 1).
      Share    — typically reduces impact (transfer of consequences); we
                 apply impact_reduction only and ignore likelihood_reduction.
    """
    if treatment.decision == "Accept":
        return ResidualRisk(
            residual_likelihood=entry.likelihood_value,
            residual_impact=entry.impact_value,
            residual_score=entry.inherent_risk_score,
            residual_level_en=entry.inherent_risk_level_en,
            rationale="Decision = Accept; residual risk equals inherent risk.",
        )

    if treatment.decision == "Avoid":
        return ResidualRisk(
            residual_likelihood=0,
            residual_impact=0,
            residual_score=0,
            residual_level_en="Eliminated",
            rationale="Decision = Avoid; risk source eliminated. Residual = 0.",
        )

    # Mitigate / Share: reduce by the band amounts.
    likelihood_red = treatment.likelihood_reduction
    impact_red = treatment.impact_reduction
    if treatment.decision == "Share":
        # Share typically transfers impact (e.g., cyber insurance) but does
        # not change likelihood of the underlying event.
        likelihood_red = 0

    new_l = max(1, entry.likelihood_value - likelihood_red)
    new_i = max(1, entry.impact_value - impact_red)
    new_score = new_l * new_i
    level_en, _ = risk_level_for_score(new_score)
    return ResidualRisk(
        residual_likelihood=new_l,
        residual_impact=new_i,
        residual_score=new_score,
        residual_level_en=level_en,
        rationale=(
            f"Decision = {treatment.decision}; "
            f"likelihood reduced by {likelihood_red} band(s) ({entry.likelihood_value} -> {new_l}); "
            f"impact reduced by {impact_red} band(s) ({entry.impact_value} -> {new_i}); "
            f"residual score = {new_l} * {new_i} = {new_score}."
        ),
    )


def apply_treatment(
    entry: RiskRegisterEntry,
    treatment: Treatment,
) -> RiskRegisterEntry:
    """Apply a §6.11 treatment to an entry; updates §6.12 residual-risk fields.

    This is the §6.16 register-update step (treatment plan + residual risk).
    The original `entry` is mutated and returned for fluent chaining.
    """
    residual = compute_residual_risk(entry, treatment)
    entry.treatment_decision = treatment.decision
    entry.treatment_plan = treatment.plan_description
    entry.treatment_priority = treatment.priority
    entry.residual_likelihood = residual.residual_likelihood
    entry.residual_impact = residual.residual_impact
    entry.residual_risk_score = residual.residual_score
    entry.residual_risk_level_en = residual.residual_level_en
    entry.treatment_status = treatment.status
    return entry


def prioritise(
    entries: list[RiskRegisterEntry],
) -> list[RiskRegisterEntry]:
    """§6.14 prioritisation: sort by inherent risk score, descending.

    Sets `treatment_priority` on each entry (1 = highest). Stable sort
    preserves register insertion order for ties.
    """
    indexed = list(enumerate(entries))
    indexed.sort(key=lambda kv: (-kv[1].inherent_risk_score, kv[0]))
    for new_priority, (_, entry) in enumerate(indexed, start=1):
        entry.treatment_priority = new_priority
    return [e for _, e in indexed]


def recommend_treatment(
    entry: RiskRegisterEntry,
    config: AcceptableRiskConfig | None = None,
) -> Treatment:
    """§6.13 recommend a default treatment based on inherent level + acceptable threshold.

    Heuristic:
      - If inherent <= acceptable: recommend Accept.
      - If inherent is Catastrophic AND attack is data-exfiltration capable:
        recommend Avoid + Mitigate combined (decision=Mitigate, large reduction).
      - Otherwise: recommend Mitigate with reductions sized to bring residual
        to the acceptable level (best-effort, clamped to [0, 4]).
    """
    decision_info = is_treatment_required(entry.inherent_risk_level_en, config=config)
    if not decision_info.treatment_required:
        return Treatment(
            decision="Accept",
            plan_description=(
                f"No treatment required: inherent risk ({entry.inherent_risk_level_en}) "
                f"is within acceptable level ({decision_info.acceptable_level_en})."
            ),
            rationale=decision_info.rationale,
            status="n/a",
        )

    # Mitigate to bring residual down to acceptable level (best-effort heuristic).
    # Each Likelihood band drop reduces score by impact_value;
    # each Impact band drop reduces score by likelihood_value.
    cfg = config if config is not None else AcceptableRiskConfig()
    from .acceptable_risk import _level_rank

    # Target: residual rank <= acceptable rank.
    # Brute force over band reductions to find a small (l, i) reduction
    # that achieves the goal.
    target_rank = _level_rank(cfg.acceptable_level_en)
    best: tuple[int, int] = (4, 4)  # max reduction as fallback
    best_total = 8
    for l_red in range(0, 5):
        for i_red in range(0, 5):
            new_l = max(1, entry.likelihood_value - l_red)
            new_i = max(1, entry.impact_value - i_red)
            new_score = new_l * new_i
            new_level, _ = risk_level_for_score(new_score)
            if _level_rank(new_level) <= target_rank:
                total = l_red + i_red
                if total < best_total:
                    best = (l_red, i_red)
                    best_total = total
    likelihood_red, impact_red = best

    return Treatment(
        decision="Mitigate",
        plan_description=(
            f"Mitigation: apply additional controls to reduce likelihood by "
            f"{likelihood_red} band(s) and impact by {impact_red} band(s). "
            f"Target residual: {cfg.acceptable_level_en}."
        ),
        likelihood_reduction=likelihood_red,
        impact_reduction=impact_red,
        rationale=(
            f"Inherent {entry.inherent_risk_level_en} exceeds acceptable "
            f"{cfg.acceptable_level_en}; Mitigate is the lowest-cost decision "
            f"that does not eliminate operational use of the asset."
        ),
        status="planned",
    )
