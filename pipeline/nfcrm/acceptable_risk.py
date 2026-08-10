"""NFCRM-1:2025 §6.8 acceptable risk level + treatment-required decision.

The standard requires entities to define an acceptable risk level. Risks at
or below the acceptable level may be Accepted (per §6.11); risks above
require treatment (Mitigate, Share, or Avoid).

Configuration is a single ordinal threshold expressed as a Figure 3 risk
level. Default: ``Low``. Higher values mean the entity tolerates more risk
before requiring treatment.
"""
from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Risk-level ordinal ordering per Figure 3
# ─────────────────────────────────────────────────────────────────────────────

RISK_LEVEL_ORDER: list[str] = [
    "Very Low",
    "Low",
    "Medium",
    "High",
    "Catastrophic",
]


def _level_rank(level: str) -> int:
    if level == "N/A (non-attack)":
        return -1
    if level not in RISK_LEVEL_ORDER:
        raise ValueError(
            f"unknown risk level {level!r}; expected one of {RISK_LEVEL_ORDER}"
        )
    return RISK_LEVEL_ORDER.index(level)


@dataclass
class AcceptableRiskConfig:
    """§6.8 acceptable-risk configuration.

    Attributes:
        acceptable_level_en: One of RISK_LEVEL_ORDER. Risks at or below
            this level may be accepted; risks above require treatment.
        rationale: Free text explaining the threshold choice (e.g.,
            "set to Low because we operate Critical National Infrastructure
            and tolerate minimal residual risk").
        approved_by: Who approved the threshold (per §6.8 implies the
            organisation's risk-management governance).
    """
    acceptable_level_en: str = "Low"
    rationale: str = "Default threshold; entities should override per their risk appetite."
    approved_by: str = "framework-default"

    def __post_init__(self) -> None:
        # Validate level on construction.
        _level_rank(self.acceptable_level_en)


@dataclass
class TreatmentDecision:
    """Output of §6.8 acceptable-risk evaluation."""
    treatment_required: bool
    rationale: str
    inherent_level_en: str
    acceptable_level_en: str
    clause_reference: str = "NFCRM-1:2025 §6.8"


def is_treatment_required(
    inherent_level_en: str,
    config: AcceptableRiskConfig | None = None,
) -> TreatmentDecision:
    """Decide whether a risk requires treatment per §6.8.

    Args:
        inherent_level_en: Risk level from §6.9 (Very Low / Low / Medium /
            High / Catastrophic, or "N/A (non-attack)").
        config: Acceptable-risk configuration. If None, defaults are used.

    Returns:
        TreatmentDecision with required-flag and rationale.
    """
    cfg = config if config is not None else AcceptableRiskConfig()

    if inherent_level_en == "N/A (non-attack)":
        return TreatmentDecision(
            treatment_required=False,
            rationale="Non-attack scenario; treatment not applicable.",
            inherent_level_en=inherent_level_en,
            acceptable_level_en=cfg.acceptable_level_en,
        )

    inherent_rank = _level_rank(inherent_level_en)
    acceptable_rank = _level_rank(cfg.acceptable_level_en)

    if inherent_rank <= acceptable_rank:
        return TreatmentDecision(
            treatment_required=False,
            rationale=(
                f"Inherent risk level ({inherent_level_en}) is at or below the "
                f"acceptable level ({cfg.acceptable_level_en}); risk may be Accepted "
                f"per §6.11."
            ),
            inherent_level_en=inherent_level_en,
            acceptable_level_en=cfg.acceptable_level_en,
        )

    gap = inherent_rank - acceptable_rank
    return TreatmentDecision(
        treatment_required=True,
        rationale=(
            f"Inherent risk level ({inherent_level_en}) exceeds the acceptable level "
            f"({cfg.acceptable_level_en}) by {gap} band(s); treatment is required "
            f"per §6.11. Suggested treatment options: Mitigate (apply additional "
            f"controls), Share (transfer via insurance/contract), or Avoid "
            f"(eliminate the risk source)."
        ),
        inherent_level_en=inherent_level_en,
        acceptable_level_en=cfg.acceptable_level_en,
    )
