"""NFCRM-1:2025 §6.9 risk-score computation: Likelihood × Impact → Figure 3 band."""
from __future__ import annotations

from dataclasses import dataclass

from .constants import risk_level_for_score
from .impact import ImpactResult, compute_impact
from .likelihood import LikelihoodResult, compute_likelihood


@dataclass
class RiskScore:
    """Inherent risk score per NFCRM-1:2025 §6.9.

    Attributes:
        score: Likelihood × Impact (range 1..25; 0 for non-attack scenarios).
        level_en: English risk-level label (Very Low / Low / Medium / High / Catastrophic).
        level_ar: Arabic risk-level label.
        likelihood: LikelihoodResult component.
        impact: ImpactResult component.
        clause_reference: Always "NFCRM-1:2025 §6.9".
    """
    score: int
    level_en: str
    level_ar: str
    likelihood: LikelihoodResult
    impact: ImpactResult
    clause_reference: str = "NFCRM-1:2025 §6.9"


def compute_risk_score(
    attack_type: str,
    *,
    time_period_override: int | None = None,
    exploitability_override: int | None = None,
    c_override: int | None = None,
    i_override: int | None = None,
    a_override: int | None = None,
) -> RiskScore:
    """Compute inherent risk score per NFCRM-1:2025 §6.9 (Figures 3, 4, 5).

    Args:
        attack_type: CIC-IDS2018 category.
        time_period_override / exploitability_override: Per-scenario overrides
            for likelihood components.
        c_override / i_override / a_override: Per-scenario overrides for
            impact CIA components.

    Returns:
        RiskScore object with score, level (en/ar), and component breakdown.
    """
    likelihood = compute_likelihood(
        attack_type,
        time_period_override=time_period_override,
        exploitability_override=exploitability_override,
    )
    impact = compute_impact(
        attack_type,
        c_override=c_override,
        i_override=i_override,
        a_override=a_override,
    )

    if likelihood.value == 0 or impact.value == 0:
        return RiskScore(
            score=0,
            level_en="N/A (non-attack)",
            level_ar="غير قابل للتطبيق",
            likelihood=likelihood,
            impact=impact,
        )

    score = likelihood.value * impact.value
    level_en, level_ar = risk_level_for_score(score)
    return RiskScore(
        score=score,
        level_en=level_en,
        level_ar=level_ar,
        likelihood=likelihood,
        impact=impact,
    )
