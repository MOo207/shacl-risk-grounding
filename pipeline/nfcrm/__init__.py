"""NFCRM-1:2025 compliance computations.

Module structure:
    constants.py    - Figures 3, 4, 5 lookup tables
    likelihood.py   - section 6.9 likelihood per Figure 5
    impact.py       - section 6.9 impact per Figure 4
    risk_score.py   - section 6.9 L*I -> Figure 3 band

All public symbols are re-exported here for ergonomic import.
"""
from .constants import (
    IMPACT_LEVELS,
    LIKELIHOOD_LEVELS,
    RISK_LEVEL_BANDS,
    risk_level_for_score,
)
from .impact import ATTACK_IMPACT_DEFAULTS, ImpactResult, compute_impact
from .likelihood import (
    ATTACK_LIKELIHOOD_DEFAULTS,
    LikelihoodResult,
    compute_likelihood,
)
from .acceptable_risk import (
    RISK_LEVEL_ORDER,
    AcceptableRiskConfig,
    TreatmentDecision,
    is_treatment_required,
)
from .inventories import (
    AssetEntry,
    ThreatEntry,
    VulnerabilityEntry,
    build_asset_inventory,
    build_threat_inventory,
    build_vulnerability_inventory,
    export_inventories,
)
from .monitoring import (
    STATISTICAL_DATA_CRITERIA,
    Report,
    TriggerSet,
    detect_triggers,
    generate_report,
)
from .treatment import (
    ResidualRisk,
    Treatment,
    apply_treatment,
    compute_residual_risk,
    prioritise,
    recommend_treatment,
)
from .risk_register import (
    RiskRegisterEntry,
    RiskScenario,
    build_register_entry,
    required_fields,
    validate_entry,
)
from .risk_score import RiskScore, compute_risk_score
from .asset_risk import (
    AssetScenarioRisk,
    AssetView,
    build_asset_views,
    cvss_kev_to_exploitability,
    criticality_to_cia_overrides,
    load_arg,
    score_arg,
    score_asset_scenario,
)

__all__ = [
    "AssetScenarioRisk",
    "AssetView",
    "build_asset_views",
    "criticality_to_cia_overrides",
    "cvss_kev_to_exploitability",
    "load_arg",
    "score_arg",
    "score_asset_scenario",
    "ATTACK_IMPACT_DEFAULTS",
    "ATTACK_LIKELIHOOD_DEFAULTS",
    "AcceptableRiskConfig",
    "IMPACT_LEVELS",
    "ImpactResult",
    "LIKELIHOOD_LEVELS",
    "LikelihoodResult",
    "RISK_LEVEL_BANDS",
    "RISK_LEVEL_ORDER",
    "RiskRegisterEntry",
    "RiskScenario",
    "RiskScore",
    "TreatmentDecision",
    "build_register_entry",
    "compute_impact",
    "compute_likelihood",
    "compute_risk_score",
    "is_treatment_required",
    "required_fields",
    "risk_level_for_score",
    "validate_entry",
]
