"""NFCRM-1:2025 §6.9 likelihood computation per Figure 5.

The standard defines likelihood as the higher of (time_period_score,
exploitability_score). Both factors yield a value in 1..5; the result is
the maximum.

For CIC-IDS2018 attack types we do not have per-asset frequency or
per-CVE EPSS scores. We use attack-type defaults derived from public
threat-intelligence consensus (CISA KEV, MITRE ATT&CK incidence
reporting). These defaults are overrideable per scenario.
"""
from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Default attack-type likelihood per CIC-IDS2018 categories
# ─────────────────────────────────────────────────────────────────────────────
#
# Rationale per category (see audit/nfcrm_compliance_matrix.md and
# Figure 5 of NFCRM-1:2025):
#
# DDoS, BruteForce: ready-made tools (LOIC, HOIC, Hydra, Medusa) → exploitability=5
#                   incidence: organisations report continuous BruteForce attempts
#                   on internet-facing services → time_period=5 OR 4
# DoS:            Slowloris, Slowhttptest are public tools → exploitability=5;
#                 organisational incidence: typically a few times per year → 4
# WebAttack:      SQLi/XSS payloads exist in many automated scanners
#                 (sqlmap) → exploitability=5; targeted incidence is lower → 3-4
# Infiltration:   requires advanced TTPs (lateral movement, evasion);
#                 exploitability=3-4; incidence is rarer → 3
# Benign:         not an attack — likelihood is undefined (we return 0 as sentinel)

ATTACK_LIKELIHOOD_DEFAULTS: dict[str, dict[str, int]] = {
    "DoS":          {"time_period": 4, "exploitability": 5},
    "DDoS":         {"time_period": 4, "exploitability": 5},
    "BruteForce":   {"time_period": 5, "exploitability": 5},
    "WebAttack":    {"time_period": 4, "exploitability": 5},
    "Infiltration": {"time_period": 3, "exploitability": 3},
    "Benign":       {"time_period": 0, "exploitability": 0},
}


@dataclass
class LikelihoodResult:
    """Per-scenario likelihood per Figure 5 of NFCRM-1:2025."""
    value: int                 # 1..5 (or 0 for non-attack)
    time_period: int           # 1..5 component
    exploitability: int        # 1..5 component
    rationale: str             # human-readable explanation


def compute_likelihood(
    attack_type: str,
    *,
    time_period_override: int | None = None,
    exploitability_override: int | None = None,
) -> LikelihoodResult:
    """Compute likelihood per Figure 5 of NFCRM-1:2025.

    Likelihood is the maximum of (time_period_score, exploitability_score).
    Both inputs default to attack-type heuristics; per-scenario overrides
    (e.g., from EPSS scores or organisational incidence data) take priority.

    Args:
        attack_type: One of CIC-IDS2018 categories (DoS, DDoS, BruteForce,
                     WebAttack, Infiltration, Benign).
        time_period_override: Override 1..5 for time-period component.
        exploitability_override: Override 1..5 for exploitability component.

    Returns:
        LikelihoodResult with value, components, and rationale.
    """
    if attack_type not in ATTACK_LIKELIHOOD_DEFAULTS:
        raise ValueError(
            f"unknown attack_type {attack_type!r}; "
            f"expected one of {sorted(ATTACK_LIKELIHOOD_DEFAULTS)}"
        )
    defaults = ATTACK_LIKELIHOOD_DEFAULTS[attack_type]

    tp = time_period_override if time_period_override is not None else defaults["time_period"]
    ex = exploitability_override if exploitability_override is not None else defaults["exploitability"]

    if not (0 <= tp <= 5):
        raise ValueError(f"time_period must be in [0, 5], got {tp}")
    if not (0 <= ex <= 5):
        raise ValueError(f"exploitability must be in [0, 5], got {ex}")

    value = max(tp, ex)
    if value == 0:
        rationale = (
            f"Non-attack ({attack_type}) — likelihood not applicable per Figure 5."
        )
    else:
        rationale = (
            f"Likelihood = max(time_period={tp}, exploitability={ex}) = {value} "
            f"per NFCRM-1:2025 Figure 5. Defaults for {attack_type} reflect "
            f"public threat-intelligence consensus."
        )
    return LikelihoodResult(
        value=value,
        time_period=tp,
        exploitability=ex,
        rationale=rationale,
    )
