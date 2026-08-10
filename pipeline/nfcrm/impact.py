"""NFCRM-1:2025 §6.9 impact computation per Figure 4.

Impact has three dimensions (C, I, A) each scored 1..5. The standard does
not specify how to aggregate the three dimensions into a single impact
score for the L×I matrix; we use max(C, I, A) per scenario, which is the
conservative (worst-case) choice consistent with how Figure 2 cells are
read in practice.

For CIC-IDS2018 attack types we do not have per-asset CIA classification.
We use attack-type defaults; per-scenario overrides (e.g., from CMDB
asset criticality) take priority.
"""
from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Default attack-type impact per CIC-IDS2018 categories (CIA scores 1..5)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rationale per category (see Figure 4 descriptions):
#
# DoS:          flooding/exhaustion → asset stoppage with notable to major impact
#               on the entity. C and I unaffected (no data exposure or alteration).
# DDoS:         distributed flood → asset stoppage with major impact, possibly
#               national-level when targeting CNI. C and I unaffected.
# BruteForce:   credential compromise enables data leakage and modification.
#               If the attack reaches the post-auth phase, C/I impact is high
#               on the asset. Availability impact during the attack is minor
#               (account lockouts).
# WebAttack:    SQLi/XSS/etc. allow exfiltration (C) and data alteration (I).
#               Some can also DoS via expensive queries (A=2-3).
# Infiltration: deep compromise → unrestricted lateral movement → leakage of
#               sensitive data with major impact (C=4-5), unrestricted asset
#               alteration (I=4-5), degraded operations (A=3).
# Benign:       not an attack — impact undefined (return 0 as sentinel).

ATTACK_IMPACT_DEFAULTS: dict[str, dict[str, int]] = {
    "DoS":          {"C": 1, "I": 1, "A": 4},
    "DDoS":         {"C": 1, "I": 1, "A": 5},
    "BruteForce":   {"C": 4, "I": 3, "A": 2},
    "WebAttack":    {"C": 4, "I": 4, "A": 2},
    "Infiltration": {"C": 5, "I": 5, "A": 3},
    "Benign":       {"C": 0, "I": 0, "A": 0},
}


@dataclass
class ImpactResult:
    """Per-scenario impact per Figure 4 of NFCRM-1:2025."""
    value: int             # max(C, I, A), in 1..5 (or 0 for non-attack)
    confidentiality: int   # 1..5
    integrity: int         # 1..5
    availability: int      # 1..5
    rationale: str


def compute_impact(
    attack_type: str,
    *,
    c_override: int | None = None,
    i_override: int | None = None,
    a_override: int | None = None,
) -> ImpactResult:
    """Compute impact per Figure 4 of NFCRM-1:2025.

    Three CIA dimensions are scored independently. Per-scenario overrides
    (e.g., from CMDB asset criticality or data-classification labels) take
    priority over attack-type defaults. The aggregate impact value used in
    the L×I matrix is max(C, I, A) — the worst-case dimension.

    Args:
        attack_type: One of CIC-IDS2018 categories.
        c_override / i_override / a_override: Override 1..5 per CIA dimension.

    Returns:
        ImpactResult with aggregate value, components, and rationale.
    """
    if attack_type not in ATTACK_IMPACT_DEFAULTS:
        raise ValueError(
            f"unknown attack_type {attack_type!r}; "
            f"expected one of {sorted(ATTACK_IMPACT_DEFAULTS)}"
        )
    defaults = ATTACK_IMPACT_DEFAULTS[attack_type]

    c = c_override if c_override is not None else defaults["C"]
    i = i_override if i_override is not None else defaults["I"]
    a = a_override if a_override is not None else defaults["A"]

    for label, val in [("C", c), ("I", i), ("A", a)]:
        if not (0 <= val <= 5):
            raise ValueError(f"{label} must be in [0, 5], got {val}")

    value = max(c, i, a)
    if value == 0:
        rationale = (
            f"Non-attack ({attack_type}) — impact not applicable per Figure 4."
        )
    else:
        rationale = (
            f"Impact = max(C={c}, I={i}, A={a}) = {value} per NFCRM-1:2025 "
            f"Figure 4 (worst-case CIA dimension). Defaults for {attack_type} "
            f"reflect typical CIA exposure pattern of the attack class."
        )
    return ImpactResult(
        value=value,
        confidentiality=c,
        integrity=i,
        availability=a,
        rationale=rationale,
    )
