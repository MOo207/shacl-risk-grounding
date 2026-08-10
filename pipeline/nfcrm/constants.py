"""NFCRM-1:2025 constants per Figures 3, 4, 5 of the standard.

Source: National Cybersecurity Authority (Saudi Arabia),
National Framework for Cybersecurity Risk Management (NFCRM-1: 2025),
local copy: References/2025-1-NFCRM.pdf.

These constants encode the standard's lookup tables. They are intentionally
declarative (no logic) so a reviewer can verify against the source document
line-by-line.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Likelihood levels (1..5)
# ─────────────────────────────────────────────────────────────────────────────
#
# The standard defines likelihood by EITHER time-period OR exploitability;
# the higher of the two governs.

LIKELIHOOD_LEVELS: dict[int, dict[str, str]] = {
    5: {
        "label_ar": "شبه مؤكد",
        "label_en": "Almost Certain",
        "time_period": "Multiple times per month, or several times per year",
        "exploitability": "Exploitable by unsophisticated attackers (e.g., script-kiddies) using ready-made exploit code",
    },
    4: {
        "label_ar": "مرجح",
        "label_en": "Likely",
        "time_period": "Common, approximately once per year",
        "exploitability": "Exploitable by advanced attackers via sophisticated attacks using custom tools or methods",
    },
    3: {
        "label_ar": "غير مرجح",
        "label_en": "Unlikely",
        "time_period": "Could happen at some point, every 2 to 3 years",
        "exploitability": "Exploitable by advanced attackers under specific conditions (e.g., precise vulnerability targeting, internal-network knowledge)",
    },
    2: {
        "label_ar": "نادر",
        "label_en": "Rare",
        "time_period": "Could happen approximately every 3 to 10 years",
        "exploitability": "Not directly exploitable now, but exploitability possible in the future",
    },
    1: {
        "label_ar": "نادر جدًا",
        "label_en": "Very Rare",
        "time_period": "Happens only in exceptional circumstances; less than once every 10 to 20 years",
        "exploitability": "Not currently exploitable",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Impact levels (1..5) per CIA dimension
# ─────────────────────────────────────────────────────────────────────────────

IMPACT_LEVELS: dict[int, dict[str, str]] = {
    5: {
        "label_ar": "كارثي",
        "label_en": "Catastrophic",
        "confidentiality": "Disclosure or leakage of sensitive data with major impact on the entity that the entity cannot sustain, or impact at the national level",
        "integrity": "Asset alteration with major impact on the entity that it cannot sustain, or impact at the national level",
        "availability": "Asset stoppage with major impact on the entity that it cannot sustain, or impact at the national level",
    },
    4: {
        "label_ar": "مرتفع",
        "label_en": "High",
        "confidentiality": "Disclosure or leakage of sensitive data with major impact on the entity",
        "integrity": "Asset alteration with major impact on the entity",
        "availability": "Asset stoppage with major impact on the entity",
    },
    3: {
        "label_ar": "متوسط",
        "label_en": "Medium",
        "confidentiality": "Disclosure or leakage of sensitive data with notable (tangible) impact on the entity",
        "integrity": "Asset alteration with notable (tangible) impact on the entity",
        "availability": "Asset stoppage with notable (tangible) impact on the entity",
    },
    2: {
        "label_ar": "منخفض",
        "label_en": "Low",
        "confidentiality": "Disclosure or leakage of sensitive data with minor impact on the entity",
        "integrity": "Asset alteration with minor impact on the entity",
        "availability": "Asset stoppage with minor impact on the entity",
    },
    1: {
        "label_ar": "منخفض جدًا",
        "label_en": "Very Low",
        "confidentiality": "Disclosure or leakage of non-sensitive data with negligible impact on the entity",
        "integrity": "Non-impactful asset alteration on the entity",
        "availability": "Non-impactful asset stoppage on the entity",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Risk-level bands (Risk = Likelihood × Impact, range 1..25)
# ─────────────────────────────────────────────────────────────────────────────
#
# Banding per the standard:
#   1..2   → Very Low
#   3..7   → Low
#   8..14  → Medium
#   15..19 → High
#   20..25 → Catastrophic

RISK_LEVEL_BANDS: list[tuple[int, int, str, str]] = [
    (1, 2, "Very Low", "منخفض جدًا"),
    (3, 7, "Low", "منخفض"),
    (8, 14, "Medium", "متوسط"),
    (15, 19, "High", "مرتفع"),
    (20, 25, "Catastrophic", "كارثي"),
]


def risk_level_for_score(score: int) -> tuple[str, str]:
    """Return (English label, Arabic label) for a Likelihood×Impact score.

    Per Figure 3 of NFCRM-1:2025. Scores outside [1, 25] raise ValueError.
    """
    if not (1 <= score <= 25):
        raise ValueError(f"risk score must be in [1, 25], got {score}")
    for low, high, en, ar in RISK_LEVEL_BANDS:
        if low <= score <= high:
            return en, ar
    raise AssertionError("unreachable: bands cover [1, 25]")
