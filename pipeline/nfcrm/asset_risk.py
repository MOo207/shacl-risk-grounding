"""NFCRM-1:2025 §6.9 ASSET-SPECIFIC risk scoring from the Asset Relationship Graph.

This module replaces the attack-type-constant §6.9 scoring (every DDoS flow
scores 25, every DoS 20, regardless of asset) with *per-asset* inherent and
residual risk, derived entirely from data already resident in the ARG:

    exploitability override  <- the asset's exposed CVEs (max CVSS band;
                                 CISA-KEV membership forces the top band 5,
                                 because a known-exploited CVE has ready-made
                                 exploit code per Figure 5, level 5).
    impact CIA override      <- the attack class's CIA *shape* scaled by the
                                 asset's §6.3 criticality (critical +1 band,
                                 high +0, medium -1, clamped to [1, 5]).
    residual risk            <- §6.7 currently-applied controls reached via
                                 MITIGATED_BY edges; each control reduces the
                                 likelihood by one Figure-5 band (§6.12).

It is fully OFFLINE: only ``results/multisource_arg.json`` is read; no network
calls are made. The likelihood band is built from CVSS + CISA-KEV only and
deliberately does NOT use EPSS, so that EPSS can serve as an independent
external ground truth in ``scripts/validate_asset_risk.py``.

The heavy lifting (Figure 3/4/5 tables, residual computation) is delegated to
the audited modules ``risk_score``, ``risk_register`` and ``treatment``; this
module is a composition layer that sources their override parameters from the
graph.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .impact import ATTACK_IMPACT_DEFAULTS, compute_impact
from .likelihood import ATTACK_LIKELIHOOD_DEFAULTS, compute_likelihood
from .risk_register import RiskScenario, build_register_entry
from .risk_score import RiskScore, compute_risk_score
from .treatment import Treatment, compute_residual_risk

# ─────────────────────────────────────────────────────────────────────────────
# Mapping: ARG RiskCase attack_type -> §6.9 canonical CIC-IDS2018 class
# ─────────────────────────────────────────────────────────────────────────────
# The ARG names some scenarios more finely than the six §6.9 default classes
# (e.g., FTP-BruteForce). We fold them onto the canonical class whose CIA shape
# best matches, so the attack-class CIA *prior* is well defined. Exploitability
# is overridden per asset, so this mapping only selects the impact shape.
RISKCASE_TO_CANONICAL: dict[str, str] = {
    "BruteForce": "BruteForce",
    "FTP-BruteForce": "BruteForce",
    "SSH-BruteForce": "BruteForce",
    "DoS": "DoS",
    "DDoS": "DDoS",
    "WebAttack": "WebAttack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Infiltration",   # memory-disclosure -> data exfiltration shape (C=5)
    "Bot": "Infiltration",          # botnet C2 / compromise -> compromise shape
}

# §6.3 criticality -> impact band delta (applied to each CIA dimension, clamped).
CRITICALITY_IMPACT_DELTA: dict[str, int] = {
    "critical": +1,
    "high": 0,
    "medium": -1,
    "low": -2,
}

# Each currently-applied control reduces the likelihood by this many Figure-5
# bands when computing §6.12 residual risk (conservative: baseline §6.7 controls
# lower but do not eliminate exposure).
LIKELIHOOD_REDUCTION_PER_CONTROL = 1
MAX_LIKELIHOOD_REDUCTION = 4


def cvss_kev_to_exploitability(
    max_cvss: float | None,
    *,
    in_cisa_kev: bool,
) -> tuple[int, str]:
    """Map an asset's worst exposed CVE to a Figure-5 exploitability band (1..5).

    Grounded in the NFCRM-1:2025 Figure-5 exploitability descriptors
    (``constants.LIKELIHOOD_LEVELS``):

        5  ready-made exploit code, unsophisticated attacker  <- CISA KEV
        4  advanced attacker, custom tooling                  <- CVSS >= 9.0
        3  advanced attacker, specific conditions             <- CVSS 7.0-8.9
        2  not directly exploitable now                       <- CVSS 4.0-6.9
        1  not currently exploitable                          <- CVSS < 4.0

    CISA-KEV membership dominates: a known-exploited CVE has, by definition,
    in-the-wild exploit code and therefore maps to band 5 regardless of base
    score. EPSS is intentionally NOT consulted here (held out for validation).

    Returns (band, rationale).
    """
    if in_cisa_kev:
        return 5, (
            "Exploitability=5 (Figure 5): at least one exposed CVE is in the "
            "CISA Known-Exploited-Vulnerabilities catalogue (ready-made exploit "
            "code, in-the-wild exploitation)."
        )
    if max_cvss is None:
        # No CVE exposure recorded: fall back to the attack-class default later.
        return 0, "No exposed CVE with a CVSS score; attack-class default used."
    if max_cvss >= 9.0:
        return 4, (
            f"Exploitability=4 (Figure 5): worst exposed CVE CVSS={max_cvss:.1f} "
            f"(Critical) -> exploitable by advanced attackers with custom tooling."
        )
    if max_cvss >= 7.0:
        return 3, (
            f"Exploitability=3 (Figure 5): worst exposed CVE CVSS={max_cvss:.1f} "
            f"(High) -> exploitable by advanced attackers under specific conditions."
        )
    if max_cvss >= 4.0:
        return 2, (
            f"Exploitability=2 (Figure 5): worst exposed CVE CVSS={max_cvss:.1f} "
            f"(Medium) -> not directly exploitable now."
        )
    return 1, (
        f"Exploitability=1 (Figure 5): worst exposed CVE CVSS={max_cvss:.1f} "
        f"(Low) -> not currently exploitable."
    )


def criticality_to_cia_overrides(
    attack_type: str,
    criticality: str,
) -> tuple[int, int, int, str]:
    """Scale the attack class's CIA prior by §6.3 asset criticality.

    The attack class fixes the *shape* (which of C/I/A dominates); criticality
    shifts every active dimension up or down by a band (clamped to [1, 5]).
    A dimension that is 0 in the class default (i.e. not applicable, e.g. the
    confidentiality dimension of a pure DoS) stays 0.

    Returns (c_override, i_override, a_override, rationale).
    """
    defaults = ATTACK_IMPACT_DEFAULTS[attack_type]
    delta = CRITICALITY_IMPACT_DELTA.get(criticality.lower(), 0)

    def scaled(v: int) -> int:
        if v == 0:
            return 0
        return max(1, min(5, v + delta))

    c, i, a = scaled(defaults["C"]), scaled(defaults["I"]), scaled(defaults["A"])
    rationale = (
        f"CIA shape from {attack_type} prior (C={defaults['C']},I={defaults['I']},"
        f"A={defaults['A']}) scaled by criticality '{criticality}' (delta={delta:+d}) "
        f"-> (C={c},I={i},A={a})."
    )
    return c, i, a, rationale


@dataclass
class AssetView:
    """The ARG slice relevant to one asset's §6.9 scoring."""
    asset_id: str
    hostname: str
    criticality: str
    exposed_cves: list[dict] = field(default_factory=list)   # CVE node dicts
    attack_types: list[str] = field(default_factory=list)    # canonical §6.9 classes
    applied_controls: list[str] = field(default_factory=list)  # NFCRM control ids

    @property
    def max_cvss(self) -> float | None:
        scores = [c.get("cvss_v3") for c in self.exposed_cves if c.get("cvss_v3")]
        return max(scores) if scores else None

    @property
    def in_cisa_kev(self) -> bool:
        return any(c.get("in_cisa_kev") for c in self.exposed_cves)


def load_arg(path: str | Path) -> dict:
    """Load the multi-source ARG JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_asset_views(arg: dict) -> dict[str, AssetView]:
    """Construct per-asset views (criticality, exposed CVEs, attacks, controls)."""
    nodes = {n["id"]: n for n in arg["nodes"]}
    edges = arg["edges"]

    views: dict[str, AssetView] = {}
    for n in arg["nodes"]:
        if n.get("type") == "Asset":
            views[n["id"]] = AssetView(
                asset_id=n["id"],
                hostname=n.get("hostname", n["id"]),
                criticality=n.get("criticality", "medium"),
            )

    # asset -> exposed CVEs (EXPOSED_TO_CVE)
    for e in edges:
        if e.get("relation") == "EXPOSED_TO_CVE":
            asset_id, cve_id = e["source"], e["target"]
            if asset_id in views and cve_id in nodes:
                views[asset_id].exposed_cves.append(nodes[cve_id])

    # RiskCase -> asset (AFFECTS_ASSET) gives the attack scenario(s) per asset;
    # RiskCase -> control (MITIGATED_BY) gives the §6.7 applied controls.
    riskcase_controls: dict[str, list[str]] = {}
    for e in edges:
        if e.get("relation") == "MITIGATED_BY":
            riskcase_controls.setdefault(e["source"], []).append(e["target"])

    for e in edges:
        if e.get("relation") == "AFFECTS_ASSET":
            rc_id, asset_id = e["source"], e["target"]
            if asset_id not in views or rc_id not in nodes:
                continue
            rc = nodes[rc_id]
            raw_atk = rc.get("attack_type", "")
            canonical = RISKCASE_TO_CANONICAL.get(raw_atk)
            if canonical and canonical not in views[asset_id].attack_types:
                views[asset_id].attack_types.append(canonical)
            for ctrl in riskcase_controls.get(rc_id, []):
                if ctrl not in views[asset_id].applied_controls:
                    views[asset_id].applied_controls.append(ctrl)

    return views


@dataclass
class AssetScenarioRisk:
    """Per-(asset, attack-scenario) inherent + residual §6.9 risk."""
    asset_id: str
    hostname: str
    criticality: str
    attack_type: str
    max_cvss: float | None
    in_cisa_kev: bool
    n_applied_controls: int
    inherent_score: int
    inherent_level: str
    residual_score: int
    residual_level: str
    likelihood: int
    impact: int
    exploitability: int
    rationale: str

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "hostname": self.hostname,
            "criticality": self.criticality,
            "attack_type": self.attack_type,
            "max_cvss": self.max_cvss,
            "in_cisa_kev": self.in_cisa_kev,
            "n_applied_controls": self.n_applied_controls,
            "inherent_risk_score": self.inherent_score,
            "inherent_risk_level": self.inherent_level,
            "residual_risk_score": self.residual_score,
            "residual_risk_level": self.residual_level,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "exploitability": self.exploitability,
            "rationale": self.rationale,
        }


def score_asset_scenario(view: AssetView, attack_type: str) -> AssetScenarioRisk:
    """Compute asset-specific inherent + residual §6.9 risk for one scenario."""
    # Exploitability from the asset's worst exposed CVE (+ KEV); fall back to
    # the attack-class default when the asset has no scored CVE exposure.
    expl_band, expl_rationale = cvss_kev_to_exploitability(
        view.max_cvss, in_cisa_kev=view.in_cisa_kev
    )
    expl_override = expl_band if expl_band > 0 else None

    c_ovr, i_ovr, a_ovr, cia_rationale = criticality_to_cia_overrides(
        attack_type, view.criticality
    )

    rs: RiskScore = compute_risk_score(
        attack_type,
        exploitability_override=expl_override,
        c_override=c_ovr,
        i_override=i_ovr,
        a_override=a_ovr,
    )

    # Residual: each applied §6.7 control reduces likelihood by one band.
    n_controls = len(view.applied_controls)
    scenario = RiskScenario(
        scenario_id=f"{view.asset_id}-{attack_type}",
        asset_id=view.asset_id,
        asset_name=view.hostname,
        asset_classification=view.criticality,
        threat_id=None,
        threat_description=f"{attack_type} against {view.hostname}",
        vulnerability_id=None,
        vulnerability_description=None,
        attack_type=attack_type,
    )
    entry = build_register_entry(
        scenario,
        risk_score=rs,
        currently_applied_controls=view.applied_controls,
    )
    if n_controls > 0 and rs.score > 0:
        reduction = min(MAX_LIKELIHOOD_REDUCTION,
                        n_controls * LIKELIHOOD_REDUCTION_PER_CONTROL)
        treatment = Treatment(
            decision="Mitigate",
            plan_description=(
                f"{n_controls} currently-applied NFCRM §6.7 control(s): "
                f"{', '.join(view.applied_controls)}."
            ),
            likelihood_reduction=reduction,
            rationale="§6.12 residual risk after currently-applied baseline controls.",
            status="implemented",
        )
        residual = compute_residual_risk(entry, treatment)
        residual_score, residual_level = residual.residual_score, residual.residual_level_en
    else:
        residual_score, residual_level = rs.score, rs.level_en

    return AssetScenarioRisk(
        asset_id=view.asset_id,
        hostname=view.hostname,
        criticality=view.criticality,
        attack_type=attack_type,
        max_cvss=view.max_cvss,
        in_cisa_kev=view.in_cisa_kev,
        n_applied_controls=n_controls,
        inherent_score=rs.score,
        inherent_level=rs.level_en,
        residual_score=residual_score,
        residual_level=residual_level,
        likelihood=rs.likelihood.value,
        impact=rs.impact.value,
        exploitability=rs.likelihood.exploitability,
        rationale=f"{expl_rationale} {cia_rationale}",
    )


def score_arg(arg: dict) -> list[AssetScenarioRisk]:
    """Score every (asset, attack-scenario) pair derivable from the ARG.

    Assets with a RiskCase scenario are scored against each mapped attack class.
    Assets that have CVE exposure but no RiskCase scenario are scored against an
    'Infiltration' compromise shape (a directly-exploitable CVE on an asset is a
    potential compromise), so the full CVE-exposed asset population is covered.
    """
    views = build_asset_views(arg)
    out: list[AssetScenarioRisk] = []
    for view in views.values():
        attack_types = list(view.attack_types)
        if not attack_types and view.exposed_cves:
            attack_types = ["Infiltration"]  # CVE exposure w/o named scenario
        for atk in attack_types:
            out.append(score_asset_scenario(view, atk))
    return out
