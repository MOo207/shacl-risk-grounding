"""End-to-end NFCRM-1:2025 cycle demonstration.

Runs the full Identification -> Assessment -> Treatment -> Monitoring
methodology phases on the existing demo data:

  Identification  : §6.3 / §6.4 / §6.5 inventories from the ARG
  Assessment      : §6.9 likelihood × impact + §6.10 risk register
  Treatment       : §6.8 acceptable-level decision + §6.11–§6.16
                    recommended treatment + residual risk + register update
  Monitoring      : §6.17 report + §6.18 statistics + §6.20 trigger snapshot

Outputs:
    results/nfcrm_full_cycle.json    - all artefacts in one document
    results/nfcrm_full_cycle.md      - human-readable summary
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.nfcrm import (  # noqa: E402
    AcceptableRiskConfig,
    RiskScenario,
    apply_treatment,
    build_register_entry,
    detect_triggers,
    export_inventories,
    generate_report,
    is_treatment_required,
    prioritise,
    recommend_treatment,
    STATISTICAL_DATA_CRITERIA,
)

RESULTS = REPO / "results"
ARG_PATH = RESULTS / "multisource_arg.json"
PREV_REGISTER_PATH = RESULTS / "nfcrm_risk_register.json"
OUT_JSON = RESULTS / "nfcrm_full_cycle.json"
OUT_MD = RESULTS / "nfcrm_full_cycle.md"


# Default ATT&CK technique IDs per attack class
ATTACK_TO_THREAT = {
    "DoS":          ("T1499", "Endpoint Denial of Service"),
    "DDoS":         ("T1498", "Network Denial of Service"),
    "BruteForce":   ("T1110", "Brute Force"),
    "WebAttack":    ("T1190", "Exploit Public-Facing Application"),
    "Infiltration": ("T1570", "Lateral Tool Transfer"),
}

DEFAULT_CONTROLS = {
    "DoS":          ["Rate limiting", "Reverse-proxy timeouts"],
    "DDoS":         ["CDN scrubbing service", "ISP-level black-holing"],
    "BruteForce":   ["Account lockout", "Login throttling"],
    "WebAttack":    ["WAF (OWASP CRS)", "Parameterised queries"],
    "Infiltration": ["EDR agent", "Network segmentation"],
}


def build_class_register():
    """One register entry per attack class for end-to-end demo."""
    entries = []
    for atk, (tid, tdesc) in ATTACK_TO_THREAT.items():
        s = RiskScenario(
            scenario_id=f"CLASS-{atk}",
            asset_id="GENERIC-ASSET",
            asset_name="Generic CIC-IDS2018 target",
            asset_classification="Sensitive system",
            threat_id=tid,
            threat_description=tdesc,
            vulnerability_id=None,
            vulnerability_description=None,
            attack_type=atk,
        )
        entries.append(build_register_entry(
            s,
            currently_applied_controls=DEFAULT_CONTROLS.get(atk, []),
            entry_id=f"RR-CLASS-{atk}",
        ))
    return entries


def main() -> int:
    cfg = AcceptableRiskConfig(
        acceptable_level_en="Low",
        rationale="Default thesis-demo threshold. A real organisation would set "
                  "this per its risk appetite and the §6.5 sensitivity of the asset.",
        approved_by="thesis-demo",
    )

    # Identification: §6.3, §6.4, §6.5 inventories
    inventories = export_inventories(ARG_PATH, RESULTS)

    # Assessment: §6.9 + §6.10
    register = build_class_register()

    # §6.14 prioritise by inherent score
    register = prioritise(register)

    # Treatment: §6.8 decision + §6.11–§6.16
    treatments = []
    for entry in register:
        decision_info = is_treatment_required(entry.inherent_risk_level_en, config=cfg)
        treatment = recommend_treatment(entry, config=cfg)
        apply_treatment(entry, treatment)
        treatments.append({
            "entry_id": entry.entry_id,
            "treatment_required": decision_info.treatment_required,
            "treatment": asdict(treatment),
            "decision_rationale": decision_info.rationale,
        })

    register_dicts = [e.to_dict() for e in register]

    # Monitoring: §6.17 + §6.18 + §6.20
    report = generate_report(
        register_dicts,
        acceptable_level_en=cfg.acceptable_level_en,
    )

    previous_register = None
    if PREV_REGISTER_PATH.exists():
        prev = json.loads(PREV_REGISTER_PATH.read_text(encoding="utf-8"))
        previous_register = (prev.get("class_level") or []) + (prev.get("demo_flows") or [])
    triggers = detect_triggers(previous_register, register_dicts)

    payload = {
        "clause_references": (
            "Identification: §6.3, §6.4, §6.5; "
            "Assessment: §6.9, §6.10; "
            "Treatment: §6.8, §6.11, §6.12, §6.13, §6.14, §6.15, §6.16; "
            "Monitoring: §6.17, §6.18, §6.20"
        ),
        "acceptable_risk_config": {
            "acceptable_level_en": cfg.acceptable_level_en,
            "rationale": cfg.rationale,
            "approved_by": cfg.approved_by,
            "clause_reference": "NFCRM-1:2025 §6.8",
        },
        "identification": {
            "n_assets_6_3": len(inventories["assets_6_3"]),
            "n_vulnerabilities_6_4": len(inventories["vulnerabilities_6_4"]),
            "n_threats_6_5": len(inventories["threats_6_5"]),
        },
        "assessment_register_6_10": register_dicts,
        "treatment_decisions_6_8_11_13": treatments,
        "monitoring_report_6_17": report.to_dict(),
        "monitoring_statistical_criteria_6_18": STATISTICAL_DATA_CRITERIA,
        "monitoring_triggers_6_20": triggers.to_dict(),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")

    print("NFCRM-1:2025 full cycle:")
    print(f"  §6.3-6.5 inventories: assets={payload['identification']['n_assets_6_3']}  "
          f"vulns={payload['identification']['n_vulnerabilities_6_4']}  "
          f"threats={payload['identification']['n_threats_6_5']}")
    print(f"  §6.10 register entries: {len(register_dicts)}")
    print(f"  §6.17 report: above_acceptable={report.above_acceptable_count}")
    print(f"  §6.20 triggers: any_triggered={triggers.any_triggered}")
    print(f"  -> {OUT_JSON.relative_to(REPO)}")
    print(f"  -> {OUT_MD.relative_to(REPO)}")
    return 0


def _render_md(payload: dict) -> str:
    lines = ["# NFCRM-1:2025 Full-Cycle Demo", "", payload["clause_references"], ""]
    lines.append("## Identification (§6.3, §6.4, §6.5)")
    ident = payload["identification"]
    lines.append(f"- §6.3 Assets: {ident['n_assets_6_3']}")
    lines.append(f"- §6.4 Vulnerabilities: {ident['n_vulnerabilities_6_4']}")
    lines.append(f"- §6.5 Threats: {ident['n_threats_6_5']}")
    lines.append("")
    lines.append("## §6.8 Acceptable Risk Level")
    cfg = payload["acceptable_risk_config"]
    lines.append(f"- Threshold: **{cfg['acceptable_level_en']}**")
    lines.append(f"- Rationale: {cfg['rationale']}")
    lines.append("")
    lines.append("## Assessment + Treatment (§6.9, §6.10, §6.11–§6.16)")
    lines.append("")
    lines.append("| Priority | Entry ID | Attack | Inherent | Treatment | Residual | Status |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in payload["assessment_register_6_10"]:
        lines.append(
            f"| {e.get('treatment_priority', '—')} | {e['entry_id']} | "
            f"{e['scenario']['attack_type']} | "
            f"{e['inherent_risk_score']} ({e['inherent_risk_level_en']}) | "
            f"{e.get('treatment_decision', '—')}: L-{e.get('residual_likelihood', '—')} "
            f"I-{e.get('residual_impact', '—')} | "
            f"{e.get('residual_risk_score', '—')} ({e.get('residual_risk_level_en', '—')}) | "
            f"{e.get('treatment_status', '—')} |"
        )
    lines.append("")
    lines.append("## Monitoring Report (§6.17, §6.18)")
    rep = payload["monitoring_report_6_17"]
    lines.append(f"- Total entries: {rep['n_entries']}")
    lines.append(f"- Above acceptable level: {rep['above_acceptable_count']}")
    lines.append("")
    lines.append("**Inherent risk distribution**")
    for level, n in rep["inherent_distribution"].items():
        lines.append(f"  - {level}: {n}")
    lines.append("")
    lines.append("**Residual risk distribution**")
    for level, n in rep["residual_distribution"].items():
        lines.append(f"  - {level}: {n}")
    lines.append("")
    lines.append("**Treatment-decision mix**")
    for dec, n in rep["treatment_decision_distribution"].items():
        lines.append(f"  - {dec}: {n}")
    lines.append("")
    lines.append("## §6.18 Statistical-Data Criteria")
    for c in payload["monitoring_statistical_criteria_6_18"]:
        lines.append(f"- **{c['id']} {c['name']}** — {c['computed_from']} ({c['unit']})")
    lines.append("")
    lines.append("## §6.20 Triggers")
    trig = payload["monitoring_triggers_6_20"]
    lines.append(f"- any_triggered: {trig['any_triggered']}")
    for cat in [
        "treatment_plan_changes", "threat_changes", "scenario_changes",
        "control_changes", "infrastructure_changes",
    ]:
        items = trig.get(cat, [])
        if items:
            lines.append(f"- {cat}: {len(items)}")
            for s in items[:5]:
                lines.append(f"  - {s}")
            if len(items) > 5:
                lines.append(f"  - ... +{len(items)-5} more")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
