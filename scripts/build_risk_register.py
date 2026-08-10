"""Build the NFCRM-1:2025 §6.10 cybersecurity risk register from framework outputs.

Reads:
    results/multisource_arg.json              - ARG: assets, CVEs, ATT&CK techniques
    results/intersymbolic_explanations.json   - 5-flow demo with Layer 3 mappings
    (optional) results/slm_nl_classification.json - SLM run predictions

Writes:
    results/nfcrm_risk_register.json          - register entries (machine-readable)
    results/nfcrm_risk_register.md            - register entries (human-readable)

Stdlib only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.nfcrm import (  # noqa: E402
    RiskScenario,
    build_register_entry,
    validate_entry,
)

RESULTS = REPO / "results"
ARG_PATH = RESULTS / "multisource_arg.json"
DEMO_PATH = RESULTS / "intersymbolic_explanations.json"
SLM_PATH = RESULTS / "slm_nl_classification.json"
OUT_JSON = RESULTS / "nfcrm_risk_register.json"
OUT_MD = RESULTS / "nfcrm_risk_register.md"


# Mapping: CIC-IDS2018 attack type -> default ATT&CK technique used in the ARG
ATTACK_TO_THREAT: dict[str, dict[str, str]] = {
    "DoS":          {"id": "T1499", "desc": "Endpoint Denial of Service (MITRE ATT&CK T1499)"},
    "DDoS":         {"id": "T1498", "desc": "Network Denial of Service (MITRE ATT&CK T1498)"},
    "BruteForce":   {"id": "T1110", "desc": "Brute Force (MITRE ATT&CK T1110)"},
    "WebAttack":    {"id": "T1190", "desc": "Exploit Public-Facing Application (MITRE ATT&CK T1190)"},
    "Infiltration": {"id": "T1570", "desc": "Lateral Tool Transfer (MITRE ATT&CK T1570)"},
}

# Default current-controls (§6.7) per attack type — these would normally come
# from an organisational control inventory; we use representative defaults here.
DEFAULT_CURRENT_CONTROLS: dict[str, list[str]] = {
    "DoS":          ["Rate limiting on web tier", "Reverse proxy timeouts"],
    "DDoS":         ["CDN scrubbing service (e.g., Cloudflare/Akamai)", "ISP-level black-holing"],
    "BruteForce":   ["Account lockout policy", "Login-rate throttling"],
    "WebAttack":    ["WAF rule set (OWASP CRS)", "Parameterised query enforcement"],
    "Infiltration": ["EDR endpoint agent", "Network segmentation between tiers"],
    "Benign":       [],
}


def build_demo_register() -> list[dict]:
    """Build register entries from the 5-flow demo, one entry per flow."""
    if not DEMO_PATH.exists():
        return []
    flows = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
    entries = []
    for flow in flows:
        atk = flow.get("predicted_label") or flow.get("true_label")
        if atk == "Benign" or atk not in ATTACK_TO_THREAT:
            continue
        threat = ATTACK_TO_THREAT[atk]
        cve = (flow.get("layer3", {}).get("cve_examples") or ["unknown"])[0]
        scenario = RiskScenario(
            scenario_id=f"DEMO-{flow['flow_id']:03d}-{atk}",
            asset_id=None,
            asset_name="(unknown — CIC-IDS2018 lacks per-flow asset linkage)",
            asset_classification=None,
            threat_id=threat["id"],
            threat_description=threat["desc"],
            vulnerability_id=cve,
            vulnerability_description=f"CVE example for {atk}: {cve}",
            attack_type=atk,
            rationale=(
                "Scenario derived from Layer 3 GRC mapping of a single classified flow. "
                "Asset linkage is unavailable in CIC-IDS2018; per §6.10 schema this is "
                "documented as None. Threat is the canonical MITRE ATT&CK technique for "
                "the attack class. Vulnerability is the first CVE example surfaced by "
                "Layer 3."
            ),
        )
        entry = build_register_entry(
            scenario,
            currently_applied_controls=DEFAULT_CURRENT_CONTROLS.get(atk, []),
        )
        problems = validate_entry(entry)
        if problems:
            print(f"[warn] entry {entry.entry_id} has validation problems: {problems}")
        entries.append(entry.to_dict())
    return entries


def build_attack_class_register() -> list[dict]:
    """Build one canonical entry per attack class (using attack-type defaults)."""
    entries = []
    for atk, threat in ATTACK_TO_THREAT.items():
        scenario = RiskScenario(
            scenario_id=f"CLASS-{atk}",
            asset_id="GENERIC-ASSET",
            asset_name="Generic CIC-IDS2018 target asset",
            asset_classification="undefined (CIC-IDS2018 limitation)",
            threat_id=threat["id"],
            threat_description=threat["desc"],
            vulnerability_id=None,
            vulnerability_description=f"Class-level entry; per-CVE entries are derived from {atk}-specific scenarios",
            attack_type=atk,
            rationale=(
                f"Class-level register entry showing the §6.9 inherent risk for the "
                f"{atk} attack class under default likelihood/impact values. Per-scenario "
                f"entries (asset × threat × vulnerability) are derived from this template."
            ),
        )
        entry = build_register_entry(
            scenario,
            currently_applied_controls=DEFAULT_CURRENT_CONTROLS.get(atk, []),
            entry_id=f"RR-CLASS-{atk}",
        )
        problems = validate_entry(entry)
        if problems:
            print(f"[warn] entry {entry.entry_id}: {problems}")
        entries.append(entry.to_dict())
    return entries


def render_md(register: dict) -> str:
    lines = [
        "# NFCRM-1:2025 §6.10 Cybersecurity Risk Register",
        "",
        "_Generated by `scripts/build_risk_register.py` from framework outputs._  ",
        f"_Source clauses: §6.6 (scenarios), §6.7 (controls), §6.9 (inherent risk), §6.10 (register format)._  ",
        f"_Total entries: {len(register['class_level']) + len(register['demo_flows'])}_",
        "",
        "## Class-level entries (one per CIC-IDS2018 attack class)",
        "",
        "| Entry ID | Scenario ID | Threat (T-ID) | Likelihood | Impact (C/I/A) | Score | Level |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in register["class_level"]:
        s = e["scenario"]
        lines.append(
            f"| {e['entry_id']} | {s['scenario_id']} | {s['threat_id']} | "
            f"{e['likelihood_value']} | {e['impact_value']} "
            f"({e['impact_confidentiality']}/{e['impact_integrity']}/{e['impact_availability']}) | "
            f"{e['inherent_risk_score']} | {e['inherent_risk_level_en']} |"
        )
    lines.append("")
    lines.append("## Per-flow entries (5-flow Layer 4 demo)")
    lines.append("")
    if register["demo_flows"]:
        lines.append("| Entry ID | Predicted | Threat | Vulnerability | Score | Level | Current controls |")
        lines.append("|---|---|---|---|---|---|---|")
        for e in register["demo_flows"]:
            s = e["scenario"]
            controls = ", ".join(e["currently_applied_controls"]) or "—"
            lines.append(
                f"| {e['entry_id']} | {s['attack_type']} | {s['threat_id']} | "
                f"{s['vulnerability_id']} | {e['inherent_risk_score']} | "
                f"{e['inherent_risk_level_en']} | {controls} |"
            )
    else:
        lines.append("_demo file not present_")
    lines.append("")
    lines.append("## Schema notes")
    lines.append("")
    lines.append("- Each entry follows the §6.10 schema defined in `pipeline/nfcrm/risk_register.py::RiskRegisterEntry`.")
    lines.append("- Treatment fields (decision, plan, residual risk) are present in the schema but unpopulated; sub-project E (§6.11–§6.16) will fill them.")
    lines.append("- `asset_id` is null for CIC-IDS2018 entries because the dataset does not provide per-flow asset linkage. A real deployment would join against the CMDB inventory from §6.3.")
    lines.append("- `currently_applied_controls` (§6.7) carries representative defaults per attack class. A real deployment would derive these from the organisation's control inventory.")
    return "\n".join(lines)


def main() -> int:
    register = {
        "clause_reference": "NFCRM-1:2025 §6.10",
        "schema_module": "pipeline.nfcrm.risk_register",
        "class_level": build_attack_class_register(),
        "demo_flows": build_demo_register(),
    }
    OUT_JSON.write_text(json.dumps(register, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    OUT_MD.write_text(render_md(register), encoding="utf-8")
    print(f"Risk register: {len(register['class_level'])} class-level + "
          f"{len(register['demo_flows'])} demo-flow entries")
    print(f"  -> {OUT_JSON.relative_to(REPO)}")
    print(f"  -> {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
