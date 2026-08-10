"""NFCRM-1:2025 §6.3, §6.4, §6.5 inventory exporters from the ARG.

The ARG (results/multisource_arg.json) carries asset, software, CVE,
ATT&CK technique, and NFCRM-control nodes. This module turns that node
set into the three NFCRM-required inventories:

  §6.3 — Entity asset inventory + classification
  §6.4 — Vulnerability inventory
  §6.5 — Potential-threat inventory

Each inventory is a list of structured records suitable for inclusion in
the cybersecurity risk register or for export to GRC tooling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class AssetEntry:
    """§6.3 asset inventory record."""
    asset_id: str
    asset_type: str                 # server / workstation / IoT / network / etc.
    classification: str             # NFCRM-aligned: sensitive system / OT / social media / generic
    has_software: list[str] = field(default_factory=list)  # CPE names
    notes: str = ""


@dataclass
class VulnerabilityEntry:
    """§6.4 vulnerability inventory record."""
    cve_id: str
    description: str
    severity: str | None = None      # CVSS v3 base if available
    affects_assets: list[str] = field(default_factory=list)
    in_kev: bool = False             # CISA Known Exploited Vulnerabilities flag


@dataclass
class ThreatEntry:
    """§6.5 threat inventory record."""
    threat_id: str                   # e.g., MITRE ATT&CK technique ID
    framework: str                   # "MITRE ATT&CK", "NCA TTPs", etc.
    description: str
    targets_assets: list[str] = field(default_factory=list)


def _classify_asset(node: dict) -> str:
    """Heuristic NFCRM classification per §6.3 examples."""
    name = (node.get("name") or node.get("id") or "").lower()
    asset_type = (node.get("asset_type") or "").lower()
    if "ot" in name or "scada" in name or "ics" in name:
        return "Operational Technology (OT)"
    if "social" in name or "twitter" in name or "linkedin" in name:
        return "Social-media account"
    if "server" in asset_type or "database" in asset_type or "core" in name:
        return "Sensitive system"
    if asset_type in {"workstation", "endpoint"}:
        return "Endpoint workstation"
    return "Generic asset"


def build_asset_inventory(arg_data: dict) -> list[AssetEntry]:
    """Build §6.3 asset inventory from ARG node set."""
    inv: list[AssetEntry] = []
    for node in arg_data.get("nodes", []):
        if (node.get("type") or node.get("node_type")) != "Asset":
            continue
        software_names = _software_for_asset(arg_data, node.get("id"))
        inv.append(AssetEntry(
            asset_id=str(node.get("id")),
            asset_type=str(node.get("asset_type") or node.get("type") or "Asset"),
            classification=_classify_asset(node),
            has_software=software_names,
            notes=str(node.get("notes") or ""),
        ))
    return inv


def _software_for_asset(arg_data: dict, asset_id: Any) -> list[str]:
    """Find software linked to an asset via HAS_SOFTWARE edges."""
    sw: list[str] = []
    sw_lookup = {n["id"]: n for n in arg_data.get("nodes", [])
                 if (n.get("type") or n.get("node_type")) == "Software"}
    for e in arg_data.get("edges", []):
        if e.get("relation") != "HAS_SOFTWARE":
            continue
        if str(e.get("source")) == str(asset_id):
            tgt = sw_lookup.get(e.get("target"))
            if tgt:
                sw.append(str(tgt.get("name") or tgt.get("id")))
    return sw


def build_vulnerability_inventory(arg_data: dict) -> list[VulnerabilityEntry]:
    """Build §6.4 vulnerability inventory from CVE nodes + EXPOSED_TO_CVE edges."""
    inv: list[VulnerabilityEntry] = []
    asset_for_cve: dict[str, list[str]] = {}
    for e in arg_data.get("edges", []):
        if e.get("relation") in {"EXPOSED_TO_CVE", "AFFECTED_BY_CVE"}:
            cve = str(e.get("target"))
            asset_for_cve.setdefault(cve, []).append(str(e.get("source")))

    for node in arg_data.get("nodes", []):
        if (node.get("type") or node.get("node_type")) != "CVE":
            continue
        cve_id = str(node.get("id") or node.get("cve_id"))
        inv.append(VulnerabilityEntry(
            cve_id=cve_id,
            description=str(node.get("description") or node.get("name") or ""),
            severity=str(node.get("cvss_severity") or node.get("severity") or "") or None,
            affects_assets=asset_for_cve.get(cve_id, []),
            in_kev=bool(node.get("in_kev", False)),
        ))
    return inv


def build_threat_inventory(arg_data: dict) -> list[ThreatEntry]:
    """Build §6.5 threat inventory from ATTACKTechnique nodes + TARGETS_ASSET edges."""
    inv: list[ThreatEntry] = []
    targets_for_threat: dict[str, list[str]] = {}
    for e in arg_data.get("edges", []):
        if e.get("relation") in {"TARGETS_ASSET", "AFFECTS_ASSET"}:
            t = str(e.get("source"))
            targets_for_threat.setdefault(t, []).append(str(e.get("target")))

    for node in arg_data.get("nodes", []):
        ntype = node.get("type") or node.get("node_type")
        if ntype not in {"ATTACKTechnique", "AttackTechnique", "ATT&CKTechnique"}:
            continue
        threat_id = str(node.get("id") or node.get("technique_id"))
        inv.append(ThreatEntry(
            threat_id=threat_id,
            framework="MITRE ATT&CK",
            description=str(node.get("name") or node.get("description") or threat_id),
            targets_assets=targets_for_threat.get(threat_id, []),
        ))
    return inv


def export_inventories(arg_path: Path, out_dir: Path) -> dict[str, Any]:
    """Build all three inventories and write to JSON files in out_dir."""
    arg_data = json.loads(arg_path.read_text(encoding="utf-8"))
    assets = build_asset_inventory(arg_data)
    vulns = build_vulnerability_inventory(arg_data)
    threats = build_threat_inventory(arg_data)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_arg": str(arg_path),
        "clause_reference": "NFCRM-1:2025 §6.3 (assets), §6.4 (vulnerabilities), §6.5 (threats)",
        "assets_6_3": [_asdict(a) for a in assets],
        "vulnerabilities_6_4": [_asdict(v) for v in vulns],
        "threats_6_5": [_asdict(t) for t in threats],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nfcrm_inventories.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _asdict(obj: Any) -> dict:
    """Lightweight dataclass-to-dict (avoids dataclasses.asdict overhead)."""
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)
