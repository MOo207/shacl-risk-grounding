"""Build multi-source Asset Relationship Graph from 5 data layers.

Layers:
  1. CIC-IDS2018 (attack types — already in results)
  2. CIC-IDS2018 Testbed CMDB (data/external/cmdb_assets.json) — Sharafaldin et al. 2018 + Rapid7
  3. CISA KEV + NVD (data/external/cisa_kev.json, nvd_enrichment.json)
  4. MITRE ATT&CK (data/external/enterprise-attack.json)
  5. NFCRM-1:2025 (data/external/nfcrm_clause_mapping.json)

Output: results/multisource_arg.json
"""
import json
import os
import time
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_arg():
    t0 = time.time()

    # Load all sources
    cmdb = load_json("cmdb_assets.json")
    nvd = load_json("nvd_enrichment.json")
    nfcrm = load_json("nfcrm_clause_mapping.json")
    kev = load_json("cisa_kev.json")

    # Authoritative per-CVE enrichment (CWE, severity string, CVSS version).
    # The CMDB fixture carries only cve_id/description/cvss, which leaves
    # grc:cweId and grc:severity unpopulated and the graph non-conformant.
    # scripts/enrich_arg_cves_nvd.py fetches these from NVD API 2.0.
    try:
        nvd_cves = load_json("nvd_arg_cves.json").get("cves", {})
    except FileNotFoundError:
        nvd_cves = {}
        print("  WARNING: data/external/nvd_arg_cves.json missing; CVE nodes will "
              "lack cwe/severity. Run scripts/enrich_arg_cves_nvd.py first.")

    build_stamp = datetime.now().replace(microsecond=0).isoformat()

    # ATT&CK: extract only techniques (not all 24K objects)
    attack_raw = load_json("enterprise-attack.json")
    attack_techniques = {}
    for obj in attack_raw["objects"]:
        if obj.get("type") == "attack-pattern" and not obj.get("revoked", False):
            ext_refs = obj.get("external_references", [])
            for ref in ext_refs:
                if ref.get("source_name") == "mitre-attack":
                    attack_techniques[ref["external_id"]] = {
                        "technique_id": ref["external_id"],
                        "name": obj.get("name", ""),
                        "description": obj.get("description", "")[:200],
                        "kill_chain_phases": [
                            p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])
                        ],
                    }
                    break

    nodes = []
    edges = []
    node_ids = set()

    def add_node(node_id, node_type, **props):
        if node_id not in node_ids:
            nodes.append({"id": node_id, "type": node_type, **props})
            node_ids.add(node_id)

    def add_edge(source, target, relation, **props):
        edges.append({"source": source, "target": target, "relation": relation, **props})

    # --- Layer 2: CMDB Assets ---
    # grc:assetType is constrained by AssetShape to (software|hardware|network).
    # The CMDB's own vocabulary is deployment-role, not ontology-class, so the
    # mapping is stated here rather than left implicit in the converter.
    ASSET_TYPE_MAP = {
        "server": "hardware",
        "workstation": "hardware",
        "network": "network",
        "network_device": "network",
        "software": "software",
    }

    for asset in cmdb["assets"]:
        add_node(asset["asset_id"], "Asset",
                 hostname=asset["hostname"], ip=asset["ip"],
                 os=asset["os"], criticality=asset["criticality"],
                 location=asset["location"], business_unit=asset["business_unit"],
                 asset_type=asset["asset_type"],
                 ontology_asset_type=ASSET_TYPE_MAP.get(asset["asset_type"], "hardware"),
                 provenance_source="CIC-IDS2018 testbed CMDB",
                 provenance_timestamp=build_stamp)

        for sw in asset["software"]:
            sw_id = f"SW-{sw['name'].replace(' ', '_')}-{sw['version']}"
            add_node(sw_id, "Software",
                     name=sw["name"], version=sw["version"], cpe=sw["cpe"],
                     provenance_source="CIC-IDS2018 testbed CMDB",
                     provenance_timestamp=build_stamp)
            add_edge(asset["asset_id"], sw_id, "HAS_SOFTWARE")

            # --- Layer 3: CVEs — NVD lookup first, fall back to CMDB direct entries ---
            cpe_cves = nvd.get("cpe_to_cves", {}).get(sw["cpe"], [])
            # Normalise CMDB direct entries to the same shape as NVD entries
            cmdb_cves = []
            for c in sw.get("cves", []):
                cmdb_cves.append({
                    "cve_id": c["cve_id"],
                    "cvss_v3": c.get("cvss", 0.0),
                    "severity": c.get("severity", ""),
                    "description": c.get("description", ""),
                    "cwe": c.get("cwe", ""),
                    "in_cisa_kev": c.get("in_cisa_kev", False),
                    "published": c.get("published", ""),
                })
            # Merge: NVD wins on overlap (more complete enrichment)
            seen_cve_ids = {c["cve_id"] for c in cpe_cves}
            merged_cves = cpe_cves + [c for c in cmdb_cves if c["cve_id"] not in seen_cve_ids]
            for cve in merged_cves:
                # NVD is authoritative for cwe/severity/score where it has a record;
                # the CMDB fixture supplies only id/description/cvss.
                auth = nvd_cves.get(cve["cve_id"], {})
                add_node(cve["cve_id"], "CVE",
                         cvss_v3=auth.get("cvss_score", cve["cvss_v3"]),
                         cvss_version=auth.get("cvss_version", ""),
                         severity=auth.get("severity") or cve["severity"],
                         description=cve["description"],
                         cwe=auth.get("cwe") or cve.get("cwe", ""),
                         in_cisa_kev=cve.get("in_cisa_kev", False),
                         published=auth.get("published") or cve.get("published", ""),
                         provenance_source="NVD API 2.0" if auth else "CIC-IDS2018 testbed CMDB",
                         provenance_timestamp=build_stamp)
                add_edge(sw_id, cve["cve_id"], "AFFECTED_BY_CVE")
                add_edge(asset["asset_id"], cve["cve_id"], "EXPOSED_TO_CVE")

    # --- Layer 4: MITRE ATT&CK Techniques ---
    for attack_type, mapping in cmdb.get("attack_technique_map", {}).items():
        tid = mapping["technique_id"]
        technique_data = attack_techniques.get(tid, {})
        add_node(tid, "ATTACKTechnique",
                 technique_id=tid,
                 name=mapping["technique_name"],
                 tactic=mapping.get("tactic", ""),
                 description=technique_data.get("description", "")[:200],
                 kill_chain_phases=technique_data.get("kill_chain_phases", []))

        # Link technique to targeted assets
        targeted_assets = cmdb.get("attack_asset_map", {}).get(attack_type, [])
        for asset_id in targeted_assets:
            add_edge(tid, asset_id, "TARGETS_ASSET",
                     attack_type=attack_type)

    # --- Layer 5: NFCRM-1:2025 Controls & RiskCases ---
    for phase_name, phase_data in nfcrm.get("phases", {}).items():
        clauses = phase_data.get("clauses", [])
        # Handle both list and dict formats
        if isinstance(clauses, list):
            for clause_obj in clauses:
                clause_id = clause_obj.get("id", "").replace("§", "")
                clause_desc = clause_obj.get("description", "")
                control_id = f"NFCRM-{clause_id.replace('.', '-')}"
                add_node(control_id, "NFCRMControl",
                         clause_id=f"NFCRM-1:2025-\u00a7{clause_id}",
                         clause_description=clause_desc,
                         phase=phase_name,
                         standard="NFCRM-1:2025",
                         provenance_source="NFCRM-1:2025",
                         provenance_timestamp=build_stamp)
        else:
            for clause_id, clause_desc in clauses.items():
                control_id = f"NFCRM-{clause_id}"
                add_node(control_id, "NFCRMControl",
                         clause_id=f"NFCRM-1:2025-\u00a7{clause_id}",
                         clause_description=clause_desc,
                         phase=phase_name,
                         standard="NFCRM-1:2025",
                         provenance_source="NFCRM-1:2025",
                         provenance_timestamp=build_stamp)

    # Create RiskCases for each attack scenario
    for attack_type, targeted_assets in cmdb.get("attack_asset_map", {}).items():
        risk_id = f"RISK-{attack_type}"
        technique = cmdb.get("attack_technique_map", {}).get(attack_type, {})
        nfcrm_mapping = cmdb.get("attack_nfcrm_map", {}).get(attack_type, {})

        # Compute risk score using NFCRM 5x5 matrix
        # Impact: based on asset criticality (critical=5, high=4, medium=3)
        max_criticality = max(
            (5 if next((a for a in cmdb["assets"] if a["asset_id"] == aid), {}).get("criticality") == "critical"
             else 4 if next((a for a in cmdb["assets"] if a["asset_id"] == aid), {}).get("criticality") == "high"
             else 3)
            for aid in targeted_assets
        )
        # Likelihood: based on whether CVEs exist and are in KEV
        has_kev_cve = False
        for aid in targeted_assets:
            asset_obj = next((a for a in cmdb["assets"] if a["asset_id"] == aid), None)
            if asset_obj:
                for sw in asset_obj["software"]:
                    for cve in nvd.get("cpe_to_cves", {}).get(sw["cpe"], []):
                        if cve.get("in_cisa_kev"):
                            has_kev_cve = True
        likelihood = 5 if has_kev_cve else 3
        risk_score = max_criticality * likelihood
        risk_level = (
            "Catastrophic" if risk_score >= 20 else
            "High" if risk_score >= 15 else
            "Medium" if risk_score >= 10 else
            "Low" if risk_score >= 5 else "Very Low"
        )

        add_node(risk_id, "RiskCase",
                 attack_type=attack_type,
                 technique_id=technique.get("technique_id", ""),
                 nfcrm_clause=f"NFCRM-1:2025-\u00a7{nfcrm_mapping.get('clause', '')}",
                 impact=max_criticality, likelihood=likelihood,
                 risk_score=risk_score, risk_level=risk_level,
                 # These RiskCases are produced by a deterministic rule over
                 # complete inputs (asset criticality x KEV membership), not by a
                 # probabilistic estimator, so assessment confidence is 1.0 by
                 # construction. It is recorded rather than left absent because
                 # grc:RiskCaseShape requires it; it is not a model score.
                 confidence=1.0,
                 confidence_basis="deterministic rule derivation",
                 provenance_source="NFCRM-1:2025 \u00a76.9 rule engine",
                 provenance_timestamp=build_stamp)

        # Edges: RiskCase -> assets, technique, controls
        for aid in targeted_assets:
            add_edge(risk_id, aid, "AFFECTS_ASSET")
        if technique.get("technique_id"):
            add_edge(risk_id, technique["technique_id"], "USES_TECHNIQUE")
        if nfcrm_mapping.get("clause"):
            # Control node ids are dash-style (NFCRM-6-7); the clause mapping is
            # dot-style (6.7). Normalising here fixes the dangling MITIGATED_BY
            # edges that the competency-question evaluation surfaced (CQ4/CQ8).
            clause_ref = str(nfcrm_mapping["clause"]).replace("§", "").replace(".", "-")
            add_edge(risk_id, f"NFCRM-{clause_ref}", "MITIGATED_BY")

        # RiskCase provenance and the vulnerability linkage that
        # grc:RiskCaseMustLinkToVulnerabilityOrEventConstraint requires.
        for aid in targeted_assets:
            for e in list(edges):
                if e["source"] == aid and e["relation"] == "EXPOSED_TO_CVE":
                    add_edge(e["target"], risk_id, "RELATES_TO_RISK")

    # --- Statistics ---
    build_time = time.time() - t0
    nodes_by_type = {}
    for n in nodes:
        t = n["type"]
        nodes_by_type[t] = nodes_by_type.get(t, 0) + 1
    edges_by_relation = {}
    for e in edges:
        r = e["relation"]
        edges_by_relation[r] = edges_by_relation.get(r, 0) + 1

    kev_cves = [n for n in nodes if n["type"] == "CVE" and n.get("in_cisa_kev")]

    output = {
        "generated": datetime.now().isoformat(),
        "build_time_seconds": round(build_time, 3),
        "statistics": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes_by_type": nodes_by_type,
            "edges_by_relation": edges_by_relation,
            "cves_in_kev": len(kev_cves),
            "attack_techniques_mapped": len(cmdb.get("attack_technique_map", {})),
            "nfcrm_controls": sum(1 for n in nodes if n["type"] == "NFCRMControl"),
            "risk_cases": sum(1 for n in nodes if n["type"] == "RiskCase"),
        },
        "data_sources": [
            {"name": "CIC-IDS2018", "type": "Network IDS", "records": "16.2M flows (6 attack types used)"},
            {"name": "CIC-IDS2018 Testbed CMDB", "type": "Asset Inventory", "records": f"{cmdb['total_assets']} assets (Sharafaldin et al. 2018 + Rapid7)"},
            {"name": "CISA KEV", "type": "Threat Intelligence", "records": f"{len(kev.get('vulnerabilities', []))} CVEs"},
            {"name": "NVD", "type": "Vulnerability Data", "records": f"{nvd['total_cves']} CVEs across {len(nvd.get('cpe_to_cves', {}))} CPEs"},
            {"name": "NFCRM-1:2025", "type": "Standards/Controls", "records": f"{sum(len(p['clauses']) for p in nfcrm['phases'].values())} clauses"},
        ],
        "risk_summary": [
            {k: v for k, v in n.items() if k != "description"}
            for n in nodes if n["type"] == "RiskCase"
        ],
        "nodes": nodes,
        "edges": edges,
    }

    dest = os.path.join(RESULTS_DIR, "multisource_arg.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Multi-source ARG built in {build_time:.3f}s")
    print(f"  Nodes: {len(nodes)} ({nodes_by_type})")
    print(f"  Edges: {len(edges)} ({edges_by_relation})")
    print(f"  CVEs in CISA KEV: {len(kev_cves)}")
    print(f"  Risk Cases: {len([n for n in nodes if n['type'] == 'RiskCase'])}")
    print(f"  -> {dest}")


if __name__ == "__main__":
    build_arg()
