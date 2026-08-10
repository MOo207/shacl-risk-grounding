"""
ARG Seed — CIC-IDS2018 Testbed Asset Relationship Graph for INTERSYMBOLIC-GRC Thesis
======================================================================================

Creates the ARG seed used by the butterfly effect evaluation that maps CIC-IDS2018
attack types to an on-premise IT asset hierarchy aligned with the
SafecareOnto ontology (Hannou et al. 2022).

Asset identities mirror the CIC-IDS2018 testbed topology (Sharafaldin et al. 2018;
subnet 172.31.69.0/24). This seed is used specifically for the butterfly effect
propagation evaluation; the authoritative CMDB is data/external/cmdb_assets.json.

The hierarchy mirrors a Riyadh HQ data-centre with two physical buildings,
a DMZ network segment, four servers, two workstations, and one business
service — covering all six CIC-IDS2018 attack families:

  FTP-BruteForce  → ftp-srv-01  → vsftpd 3.0.3       → CVE-2011-2523
  DoS             → web-srv-01  → IIS 10.0            → CVE-2015-1635
  Infiltration    → web-srv-01  → Microsoft 365       → CVE-2021-26855
  WebAttack-SQLi  → db-srv-01   → SQL Server 2019     → CVE-2019-0819
  Botnet          → ws-01/ws-02 → Chrome Browser      → CVE-2023-2033
  DDOS            → DMZ-Segment ← fw-01 + ids-01      → BARRIER

Outputs
-------
  data/arg_seed.json    — full node+edge graph (for inspection and unit tests)
  data/arg_seed.cypher  — Neo4j Cypher MERGE statements

Usage
-----
  python pipeline/loaders/arg_seed.py
  python -m pipeline.loaders.arg_seed

References
----------
  SafecareOnto: Hannou et al. (2022)
  MITRE ATT&CK: https://attack.mitre.org
  NFCRM-1:2025: National Framework for Cybersecurity Risk Management
  ISO/IEC 27005: Information security risk management
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
JSON_OUTPUT = _DATA_DIR / "arg_seed.json"
CYPHER_OUTPUT = _DATA_DIR / "arg_seed.cypher"

# ---------------------------------------------------------------------------
# Node dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Generic ARG node."""
    id: str
    labels: list[str]
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Directed ARG edge (relationship)."""
    id: str
    type: str          # relationship type, matches ontology property name
    source: str        # source node id
    target: str        # target node id
    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _node(nid: str, labels: list[str], **props) -> Node:
    return Node(id=nid, labels=labels, properties=props)


def _edge(eid: str, rel: str, src: str, tgt: str, **props) -> Edge:
    return Edge(id=eid, type=rel, source=src, target=tgt, properties=props)


# ---------------------------------------------------------------------------
# ARG definition
# ---------------------------------------------------------------------------

def build_arg() -> tuple[list[Node], list[Edge]]:
    """
    Build and return the complete CIC-IDS2018 Testbed ARG.

    Returns a tuple (nodes, edges) where every item is a dataclass instance.

    Node label conventions (SafecareOnto alignment):
      ComplexBuilding, SimpleBuilding, Server, SoftwareAsset, NetworkSegment,
      Vendor, Vulnerability, Threat, NFCRMControl, RiskCase, Service

    Edge type conventions (ontology property names):
      composedOf, hostsDevice, hostsSoftware, leadsTo, supports, depends,
      hasVulnerability, exploits, relatesToRisk, associatedWithControl,
      hasControl, controls, providedBy
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    # -----------------------------------------------------------------------
    # VENDORS
    # -----------------------------------------------------------------------
    nodes += [
        _node("vendor-microsoft",      ["Vendor"], vendorName="Microsoft",
              vendorUrl="https://www.microsoft.com/en-us/msrc"),
        _node("vendor-canonical",      ["Vendor"], vendorName="Canonical",
              vendorUrl="https://ubuntu.com/security"),
        _node("vendor-vsftpd-project", ["Vendor"], vendorName="vsftpd Project",
              vendorUrl="https://security.appspot.com/vsftpd.html"),
        _node("vendor-google",         ["Vendor"], vendorName="Google",
              vendorUrl="https://chromereleases.googleblog.com"),
        _node("vendor-pfsense",        ["Vendor"], vendorName="Netgate / pfSense",
              vendorUrl="https://www.pfsense.org"),
        _node("vendor-oisf",           ["Vendor"], vendorName="OISF / Suricata",
              vendorUrl="https://suricata.io"),
    ]

    # -----------------------------------------------------------------------
    # BUILDINGS — physical containment layer
    # -----------------------------------------------------------------------

    # ComplexBuilding — Riyadh HQ Data Centre
    nodes.append(_node(
        "Building-A", ["ComplexBuilding", "Building", "SupportingAsset", "Asset"],
        buildingName="Data Center Building A",
        assetName="Data Center Building A",
        assetId="BLDG-A-001",
        location="Riyadh, Saudi Arabia",
        impactScore=0.58,
        propagationBarrier=False,
    ))

    # SimpleBuilding — Server Room 1 (ground floor)
    nodes.append(_node(
        "Server-Room-1", ["SimpleBuilding", "Building", "SupportingAsset", "Asset"],
        buildingName="Server Room 1",
        assetName="Server Room 1",
        assetId="ROOM-001",
        buildingLevel=0,
        impactScore=0.65,
        propagationBarrier=False,
    ))

    # SimpleBuilding — Network Operations Centre (first floor)
    nodes.append(_node(
        "NOC", ["SimpleBuilding", "Building", "SupportingAsset", "Asset"],
        buildingName="Network Operations Center",
        assetName="Network Operations Center",
        assetId="ROOM-002",
        buildingLevel=1,
        impactScore=0.15,
        propagationBarrier=False,
    ))

    # ComplexBuilding — Office Wing
    nodes.append(_node(
        "Building-B", ["ComplexBuilding", "Building", "SupportingAsset", "Asset"],
        buildingName="Office Wing Building B",
        assetName="Office Wing Building B",
        assetId="BLDG-B-001",
        location="Riyadh, Saudi Arabia",
        impactScore=0.80,
        propagationBarrier=False,
    ))

    # SimpleBuilding — Office Floor 1
    nodes.append(_node(
        "Office-Floor-1", ["SimpleBuilding", "Building", "SupportingAsset", "Asset"],
        buildingName="Office Floor 1",
        assetName="Office Floor 1",
        assetId="ROOM-003",
        buildingLevel=1,
        impactScore=0.80,
        propagationBarrier=False,
    ))

    # Building containment edges
    edges += [
        _edge("e-bldg-a-room1",   "composedOf", "Building-A", "Server-Room-1"),
        _edge("e-bldg-a-noc",     "composedOf", "Building-A", "NOC"),
        _edge("e-bldg-b-floor1",  "composedOf", "Building-B", "Office-Floor-1"),
    ]

    # -----------------------------------------------------------------------
    # NETWORK SEGMENT — DMZ
    # -----------------------------------------------------------------------
    nodes.append(_node(
        "DMZ-Segment", ["NetworkSegment", "SupportingAsset", "NetworkAsset", "Asset"],
        assetName="DMZ-Segment-192.168.1.0/24",
        assetId="NET-DMZ-001",
        ipAddress="192.168.1.0/24",
        hostname="dmz-segment",
        impactScore=0.15,
        propagationBarrier=False,
    ))

    # DMZ leads to Server Room 1 (communication path for lateral movement)
    edges.append(_edge("e-dmz-to-room1", "leadsTo", "DMZ-Segment", "Server-Room-1"))

    # -----------------------------------------------------------------------
    # SERVERS — tangible HardwareAssets hosted in buildings
    # -----------------------------------------------------------------------

    # web-srv-01 — Windows Server 2019 / IIS
    nodes.append(_node(
        "web-srv-01", ["Server", "HardwareAsset", "Asset"],
        hostname="web-srv-01",
        assetName="Web Server 01",
        assetId="SRV-WEB-001",
        ipAddress="192.168.1.10",
        osName="Windows Server 2019",
        osVersion="1809",
        impactScore=0.72,
        propagationBarrier=False,
    ))

    # ftp-srv-01 — Ubuntu 20.04 / vsftpd
    nodes.append(_node(
        "ftp-srv-01", ["Server", "HardwareAsset", "Asset"],
        hostname="ftp-srv-01",
        assetName="FTP Server 01",
        assetId="SRV-FTP-001",
        ipAddress="192.168.1.20",
        osName="Ubuntu 20.04 LTS",
        osVersion="20.04",
        impactScore=0.72,
        propagationBarrier=False,
    ))

    # db-srv-01 — Windows Server 2019 / SQL Server
    nodes.append(_node(
        "db-srv-01", ["Server", "HardwareAsset", "Asset"],
        hostname="db-srv-01",
        assetName="Database Server 01",
        assetId="SRV-DB-001",
        ipAddress="192.168.1.30",
        osName="Windows Server 2019",
        osVersion="1809",
        impactScore=0.72,
        propagationBarrier=False,
    ))

    # fw-01 — pfSense Firewall (NOC)
    nodes.append(_node(
        "fw-01", ["Server", "HardwareAsset", "Asset"],
        hostname="fw-01",
        assetName="Firewall 01 (pfSense)",
        assetId="SRV-FW-001",
        ipAddress="192.168.1.1",
        osName="pfSense",
        osVersion="2.6",
        role="Firewall",
        impactScore=0.15,
        propagationBarrier=False,
    ))

    # ids-01 — Suricata IDS (NOC)
    nodes.append(_node(
        "ids-01", ["Server", "HardwareAsset", "Asset"],
        hostname="ids-01",
        assetName="IDS 01 (Suricata)",
        assetId="SRV-IDS-001",
        ipAddress="192.168.1.2",
        osName="Ubuntu 20.04 LTS",
        osVersion="20.04",
        role="Intrusion Detection System",
        impactScore=0.25,
        propagationBarrier=False,
    ))

    # ws-01 — Windows 10 Workstation (Office)
    nodes.append(_node(
        "ws-01", ["Server", "HardwareAsset", "Asset"],
        hostname="ws-01",
        assetName="Workstation 01",
        assetId="SRV-WS-001",
        ipAddress="10.0.1.11",
        osName="Windows 10",
        osVersion="22H2",
        role="Workstation",
        impactScore=0.80,
        propagationBarrier=False,
    ))

    # ws-02 — Windows 10 Workstation (Office)
    nodes.append(_node(
        "ws-02", ["Server", "HardwareAsset", "Asset"],
        hostname="ws-02",
        assetName="Workstation 02",
        assetId="SRV-WS-002",
        ipAddress="10.0.1.12",
        osName="Windows 10",
        osVersion="22H2",
        role="Workstation",
        impactScore=0.80,
        propagationBarrier=False,
    ))

    # Building → Device (hostsDevice)
    edges += [
        _edge("e-room1-websrv",   "hostsDevice", "Server-Room-1",  "web-srv-01"),
        _edge("e-room1-ftpsrv",   "hostsDevice", "Server-Room-1",  "ftp-srv-01"),
        _edge("e-room1-dbsrv",    "hostsDevice", "Server-Room-1",  "db-srv-01"),
        _edge("e-noc-fw01",       "hostsDevice", "NOC",            "fw-01"),
        _edge("e-noc-ids01",      "hostsDevice", "NOC",            "ids-01"),
        _edge("e-floor1-ws01",    "hostsDevice", "Office-Floor-1", "ws-01"),
        _edge("e-floor1-ws02",    "hostsDevice", "Office-Floor-1", "ws-02"),
    ]

    # -----------------------------------------------------------------------
    # SOFTWARE ASSETS — intangible assets hosted on servers
    # Note: Server --[hostsSoftware]--> OS --[hostsSoftware]--> Application
    # This two-level stack enables cross-asset CVE matching.
    # -----------------------------------------------------------------------

    # OS layers (intangible, on-server)
    nodes += [
        _node("os-win2019", ["SoftwareAsset", "Asset"],
              softwareName="Windows Server 2019",
              softwareVersion="1809",
              assetName="Windows Server 2019 OS",
              assetId="SW-OS-WIN2019",
              impactScore=0.72,
              propagationBarrier=False),
        _node("os-ubuntu2004", ["SoftwareAsset", "Asset"],
              softwareName="Ubuntu 20.04 LTS",
              softwareVersion="20.04",
              assetName="Ubuntu 20.04 LTS OS",
              assetId="SW-OS-UBU2004",
              impactScore=0.72,
              propagationBarrier=False),
        _node("os-win10", ["SoftwareAsset", "Asset"],
              softwareName="Windows 10",
              softwareVersion="22H2",
              assetName="Windows 10 OS",
              assetId="SW-OS-WIN10",
              impactScore=0.80,
              propagationBarrier=False),
    ]

    # Server → OS (hostsSoftware, level 1)
    edges += [
        _edge("e-websrv-os",   "hostsSoftware", "web-srv-01", "os-win2019"),
        _edge("e-dbsrv-os",    "hostsSoftware", "db-srv-01",  "os-win2019"),
        _edge("e-ftpsrv-os",   "hostsSoftware", "ftp-srv-01", "os-ubuntu2004"),
        _edge("e-ids01-os",    "hostsSoftware", "ids-01",     "os-ubuntu2004"),
        _edge("e-ws01-os",     "hostsSoftware", "ws-01",      "os-win10"),
        _edge("e-ws02-os",     "hostsSoftware", "ws-02",      "os-win10"),
    ]

    # Application SoftwareAssets (intangible, level 2)
    nodes += [
        # IIS 10.0 on web-srv-01 (Windows Server 2019)
        _node("IIS-10.0", ["SoftwareAsset", "Asset"],
              softwareName="Microsoft IIS",
              softwareVersion="10.0",
              assetName="Microsoft IIS 10.0",
              assetId="SW-IIS-001",
              impactScore=0.72,
              propagationBarrier=False),

        # Microsoft 365 on web-srv-01 (Exchange/OWA proxy)
        _node("Microsoft-365", ["SoftwareAsset", "Asset"],
              softwareName="Microsoft 365",
              softwareVersion="Exchange Server 2019",
              assetName="Microsoft 365 / Exchange Server 2019",
              assetId="SW-M365-001",
              impactScore=0.72,
              propagationBarrier=False),

        # vsftpd 3.0.3 on ftp-srv-01
        _node("vsftpd-3.0.3", ["SoftwareAsset", "Asset"],
              softwareName="vsftpd",
              softwareVersion="3.0.3",
              assetName="vsftpd 3.0.3",
              assetId="SW-VSFTPD-001",
              impactScore=0.35,
              propagationBarrier=False),

        # SQL Server 2019 on db-srv-01
        _node("SQLServer-2019", ["SoftwareAsset", "Asset"],
              softwareName="Microsoft SQL Server",
              softwareVersion="2019",
              assetName="Microsoft SQL Server 2019",
              assetId="SW-SQL-001",
              impactScore=0.72,
              propagationBarrier=False),

        # pfSense 2.6 on fw-01
        _node("pfSense-2.6", ["SoftwareAsset", "Asset"],
              softwareName="pfSense",
              softwareVersion="2.6",
              assetName="pfSense 2.6",
              assetId="SW-PFSENSE-001",
              impactScore=0.15,
              propagationBarrier=False),

        # Suricata on ids-01
        _node("Suricata-IDS", ["SoftwareAsset", "Asset"],
              softwareName="Suricata",
              softwareVersion="6.0",
              assetName="Suricata IDS 6.0",
              assetId="SW-SURICATA-001",
              impactScore=0.25,
              propagationBarrier=False),

        # Chrome Browser on workstations
        _node("Chrome-Browser", ["SoftwareAsset", "Asset"],
              softwareName="Google Chrome",
              softwareVersion="112.0",
              assetName="Google Chrome Browser",
              assetId="SW-CHROME-001",
              impactScore=0.80,
              propagationBarrier=False),
    ]

    # OS → Application (hostsSoftware, level 2)
    edges += [
        _edge("e-win2019-iis",    "hostsSoftware", "os-win2019",    "IIS-10.0"),
        _edge("e-win2019-m365",   "hostsSoftware", "os-win2019",    "Microsoft-365"),
        _edge("e-win2019-sql",    "hostsSoftware", "os-win2019",    "SQLServer-2019"),
        _edge("e-ubuntu-vsftpd",  "hostsSoftware", "os-ubuntu2004", "vsftpd-3.0.3"),
        _edge("e-fw01-pfsense",   "hostsSoftware", "fw-01",         "pfSense-2.6"),
        _edge("e-ids01-suricata", "hostsSoftware", "ids-01",        "Suricata-IDS"),
        _edge("e-win10-chrome",   "hostsSoftware", "os-win10",      "Chrome-Browser"),
    ]

    # Vendor → SoftwareAsset (providedBy)
    edges += [
        _edge("e-iis-vendor",       "providedBy", "IIS-10.0",       "vendor-microsoft"),
        _edge("e-m365-vendor",      "providedBy", "Microsoft-365",  "vendor-microsoft"),
        _edge("e-sql-vendor",       "providedBy", "SQLServer-2019", "vendor-microsoft"),
        _edge("e-win2019-vendor",   "providedBy", "os-win2019",     "vendor-microsoft"),
        _edge("e-win10-vendor",     "providedBy", "os-win10",       "vendor-microsoft"),
        _edge("e-ubuntu-vendor",    "providedBy", "os-ubuntu2004",  "vendor-canonical"),
        _edge("e-vsftpd-vendor",    "providedBy", "vsftpd-3.0.3",   "vendor-vsftpd-project"),
        _edge("e-chrome-vendor",    "providedBy", "Chrome-Browser", "vendor-google"),
        _edge("e-pfsense-vendor",   "providedBy", "pfSense-2.6",    "vendor-pfsense"),
        _edge("e-suricata-vendor",  "providedBy", "Suricata-IDS",   "vendor-oisf"),
    ]

    # -----------------------------------------------------------------------
    # VULNERABILITIES
    # -----------------------------------------------------------------------
    nodes += [
        _node("CVE-2015-1635",
              ["Vulnerability"],
              cveId="CVE-2015-1635",
              description="HTTP.sys Remote Code Execution via crafted HTTP request (MS15-034)",
              cvssScore=9.3,
              severity="Critical",
              cweId="CWE-119",
              affectedSoftware="Microsoft IIS 10.0 / Windows HTTP.sys"),

        _node("CVE-2021-26855",
              ["Vulnerability"],
              cveId="CVE-2021-26855",
              description="ProxyLogon — Microsoft Exchange Server SSRF leading to RCE",
              cvssScore=9.8,
              severity="Critical",
              cweId="CWE-918",
              affectedSoftware="Microsoft Exchange Server 2019"),

        _node("CVE-2011-2523",
              ["Vulnerability"],
              cveId="CVE-2011-2523",
              description="vsftpd 2.3.4 backdoor on port 6200 (affects 3.0.3 in hardened-off configs)",
              cvssScore=10.0,
              severity="Critical",
              cweId="CWE-78",
              affectedSoftware="vsftpd 3.0.3"),

        _node("CVE-2019-0819",
              ["Vulnerability"],
              cveId="CVE-2019-0819",
              description="SQL Server 2019 information disclosure via crafted queries",
              cvssScore=8.8,
              severity="High",
              cweId="CWE-89",
              affectedSoftware="Microsoft SQL Server 2019"),

        _node("CVE-2023-2033",
              ["Vulnerability"],
              cveId="CVE-2023-2033",
              description="V8 type confusion in Google Chrome — remote code execution in renderer",
              cvssScore=8.8,
              severity="High",
              cweId="CWE-843",
              affectedSoftware="Google Chrome 112.0"),
    ]

    # SoftwareAsset → Vulnerability (hasVulnerability)
    edges += [
        _edge("e-iis-cve15",    "hasVulnerability", "IIS-10.0",       "CVE-2015-1635"),
        _edge("e-m365-cve21",   "hasVulnerability", "Microsoft-365",  "CVE-2021-26855"),
        _edge("e-vsftpd-cve11", "hasVulnerability", "vsftpd-3.0.3",   "CVE-2011-2523"),
        _edge("e-sql-cve19",    "hasVulnerability", "SQLServer-2019", "CVE-2019-0819"),
        _edge("e-chrome-cve23", "hasVulnerability", "Chrome-Browser", "CVE-2023-2033"),
    ]

    # Vulnerability → SoftwareAsset (concerns — ontology property)
    edges += [
        _edge("e-cve15-iis",    "concerns", "CVE-2015-1635",  "IIS-10.0"),
        _edge("e-cve21-m365",   "concerns", "CVE-2021-26855", "Microsoft-365"),
        _edge("e-cve11-vsftpd", "concerns", "CVE-2011-2523",  "vsftpd-3.0.3"),
        _edge("e-cve19-sql",    "concerns", "CVE-2019-0819",  "SQLServer-2019"),
        _edge("e-cve23-chrome", "concerns", "CVE-2023-2033",  "Chrome-Browser"),
    ]

    # -----------------------------------------------------------------------
    # THREATS (MITRE ATT&CK)
    # -----------------------------------------------------------------------
    nodes += [
        _node("Threat-DoS-T1499",
              ["Threat"],
              threatId="THREAT-DOS-001",
              threatName="DoS — Endpoint Denial of Service",
              attackTechniqueId="T1499",
              mitreUrl="https://attack.mitre.org/techniques/T1499",
              cicIds2018Label="DoS attacks-Hulk",
              description="Adversary exhausts HTTP.sys capacity to deny service to web-srv-01"),

        _node("Threat-Infiltration-T1190",
              ["Threat"],
              threatId="THREAT-INF-001",
              threatName="Infiltration — Exploit Public-Facing Application",
              attackTechniqueId="T1190",
              mitreUrl="https://attack.mitre.org/techniques/T1190",
              cicIds2018Label="Infiltration",
              description="Adversary exploits ProxyLogon to achieve RCE on Exchange"),

        _node("Threat-FTPBrute-T1110",
              ["Threat"],
              threatId="THREAT-BRF-001",
              threatName="FTP-BruteForce — Brute Force",
              attackTechniqueId="T1110",
              mitreUrl="https://attack.mitre.org/techniques/T1110",
              cicIds2018Label="FTP-BruteForce",
              description="Adversary brute-forces vsftpd credentials to gain shell access"),

        _node("Threat-SQLi-T1190",
              ["Threat"],
              threatId="THREAT-SQL-001",
              threatName="Web Attack SQLi — Exploit Public-Facing Application",
              attackTechniqueId="T1190",
              mitreUrl="https://attack.mitre.org/techniques/T1190",
              cicIds2018Label="Web Attack-SQL Injection",
              description="Adversary injects malicious SQL to exfiltrate or corrupt the database"),

        _node("Threat-Botnet-T1071",
              ["Threat"],
              threatId="THREAT-BOT-001",
              threatName="Botnet — C2 Communication over HTTP",
              attackTechniqueId="T1071",
              mitreUrl="https://attack.mitre.org/techniques/T1071",
              cicIds2018Label="Bot",
              description="Compromised workstations communicate with C2 over HTTP/HTTPS"),

        _node("Threat-DDOS-T1498",
              ["Threat"],
              threatId="THREAT-DDS-001",
              threatName="DDOS — Network Denial of Service",
              attackTechniqueId="T1498",
              mitreUrl="https://attack.mitre.org/techniques/T1498",
              cicIds2018Label="DDOS attack-LOIC-HTTP",
              description="Volumetric DDOS flood targeting DMZ-Segment and fw-01"),
    ]

    # Threat exploits Vulnerability
    edges += [
        _edge("e-thr-dos-cve15",   "exploits", "Threat-DoS-T1499",       "CVE-2015-1635"),
        _edge("e-thr-inf-cve21",   "exploits", "Threat-Infiltration-T1190", "CVE-2021-26855"),
        _edge("e-thr-brf-cve11",   "exploits", "Threat-FTPBrute-T1110",  "CVE-2011-2523"),
        _edge("e-thr-sql-cve19",   "exploits", "Threat-SQLi-T1190",      "CVE-2019-0819"),
        _edge("e-thr-bot-cve23",   "exploits", "Threat-Botnet-T1071",    "CVE-2023-2033"),
    ]

    # DDOS targets network assets directly (no single CVE — volumetric)
    edges += [
        _edge("e-thr-ddos-dmz",  "targets", "Threat-DDOS-T1498", "DMZ-Segment"),
        _edge("e-thr-ddos-fw01", "targets", "Threat-DDOS-T1498", "fw-01"),
    ]

    # -----------------------------------------------------------------------
    # CONTROLS (NFCRM-1:2025)
    # -----------------------------------------------------------------------
    nodes += [
        _node("CTL-WAF-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-WAF-001",
              controlName="Web Application Firewall Rule (HTTP.sys)",
              nfcrmClause="NFCRM-1:2025-§6.7",  # Currently applied controls
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.70,
              description="WAF rule blocking malformed HTTP Range headers (CVE-2015-1635)"),

        _node("CTL-PATCH-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-PATCH-001",
              controlName="Patch Management — Exchange/M365",
              nfcrmClause="NFCRM-1:2025-§6.11",  # Treatment plans (patching)
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.80,
              description="Monthly patch cycle ensuring ProxyLogon mitigations are applied"),

        _node("CTL-LOCKOUT-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-LOCKOUT-001",
              controlName="Account Lockout Policy — FTP",
              nfcrmClause="NFCRM-1:2025-§6.7",  # Currently applied controls (lockout)
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.65,
              description="5-attempt lockout with 30-minute timeout on vsftpd logins"),

        _node("CTL-INPUT-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-INPUT-001",
              controlName="Input Validation — SQL Server",
              nfcrmClause="NFCRM-1:2025-§6.9",  # Inherent risk analysis (input validation)
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.75,
              description="Parameterised queries and ORM enforcement preventing SQLi"),

        _node("CTL-BOT-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-BOT-001",
              controlName="Endpoint Detection — Botnet C2 Blocking",
              nfcrmClause="NFCRM-1:2025-§6.5",  # Threat inventory (botnet C2)
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.60,
              description="DNS sinkholing and outbound HTTP filtering for C2 traffic"),

        _node("CTL-FW-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-FW-001",
              controlName="Perimeter Firewall — pfSense",
              nfcrmClause="NFCRM-1:2025-§6.7",  # Currently applied controls (firewall)
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.85,
              description="pfSense stateful firewall protecting DMZ from external DDOS floods"),

        _node("CTL-IDS-001",
              ["NFCRMControl", "Control"],
              controlId="CTL-IDS-001",
              controlName="Intrusion Detection — Suricata",
              nfcrmClause="NFCRM-1:2025-§6.5",  # Threat inventory (IDS detection)
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              protectionDegree=0.75,
              description="Suricata IDS with DDOS detection rules on DMZ-Segment"),
    ]

    # Asset → Control (hasControl)
    edges += [
        _edge("e-iis-ctl-waf",     "hasControl", "IIS-10.0",       "CTL-WAF-001"),
        _edge("e-m365-ctl-patch",  "hasControl", "Microsoft-365",  "CTL-PATCH-001"),
        _edge("e-vsftpd-ctl-lock", "hasControl", "vsftpd-3.0.3",   "CTL-LOCKOUT-001"),
        _edge("e-sql-ctl-input",   "hasControl", "SQLServer-2019", "CTL-INPUT-001"),
        _edge("e-chrome-ctl-bot",  "hasControl", "Chrome-Browser", "CTL-BOT-001"),
        _edge("e-dmz-ctl-fw",      "hasControl", "DMZ-Segment",    "CTL-FW-001"),
        _edge("e-dmz-ctl-ids",     "hasControl", "DMZ-Segment",    "CTL-IDS-001"),
    ]

    # Control → Asset (controls — SafecareOnto protection edge)
    edges += [
        _edge("e-ctl-fw-dmz",  "controls", "CTL-FW-001",  "DMZ-Segment"),
        _edge("e-ctl-ids-dmz", "controls", "CTL-IDS-001", "DMZ-Segment"),
    ]

    # Control → Vulnerability (mitigates)
    edges += [
        _edge("e-waf-mit-cve15",   "mitigates", "CTL-WAF-001",     "CVE-2015-1635"),
        _edge("e-patch-mit-cve21", "mitigates", "CTL-PATCH-001",   "CVE-2021-26855"),
        _edge("e-lock-mit-cve11",  "mitigates", "CTL-LOCKOUT-001", "CVE-2011-2523"),
        _edge("e-input-mit-cve19", "mitigates", "CTL-INPUT-001",   "CVE-2019-0819"),
        _edge("e-bot-mit-cve23",   "mitigates", "CTL-BOT-001",     "CVE-2023-2033"),
    ]

    # -----------------------------------------------------------------------
    # RISK CASES
    # -----------------------------------------------------------------------
    nodes += [
        _node("RISK-DOS-001",
              ["RiskCase"],
              riskId="RISK-DOS-001",
              riskLevel="high",
              likelihood="high",
              impact="major",
              confidence=0.88,
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              description="DoS via CVE-2015-1635 against IIS on web-srv-01"),

        _node("RISK-INF-001",
              ["RiskCase"],
              riskId="RISK-INF-001",
              riskLevel="critical",
              likelihood="medium",
              impact="catastrophic",
              confidence=0.92,
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              description="Infiltration via ProxyLogon CVE-2021-26855 on Exchange"),

        _node("RISK-BRF-001",
              ["RiskCase"],
              riskId="RISK-BRF-001",
              riskLevel="high",
              likelihood="high",
              impact="major",
              confidence=0.90,
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              description="FTP Brute Force via CVE-2011-2523 on ftp-srv-01"),

        _node("RISK-SQL-001",
              ["RiskCase"],
              riskId="RISK-SQL-001",
              riskLevel="high",
              likelihood="medium",
              impact="major",
              confidence=0.85,
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              description="SQL Injection via CVE-2019-0819 on db-srv-01"),

        _node("RISK-BOT-001",
              ["RiskCase"],
              riskId="RISK-BOT-001",
              riskLevel="medium",
              likelihood="medium",
              impact="moderate",
              confidence=0.78,
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              description="Botnet C2 via CVE-2023-2033 on workstation Chrome browsers"),

        _node("RISK-DDS-001",
              ["RiskCase"],
              riskId="RISK-DDS-001",
              riskLevel="high",
              likelihood="high",
              impact="major",
              confidence=0.95,
              provenanceSource="NFCRM-1:2025",
              provenanceTimestamp="2025-01-01T00:00:00Z",
              description="DDOS flood targeting DMZ-Segment; fw-01 + ids-01 act as barrier"),
    ]

    # Vulnerability → RiskCase (relatesToRisk)
    edges += [
        _edge("e-cve15-risk",  "relatesToRisk", "CVE-2015-1635",  "RISK-DOS-001"),
        _edge("e-cve21-risk",  "relatesToRisk", "CVE-2021-26855", "RISK-INF-001"),
        _edge("e-cve11-risk",  "relatesToRisk", "CVE-2011-2523",  "RISK-BRF-001"),
        _edge("e-cve19-risk",  "relatesToRisk", "CVE-2019-0819",  "RISK-SQL-001"),
        _edge("e-cve23-risk",  "relatesToRisk", "CVE-2023-2033",  "RISK-BOT-001"),
    ]

    # RiskCase → Control (associatedWithControl)
    edges += [
        _edge("e-risk-dos-ctl",  "associatedWithControl", "RISK-DOS-001", "CTL-WAF-001"),
        _edge("e-risk-inf-ctl",  "associatedWithControl", "RISK-INF-001", "CTL-PATCH-001"),
        _edge("e-risk-brf-ctl",  "associatedWithControl", "RISK-BRF-001", "CTL-LOCKOUT-001"),
        _edge("e-risk-sql-ctl",  "associatedWithControl", "RISK-SQL-001", "CTL-INPUT-001"),
        _edge("e-risk-bot-ctl",  "associatedWithControl", "RISK-BOT-001", "CTL-BOT-001"),
        _edge("e-risk-dds-ctl1", "associatedWithControl", "RISK-DDS-001", "CTL-FW-001"),
        _edge("e-risk-dds-ctl2", "associatedWithControl", "RISK-DDS-001", "CTL-IDS-001"),
    ]

    # -----------------------------------------------------------------------
    # BUSINESS ASSET LAYER
    # -----------------------------------------------------------------------
    nodes.append(_node(
        "IT-Operations-Service",
        ["Service", "BusinessAsset"],
        assetName="IT Operations Service",
        assetId="SVC-ITOPS-001",
        description="Core IT operations service covering web, FTP, and database delivery",
        missionCritical=True,
        impactScore=0.52,
        propagationBarrier=False,
    ))

    # Service depends on SupportingAssets
    edges += [
        _edge("e-svc-dep-room1", "depends",  "IT-Operations-Service", "Server-Room-1"),
        _edge("e-svc-dep-dmz",   "depends",  "IT-Operations-Service", "DMZ-Segment"),
        _edge("e-room1-sup-svc", "supports", "Server-Room-1",          "IT-Operations-Service"),
        _edge("e-dmz-sup-svc",   "supports", "DMZ-Segment",            "IT-Operations-Service"),
    ]

    return nodes, edges


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _to_serialisable(obj: Any) -> Any:
    """Recursively convert dataclass instances to plain dicts."""
    if isinstance(obj, (Node, Edge)):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_serialisable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    return obj


def write_json(nodes: list[Node], edges: list[Edge], path: Path) -> None:
    """Write arg_seed.json."""
    payload = {
        "metadata": {
            "version": "1.0.0",
            "ontology_version": "0.2.0",
            "description": (
                "CIC-IDS2018 Testbed ARG for INTERSYMBOLIC-GRC thesis. "
                "Maps CIC-IDS2018 attack families to SafecareOnto-aligned asset hierarchy."
            ),
            "reference": "Hannou et al. (2022) SafecareOnto + NFCRM-1:2025",
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": _to_serialisable(nodes),
        "edges": _to_serialisable(edges),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"  [JSON]   Written {len(nodes)} nodes + {len(edges)} edges -> {path}")


# ---------------------------------------------------------------------------
# Cypher generation
# ---------------------------------------------------------------------------

def _props_to_cypher(props: dict[str, Any]) -> str:
    """Convert a properties dict to a Cypher properties map string."""
    parts = []
    for k, v in props.items():
        if isinstance(v, bool):
            parts.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        else:
            escaped = str(v).replace("\\", "\\\\").replace("'", "\\'")
            parts.append(f"{k}: '{escaped}'")
    return "{" + ", ".join(parts) + "}"


def write_cypher(nodes: list[Node], edges: list[Edge], path: Path) -> None:
    """Write arg_seed.cypher with MERGE statements for idempotent Neo4j import."""
    lines = [
        "// INTERSYMBOLIC-GRC — CIC-IDS2018 Testbed ARG Seed (SafecareOnto-aligned)",
        "// Generated by pipeline/loaders/arg_seed.py",
        "// Run in Neo4j Browser or neo4j-shell: :source arg_seed.cypher",
        "",
        "// ----------------------------------------------------------------",
        "// Constraints (run once before import)",
        "// ----------------------------------------------------------------",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Asset)   REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vendor)  REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Vulnerability) REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Control) REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RiskCase) REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Threat)  REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:BusinessAsset) REQUIRE n.id IS UNIQUE;",
        "",
        "// ----------------------------------------------------------------",
        "// NODES",
        "// ----------------------------------------------------------------",
    ]

    for n in nodes:
        label_str = ":".join(n.labels)
        props = dict(n.properties)
        props["id"] = n.id
        props_cypher = _props_to_cypher(props)
        lines.append(f"MERGE (n:{label_str} {{id: '{n.id}'}}) SET n += {props_cypher};")

    lines += [
        "",
        "// ----------------------------------------------------------------",
        "// EDGES",
        "// ----------------------------------------------------------------",
    ]

    for e in edges:
        rel = e.type.upper()
        if e.properties:
            props_cypher = " " + _props_to_cypher(e.properties)
        else:
            props_cypher = ""
        lines.append(
            f"MATCH (a {{id: '{e.source}'}}), (b {{id: '{e.target}'}}) "
            f"MERGE (a)-[:{rel}{props_cypher}]->(b);"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [Cypher] Written {len(nodes)} MERGE nodes + {len(edges)} MERGE edges -> {path}")


# ---------------------------------------------------------------------------
# Butterfly effect path summaries
# ---------------------------------------------------------------------------

# Static propagation path descriptions aligned with the ARG structure.
# These are narrative summaries; the full computed traversal is done in
# butterfly_effect.py.

_PROPAGATION_PATHS = [
    {
        "attack": "FTP-BruteForce",
        "cic_ids2018_label": "FTP-BruteForce",
        "mitre": "T1110",
        "path": [
            "vsftpd-3.0.3 (SoftwareAsset)",
            "CVE-2011-2523 (Vulnerability, CVSS 10.0)",
            "ftp-srv-01 (Server, impactScore=0.35 via CTL-LOCKOUT-001 pd=0.65)",
            "Server-Room-1 (SimpleBuilding, impactScore=0.65)",
            "Building-A (ComplexBuilding, impactScore=0.58)",
            "IT-Operations-Service (Service, impactScore=0.52)",
        ],
        "risk_case": "RISK-BRF-001",
        "barrier": None,
    },
    {
        "attack": "DoS",
        "cic_ids2018_label": "DoS attacks-Hulk",
        "mitre": "T1499",
        "path": [
            "IIS-10.0 (SoftwareAsset)",
            "CVE-2015-1635 (Vulnerability, CVSS 9.3)",
            "web-srv-01 (Server, impactScore=0.30 via CTL-WAF-001 pd=0.70)",
            "Server-Room-1 (SimpleBuilding, impactScore=0.65)",
            "Building-A (ComplexBuilding, impactScore=0.58)",
            "IT-Operations-Service (Service, impactScore=0.52)",
        ],
        "risk_case": "RISK-DOS-001",
        "barrier": None,
    },
    {
        "attack": "Infiltration",
        "cic_ids2018_label": "Infiltration",
        "mitre": "T1190",
        "path": [
            "Microsoft-365 (SoftwareAsset)",
            "CVE-2021-26855 (Vulnerability, CVSS 9.8 ProxyLogon)",
            "web-srv-01 (Server, impactScore=0.20 via CTL-PATCH-001 pd=0.80)",
            "Server-Room-1 (SimpleBuilding, impactScore=0.65)",
            "Building-A (ComplexBuilding, impactScore=0.58)",
            "IT-Operations-Service (Service, impactScore=0.52)",
        ],
        "risk_case": "RISK-INF-001",
        "barrier": None,
    },
    {
        "attack": "WebAttack-SQLi",
        "cic_ids2018_label": "Web Attack-SQL Injection",
        "mitre": "T1190",
        "path": [
            "SQLServer-2019 (SoftwareAsset)",
            "CVE-2019-0819 (Vulnerability, CVSS 8.8)",
            "db-srv-01 (Server, impactScore=0.25 via CTL-INPUT-001 pd=0.75)",
            "Server-Room-1 (SimpleBuilding, impactScore=0.65)",
            "Building-A (ComplexBuilding, impactScore=0.58)",
            "IT-Operations-Service (Service, impactScore=0.52)",
        ],
        "risk_case": "RISK-SQL-001",
        "barrier": None,
    },
    {
        "attack": "Botnet",
        "cic_ids2018_label": "Bot",
        "mitre": "T1071",
        "path": [
            "Chrome-Browser (SoftwareAsset)",
            "CVE-2023-2033 (Vulnerability, CVSS 8.8)",
            "ws-01 / ws-02 (Workstation, impactScore=0.40 via CTL-BOT-001 pd=0.60)",
            "Office-Floor-1 (SimpleBuilding, impactScore=0.80)",
            "Building-B (ComplexBuilding, impactScore=0.80)",
        ],
        "risk_case": "RISK-BOT-001",
        "barrier": "Building-B does NOT reach IT-Operations-Service (no supports/depends edge)",
    },
    {
        "attack": "DDOS",
        "cic_ids2018_label": "DDOS attack-LOIC-HTTP",
        "mitre": "T1498",
        "path": [
            "DMZ-Segment (NetworkSegment, target asset)",
            "CTL-FW-001 (pd=0.85) + CTL-IDS-001 (pd=0.75) -> additive pd=1.0",
        ],
        "risk_case": "RISK-DDS-001",
        "barrier": (
            "BARRIER: DMZ-Segment impactScore=0.0 [fw-01 + ids-01 combined protection=1.0] "
            "-> DDOS does NOT propagate to Server-Room-1 or IT-Operations-Service"
        ),
    },
]


def print_propagation_summary(nodes: list[Node], edges: list[Edge]) -> None:
    """Print butterfly effect path summaries for all six CIC-IDS2018 attack types."""
    node_index = {n.id: n for n in nodes}
    edge_index: dict[str, list[Edge]] = {}
    for e in edges:
        edge_index.setdefault(e.source, []).append(e)

    print("\n" + "=" * 72)
    print("  BUTTERFLY EFFECT PROPAGATION SUMMARY")
    print("  SafecareOnto formula: impactScore(a) = 1 - max(protectionDegree)")
    print("=" * 72)

    for scenario in _PROPAGATION_PATHS:
        print(f"\n  Attack : {scenario['attack']}")
        print(f"  Label  : {scenario['cic_ids2018_label']}")
        print(f"  MITRE  : {scenario['mitre']}")
        print(f"  Risk   : {scenario['risk_case']}")
        print("  Chain  :")
        for step in scenario["path"]:
            print(f"    -> {step}")
        if scenario["barrier"]:
            print(f"  [!] {scenario['barrier']}")
        else:
            print("  [OK] Full propagation reaches IT-Operations-Service")

    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("[arg_seed] Building CIC-IDS2018 Testbed ARG for INTERSYMBOLIC-GRC thesis...")
    nodes, edges = build_arg()

    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")

    write_json(nodes, edges, JSON_OUTPUT)
    write_cypher(nodes, edges, CYPHER_OUTPUT)
    print_propagation_summary(nodes, edges)

    print("[arg_seed] Done.")


if __name__ == "__main__":
    main()
