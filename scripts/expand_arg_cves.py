"""
Expand ARG with real CVEs from NVD API (T3.2)
==============================================
Fetches 30 additional real CVEs from the NVD REST API 2.0
covering network-related attack categories and adds them to
results/multisource_arg.json.

Target CVE categories:
  - HTTP/web server CVEs (DoS, RCE via web)
  - SSH/FTP brute-force related
  - Network infrastructure (DNS, SMTP, proxy)
  - Intrusion/infiltration relevant (C2, lateral movement)

Usage: python scripts/expand_arg_cves.py
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent


def fetch_cves_nvd(keyword: str, max_results: int = 8) -> list:
    """Fetch CVEs from NVD API 2.0 by keyword."""
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
        "keywordSearch": keyword,
        "resultsPerPage": max_results,
        "startIndex": 0,
    }
    try:
        resp = requests.get(url, params=params, timeout=30,
                            headers={"User-Agent": "INTERSYMBOLIC-GRC-Thesis/1.0"})
        resp.raise_for_status()
        data = resp.json()
        cves = []
        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            desc_list = cve.get("descriptions", [])
            desc = next((d["value"] for d in desc_list if d["lang"] == "en"), "")[:120]
            metrics = cve.get("metrics", {})
            cvss_v3 = 0.0
            severity = "UNKNOWN"
            cwe = ""
            # Try CVSS v3.1, then v3.0
            for key in ["cvssMetricV31", "cvssMetricV30"]:
                if key in metrics and metrics[key]:
                    m = metrics[key][0]
                    cvss_v3 = m.get("cvssData", {}).get("baseScore", 0.0)
                    severity = m.get("cvssData", {}).get("baseSeverity", "UNKNOWN")
                    break
            # CWE
            weaknesses = cve.get("weaknesses", [])
            if weaknesses:
                descs = weaknesses[0].get("description", [])
                cwe = next((d["value"] for d in descs if d["value"].startswith("CWE-")), "")
            # Published date
            published = cve.get("published", "")[:10]
            if cve_id and desc:
                cves.append({
                    "cve_id": cve_id,
                    "cvss_v3": cvss_v3,
                    "severity": severity.upper() if severity != "UNKNOWN" else "MEDIUM",
                    "description": desc,
                    "cwe": cwe,
                    "in_cisa_kev": False,  # conservative — KEV requires separate API call
                    "published": published,
                    "source": "NVD_API_2.0",
                })
        return cves
    except Exception as e:
        print(f"  NVD fetch error for '{keyword}': {e}")
        return []


# Curated real CVEs to add (verified, network-relevant, recent)
# These supplement NVD API results with high-quality entries
CURATED_CVES = [
    {
        "cve_id": "CVE-2023-44487",
        "cvss_v3": 7.5, "severity": "HIGH",
        "description": "HTTP/2 Rapid Reset Attack: zero-day DDoS via stream cancellation at scale",
        "cwe": "CWE-400", "in_cisa_kev": True, "published": "2023-10-10",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2022-26134",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "Atlassian Confluence OGNL injection RCE via HTTP request (actively exploited)",
        "cwe": "CWE-74", "in_cisa_kev": True, "published": "2022-06-02",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2021-44228",
        "cvss_v3": 10.0, "severity": "CRITICAL",
        "description": "Log4Shell: Apache Log4j2 JNDI injection RCE via crafted log messages",
        "cwe": "CWE-917", "in_cisa_kev": True, "published": "2021-12-10",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2023-34362",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "MOVEit Transfer SQL injection leading to RCE and data exfiltration (Cl0p ransomware)",
        "cwe": "CWE-89", "in_cisa_kev": True, "published": "2023-06-02",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2024-21762",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "Fortinet FortiOS SSL VPN out-of-bounds write RCE (actively exploited)",
        "cwe": "CWE-787", "in_cisa_kev": True, "published": "2024-02-08",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2023-46604",
        "cvss_v3": 10.0, "severity": "CRITICAL",
        "description": "Apache ActiveMQ RCE via ClassPathXmlApplicationContext (ransomware delivery)",
        "cwe": "CWE-502", "in_cisa_kev": True, "published": "2023-10-27",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2022-47966",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "Zoho ManageEngine multiple products RCE via SAML authentication bypass",
        "cwe": "CWE-287", "in_cisa_kev": True, "published": "2023-01-10",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2023-27350",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "PaperCut MF/NG authentication bypass and RCE via SetupCompleted flag",
        "cwe": "CWE-284", "in_cisa_kev": True, "published": "2023-04-20",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2024-3400",
        "cvss_v3": 10.0, "severity": "CRITICAL",
        "description": "Palo Alto GlobalProtect command injection via crafted SESSID cookie (Operation MidnightEclipse)",
        "cwe": "CWE-77", "in_cisa_kev": True, "published": "2024-04-12",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2022-22954",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "VMware Workspace ONE Access SSTI leading to RCE without authentication",
        "cwe": "CWE-94", "in_cisa_kev": True, "published": "2022-04-11",
        "source": "curated",
    },
    {
        "cve_id": "CVE-2021-26084",
        "cvss_v3": 9.8, "severity": "CRITICAL",
        "description": "Atlassian Confluence OGNL injection pre-auth RCE via WebWork namespace",
        "cwe": "CWE-74", "in_cisa_kev": True, "published": "2021-08-25",
        "source": "curated",
    },
]


def main():
    arg_path = ROOT / "results" / "multisource_arg.json"
    print(f"[T3.2] Loading ARG from {arg_path}")
    with open(arg_path) as f:
        arg = json.load(f)

    existing_cve_ids = {n["id"] for n in arg["nodes"] if n["type"] == "CVE"}
    print(f"  Existing CVEs: {len(existing_cve_ids)}")

    # NVD API queries for network-relevant CVEs
    queries = [
        ("HTTP denial of service", 5),
        ("SSH brute force authentication", 4),
        ("web application SQL injection", 4),
        ("network infiltration lateral movement", 4),
    ]

    new_cves = []

    # Add curated CVEs
    for cve in CURATED_CVES:
        if cve["cve_id"] not in existing_cve_ids:
            new_cves.append(cve)
            existing_cve_ids.add(cve["cve_id"])

    # Fetch from NVD API
    for query, max_r in queries:
        print(f"  Fetching NVD: '{query}' (max {max_r})")
        fetched = fetch_cves_nvd(query, max_r)
        for cve in fetched:
            if cve["cve_id"] not in existing_cve_ids and cve["cvss_v3"] >= 5.0:
                new_cves.append(cve)
                existing_cve_ids.add(cve["cve_id"])
        time.sleep(1)  # NVD rate limit: 5 req/30s without API key

    print(f"\n  New CVEs to add: {len(new_cves)}")
    for c in new_cves:
        print(f"    {c['cve_id']} CVSS={c['cvss_v3']} {c['severity']} — {c['description'][:60]}")

    # Add CVE nodes to ARG
    for cve in new_cves:
        node = {
            "id": cve["cve_id"],
            "type": "CVE",
            "cvss_v3": cve["cvss_v3"],
            "severity": cve["severity"],
            "description": cve["description"],
            "cwe": cve.get("cwe", ""),
            "in_cisa_kev": cve.get("in_cisa_kev", False),
            "published": cve.get("published", ""),
            "source": cve.get("source", "NVD_API_2.0"),
        }
        arg["nodes"].append(node)
        # Link to generic network asset (Asset-1 = web server in synthetic ARG)
        arg["edges"].append({
            "source": "Asset-1",
            "target": cve["cve_id"],
            "relation": "EXPOSED_TO_CVE",
        })

    # Update statistics
    arg["statistics"]["total_nodes"] = len(arg["nodes"])
    arg["statistics"]["total_edges"] = len(arg["edges"])
    arg["statistics"]["cve_count"] = len([n for n in arg["nodes"] if n["type"] == "CVE"])
    arg["statistics"]["expansion_timestamp"] = datetime.now().isoformat()
    arg["statistics"]["expansion_note"] = (
        f"T3.2: Added {len(new_cves)} real CVEs from NVD API 2.0 and curated list. "
        "Total CVE coverage now includes HTTP/2 rapid reset, Log4Shell, MOVEit, "
        "FortiOS, ActiveMQ, and other actively-exploited network CVEs."
    )

    with open(arg_path, "w") as f:
        json.dump(arg, f, indent=2)

    total_cves = len([n for n in arg["nodes"] if n["type"] == "CVE"])
    print(f"\n[T3.2] ARG expanded:")
    print(f"  Total nodes: {len(arg['nodes'])} (was {len(arg['nodes']) - len(new_cves)})")
    print(f"  Total edges: {len(arg['edges'])}")
    print(f"  Total CVEs:  {total_cves}")
    print(f"[T3.2] Saved to {arg_path}")


if __name__ == "__main__":
    main()
