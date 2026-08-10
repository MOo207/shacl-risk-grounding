"""CIC-IDS2018 Testbed CMDB — regeneration / audit script.

The authoritative CMDB is data/external/cmdb_assets.json.
This script validates and prints a summary of its contents.

Sources:
  Metasploitable 2 (SRV-001 -- SRV-027):
    Rapid7 Metasploitable 2 Exploitability Guide
    https://docs.rapid7.com/metasploit/metasploitable-2-exploitability-guide/

  CIC infrastructure hosts (SRV-028 -- SRV-033):
    Sharafaldin, Lashkari & Ghorbani, ICISSP 2018
    DOI: 10.5220/0006639801080116

  Network topology confirmed by:
    Distrinet/KU Leuven PCAP re-analysis
    https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
OUTPUT = os.path.join(DATA_DIR, "cmdb_assets.json")


def main():
    with open(OUTPUT, encoding="utf-8") as f:
        data = json.load(f)

    print(f"CIC-IDS2018 Testbed CMDB v{data['cmdb_version']}")
    print(f"Total assets : {data['total_assets']}")
    print(f"MS2 services : {data['asset_breakdown']['metasploitable2_services']}")
    print(f"CIC hosts    : {data['asset_breakdown']['cic_infrastructure_hosts']}")
    print(f"Subnet       : {data['asset_breakdown']['network_subnet']}")
    print()

    all_cves = []
    for asset in data["assets"]:
        cves = [c["cve_id"] for sw in asset["software"] for c in sw.get("cves", [])]
        all_cves.extend(cves)
        tag = f"{asset['asset_id']:8s} {asset['hostname']:28s} {asset['ip']}:{asset.get('port') or '-':<5}"
        print(f"  {tag}  CVEs={cves or '[]'}")

    print(f"\nUnique CVEs across all assets: {len(set(all_cves))}")
    print(f"Attack asset map keys: {list(data['attack_asset_map'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
