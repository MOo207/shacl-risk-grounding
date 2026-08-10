"""Extend data/external/nvd_enrichment.json with CVSS data for ~60 KEV CVEs.

Reads existing nvd_enrichment.json, fetches missing CVE data from NVD API
for the CVEs listed in tests/llm_experiment/kev_attack_mapping.json (after
that fixture is built in Task 3), merges, writes back.

This script is idempotent: running it twice produces the same output if
the upstream NVD API is unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def load_existing_enrichment(path: str) -> dict[str, dict]:
    """Flatten the existing nvd_enrichment.json into {cve_id: record}."""
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    out: dict[str, dict] = {}
    for _cpe, records in data.get("cpe_to_cves", {}).items():
        for rec in records:
            out[rec["cve_id"]] = rec
    return out


def fetch_cve_from_nvd(cve_id: str, sleep_sec: float = 6.0) -> dict[str, Any] | None:
    url = f"{NVD_API}?cveId={cve_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  [warn] NVD fetch failed for {cve_id}: {e}")
        time.sleep(sleep_sec)
        return None
    time.sleep(sleep_sec)
    items = payload.get("vulnerabilities") or []
    if not items:
        return None
    cve = items[0]["cve"]
    metrics = (cve.get("metrics") or {}).get("cvssMetricV31") or []
    cvss = metrics[0]["cvssData"] if metrics else {}
    descr = next(
        (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
        "",
    )
    return {
        "cve_id": cve_id,
        "cvss_v3": cvss.get("baseScore"),
        "severity": cvss.get("baseSeverity"),
        "attack_vector": cvss.get("attackVector"),
        "attack_complexity": cvss.get("attackComplexity"),
        "privileges_required": cvss.get("privilegesRequired"),
        "user_interaction": cvss.get("userInteraction"),
        "scope": cvss.get("scope"),
        "confidentiality_impact": cvss.get("confidentialityImpact"),
        "integrity_impact": cvss.get("integrityImpact"),
        "availability_impact": cvss.get("availabilityImpact"),
        "exploitability_score": (metrics[0].get("exploitabilityScore") if metrics else None),
        "vector_string": cvss.get("vectorString"),
        "description": descr,
        "in_cisa_kev": True,
    }


def merge_cve_records(existing: dict[str, dict], new_records: list[dict]) -> dict[str, dict]:
    """Merge new CVE records into existing dict (new fields win on overlap)."""
    out = dict(existing)
    for rec in new_records:
        cve_id = rec["cve_id"]
        if cve_id in out:
            out[cve_id] = {**out[cve_id], **rec}
        else:
            out[cve_id] = rec
    return out


def write_merged_enrichment(merged: dict[str, dict], path: str) -> None:
    """Write merged map back into the cpe_to_cves shape."""
    bucket: dict[str, list[dict]] = {"cpe:2.3:_kev_extended:_:_:_:_:_:_:_:_:_:_": []}
    for rec in merged.values():
        bucket["cpe:2.3:_kev_extended:_:_:_:_:_:_:_:_:_:_"].append(rec)
    out = {
        "source": "NVD API 2.0 + CISA KEV cross-reference (extended for LLM experiment)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_cves": len(merged),
        "cves_in_cisa_kev": sum(1 for r in merged.values() if r.get("in_cisa_kev")),
        "cpe_to_cves": bucket,
    }
    Path(path).write_text(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="tests/llm_experiment/kev_attack_mapping.json")
    ap.add_argument("--enrichment", default="data/external/nvd_enrichment.json")
    args = ap.parse_args()

    existing = load_existing_enrichment(args.enrichment)
    print(f"Loaded {len(existing)} existing CVE records")

    mapping = json.loads(Path(args.mapping).read_text())
    needed = sorted({entry["cve_id"] for entries in mapping.values() for entry in entries})
    missing = [c for c in needed if c not in existing]
    print(f"Need {len(needed)} CVEs total; {len(missing)} not yet enriched")

    new_records = []
    for i, cve_id in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] fetching {cve_id}")
        rec = fetch_cve_from_nvd(cve_id)
        if rec:
            new_records.append(rec)

    merged = merge_cve_records(existing, new_records)
    write_merged_enrichment(merged, args.enrichment)
    print(f"Wrote {len(merged)} CVEs to {args.enrichment}")


if __name__ == "__main__":
    main()
