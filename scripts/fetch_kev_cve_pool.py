"""Fetch real NVD records for a sample of the CISA KEV catalogue.

The enterprise-scale ARG study needs a CVE pool large enough that vulnerability
nodes are not the binding constraint on graph size, and every CVE in it must be
real: KEV supplies the identifier, vendor, product and CWE, but carries no CVSS
score, and grc:VulnerabilityShape requires one. This pulls the missing fields
from NVD API 2.0.

Sampling is seeded and stratified by vendor so the pool is not dominated by the
one vendor with 361 KEV entries. Re-running extends the cache rather than
refetching it.

Usage: python scripts/fetch_kev_cve_pool.py [--target 300] [--seed 42]
Output: data/external/nvd_kev_pool.json
"""
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "external")
CACHE = os.path.join(DATA_DIR, "nvd_kev_pool.json")

API = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId="
SLEEP_SECONDS = 6.5
TIMEOUT = 30
RETRIES = 3


def stratified_sample(kev_records, target, seed):
    """Round-robin over vendors so no single vendor dominates the pool."""
    by_vendor = defaultdict(list)
    for r in kev_records:
        by_vendor[r["vendorProject"]].append(r)
    rng = random.Random(seed)
    for v in by_vendor.values():
        rng.shuffle(v)
    vendors = sorted(by_vendor)
    rng.shuffle(vendors)

    picked, exhausted = [], False
    while len(picked) < target and not exhausted:
        exhausted = True
        for v in vendors:
            if by_vendor[v]:
                picked.append(by_vendor[v].pop())
                exhausted = False
                if len(picked) >= target:
                    break
    return picked


def fetch_one(cve_id):
    last_err = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(API + cve_id, timeout=TIMEOUT) as r:
                payload = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(SLEEP_SECONDS * (attempt + 1))
    else:
        return None, f"failed after {RETRIES} attempts: {last_err}"

    vulns = payload.get("vulnerabilities", [])
    if not vulns:
        return None, "not present in NVD"
    cve = vulns[0]["cve"]

    cwe = ""
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            if d.get("value", "").startswith("CWE-"):
                cwe = d["value"]
                break
        if cwe:
            break

    metrics = cve.get("metrics", {})
    score, severity, version = None, "", ""
    for key, ver in (("cvssMetricV31", "3.1"), ("cvssMetricV30", "3.0")):
        if metrics.get(key):
            d = metrics[key][0]["cvssData"]
            score, severity, version = d["baseScore"], d["baseSeverity"], ver
            break
    if score is None and metrics.get("cvssMetricV2"):
        m = metrics["cvssMetricV2"][0]
        score = m["cvssData"]["baseScore"]
        severity = m.get("baseSeverity", "")
        version = "2.0"
    if score is None:
        return None, "no CVSS metric in NVD"

    desc = ""
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            desc = d.get("value", "")[:300]
            break

    return {
        "cve_id": cve.get("id", cve_id),
        "cwe": cwe,
        "cvss_score": score,
        "cvss_version": version,
        "severity": (severity or "").lower(),
        "description": desc,
        "published": cve.get("published", ""),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "cisa_kev.json"), encoding="utf-8") as f:
        kev = json.load(f)
    records = kev["vulnerabilities"]

    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {"source": "CISA KEV + NVD API 2.0", "seed": args.seed,
                 "cves": {}, "skipped": {}}

    sample = stratified_sample(records, args.target, args.seed)
    todo = [r for r in sample
            if r["cveID"] not in cache["cves"] and r["cveID"] not in cache["skipped"]]
    print(f"KEV catalogue: {len(records)} entries; stratified sample {len(sample)}; "
          f"{len(todo)} still to fetch (~{len(todo) * SLEEP_SECONDS / 60:.0f} min)")

    for i, rec in enumerate(todo, 1):
        cid = rec["cveID"]
        got, err = fetch_one(cid)
        if got:
            got["vendor"] = rec["vendorProject"]
            got["product"] = rec["product"]
            got["in_cisa_kev"] = True
            got["known_ransomware"] = rec.get("knownRansomwareCampaignUse", "Unknown")
            cache["cves"][cid] = got
        else:
            cache["skipped"][cid] = err
        if i % 10 == 0 or i == len(todo):
            with open(CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            print(f"  [{i}/{len(todo)}] cached={len(cache['cves'])} skipped={len(cache['skipped'])}")
        if i < len(todo):
            time.sleep(SLEEP_SECONDS)

    cves = cache["cves"]
    cache["summary"] = {
        "pool_size": len(cves),
        "with_cwe": sum(1 for c in cves.values() if c["cwe"]),
        "distinct_vendors": len({c["vendor"] for c in cves.values()}),
        "distinct_products": len({(c["vendor"], c["product"]) for c in cves.values()}),
        "cvss_versions": {
            v: sum(1 for c in cves.values() if c["cvss_version"] == v)
            for v in ("3.1", "3.0", "2.0")
        },
    }
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\nPool: {cache['summary']}")
    print(f"  -> {CACHE}")


if __name__ == "__main__":
    main()
