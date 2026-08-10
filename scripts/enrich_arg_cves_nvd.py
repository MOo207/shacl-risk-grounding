"""Fetch real CWE + CVSS + severity from NVD API 2.0 for every CVE node in the ARG.

The testbed CMDB fixture (data/external/cmdb_assets.json) carries only
cve_id/description/cvss for its CVE entries -- no CWE and no severity string.
Those two fields are required by grc:VulnerabilityShape, so without them the
graph cannot conform. We fetch them from the authoritative source rather than
deriving severity from the CVSS band or inventing a CWE.

Writes data/external/nvd_arg_cves.json (a cache keyed by CVE id), which
build_multisource_arg.py consumes. Re-running is idempotent: cached CVEs are
not re-fetched unless --refresh is passed.

Usage: python scripts/enrich_arg_cves_nvd.py [--refresh]
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "external")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE = os.path.join(DATA_DIR, "nvd_arg_cves.json")

API = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId="
# NVD asks for <=5 requests per 30s without an API key.
SLEEP_SECONDS = 6.5
TIMEOUT = 30
RETRIES = 3


def arg_cve_ids():
    with open(os.path.join(RESULTS_DIR, "multisource_arg.json"), encoding="utf-8") as f:
        arg = json.load(f)
    return sorted({n["id"] for n in arg["nodes"] if n["type"] == "CVE"})


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {"source": "NVD API 2.0", "fetched": {}, "cves": {}}


def fetch_one(cve_id):
    """Return the NVD record for one CVE, or None if NVD has no usable entry."""
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
        print(f"  {cve_id}: FAILED after {RETRIES} attempts ({last_err})")
        return None

    vulns = payload.get("vulnerabilities", [])
    if not vulns:
        print(f"  {cve_id}: not present in NVD")
        return None
    cve = vulns[0]["cve"]

    # CWE: first non-informational weakness description
    cwe = ""
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            v = d.get("value", "")
            if v.startswith("CWE-"):
                cwe = v
                break
        if cwe:
            break

    # CVSS: prefer v3.1, then v3.0, then v2. Record which one was used --
    # a v2-only CVE has no v3 base score and saying otherwise would be wrong.
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

    return {
        "cve_id": cve.get("id", cve_id),
        "cwe": cwe,
        "cvss_score": score,
        "cvss_version": version,
        "severity": severity.lower() if severity else "",
        "published": cve.get("published", ""),
        "vuln_status": cve.get("vulnStatus", ""),
    }


def main():
    refresh = "--refresh" in sys.argv
    ids = arg_cve_ids()
    cache = load_cache()
    todo = [c for c in ids if refresh or c not in cache["cves"]]
    print(f"{len(ids)} CVE nodes in ARG; {len(todo)} to fetch from NVD "
          f"(~{len(todo) * SLEEP_SECONDS / 60:.1f} min at {SLEEP_SECONDS}s spacing)")

    for i, cve_id in enumerate(todo, 1):
        rec = fetch_one(cve_id)
        if rec:
            cache["cves"][cve_id] = rec
            cache["fetched"][cve_id] = time.strftime("%Y-%m-%dT%H:%M:%S")
            print(f"  [{i}/{len(todo)}] {cve_id}: cwe={rec['cwe'] or '-'} "
                  f"cvss={rec['cvss_score']} (v{rec['cvss_version']}) sev={rec['severity'] or '-'}")
        if i < len(todo):
            time.sleep(SLEEP_SECONDS)

    got = [c for c in ids if c in cache["cves"]]
    with_cwe = [c for c in got if cache["cves"][c]["cwe"]]
    with_sev = [c for c in got if cache["cves"][c]["severity"]]
    cache["coverage"] = {
        "arg_cve_nodes": len(ids),
        "resolved_in_nvd": len(got),
        "with_cwe": len(with_cwe),
        "with_severity": len(with_sev),
        "missing": [c for c in ids if c not in cache["cves"]],
        "missing_cwe": [c for c in got if not cache["cves"][c]["cwe"]],
    }
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"\nResolved {len(got)}/{len(ids)} in NVD; "
          f"{len(with_cwe)} carry a CWE, {len(with_sev)} carry a severity string")
    if cache["coverage"]["missing"]:
        print(f"  not in NVD: {cache['coverage']['missing']}")
    if cache["coverage"]["missing_cwe"]:
        print(f"  no CWE assigned by NVD: {cache['coverage']['missing_cwe']}")
    print(f"  -> {CACHE}")


if __name__ == "__main__":
    main()
