"""Download external data sources for multi-source ARG integration.

Sources:
  - CISA KEV: Known Exploited Vulnerabilities catalog
  - MITRE ATT&CK: Enterprise ATT&CK STIX 2.1 JSON
"""
import urllib.request
import json
import os
import ssl
import sys

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
os.makedirs(EXTERNAL_DIR, exist_ok=True)

SOURCES = {
    "cisa_kev.json": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "enterprise-attack.json": "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json",
}


def download_all():
    # Allow unverified SSL for environments without certs
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for filename, url in SOURCES.items():
        dest = os.path.join(EXTERNAL_DIR, filename)
        if os.path.exists(dest):
            size_kb = os.path.getsize(dest) // 1024
            print(f"[SKIP] {filename} already exists ({size_kb} KB)")
            continue
        print(f"[DOWNLOAD] {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            # Retry with no-verify context
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx) as resp:
                with open(dest, "wb") as f:
                    f.write(resp.read())
        # Validate JSON
        with open(dest, encoding="utf-8") as f:
            data = json.load(f)
        size_kb = os.path.getsize(dest) // 1024
        print(f"[OK] {filename}: {size_kb} KB")

    # Print summary
    print("\n--- Summary ---")
    for filename in SOURCES:
        dest = os.path.join(EXTERNAL_DIR, filename)
        if os.path.exists(dest):
            with open(dest, encoding="utf-8") as f:
                data = json.load(f)
            if filename == "cisa_kev.json":
                print(f"CISA KEV: {len(data.get('vulnerabilities', []))} CVEs")
            elif filename == "enterprise-attack.json":
                print(f"ATT&CK: {len(data.get('objects', []))} STIX objects")


if __name__ == "__main__":
    download_all()
