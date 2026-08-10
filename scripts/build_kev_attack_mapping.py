"""Build a frozen KEV-to-attack-class mapping.

Selects ≥12 CVEs per attack class from data/external/cisa_kev.json.
Each entry includes a one-line rationale.

Output: tests/llm_experiment/kev_attack_mapping.json

Field names confirmed from cisa_kev.json:
  cveID, vulnerabilityName, cwes (list of strings), shortDescription,
  vendorProject, product
"""
from __future__ import annotations

import json
from pathlib import Path

KEV_PATH = "data/external/cisa_kev.json"
OUT_PATH = "tests/llm_experiment/kev_attack_mapping.json"

# ---------------------------------------------------------------------------
# Attack-class match patterns
#
# Each class has:
#   cwes        – set of CWE IDs (match on cwes field, a list of strings)
#   name_kw     – substrings to match in vulnerabilityName (lowercase)
#   desc_kw     – substrings to match in shortDescription (lowercase)
#   vendor_kw   – vendorProject substrings that, combined with DoS signal,
#                 make a match (used only for DoS/DDoS classes)
#   exclude_kw  – any matching name substring disqualifies the entry
#
# Priority: BruteForce → DDoS → DoS → Infiltration (first match wins)
# so more-specific classes are evaluated first to prevent overlap.
# ---------------------------------------------------------------------------

PATTERNS: dict[str, dict] = {
    "BruteForce": {
        "cwes": {"CWE-287", "CWE-307", "CWE-521", "CWE-798", "CWE-922", "CWE-308"},
        "name_kw": [
            "authentication bypass", "credential", "password", "default credentials",
            "weak credentials", "brute", "hard-coded", "hardcoded",
            "improper authentication",
        ],
        "desc_kw": [
            "brute force", "credential stuffing", "default password",
            "hard-coded credential", "hardcoded credential",
        ],
        "vendor_kw": [],
        "exclude_kw": [],
    },
    # DDoS evaluated BEFORE DoS to capture amplification/reflection first
    "DDoS": {
        "cwes": {"CWE-940"},          # SLP/protocol-source confusion
        "name_kw": [
            "amplification", "reflection", "ddos", "rapid reset",
            "http/2 rapid reset", "distributed denial",
        ],
        "desc_kw": [
            "amplification", "distributed denial", "ddos",
            "rapid reset", "reflected amplification",
            "flood", "reflection attack",
        ],
        # Network infrastructure vendors where protocol-level DoS = DDoS-capable
        "vendor_kw": ["cisco", "juniper", "palo alto", "f5", "citrix", "barracuda",
                      "sonicwall", "fortinet", "netgear", "zyxel", "aruba"],
        # For vendor_kw matches we also require one of these in name/desc
        "vendor_dos_kw": [
            "denial of service", "denial-of-service", "dos vulnerability",
            "bgp", "ospf", "snmp", "routing protocol", "load balan",
        ],
        "exclude_kw": [],
    },
    "DoS": {
        "cwes": {"CWE-400", "CWE-770", "CWE-835", "CWE-674", "CWE-405",
                 "CWE-416", "CWE-476", "CWE-617", "CWE-772", "CWE-835",
                 "CWE-399"},
        "name_kw": [
            "denial of service", "denial-of-service",
            "resource exhaustion", "infinite loop",
            "uncontrolled resource", "rate limit",
            "use-after-free", "use after free",
            "null pointer", "null dereference",
        ],
        "desc_kw": [
            "denial of service", "denial-of-service",
            "resource exhaustion", "cause a crash", "system crash",
            "service disruption",
        ],
        "vendor_kw": [],
        "exclude_kw": ["distributed denial", "amplification", "reflection"],
    },
    "Infiltration": {
        "cwes": {
            "CWE-78", "CWE-94", "CWE-269", "CWE-502", "CWE-22", "CWE-918",
            "CWE-119", "CWE-787", "CWE-434", "CWE-89", "CWE-352",
            "CWE-611", "CWE-863", "CWE-862",
        },
        "name_kw": [
            "remote code execution", "rce", "privilege escalation",
            "command injection", "deserialization", "path traversal",
            "ssrf", "buffer overflow", "code execution", "arbitrary code",
            "improper access control", "unrestricted upload", "sql injection",
            "xml external entity", "xxe",
        ],
        "desc_kw": [
            "remote code execution", "execute arbitrary code",
            "privilege escalation", "command injection",
        ],
        "vendor_kw": [],
        "exclude_kw": [],
    },
}

# Ordered so more-specific classes are tested first
CLASS_ORDER = ["BruteForce", "DDoS", "DoS", "Infiltration"]


def _matches_ddos(entry: dict, pattern: dict) -> bool:
    cwes = set(entry.get("cwes") or [])
    name = (entry.get("vulnerabilityName") or "").lower()
    desc = (entry.get("shortDescription") or "").lower()
    vp = (entry.get("vendorProject") or "").lower()

    # CWE match
    if cwes & pattern["cwes"]:
        return True
    # Name keyword match
    if any(kw in name for kw in pattern["name_kw"]):
        return True
    # Description keyword match
    if any(kw in desc for kw in pattern["desc_kw"]):
        return True
    # Infrastructure vendor + DoS keyword = DDoS-capable
    if any(v in vp for v in pattern["vendor_kw"]):
        dos_kw = pattern.get("vendor_dos_kw", [])
        if any(kw in name or kw in desc for kw in dos_kw):
            return True
    return False


def _matches_generic(entry: dict, pattern: dict) -> bool:
    cwes = set(entry.get("cwes") or [])
    name = (entry.get("vulnerabilityName") or "").lower()
    desc = (entry.get("shortDescription") or "").lower()

    exclude = pattern.get("exclude_kw", [])
    if any(ex in name for ex in exclude) or any(ex in desc for ex in exclude):
        return False

    if cwes & pattern["cwes"]:
        return True
    if any(kw in name for kw in pattern["name_kw"]):
        return True
    if any(kw in desc for kw in pattern.get("desc_kw", [])):
        return True
    return False


def matches(cls: str, entry: dict, pattern: dict) -> bool:
    if cls == "DDoS":
        return _matches_ddos(entry, pattern)
    return _matches_generic(entry, pattern)


def main() -> None:
    kev = json.loads(Path(KEV_PATH).read_text())
    kev_entries = kev.get("vulnerabilities") if isinstance(kev, dict) else kev

    mapping: dict[str, list[dict]] = {cls: [] for cls in CLASS_ORDER}
    used: set[str] = set()

    # Two passes: first pass caps at 16 per class; second pass (if needed)
    # relaxes the cap to fill up to 12.
    for cls in CLASS_ORDER:
        pattern = PATTERNS[cls]
        for entry in kev_entries:
            cve_id = entry.get("cveID")
            if not cve_id or cve_id in used:
                continue
            if not matches(cls, entry, pattern):
                continue
            rationale = (
                f"CWEs={entry.get('cwes', [])}; "
                f"name={entry.get('vulnerabilityName', '')[:80]}"
            )
            mapping[cls].append({
                "cve_id": cve_id,
                "rationale": rationale,
                "vendor": entry.get("vendorProject"),
                "product": entry.get("product"),
            })
            used.add(cve_id)
            if len(mapping[cls]) >= 16:
                break

    for cls, entries_for_cls in mapping.items():
        if len(entries_for_cls) < 12:
            raise SystemExit(
                f"FAIL: only {len(entries_for_cls)} CVEs for {cls}; extend PATTERNS."
            )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(mapping, indent=2))
    print(
        "Wrote " + OUT_PATH + ": "
        + ", ".join(f"{cls}={len(es)}" for cls, es in mapping.items())
    )


if __name__ == "__main__":
    main()
