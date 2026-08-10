"""Build the 80-case frozen test set for the LLM Risk-Artifact Experiment.

Adaptations from API inspection (2026-05-10):
- compute_risk_score(attack_type) returns RiskScore with .level_en (str).
  'Benign' returns "N/A (non-attack)" which is not in RISK_LEVELS, so
  the fallback table maps Benign -> "Very Low".
- CMDB assets keyed under d['assets'] (list), with fields: asset_id, asset_type,
  criticality, ip (no ip_segment/subnet; use ip field instead).
- NVD enrichment CVE fields: cve_id, cvss_v3, severity, description, cwe,
  published, in_cisa_kev.  No attack_vector/attack_complexity decomposition.
- CISA KEV uses shortDescription (confirmed present in all 64 KEV mapping CVEs).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path so pipeline.* is importable when
# this script is invoked directly as 'python scripts/build_llm_artifact_test_set.py'.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.nfcrm.risk_score import compute_risk_score as _compute_risk_score

PREDICTIONS = "results/llm_experiment/rf_test_predictions.jsonl"
KEV_MAPPING = "tests/llm_experiment/kev_attack_mapping.json"
KEV_RAW = "data/external/cisa_kev.json"
NVD_ENRICHMENT = "data/external/nvd_enrichment.json"
CMDB = "data/external/cmdb_assets.json"
ARG = "results/multisource_arg.json"
OUT_TEST_SET = "results/llm_experiment/test_set.json"
OUT_MASKING = "tests/llm_experiment/masking_schedule.json"

CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Infiltration"]
PER_CLASS = 16
SET_B_PER_CLASS = 6  # 6 × 5 = 30 ambiguous / masked cases
MASKABLE_FIELDS = [
    "asset_criticality", "control_coverage", "cvss_components",
    "arg_neighborhood", "asset_type",
]

RISK_LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]

# Criticality integer used by compute_risk_score (c_override, i_override, a_override)
_CRIT_INT: dict[str, int] = {"LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def stratified_sample(predictions: list[dict], rng: random.Random) -> list[dict]:
    by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for p in predictions:
        cls = p["true_label"]
        if cls in by_class:
            by_class[cls].append(p)
    sampled: list[dict] = []
    for cls in CLASSES:
        pool = by_class[cls]
        if len(pool) < PER_CLASS:
            raise SystemExit(f"Only {len(pool)} cases for class '{cls}', need {PER_CLASS}")
        rng.shuffle(pool)
        sampled.extend(pool[:PER_CLASS])
    return sampled


def load_kev_index() -> dict[str, dict]:
    """Return dict keyed by CVE-ID from CISA KEV catalogue."""
    raw = json.loads(Path(KEV_RAW).read_text())
    items = raw.get("vulnerabilities") if isinstance(raw, dict) else raw
    return {item.get("cveID", ""): item for item in items if item.get("cveID")}


def load_nvd_index() -> dict[str, dict]:
    """Return dict keyed by CVE-ID from NVD enrichment file."""
    raw = json.loads(Path(NVD_ENRICHMENT).read_text())
    out: dict[str, dict] = {}
    for _cpe, recs in raw.get("cpe_to_cves", {}).items():
        for rec in recs:
            out[rec["cve_id"]] = rec
    return out


def load_cmdb() -> list[dict]:
    """Return list of asset dicts from CMDB."""
    raw = json.loads(Path(CMDB).read_text())
    # Top-level dict with 'assets' list
    if isinstance(raw, dict) and "assets" in raw:
        return raw["assets"]
    return raw if isinstance(raw, list) else list(raw.values())


def load_arg_summary() -> dict:
    """Extract a small neighbourhood summary from the ARG."""
    arg = json.loads(Path(ARG).read_text())
    nodes = arg.get("nodes", []) if isinstance(arg, dict) else []
    cve_nodes = [n for n in nodes if n.get("type") == "CVE"][:5]
    # ARG node fields: id, type, cvss_v3, severity, description
    return {
        "related_cves": [
            {"id": n.get("id"), "cvss_v3": n.get("cvss_v3"), "severity": n.get("severity")}
            for n in cve_nodes
        ],
    }


# ---------------------------------------------------------------------------
# Risk level computation
# ---------------------------------------------------------------------------

def compute_ground_truth_risk(attack_class: str, criticality: str = "MEDIUM") -> str:
    """Criticality-adjusted NFCRM §6.9 ground truth.

    Uses the same compute_risk_score path as the Rule arm so the two are
    aligned. Falls back to MEDIUM criticality when criticality is unknown.
    'Benign' always returns 'Very Low' (compute_risk_score returns 'N/A').
    """
    if attack_class == "Benign":
        return "Very Low"
    c_val = _CRIT_INT.get((criticality or "MEDIUM").upper(), 3)
    result = _compute_risk_score(attack_class, c_override=c_val,
                                 i_override=c_val, a_override=c_val)
    if result.level_en in RISK_LEVELS:
        return result.level_en
    return "Medium"


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def build_masking_schedule(cases: list[dict], rng: random.Random) -> dict[str, list[str]]:
    """Set-B cases always hide true_attack_class plus 0-1 additional fields."""
    schedule: dict[str, list[str]] = {}
    by_class: dict[str, list[dict]] = {}
    for c in cases:
        by_class.setdefault(c["true_attack_class"], []).append(c)
    for cls in CLASSES:
        items = by_class.get(cls, [])
        rng.shuffle(items)
        for c in items[:SET_B_PER_CLASS]:
            extra = rng.sample(MASKABLE_FIELDS, rng.randint(0, 1))
            schedule[c["case_id"]] = ["true_attack_class"] + extra
        for c in items[SET_B_PER_CLASS:]:
            schedule[c["case_id"]] = []
    return schedule


def apply_masking(cases: list[dict], schedule: dict[str, list[str]]) -> None:
    for c in cases:
        c["masked_fields"] = schedule.get(c["case_id"], [])
        c["subset"] = "B" if c["masked_fields"] else "A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(42)

    # --- 1. Sample predictions ---
    preds = load_predictions(PREDICTIONS)
    sampled = stratified_sample(preds, rng)

    cases: list[dict] = []
    for i, p in enumerate(sampled):
        cases.append({
            "case_id": f"case_{i:03d}",
            "feature_row_index": p["feature_row_index"],
            "true_attack_class": p["true_label"],
            "rf_predicted_class": p["predicted_label"],
        })

    # --- 2. Enrich with CMDB asset ---
    cmdb = load_cmdb()
    rng2 = random.Random(42)
    rng2.shuffle(cmdb)
    for i, c in enumerate(cases):
        a = cmdb[i % len(cmdb)]
        c["asset"] = {
            "id": a.get("asset_id") or f"asset_{i}",
            "type": a.get("asset_type", "server"),
            "criticality": a.get("criticality", "MEDIUM"),
            # CMDB uses 'ip' field; extract /24 subnet as ip_segment
            "ip_segment": ".".join(a.get("ip", "0.0.0.0").split(".")[:3]) + ".0/24",
        }

    # --- 3. Controls (empty per spec: §6.9 control coverage experiment) ---
    for c in cases:
        c["controls"] = []

    # --- 4. ARG neighbourhood ---
    arg_summary = load_arg_summary()
    for c in cases:
        c["arg_neighborhood"] = arg_summary

    # --- 5. KEV CVE pairing (non-Benign only) ---
    mapping = json.loads(Path(KEV_MAPPING).read_text())
    kev_idx = load_kev_index()
    nvd_idx = load_nvd_index()

    pools: dict[str, list[dict]] = {cls: list(entries) for cls, entries in mapping.items()}
    for cls in pools:
        random.Random(42).shuffle(pools[cls])
    cursors: dict[str, int] = {cls: 0 for cls in pools}

    for c in cases:
        cls = c["true_attack_class"]
        if cls == "Benign":
            c["paired_cve"] = None
            continue
        pool = pools.get(cls, [])
        if not pool:
            c["paired_cve"] = None
            continue
        cve_id = pool[cursors[cls] % len(pool)]["cve_id"]
        cursors[cls] += 1

        kev_entry = kev_idx.get(cve_id, {})
        nvd_entry = nvd_idx.get(cve_id, {})

        # shortDescription is the primary field in CISA KEV; fall back to
        # NVD description if for any reason the KEV entry is missing.
        desc = (kev_entry.get("shortDescription")
                or nvd_entry.get("description", ""))

        c["paired_cve"] = {
            "cve_id": cve_id,
            "shortDescription": desc,
            "requiredAction": kev_entry.get("requiredAction", ""),
            "vendor": kev_entry.get("vendorProject", ""),
            "product": kev_entry.get("product", ""),
            # NVD enrichment fields (no attack_vector decomposition in this dataset)
            "cvss_v3": nvd_entry.get("cvss_v3"),
            "severity": nvd_entry.get("severity"),
            "cwe": nvd_entry.get("cwe"),
            "in_cisa_kev": nvd_entry.get("in_cisa_kev"),
        }

    # --- 6. Ground truth risk level ---
    for c in cases:
        # Always use true (unmasked) criticality for GT, regardless of Set B masking.
        crit = c.get("asset", {}).get("criticality", "MEDIUM")
        c["ground_truth_risk_level"] = compute_ground_truth_risk(c["true_attack_class"], crit)

    # --- 7. Masking schedule + subset assignment ---
    schedule = build_masking_schedule(cases, rng)
    apply_masking(cases, schedule)

    # --- 8. Write outputs ---
    n_a = sum(1 for c in cases if c["subset"] == "A")
    n_b = sum(1 for c in cases if c["subset"] == "B")

    Path(OUT_MASKING).write_text(json.dumps(schedule, indent=2))
    Path(OUT_TEST_SET).write_text(json.dumps({
        "seed": 42,
        "n_cases": len(cases),
        "n_set_a": n_a,
        "n_set_b": n_b,
        "cases": cases,
    }, indent=2))

    print(f"Wrote {OUT_TEST_SET}: {len(cases)} cases (Set A={n_a}, Set B={n_b})")
    print(f"Wrote {OUT_MASKING}")


if __name__ == "__main__":
    main()
