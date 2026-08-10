"""Build the v3 test set for the LLM Risk-Artifact Experiment.

Design changes vs v2:
  - Set B is ATTACK-ONLY: Benign excluded from masked subset
  - Domain justification: Benign events never reach the risk-assessment
    workflow (no alert → no artifact). Set B tests LLM reasoning under
    partial threat intelligence for *confirmed* security events only.
  - Set A: all 5 classes (attack class visible) — 10 per class = 50 cases
  - Set B: 4 attack classes (BruteForce, DoS, DDoS, Infiltration), masked
           — 10 per class = 40 cases
  - Total: 90 cases
  - seed = 44  (new independent sample, pre-registered)
  - Outputs: test_set_v3.json, masking_schedule_v3.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.nfcrm.risk_score import compute_risk_score as _compute_risk_score

PREDICTIONS = "results/llm_experiment/rf_test_predictions.jsonl"
KEV_MAPPING = "tests/llm_experiment/kev_attack_mapping.json"
KEV_RAW = "data/external/cisa_kev.json"
NVD_ENRICHMENT = "data/external/nvd_enrichment.json"
CMDB = "data/external/cmdb_assets.json"
ARG = "results/multisource_arg.json"
OUT_TEST_SET = "results/llm_experiment/test_set_v3.json"
OUT_MASKING = "tests/llm_experiment/masking_schedule_v3.json"

ALL_CLASSES = ["Benign", "BruteForce", "DoS", "DDoS", "Infiltration"]
ATTACK_CLASSES = ["BruteForce", "DoS", "DDoS", "Infiltration"]   # Set B only
PER_CLASS_A = 10    # Set A — all 5 classes → 50 cases
PER_CLASS_B = 10    # Set B — 4 attack classes → 40 cases
PER_CLASS = PER_CLASS_A + PER_CLASS_B  # total per attack class from pool

MASKABLE_FIELDS = [
    "asset_criticality", "control_coverage", "cvss_components",
    "arg_neighborhood", "asset_type",
]

RISK_LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
_CRIT_INT: dict[str, int] = {"LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}


def load_predictions(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def stratified_sample(predictions: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Return dict[class -> shuffled pool], PER_CLASS per attack class, PER_CLASS_A for Benign."""
    by_class: dict[str, list[dict]] = {c: [] for c in ALL_CLASSES}
    for p in predictions:
        cls = p["true_label"]
        if cls in by_class:
            by_class[cls].append(p)
    sampled: dict[str, list[dict]] = {}
    for cls in ALL_CLASSES:
        pool = by_class[cls]
        need = PER_CLASS if cls in ATTACK_CLASSES else PER_CLASS_A
        if len(pool) < need:
            raise SystemExit(f"Only {len(pool)} cases for class '{cls}', need {need}")
        rng.shuffle(pool)
        sampled[cls] = pool[:need]
    return sampled


def load_kev_index() -> dict[str, dict]:
    raw = json.loads(Path(KEV_RAW).read_text())
    items = raw.get("vulnerabilities") if isinstance(raw, dict) else raw
    return {item.get("cveID", ""): item for item in items if item.get("cveID")}


def load_nvd_index() -> dict[str, dict]:
    raw = json.loads(Path(NVD_ENRICHMENT).read_text())
    out: dict[str, dict] = {}
    for _cpe, recs in raw.get("cpe_to_cves", {}).items():
        for rec in recs:
            out[rec["cve_id"]] = rec
    return out


def load_cmdb() -> list[dict]:
    raw = json.loads(Path(CMDB).read_text())
    if isinstance(raw, dict) and "assets" in raw:
        return raw["assets"]
    return raw if isinstance(raw, list) else list(raw.values())


def load_arg_summary() -> dict:
    arg = json.loads(Path(ARG).read_text())
    nodes = arg.get("nodes", []) if isinstance(arg, dict) else []
    cve_nodes = [n for n in nodes if n.get("type") == "CVE"][:5]
    return {
        "related_cves": [
            {"id": n.get("id"), "cvss_v3": n.get("cvss_v3"), "severity": n.get("severity")}
            for n in cve_nodes
        ],
    }


def compute_ground_truth_risk(attack_class: str, criticality: str = "MEDIUM") -> str:
    if attack_class == "Benign":
        return "Very Low"
    c_val = _CRIT_INT.get((criticality or "MEDIUM").upper(), 3)
    result = _compute_risk_score(attack_class, c_override=c_val,
                                 i_override=c_val, a_override=c_val)
    if result.level_en in RISK_LEVELS:
        return result.level_en
    return "Medium"


def build_masking_schedule(cases: list[dict], rng: random.Random) -> dict[str, list[str]]:
    """Set B = attack classes only. Each Set B case gets true_attack_class masked
    plus optionally one extra field."""
    schedule: dict[str, list[str]] = {}
    by_class: dict[str, list[dict]] = {}
    for c in cases:
        by_class.setdefault(c["true_attack_class"], []).append(c)

    # Benign — all in Set A (no masking)
    for c in by_class.get("Benign", []):
        schedule[c["case_id"]] = []

    # Attack classes — first PER_CLASS_B go to Set B, rest to Set A
    for cls in ATTACK_CLASSES:
        items = by_class.get(cls, [])
        rng.shuffle(items)
        for c in items[:PER_CLASS_B]:
            extra = rng.sample(MASKABLE_FIELDS, rng.randint(0, 1))
            schedule[c["case_id"]] = ["true_attack_class"] + extra
        for c in items[PER_CLASS_B:]:
            schedule[c["case_id"]] = []

    return schedule


def apply_masking(cases: list[dict], schedule: dict[str, list[str]]) -> None:
    for c in cases:
        c["masked_fields"] = schedule.get(c["case_id"], [])
        c["subset"] = "B" if c["masked_fields"] else "A"


def main() -> None:
    rng = random.Random(44)

    preds = load_predictions(PREDICTIONS)
    by_class = stratified_sample(preds, rng)

    cases: list[dict] = []
    idx = 0
    for cls in ALL_CLASSES:
        for p in by_class[cls]:
            cases.append({
                "case_id": f"v3_case_{idx:03d}",
                "feature_row_index": p["feature_row_index"],
                "true_attack_class": p["true_label"],
                "rf_predicted_class": p["predicted_label"],
            })
            idx += 1

    cmdb = load_cmdb()
    rng2 = random.Random(44)
    rng2.shuffle(cmdb)
    for i, c in enumerate(cases):
        a = cmdb[i % len(cmdb)]
        c["asset"] = {
            "id": a.get("asset_id") or f"asset_{i}",
            "type": a.get("asset_type", "server"),
            "criticality": a.get("criticality", "MEDIUM"),
            "ip_segment": ".".join(a.get("ip", "0.0.0.0").split(".")[:3]) + ".0/24",
        }

    for c in cases:
        c["controls"] = []

    arg_summary = load_arg_summary()
    for c in cases:
        c["arg_neighborhood"] = arg_summary

    mapping = json.loads(Path(KEV_MAPPING).read_text())
    kev_idx = load_kev_index()
    nvd_idx = load_nvd_index()

    pools: dict[str, list[dict]] = {cls: list(entries) for cls, entries in mapping.items()}
    for cls in pools:
        random.Random(44).shuffle(pools[cls])
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
        desc = kev_entry.get("shortDescription") or nvd_entry.get("description", "")
        c["paired_cve"] = {
            "cve_id": cve_id,
            "shortDescription": desc,
            "requiredAction": kev_entry.get("requiredAction", ""),
            "vendor": kev_entry.get("vendorProject", ""),
            "product": kev_entry.get("product", ""),
            "cvss_v3": nvd_entry.get("cvss_v3"),
            "severity": nvd_entry.get("severity"),
            "cwe": nvd_entry.get("cwe"),
            "in_cisa_kev": nvd_entry.get("in_cisa_kev"),
        }

    for c in cases:
        crit = c.get("asset", {}).get("criticality", "MEDIUM")
        c["ground_truth_risk_level"] = compute_ground_truth_risk(c["true_attack_class"], crit)

    schedule = build_masking_schedule(cases, rng)
    apply_masking(cases, schedule)

    n_a = sum(1 for c in cases if c["subset"] == "A")
    n_b = sum(1 for c in cases if c["subset"] == "B")

    Path(OUT_MASKING).write_text(json.dumps(schedule, indent=2))
    Path(OUT_TEST_SET).write_text(json.dumps({
        "seed": 44,
        "design": "attack-only-set-b",
        "n_cases": len(cases),
        "n_set_a": n_a,
        "n_set_b": n_b,
        "classes_set_a": ALL_CLASSES,
        "classes_set_b": ATTACK_CLASSES,
        "cases": cases,
    }, indent=2))

    print(f"Wrote {OUT_TEST_SET}: {len(cases)} cases (Set A={n_a}, Set B={n_b})")
    class_dist = {}
    for c in cases:
        key = f"{c['true_attack_class']}/{c['subset']}"
        class_dist[key] = class_dist.get(key, 0) + 1
    for k, v in sorted(class_dist.items()):
        print(f"  {k}: {v}")
    print(f"Wrote {OUT_MASKING}")


if __name__ == "__main__":
    main()
