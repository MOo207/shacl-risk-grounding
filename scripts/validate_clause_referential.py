#!/usr/bin/env python3
"""Referential clause validation, applied retroactively to stored LLM outputs.

The deployed shapes validated clause references syntactically (pattern
^SS[56]\\.[0-9]+$). After clause-as-resource remodelling
(ontology/ttl/nfcrm-clauses.ttl) validation is referential (sh:in over the
26 real NFCRM-1:2025 clauses; source of truth
data/external/nfcrm_clause_mapping.json). This script re-checks every clause
reference in the stored Experiment 2 and Experiment 3 artefacts against the
enumeration, converting the paper's "no accepted output was observed to cite
an out-of-enumeration clause" claim from an observation into a measured
result. Historical result files are read-only inputs; this script writes
only results/clause_referential_validation.json.
"""
import json
import os
import re
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 26 real clauses: SS5.1-SS5.6, SS6.1-SS6.20 (nfcrm_clause_mapping.json).
VALID = {f"SS5.{i}" for i in range(1, 7)} | {f"SS6.{i}" for i in range(1, 21)}
PATTERN = re.compile(r"^SS[56]\.\d+$")


def extract_refs_exp2(parsed):
    """Experiment 2 (risk generation): nfcrm_clause + recommended_controls."""
    refs = []
    c = parsed.get("nfcrm_clause")
    if c is not None:
        refs.append(str(c))
    rc = parsed.get("recommended_controls", [])
    if isinstance(rc, list):
        refs.extend(str(x) for x in rc)
    return refs


def extract_refs_exp3(parsed):
    """Experiment 3 (control recommendation): recommendations[].control_clause_id."""
    refs = []
    recs = parsed.get("recommendations", [])
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict) and "control_clause_id" in r:
                refs.append(str(r["control_clause_id"]))
    return refs


def check_file(path, extractor):
    with open(path) as f:
        data = json.load(f)
    stats = {
        "n_artifacts": 0,
        "n_artifacts_with_parse": 0,
        "n_refs": 0,
        "n_syntactic_pass": 0,
        "n_out_of_enumeration": 0,  # syntactically valid but nonexistent clause
        "n_syntactic_fail": 0,
        "out_of_enumeration_ids": [],
        "syntactic_fail_ids": [],
    }
    for case in data["results"]:
        stats["n_artifacts"] += 1
        parsed = case.get("parsed_response")
        if not isinstance(parsed, dict):
            continue
        stats["n_artifacts_with_parse"] += 1
        for ref in extractor(parsed):
            stats["n_refs"] += 1
            if PATTERN.match(ref):
                stats["n_syntactic_pass"] += 1
                if ref not in VALID:
                    stats["n_out_of_enumeration"] += 1
                    stats["out_of_enumeration_ids"].append(ref)
            else:
                stats["n_syntactic_fail"] += 1
                stats["syntactic_fail_ids"].append(ref)
    return stats


def main():
    files = [
        ("exp2_risk_generation", "results/llm_risk_generation.json", extract_refs_exp2),
        ("exp3_control_recommendation", "results/llm_control_recommendation.json", extract_refs_exp3),
    ]
    per_file = {}
    for name, rel, extractor in files:
        path = os.path.join(PROJECT_ROOT, rel)
        per_file[name] = {"file": rel, **check_file(path, extractor)}

    totals = {
        k: sum(v[k] for v in per_file.values())
        for k in ["n_artifacts", "n_artifacts_with_parse", "n_refs",
                  "n_syntactic_pass", "n_out_of_enumeration", "n_syntactic_fail"]
    }
    out = {
        "description": __doc__.strip().split("\n")[0],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_clause_count": len(VALID),
        "totals": totals,
        "per_file": per_file,
    }
    out_path = os.path.join(PROJECT_ROOT, "results", "clause_referential_validation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Enumeration size: {len(VALID)}")
    for name, s in per_file.items():
        print(f"{name}: artifacts={s['n_artifacts']} parsed={s['n_artifacts_with_parse']} "
              f"refs={s['n_refs']} syntactic_pass={s['n_syntactic_pass']} "
              f"OUT_OF_ENUM={s['n_out_of_enumeration']} syntactic_fail={s['n_syntactic_fail']}")
        if s["out_of_enumeration_ids"]:
            print("  out-of-enumeration:", sorted(set(s["out_of_enumeration_ids"])))
        if s["syntactic_fail_ids"]:
            print("  syntactic-fail:", sorted(set(s["syntactic_fail_ids"])))
    print(f"TOTAL refs={totals['n_refs']} out_of_enumeration={totals['n_out_of_enumeration']}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
