#!/usr/bin/env python3
"""Grounding-depth control: Exp3's control-recommendation task WITH Exp2's
rich 2-hop ARG context.

Pre-registered design (docs/superpowers/plans/2026-07-26-dke-review-remediation.md,
Task 3 Step 1, fixed before any LLM call): same task, same output schema, same
validators and scoring as Experiment 3; the only manipulation is the addition
of the serialized 2-hop ARG subgraph of the target asset (Experiment 2's
grounding context, Experiment 2's exact serialization) to the prompt. The 99
stored (asset_id, threat_class, cve_id) triples from
results/llm_control_recommendation.json are reused verbatim in stored order
(paired design), so per-case paired comparison against the stored sparse-
grounding outcomes is possible. Hypothesis (directional): SHACL conformance
rises above the sparse condition's 71.7%.

Historical scripts and result files are untouched; output goes to
results/grounding_depth_control.json.
"""
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.claude_cli_client import ClaudeCLIClient
# Validators imported UNCHANGED from Exp3 (identical scoring is the point).
from scripts.run_llm_control_recommendation import (
    load_arg, parse_json_response, validate_json_schema, validate_shacl,
    compute_precision,
)
# 2-hop subgraph extraction imported UNCHANGED from Exp2.
from scripts.run_llm_risk_generation import extract_2hop_subgraph

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stored_cases():
    """The 99 stored Exp3 cases: (asset_id, threat_class, cve_id, shacl_valid)."""
    path = os.path.join(PROJECT_ROOT, "results", "llm_control_recommendation.json")
    with open(path) as f:
        data = json.load(f)
    return [
        {
            "run": r["run"],
            "asset_id": r["asset_id"],
            "threat_class": r["threat_class"],
            "cve_id": r["cve_id"],
            "sparse_shacl_valid": r["shacl_valid"],
            "sparse_json_valid": r["json_schema_valid"],
        }
        for r in data["results"]
    ]


def load_template():
    path = os.path.join(PROJECT_ROOT, "scripts", "templates",
                        "control_recommendation_rich_prompt.txt")
    with open(path) as f:
        return f.read()


def fill_prompt(template, asset, threat_class, cve_id, subgraph):
    return (
        template
        .replace("{asset_id}", asset["id"])
        .replace("{asset_type}", asset.get("asset_type", "unknown"))
        .replace("{asset_os}", asset.get("os", "unknown"))
        .replace("{criticality}", asset.get("criticality", "medium"))
        .replace("{threat_class}", threat_class)
        .replace("{vulnerability_cve}", cve_id)
        .replace("{arg_context}", json.dumps(subgraph, indent=2))
    )


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar p-value (binomial on discordant pairs)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run(model, limit=None):
    print("=== Grounding-depth control: Exp3 task + Exp2 2-hop ARG context ===")
    print(f"Model: {model} | Started: {datetime.now().isoformat()}")

    arg = load_arg(os.path.join(PROJECT_ROOT, "results", "multisource_arg.json"))
    arg_node_ids = {n["id"] for n in arg["nodes"]}
    assets_by_id = {n["id"]: n for n in arg["nodes"] if n["type"] == "Asset"}
    template = load_template()
    cases = load_stored_cases()
    if limit:
        cases = cases[:limit]

    client = ClaudeCLIClient(model=model, temperature=0.3, max_tokens=2048)
    system_msg = (
        "You are a cybersecurity GRC analyst specializing in control recommendation "
        "per NFCRM-1:2025. Respond with ONLY valid JSON."
    )

    # Per-case checkpoint (JSONL) so a killed run resumes without re-spending calls.
    ckpt_path = os.path.join(PROJECT_ROOT, "results",
                             "grounding_depth_control_cases.jsonl")
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["case_index"]] = rec
        print(f"Resuming: {len(done)} cases already checkpointed")

    results = []
    for i, case in enumerate(cases):
        if i in done:
            results.append(done[i])
            continue
        asset = assets_by_id[case["asset_id"]]
        subgraph = extract_2hop_subgraph(arg, case["asset_id"])
        user_msg = fill_prompt(template, asset, case["threat_class"],
                               case["cve_id"], subgraph)
        print(f"  [{i+1}/{len(cases)}] {case['asset_id']} | {case['threat_class']}"
              f" | {case['cve_id']}...", end=" ", flush=True)

        t0 = time.time()
        raw = client.generate(system_msg, user_msg)
        latency = time.time() - t0

        parsed = parse_json_response(raw)
        parseable = parsed is not None
        json_valid, json_violations = (False, [])
        shacl_valid, shacl_violations = (False, [])
        precision = 0.0
        if parseable:
            json_valid, json_violations = validate_json_schema(parsed)
            if json_valid:
                shacl_valid, shacl_violations = validate_shacl(parsed, arg_node_ids)
                precision = compute_precision(parsed)

        status = ("SHACL_OK" if shacl_valid else
                  ("JSON_OK" if json_valid else ("PARSED" if parseable else "FAIL")))
        print(f"{status} prec={precision}% ({latency:.1f}s)")

        rec = {
            "case_index": i,
            "sparse_run": case["run"],
            "asset_id": case["asset_id"],
            "threat_class": case["threat_class"],
            "cve_id": case["cve_id"],
            "parseable": parseable,
            "json_schema_valid": json_valid,
            "shacl_valid": shacl_valid,
            "precision_pct": precision,
            "violations": json_violations + shacl_violations,
            "sparse_shacl_valid": case["sparse_shacl_valid"],
            "latency_seconds": round(latency, 2),
            "raw_response": raw[:500] if raw else None,
            "parsed_response": parsed,
        }
        results.append(rec)
        with open(ckpt_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    n = len(results)
    n_shacl = sum(r["shacl_valid"] for r in results)
    n_json = sum(r["json_schema_valid"] for r in results)
    n_parse = sum(r["parseable"] for r in results)
    # Paired discordance vs stored sparse condition
    b = sum(1 for r in results if r["shacl_valid"] and not r["sparse_shacl_valid"])
    c = sum(1 for r in results if not r["shacl_valid"] and r["sparse_shacl_valid"])
    p = mcnemar_exact_two_sided(b, c)
    # threat_class error class (the 28/99 error class in sparse Exp3)
    tc_violations = sum(
        1 for r in results
        if any("threat_class" in v for v in r["violations"])
    )
    v_counts = defaultdict(int)
    for r in results:
        for v in r["violations"]:
            import re as _re
            v_counts[_re.sub(r"'[^']*'", "'...'", v)] += 1

    summary = {
        "model": model,
        "design": "pre-registered paired grounding-depth control; same task/validators/"
                  "scoring as Exp3; manipulation = 2-hop ARG subgraph added to prompt; "
                  "99 stored Exp3 (asset, threat, cve) triples reused verbatim",
        "n": n,
        "parseable_pct": round(100 * n_parse / n, 1),
        "json_schema_valid_pct": round(100 * n_json / n, 1),
        "shacl_valid_pct": round(100 * n_shacl / n, 1),
        "sparse_shacl_valid_pct": round(
            100 * sum(r["sparse_shacl_valid"] for r in results) / n, 1),
        "paired_mcnemar": {"b_rich_only_valid": b, "c_sparse_only_valid": c,
                           "p_two_sided": p},
        "threat_class_violation_count": tc_violations,
        "violation_counts": dict(sorted(v_counts.items(), key=lambda x: -x[1])),
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in results) / n, 2),
    }
    out = {
        "experiment": "Grounding-depth control (Exp3 task, Exp2 context)",
        "pre_registered": True,
        "pre_registration": "docs/superpowers/plans/2026-07-26-dke-review-remediation.md Task 3 Step 1",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": results,
    }
    out_path = os.path.join(PROJECT_ROOT, "results", "grounding_depth_control.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSHACL rich: {summary['shacl_valid_pct']}% vs sparse "
          f"{summary['sparse_shacl_valid_pct']}% | McNemar b={b} c={c} p={p:.4g}")
    print(f"Saved: {out_path}")
    return out


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "haiku"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(model, limit)
