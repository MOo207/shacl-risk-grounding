#!/usr/bin/env python3
"""Experiment 3: LLM Control Recommendation with SHACL + JSON Schema validation."""
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.claude_cli_client import ClaudeCLIClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ATTACK_CLASSES = ["BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"]

VALID_CLAUSE_PATTERN = re.compile(r"^SS[56]\.\d+$")


def load_arg(path):
    with open(path) as f:
        return json.load(f)


def get_asset_nodes(arg):
    return [n for n in arg["nodes"] if n["type"] == "Asset"]


def get_cve_for_asset(arg, asset_id):
    """Find a CVE linked to this asset via EXPOSED_TO_CVE edge, else fallback."""
    for e in arg["edges"]:
        if e["source"] == asset_id and e["relation"] == "EXPOSED_TO_CVE":
            return e["target"]
    # Fallback: any CVE in the graph
    cves = [n["id"] for n in arg["nodes"] if n["type"] == "CVE"]
    if cves:
        return random.choice(cves)
    return "CVE-2024-XXXXX"


def load_prompt_template():
    path = os.path.join(PROJECT_ROOT, "scripts", "templates",
                        "control_recommendation_prompt.txt")
    with open(path) as f:
        return f.read()


def fill_prompt(template, asset, threat_class, cve_id):
    return (
        template
        .replace("{asset_id}", asset["id"])
        .replace("{asset_type}", asset.get("asset_type", "unknown"))
        .replace("{asset_os}", asset.get("os", "unknown"))
        .replace("{criticality}", asset.get("criticality", "medium"))
        .replace("{threat_class}", threat_class)
        .replace("{vulnerability_cve}", cve_id)
    )


def parse_json_response(raw):
    """Try to extract JSON from LLM response, handling markdown fences."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


REQUIRED_FIELDS = ["target_asset", "threat_class", "recommendations"]


def validate_json_schema(parsed):
    """Structural JSON Schema validation: check required fields exist."""
    if not isinstance(parsed, dict):
        return False, ["Response is not a JSON object"]
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        return False, [f"Missing required field: {f}" for f in missing]
    # recommendations must be a list
    recs = parsed.get("recommendations", [])
    if not isinstance(recs, list):
        return False, ["recommendations is not a list"]
    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            return False, [f"recommendations[{i}] is not an object"]
        for key in ["control_clause_id", "justification"]:
            if key not in rec:
                return False, [f"recommendations[{i}] missing '{key}'"]
    return True, []


def validate_shacl(parsed, arg_node_ids):
    """Programmatic SHACL-like constraint validation."""
    violations = []

    # 1. target_asset must exist in ARG nodes
    ta = parsed.get("target_asset", "")
    if ta not in arg_node_ids:
        violations.append(f"target_asset '{ta}' not found in ARG nodes")

    # 2. threat_class must be in valid set
    tc = parsed.get("threat_class", "")
    if tc not in ATTACK_CLASSES:
        violations.append(f"threat_class '{tc}' not in valid CIC-IDS2018 set")

    # 3. All control_clause_id must match SS[56].\d+ pattern
    recs = parsed.get("recommendations", [])
    if isinstance(recs, list):
        for rec in recs:
            cid = str(rec.get("control_clause_id", ""))
            if not VALID_CLAUSE_PATTERN.match(cid):
                violations.append(
                    f"control_clause_id '{cid}' does not match SS[56].d+ pattern"
                )
            # justification must be present and non-empty
            just = rec.get("justification", "")
            if not just or not str(just).strip():
                violations.append(f"Empty justification for clause '{cid}'")
    else:
        violations.append("recommendations is not a list")

    return len(violations) == 0, violations


def compute_precision(parsed):
    """Precision = % of recommended controls with valid NFCRM clause IDs."""
    recs = parsed.get("recommendations", [])
    if not isinstance(recs, list) or len(recs) == 0:
        return 0.0
    valid = sum(
        1 for r in recs
        if VALID_CLAUSE_PATTERN.match(str(r.get("control_clause_id", "")))
    )
    return round(100.0 * valid / len(recs), 1)


def run_experiment(model, n_prompts):
    print(f"=== Experiment 3: LLM Control Recommendation ===")
    print(f"Model: {model}, Prompts: {n_prompts}, Runs: 3")
    print(f"Started: {datetime.now().isoformat()}")

    # Load data
    arg_path = os.path.join(PROJECT_ROOT, "results", "multisource_arg.json")
    arg = load_arg(arg_path)
    assets = get_asset_nodes(arg)
    arg_node_ids = {n["id"] for n in arg["nodes"]}
    template = load_prompt_template()

    # Select N assets (cycle if n > len(assets))
    selected = []
    for i in range(n_prompts):
        selected.append(assets[i % len(assets)])

    # Init client
    client = ClaudeCLIClient(model=model, temperature=0.3, max_tokens=2048)
    system_msg = (
        "You are a cybersecurity GRC analyst specializing in control recommendation "
        "per NFCRM-1:2025. Respond with ONLY valid JSON."
    )

    all_results = []
    run_summaries = []

    for run_idx in range(3):
        print(f"\n--- Run {run_idx + 1}/3 ---")
        run_results = []
        run_precisions = []

        for i, asset in enumerate(selected):
            asset_id = asset["id"]
            threat_class = random.choice(ATTACK_CLASSES)
            cve_id = get_cve_for_asset(arg, asset_id)
            print(
                f"  [{i+1}/{n_prompts}] Asset: {asset_id} | Threat: {threat_class}"
                f" | CVE: {cve_id}...",
                end=" ",
                flush=True,
            )

            # Fill prompt
            user_msg = fill_prompt(template, asset, threat_class, cve_id)

            # Call LLM
            t0 = time.time()
            raw_response = client.generate(system_msg, user_msg)
            latency = time.time() - t0

            # Parse
            parsed = parse_json_response(raw_response)
            parseable = parsed is not None

            # Validate
            json_valid = False
            json_violations = []
            shacl_valid = False
            shacl_violations = []
            precision = 0.0

            if parseable:
                json_valid, json_violations = validate_json_schema(parsed)
                if json_valid:
                    shacl_valid, shacl_violations = validate_shacl(parsed, arg_node_ids)
                    precision = compute_precision(parsed)
                    run_precisions.append(precision)

            status = (
                "SHACL_OK"
                if shacl_valid
                else ("JSON_OK" if json_valid else ("PARSED" if parseable else "FAIL"))
            )
            print(f"{status} prec={precision}% ({latency:.1f}s)")

            result = {
                "run": run_idx + 1,
                "asset_id": asset_id,
                "threat_class": threat_class,
                "cve_id": cve_id,
                "parseable": parseable,
                "json_schema_valid": json_valid,
                "shacl_valid": shacl_valid,
                "precision_pct": precision,
                "violations": json_violations + shacl_violations,
                "latency_seconds": round(latency, 2),
                "raw_response": raw_response[:500] if raw_response else None,
                "parsed_response": parsed,
            }
            run_results.append(result)

        # Run summary
        n_total = len(run_results)
        n_parseable = sum(1 for r in run_results if r["parseable"])
        n_json_valid = sum(1 for r in run_results if r["json_schema_valid"])
        n_shacl_valid = sum(1 for r in run_results if r["shacl_valid"])
        avg_latency = sum(r["latency_seconds"] for r in run_results) / max(n_total, 1)
        avg_precision = (
            round(sum(run_precisions) / len(run_precisions), 1)
            if run_precisions
            else 0.0
        )

        run_summary = {
            "run": run_idx + 1,
            "total_prompts": n_total,
            "parseable": n_parseable,
            "parseable_pct": round(100 * n_parseable / max(n_total, 1), 1),
            "json_schema_valid": n_json_valid,
            "json_schema_valid_pct": round(100 * n_json_valid / max(n_total, 1), 1),
            "shacl_valid": n_shacl_valid,
            "shacl_valid_pct": round(100 * n_shacl_valid / max(n_total, 1), 1),
            "avg_precision_pct": avg_precision,
            "avg_latency_seconds": round(avg_latency, 2),
        }
        run_summaries.append(run_summary)
        all_results.extend(run_results)
        print(
            f"  Run {run_idx+1}: parse={n_parseable}/{n_total}"
            f" json={n_json_valid}/{n_total}"
            f" shacl={n_shacl_valid}/{n_total}"
            f" prec={avg_precision}% avg_lat={avg_latency:.1f}s"
        )

    # Aggregate summary
    total = len(all_results)
    all_precisions = [r["precision_pct"] for r in all_results if r["json_schema_valid"]]
    summary = {
        "model": model,
        "n_prompts_per_run": n_prompts,
        "total_runs": 3,
        "total_prompts": total,
        "overall_parseable_pct": round(
            100 * sum(r["parseable"] for r in all_results) / max(total, 1), 1
        ),
        "overall_json_schema_valid_pct": round(
            100 * sum(r["json_schema_valid"] for r in all_results) / max(total, 1), 1
        ),
        "overall_shacl_valid_pct": round(
            100 * sum(r["shacl_valid"] for r in all_results) / max(total, 1), 1
        ),
        "overall_precision_pct": round(
            sum(all_precisions) / len(all_precisions), 1
        ) if all_precisions else 0.0,
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in all_results) / max(total, 1), 2
        ),
        "per_run": run_summaries,
        "violation_counts": {},
    }

    # Count violation types
    v_counts = defaultdict(int)
    for r in all_results:
        for v in r["violations"]:
            key = re.sub(r"'[^']*'", "'...'", v)
            v_counts[key] += 1
    summary["violation_counts"] = dict(sorted(v_counts.items(), key=lambda x: -x[1]))

    # Save
    output = {
        "experiment": "LLM Control Recommendation (Exp 3)",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": all_results,
    }
    out_path = os.path.join(PROJECT_ROOT, "results", "llm_control_recommendation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n=== Results saved to {out_path} ===")
    print(
        f"Parse: {summary['overall_parseable_pct']}% | "
        f"JSON: {summary['overall_json_schema_valid_pct']}% | "
        f"SHACL: {summary['overall_shacl_valid_pct']}% | "
        f"Precision: {summary['overall_precision_pct']}%"
    )
    return output


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "haiku"
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_experiment(model, n_prompts)
