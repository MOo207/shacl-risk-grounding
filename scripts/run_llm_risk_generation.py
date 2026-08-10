#!/usr/bin/env python3
"""Experiment 2: LLM Risk Scenario Generation with SHACL + JSON Schema validation."""
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

VALID_CLAUSE_PATTERN = re.compile(r"^SS[56]\.\d+$")
VALID_IMPACT_LEVELS = ["critical", "high", "medium", "low"]


def load_arg(path):
    with open(path) as f:
        return json.load(f)


def get_asset_nodes(arg):
    return [n for n in arg["nodes"] if n["type"] == "Asset"]


def extract_2hop_subgraph(arg, asset_id):
    """Extract target asset node and all nodes within 2 hops."""
    node_map = {n["id"]: n for n in arg["nodes"]}

    # 1-hop neighbors
    hop1_ids = set()
    for e in arg["edges"]:
        if e["source"] == asset_id:
            hop1_ids.add(e["target"])
        elif e["target"] == asset_id:
            hop1_ids.add(e["source"])

    # 2-hop neighbors
    hop2_ids = set()
    for e in arg["edges"]:
        if e["source"] in hop1_ids:
            hop2_ids.add(e["target"])
        elif e["target"] in hop1_ids:
            hop2_ids.add(e["source"])

    all_ids = {asset_id} | hop1_ids | hop2_ids
    subgraph_nodes = [node_map[nid] for nid in all_ids if nid in node_map]
    subgraph_edges = [
        e for e in arg["edges"]
        if e["source"] in all_ids and e["target"] in all_ids
    ]
    return {"nodes": subgraph_nodes, "edges": subgraph_edges}


def load_prompt_template():
    path = os.path.join(PROJECT_ROOT, "scripts", "templates",
                        "risk_scenario_prompt.txt")
    with open(path) as f:
        return f.read()


def fill_prompt(template, subgraph):
    subgraph_str = json.dumps(subgraph, indent=2)
    return template.replace("{arg_subgraph}", subgraph_str)


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


REQUIRED_FIELDS = ["scenario_title", "affected_asset", "threat_vector",
                   "impact_level", "nfcrm_clause", "recommended_controls",
                   "description"]


def validate_json_schema(parsed):
    """Structural JSON Schema validation: check required fields exist."""
    if not isinstance(parsed, dict):
        return False, ["Response is not a JSON object"]
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        return False, [f"Missing required field: {f}" for f in missing]
    # recommended_controls must be a list
    ctrls = parsed.get("recommended_controls", [])
    if not isinstance(ctrls, list):
        return False, ["recommended_controls is not a list"]
    return True, []


def validate_shacl(parsed, arg_node_ids):
    """Programmatic SHACL-like constraint validation."""
    violations = []

    # 1. affected_asset must exist in ARG nodes
    aa = parsed.get("affected_asset", "")
    if aa not in arg_node_ids:
        violations.append(f"affected_asset '{aa}' not found in ARG nodes")

    # 2. impact_level must be in valid set
    il = parsed.get("impact_level", "")
    if il not in VALID_IMPACT_LEVELS:
        violations.append(f"impact_level '{il}' not in valid set")

    # 3. nfcrm_clause must match SS[56].\d+ pattern
    clause = str(parsed.get("nfcrm_clause", ""))
    if not VALID_CLAUSE_PATTERN.match(clause):
        violations.append(
            f"nfcrm_clause '{clause}' does not match SS[56].d+ pattern"
        )

    # 4. All recommended_controls must match SS[56].\d+ pattern
    ctrls = parsed.get("recommended_controls", [])
    if isinstance(ctrls, list):
        for ctrl in ctrls:
            cid = str(ctrl)
            if not VALID_CLAUSE_PATTERN.match(cid):
                violations.append(
                    f"recommended_control '{cid}' does not match SS[56].d+ pattern"
                )
    else:
        violations.append("recommended_controls is not a list")

    return len(violations) == 0, violations


def track_hallucinations(parsed, arg_node_ids):
    """Track entity and relationship hallucinations."""
    entity_hallucinations = []
    relationship_hallucinations = []

    # Entity: affected_asset not in ARG
    aa = parsed.get("affected_asset", "")
    if aa and aa not in arg_node_ids:
        entity_hallucinations.append(f"asset:{aa}")

    # Entity: nfcrm_clause not matching pattern
    clause = str(parsed.get("nfcrm_clause", ""))
    if clause and not VALID_CLAUSE_PATTERN.match(clause):
        entity_hallucinations.append(f"clause:{clause}")

    # Relationship: recommended_controls with invalid patterns
    ctrls = parsed.get("recommended_controls", [])
    if isinstance(ctrls, list):
        for ctrl in ctrls:
            cid = str(ctrl)
            if not VALID_CLAUSE_PATTERN.match(cid):
                relationship_hallucinations.append(f"control:{cid}")

    return entity_hallucinations, relationship_hallucinations


def compute_precision(parsed):
    """Precision = % of recommended controls with valid NFCRM clause IDs."""
    ctrls = parsed.get("recommended_controls", [])
    if not isinstance(ctrls, list) or len(ctrls) == 0:
        return 0.0
    valid = sum(
        1 for c in ctrls
        if VALID_CLAUSE_PATTERN.match(str(c))
    )
    return round(100.0 * valid / len(ctrls), 1)


def run_experiment(model, n_prompts):
    print(f"=== Experiment 2: LLM Risk Scenario Generation ===")
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
        "You are a cybersecurity GRC analyst specializing in risk scenario generation "
        "per NFCRM-1:2025. Respond with ONLY valid JSON."
    )

    all_results = []
    run_summaries = []
    total_entity_hallucinations = 0
    total_relationship_hallucinations = 0

    for run_idx in range(3):
        print(f"\n--- Run {run_idx + 1}/3 ---")
        run_results = []
        run_precisions = []

        for i, asset in enumerate(selected):
            asset_id = asset["id"]
            print(
                f"  [{i+1}/{n_prompts}] Asset: {asset_id}...",
                end=" ",
                flush=True,
            )

            # Extract 2-hop subgraph
            subgraph = extract_2hop_subgraph(arg, asset_id)

            # Fill prompt
            user_msg = fill_prompt(template, subgraph)

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
            entity_h = []
            relationship_h = []

            if parseable:
                json_valid, json_violations = validate_json_schema(parsed)
                if json_valid:
                    shacl_valid, shacl_violations = validate_shacl(
                        parsed, arg_node_ids
                    )
                    precision = compute_precision(parsed)
                    run_precisions.append(precision)
                    entity_h, relationship_h = track_hallucinations(
                        parsed, arg_node_ids
                    )
                    total_entity_hallucinations += len(entity_h)
                    total_relationship_hallucinations += len(relationship_h)

            status = (
                "SHACL_OK"
                if shacl_valid
                else (
                    "JSON_OK"
                    if json_valid
                    else ("PARSED" if parseable else "FAIL")
                )
            )
            print(f"{status} prec={precision}% ({latency:.1f}s)")

            result = {
                "run": run_idx + 1,
                "asset_id": asset_id,
                "parseable": parseable,
                "json_schema_valid": json_valid,
                "shacl_valid": shacl_valid,
                "precision_pct": precision,
                "violations": json_violations + shacl_violations,
                "entity_hallucinations": entity_h,
                "relationship_hallucinations": relationship_h,
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
        avg_latency = sum(r["latency_seconds"] for r in run_results) / max(
            n_total, 1
        )
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
            "json_schema_valid_pct": round(
                100 * n_json_valid / max(n_total, 1), 1
            ),
            "shacl_valid": n_shacl_valid,
            "shacl_valid_pct": round(
                100 * n_shacl_valid / max(n_total, 1), 1
            ),
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
    all_precisions = [
        r["precision_pct"] for r in all_results if r["json_schema_valid"]
    ]
    summary = {
        "model": model,
        "n_prompts_per_run": n_prompts,
        "total_runs": 3,
        "total_prompts": total,
        "overall_parseable_pct": round(
            100 * sum(r["parseable"] for r in all_results) / max(total, 1), 1
        ),
        "overall_json_schema_valid_pct": round(
            100
            * sum(r["json_schema_valid"] for r in all_results)
            / max(total, 1),
            1,
        ),
        "overall_shacl_valid_pct": round(
            100 * sum(r["shacl_valid"] for r in all_results) / max(total, 1),
            1,
        ),
        "overall_precision_pct": round(
            sum(all_precisions) / len(all_precisions), 1
        )
        if all_precisions
        else 0.0,
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in all_results) / max(total, 1), 2
        ),
        "total_entity_hallucinations": total_entity_hallucinations,
        "total_relationship_hallucinations": total_relationship_hallucinations,
        "per_run": run_summaries,
        "violation_counts": {},
    }

    # Count violation types
    v_counts = defaultdict(int)
    for r in all_results:
        for v in r["violations"]:
            key = re.sub(r"'[^']*'", "'...'", v)
            v_counts[key] += 1
    summary["violation_counts"] = dict(
        sorted(v_counts.items(), key=lambda x: -x[1])
    )

    # Save
    output = {
        "experiment": "LLM Risk Scenario Generation (Exp 2)",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": all_results,
    }
    out_path = os.path.join(
        PROJECT_ROOT, "results", "llm_risk_generation.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n=== Results saved to {out_path} ===")
    print(
        f"Parse: {summary['overall_parseable_pct']}% | "
        f"JSON: {summary['overall_json_schema_valid_pct']}% | "
        f"SHACL: {summary['overall_shacl_valid_pct']}% | "
        f"Precision: {summary['overall_precision_pct']}%"
    )
    print(
        f"Entity hallucinations: {total_entity_hallucinations} | "
        f"Relationship hallucinations: {total_relationship_hallucinations}"
    )
    return output


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "haiku"
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_experiment(model, n_prompts)
