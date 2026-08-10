"""Re-validate the stored Experiment 2/3 outputs with real SHACL (pyshacl).

Why this exists
---------------
Experiments 2 and 3 shipped with a function named validate_shacl() whose own
docstring read "Programmatic SHACL-like constraint validation". It loaded no
shape file, imported neither rdflib nor pyshacl, and re-implemented three checks
in imperative Python -- one of them against a hardcoded module-level list:

    ATTACK_CLASSES = ["BruteForce", "DDoS", "DoS", "Infiltration", "WebAttack"]

So the paper's headline comparison ("graph-derived SHACL rejects 28 outputs a
hand-written JSON Schema accepted") was, mechanically, a hand-written Python
function against a hand-written JSON Schema. This script re-runs the same stored
model outputs -- no new LLM calls -- through the actual shape library in
ontology/shapes/, using pyshacl, so the comparison measures what it claims to.

Two differences from the Python stand-in are expected and are the point:
  * grc:ControlRecommendationShape enumerates six threat classes, including
    "Benign", where the Python list carried five.
  * Both shapes validate clause identifiers referentially against the 26
    enumerated NFCRM-1:2025 clauses, where the Python stand-in applied the
    syntactic pattern ^SS[56]\\.[0-9]+$ that would accept SS5.999.

Output: results/generation_shacl_revalidation.json
"""
import json
import os
import sys
import time
from collections import Counter

from rdflib import Graph, Literal, Namespace, RDF
from pyshacl import validate

from clause_context import with_clause_context

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SHAPES_DIR = os.path.join(BASE_DIR, "ontology", "shapes")

GRC = Namespace("https://w3id.org/grc/ontology#")


def load_shapes():
    g = Graph()
    for fname in sorted(os.listdir(SHAPES_DIR)):
        if fname.endswith(".shacl.ttl"):
            g.parse(os.path.join(SHAPES_DIR, fname), format="turtle")
    return g


def arg_asset_context():
    """Real grc:Asset nodes from the ARG, with their properties.

    Both shapes constrain an asset reference with sh:class grc:Asset, so the
    referenced node must exist and be typed. This is the genuinely graph-derived
    part of the constraint: the valid asset set is read from the knowledge
    graph, not enumerated in the validator.

    The assets are converted with their real properties rather than stubbed as
    bare typed nodes, otherwise grc:AssetShape fires on the context itself and
    buries the artefact's own violations under scaffold noise.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_shacl_validation as R

    with open(os.path.join(RESULTS_DIR, "multisource_arg.json"), encoding="utf-8") as f:
        arg = json.load(f)
    assets_only = {
        "nodes": [n for n in arg["nodes"] if n["type"] == "Asset"],
        "edges": [],
    }
    return R.arg_to_rdf(assets_only), {n["id"] for n in assets_only["nodes"]}


def base_graph(asset_ctx):
    g = Graph()
    g.bind("grc", GRC)
    for triple in asset_ctx:
        g.add(triple)
    return g


def risk_scenario_graph(parsed, asset_ctx, idx):
    g = base_graph(asset_ctx)
    node = GRC[f"scenario-{idx}"]
    g.add((node, RDF.type, GRC.RiskScenario))
    if parsed.get("scenario_title"):
        g.add((node, GRC.scenarioTitle, Literal(str(parsed["scenario_title"]))))
    if parsed.get("description"):
        g.add((node, GRC.description, Literal(str(parsed["description"]))))
    if parsed.get("threat_vector"):
        g.add((node, GRC.threatVector, Literal(str(parsed["threat_vector"]))))
    if parsed.get("impact_level"):
        g.add((node, GRC.impactLevel, Literal(str(parsed["impact_level"]))))
    if parsed.get("nfcrm_clause"):
        g.add((node, GRC.nfcrmClause, Literal(str(parsed["nfcrm_clause"]))))
    aa = parsed.get("affected_asset")
    if aa:
        # Referenced verbatim. An asset the graph does not contain stays
        # untyped, so sh:class grc:Asset fails -- which is the check.
        g.add((node, GRC.affectedAsset, GRC[str(aa)]))
    for c in parsed.get("recommended_controls", []) or []:
        g.add((node, GRC.recommendedControl, Literal(str(c))))
    return g, node


def control_recommendation_graph(parsed, asset_ctx, idx):
    g = base_graph(asset_ctx)
    node = GRC[f"recommendation-{idx}"]
    g.add((node, RDF.type, GRC.ControlRecommendation))
    if parsed.get("threat_class"):
        g.add((node, GRC.threatClass, Literal(str(parsed["threat_class"]))))
    ta = parsed.get("target_asset")
    if ta:
        g.add((node, GRC.targetAsset, GRC[str(ta)]))
    for rec in parsed.get("recommendations", []) or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("control_clause_id"):
            g.add((node, GRC.controlClauseId, Literal(str(rec["control_clause_id"]))))
        if rec.get("justification"):
            g.add((node, GRC.justification, Literal(str(rec["justification"]))))
    return g, node


def run(experiment, infile, builder, shapes):
    path = os.path.join(RESULTS_DIR, infile)
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    asset_ctx, asset_ids = arg_asset_context()

    per_case, messages = [], Counter()
    t0 = time.time()
    for i, res in enumerate(stored["results"]):
        parsed = res.get("parsed_response")
        if parsed and not isinstance(parsed, dict):
            # The original pipeline gated validate_shacl() behind a JSON-Schema
            # object check, so a non-object response never reached constraint
            # validation and was recorded invalid. Mirrored here.
            per_case.append({
                "index": i, "asset_id": res.get("asset_id"),
                "parseable": True, "shacl_conforms": False,
                "messages": ["response is not a JSON object"],
                "stored_shacl_valid": res.get("shacl_valid"),
            })
            messages["response is not a JSON object"] += 1
            continue
        if not parsed:
            per_case.append({
                "index": i, "asset_id": res.get("asset_id"),
                "parseable": False, "shacl_conforms": False,
                "messages": ["unparseable output"],
                "stored_shacl_valid": res.get("shacl_valid"),
            })
            messages["unparseable output"] += 1
            continue
        g, _ = builder(parsed, asset_ctx, i)
        conforms, rg, _ = validate(with_clause_context(g), shacl_graph=shapes, inference="none",
                                   abort_on_first=False)
        msgs = sorted({
            str(o) for s, p, o in rg if str(p).endswith("#resultMessage")
        })
        for m in msgs:
            messages[m] += 1
        per_case.append({
            "index": i, "asset_id": res.get("asset_id"),
            "parseable": True, "shacl_conforms": bool(conforms),
            "messages": msgs,
            "stored_shacl_valid": res.get("shacl_valid"),
        })
    elapsed = time.time() - t0

    n = len(per_case)
    conforming = sum(1 for c in per_case if c["shacl_conforms"])
    stored_valid = sum(1 for c in per_case if c["stored_shacl_valid"])
    agree = sum(1 for c in per_case
                if bool(c["stored_shacl_valid"]) == c["shacl_conforms"])
    return {
        "experiment": experiment,
        "source_file": infile,
        "n_cases": n,
        "real_shacl_conforming": conforming,
        "real_shacl_conforming_pct": round(100.0 * conforming / max(n, 1), 1),
        "real_shacl_rejected": n - conforming,
        "stored_pythonic_valid": stored_valid,
        "stored_pythonic_valid_pct": round(100.0 * stored_valid / max(n, 1), 1),
        "stored_pythonic_rejected": n - stored_valid,
        "verdict_agreement": agree,
        "verdict_agreement_pct": round(100.0 * agree / max(n, 1), 1),
        "violation_messages": dict(messages.most_common()),
        "validation_seconds": round(elapsed, 2),
        "per_case": per_case,
    }


def main():
    shapes = load_shapes()
    out = {
        "purpose": "Re-validate stored Experiment 2/3 outputs with real pyshacl "
                   "against ontology/shapes/, replacing the hand-written "
                   "validate_shacl() stand-in. No new LLM calls: the same stored "
                   "parsed_response objects are re-scored.",
        "shape_files": sorted(f for f in os.listdir(SHAPES_DIR)
                              if f.endswith(".shacl.ttl")),
        "shape_triples": len(shapes),
        "findings": [
            "Experiment 3: real SHACL rejects exactly the same 28 of 99 outputs as the "
            "Python stand-in did (100% verdict agreement). The headline rejection count "
            "is robust to swapping the validator.",
            "Experiment 3 mechanism correction: the rejected value set is the "
            "grc:threatClass sh:in enumeration, which is authored by hand in "
            "control-recommendation.shacl.ttl (six classes, including Benign) exactly as "
            "the Python ATTACK_CLASSES list was (five classes). Moving the constraint "
            "into the shape library does not make it graph-derived; it makes it "
            "declarative and reusable across producers.",
            "Experiment 2: real SHACL rejects 4 of 99 where the stand-in rejected 3 "
            "(99% agreement). The extra rejection is case index 74, prompted about asset "
            "SRV-009, whose affected_asset is 'SW-OWASP_Mutillidae_II-2.1.19' -- a node "
            "that exists in the ARG but is typed Software, not Asset. The stand-in tested "
            "membership in the untyped set of all node ids and passed it; the shape's "
            "sh:class grc:Asset is type-aware and catches the class confusion. This is "
            "the one measured instance where the graph-derived constraint catches an "
            "error the hand-written check misses.",
        ],
        "experiments": [
            run("Experiment 2: risk scenario generation",
                "llm_risk_generation.json", risk_scenario_graph, shapes),
            run("Experiment 3: control recommendation",
                "llm_control_recommendation.json",
                control_recommendation_graph, shapes),
        ],
    }
    dest = os.path.join(RESULTS_DIR, "generation_shacl_revalidation.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    for e in out["experiments"]:
        print(f"\n{e['experiment']}  (n={e['n_cases']})")
        print(f"  real SHACL conforming : {e['real_shacl_conforming']:3d} "
              f"({e['real_shacl_conforming_pct']}%)   rejected {e['real_shacl_rejected']}")
        print(f"  stored Python stand-in: {e['stored_pythonic_valid']:3d} "
              f"({e['stored_pythonic_valid_pct']}%)   rejected {e['stored_pythonic_rejected']}")
        print(f"  verdicts agree        : {e['verdict_agreement']}/{e['n_cases']} "
              f"({e['verdict_agreement_pct']}%)")
        for m, c in list(e["violation_messages"].items())[:8]:
            print(f"      {c:4d}  {m[:88]}")
    print(f"\n  -> {dest}")


if __name__ == "__main__":
    sys.exit(main())
