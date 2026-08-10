"""Measure how the ontology + SHACL layer behaves as the ARG grows.

Sweeps asset counts, and for each size records:
  * build time, node and edge counts, edge:node ratio
  * JSON -> RDF conversion time and triple count
  * SHACL validation time, conformance, violation count and classes
  * shape coverage (nodes actually targeted by a sh:targetClass)
  * degree distribution and measured traversal depth on the canonical
    Asset -> RiskCase -> Control chain

The question this answers is not whether a synthetic enterprise graph looks like
a real one -- it does not, and build_enterprise_arg.py says so -- but whether the
constraint layer's cost and behaviour hold up as the graph grows by orders of
magnitude, which is the property the architecture claims and has never been
tested beyond 138 nodes.

Usage: python scripts/measure_arg_scaling.py [--sizes 100 500 1000 5000 10000]
Output: results/arg_scaling/scaling_summary.json
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict, deque

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "arg_scaling")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdflib import RDF  # noqa: E402
from pyshacl import validate  # noqa: E402

from clause_context import with_clause_context  # noqa: E402

import build_enterprise_arg as B  # noqa: E402
import run_shacl_validation as R  # noqa: E402


def topology(arg):
    """Degree distribution and measured hop depth of the traceability chain."""
    out_adj = defaultdict(list)
    deg = Counter()
    for e in arg["edges"]:
        out_adj[e["source"]].append((e["target"], e["relation"]))
        deg[e["source"]] += 1
        deg[e["target"]] += 1

    types = {n["id"]: n["type"] for n in arg["nodes"]}
    assets = [n["id"] for n in arg["nodes"] if n["type"] == "Asset"]
    controls = {n["id"] for n in arg["nodes"] if n["type"] == "NFCRMControl"}

    # Shortest Asset -> ... -> Control path, over a sample of assets so the
    # measurement stays linear in graph size rather than quadratic.
    sample = assets[:: max(1, len(assets) // 200)][:200]
    depths = []
    for start in sample:
        seen_n, q = {start}, deque([(start, 0)])
        while q:
            node, d = q.popleft()
            if d > 8:
                break
            if node in controls:
                depths.append(d)
                break
            for nxt, _rel in out_adj.get(node, ()):
                if nxt not in seen_n:
                    seen_n.add(nxt)
                    q.append((nxt, d + 1))

    degs = list(deg.values())
    return {
        "assets_sampled_for_depth": len(sample),
        "assets_reaching_a_control": len(depths),
        "mean_hops_asset_to_control": round(statistics.mean(depths), 2) if depths else None,
        "max_hops_asset_to_control": max(depths) if depths else None,
        "mean_degree": round(statistics.mean(degs), 2) if degs else 0,
        "median_degree": statistics.median(degs) if degs else 0,
        "max_degree": max(degs) if degs else 0,
        "isolated_nodes": sum(1 for n in arg["nodes"] if deg[n["id"]] == 0),
        "node_types_present": dict(Counter(types.values())),
    }


def strip_sparql_constraints(shapes):
    """Copy of the shape graph with sh:sparql constraints removed.

    Lets validation cost be attributed between ordinary property constraints,
    which pyshacl evaluates per focus node, and the graph-level SPARQL
    constraints, which issue a query per focus node and are the expected source
    of any superlinear growth.
    """
    from rdflib import Graph as _G
    out = _G()
    sparql_objs = {o for s, p, o in shapes if str(p).endswith("#sparql")}
    drop = set(sparql_objs)
    # blank-node closure of each sh:sparql value
    frontier = list(sparql_objs)
    while frontier:
        node = frontier.pop()
        for s, p, o in shapes.triples((node, None, None)):
            if o not in drop and not isinstance(o, str):
                drop.add(o)
                frontier.append(o)
    for s, p, o in shapes:
        if str(p).endswith("#sparql") or s in drop:
            continue
        out.add((s, p, o))
    return out


def measure(n_assets, seed, cve_pool, techniques, nfcrm, shapes, keep_graph,
            shapes_no_sparql=None):
    arg = B.build(n_assets, seed, cve_pool, techniques, nfcrm)

    t = time.time()
    data = R.arg_to_rdf(arg)
    convert_s = time.time() - t

    # The clause set joins in outside the timed region. It is a fixed 116-triple
    # cost independent of graph size, so folding it into the measurement would
    # add a constant to every point and flatten the reported scaling curve.
    data_v = with_clause_context(data)

    t = time.time()
    conforms, rg, _ = validate(data_v, shacl_graph=shapes, inference="none",
                               abort_on_first=False)
    validate_s = time.time() - t

    no_sparql_s = None
    if shapes_no_sparql is not None:
        t = time.time()
        validate(data_v, shacl_graph=shapes_no_sparql, inference="none",
                 abort_on_first=False)
        no_sparql_s = round(time.time() - t, 3)

    target_classes = {
        str(o) for s, p, o in shapes
        if str(p) == "http://www.w3.org/ns/shacl#targetClass"
    }
    targeted = {str(s) for s, p, o in data
                if p == RDF.type and str(o) in target_classes}
    failing = {str(o) for s, p, o in rg if str(p).endswith("#focusNode")}
    msgs = Counter(str(o) for s, p, o in rg if str(p).endswith("#resultMessage"))

    st = arg["statistics"]
    row = {
        "n_assets": n_assets,
        "seed": seed,
        "nodes": st["total_nodes"],
        "edges": st["total_edges"],
        "edge_node_ratio": st["edge_node_ratio"],
        "nodes_by_type": st["nodes_by_type"],
        "build_seconds": arg["build_time_seconds"],
        "rdf_convert_seconds": round(convert_s, 3),
        "rdf_triples": len(data),
        "shacl_validate_seconds": round(validate_s, 3),
        "shacl_validate_seconds_without_sparql": no_sparql_s,
        "sparql_constraint_share": (
            round(1.0 - no_sparql_s / validate_s, 3)
            if no_sparql_s is not None and validate_s > 0 else None),
        "conforms": bool(conforms),
        "nodes_targeted": len(targeted),
        "shape_coverage_percent": round(100.0 * len(targeted) / max(st["total_nodes"], 1), 1),
        "nodes_with_violations": len(failing),
        "total_violations": sum(msgs.values()),
        "violations_by_message": dict(msgs.most_common(10)),
        "microseconds_per_triple": round(1e6 * validate_s / max(len(data), 1), 2),
        "topology": topology(arg),
    }

    if keep_graph:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        dest = os.path.join(RESULTS_DIR, f"arg_{n_assets}.json")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(arg, f)
        row["graph_file"] = os.path.relpath(dest, BASE_DIR)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 500, 1000, 5000, 10000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-largest", action="store_true",
                    help="persist the largest generated graph to disk")
    ap.add_argument("--output", default=None,
                    help="output path (default: results/arg_scaling/scaling_summary.json). "
                         "Use this for partial/exploratory runs so the committed summary "
                         "is never silently overwritten.")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting the default output path with a --sizes list "
                         "that doesn't match the file already there")
    args = ap.parse_args()

    dest = args.output or os.path.join(RESULTS_DIR, "scaling_summary.json")
    if args.output is None and os.path.exists(dest) and not args.force:
        with open(dest, encoding="utf-8") as f:
            existing_sizes = sorted(r.get("n_assets") for r in json.load(f).get("rows", []))
        if sorted(args.sizes) != existing_sizes:
            raise SystemExit(
                f"Refusing to overwrite {dest}: its rows cover sizes {existing_sizes}, "
                f"this run covers {sorted(args.sizes)}. Pass --output to write elsewhere, "
                f"or --force to overwrite anyway.")

    cve_pool = B.load_cve_pool()
    techniques = B.attack_techniques()
    nfcrm = B.load_json("nfcrm_clause_mapping.json")
    shapes = R.load_shacl_shapes()
    shapes_no_sparql = strip_sparql_constraints(shapes)

    rows = []
    for n in sorted(args.sizes):
        keep = args.keep_largest and n == max(args.sizes)
        print(f"\n=== {n} assets ===")
        row = measure(n, args.seed, cve_pool, techniques, nfcrm, shapes, keep,
                      shapes_no_sparql)
        rows.append(row)
        print(f"  nodes={row['nodes']} edges={row['edges']} "
              f"triples={row['rdf_triples']} ratio={row['edge_node_ratio']}")
        print(f"  build={row['build_seconds']}s convert={row['rdf_convert_seconds']}s "
              f"validate={row['shacl_validate_seconds']}s "
              f"(sparql share {row['sparql_constraint_share']}) "
              f"({row['microseconds_per_triple']} us/triple)")
        print(f"  conforms={row['conforms']} violations={row['total_violations']} "
              f"coverage={row['shape_coverage_percent']}%")
        tp = row["topology"]
        print(f"  mean degree={tp['mean_degree']} max={tp['max_degree']} "
              f"asset->control hops mean={tp['mean_hops_asset_to_control']}")

    os.makedirs(os.path.dirname(dest) or RESULTS_DIR, exist_ok=True)
    out = {
        "purpose": "Scaling behaviour of the OWL/SHACL constraint layer as the ARG "
                   "grows. Threat-side data is real (CISA KEV + NVD + MITRE ATT&CK + "
                   "NFCRM-1:2025); organisation-side inventory is synthetic and seeded. "
                   "Supports claims about validation cost and conformance at scale, not "
                   "about real enterprise topology.",
        "cve_pool_size": len(cve_pool),
        "shape_triples": len(shapes),
        "seed": args.seed,
        "rows": rows,
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {dest}")


if __name__ == "__main__":
    main()
