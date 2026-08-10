"""Run SHACL validation on the multi-source ARG using pyshacl.

Converts ARG JSON nodes to RDF triples, validates against ontology SHACL shapes.

Output: results/shacl_validation.json
"""
import json
import os
import time
from decimal import Decimal

from rdflib import Graph, Namespace, Literal, URIRef, RDF, XSD
from pyshacl import validate

from clause_context import with_clause_context

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ONTOLOGY_DIR = os.path.join(BASE_DIR, "ontology")

# NOTE: must match the namespace used by ontology/ttl/grc-ontology.ttl and every
# file in ontology/shapes/. A prior value ("http://intersymbolic-grc.org/ontology#")
# matched no sh:targetClass, so validation silently targeted zero nodes and
# reported conforms=true vacuously.
GRC = Namespace("https://w3id.org/grc/ontology#")
SHACL = Namespace("http://www.w3.org/ns/shacl#")


def load_arg():
    with open(os.path.join(RESULTS_DIR, "multisource_arg.json"), encoding="utf-8") as f:
        return json.load(f)


# NFCRM-1:2025 §6.9 stores impact and likelihood as 1-5 ordinals; the shapes
# enumerate them as words. Mapping is stated here rather than left implicit.
IMPACT_WORDS = {1: "negligible", 2: "minor", 3: "moderate", 4: "major", 5: "catastrophic"}
LIKELIHOOD_WORDS = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}


def _str(g, s, p, v):
    """Add a string literal.

    Written as a plain (simple) literal, not xsd:string-typed. RDF 1.1 treats the
    two as identical, but rdflib keeps them as distinct terms, and SHACL sh:in
    membership is exact term matching -- so an xsd:string-typed value never
    matches an enumeration written with plain literals, as every shape here is.
    A sh:datatype xsd:string constraint still accepts a plain literal.
    """
    if v not in (None, ""):
        g.add((s, p, Literal(str(v))))


def arg_to_rdf(arg_data):
    """Convert ARG JSON nodes/edges to RDF using the ontology's own vocabulary.

    Property names and datatypes here must match ontology/ttl/grc-ontology.ttl
    and the shapes in ontology/shapes/. Earlier revisions emitted convenience
    names (hostname where the shape requires assetId/assetName) and xsd:float
    where the shape requires xsd:decimal; both are corrected below.

    Class typing is asserted explicitly rather than left to a reasoner: the data
    graph carries no rdfs:subClassOf triples, so an sh:class grc:Control
    constraint is only satisfied if the node is directly typed grc:Control.
    """
    g = Graph()
    g.bind("grc", GRC)

    # Direct types plus every superclass the shapes reference by sh:class.
    type_map = {
        "Asset": [GRC.Asset],
        "Software": [GRC.SoftwareAsset, GRC.Asset],
        "CVE": [GRC.Vulnerability],
        "ATTACKTechnique": [GRC.AttackTechnique],  # class declared as of ontology v0.3.0
        "NFCRMControl": [GRC.NFCRMControl, GRC.Control],
        "RiskCase": [GRC.RiskCase],
    }

    for node in arg_data.get("nodes", []):
        node_uri = GRC[node["id"].replace(" ", "_")]
        for t in type_map.get(node["type"], []):
            g.add((node_uri, RDF.type, t))

        ntype = node["type"]

        # Provenance is required on Asset, Vulnerability, Control and RiskCase.
        if ntype in ("Asset", "Software", "CVE", "NFCRMControl", "RiskCase"):
            _str(g, node_uri, GRC.provenanceSource, node.get("provenance_source"))
            if node.get("provenance_timestamp"):
                g.add((node_uri, GRC.provenanceTimestamp,
                       Literal(node["provenance_timestamp"], datatype=XSD.dateTime)))

        if ntype == "Asset":
            _str(g, node_uri, GRC.assetId, node["id"])
            _str(g, node_uri, GRC.assetName, node.get("hostname") or node["id"])
            _str(g, node_uri, GRC.assetType, node.get("ontology_asset_type"))
            _str(g, node_uri, GRC.hostname, node.get("hostname"))
            _str(g, node_uri, GRC.ipAddress, node.get("ip"))
            _str(g, node_uri, GRC.osName, node.get("os"))
            # NOTE: asset criticality and CISA-KEV membership have no declared
            # datatype property (disclosed vocabulary gaps); not emitted.

        elif ntype == "Software":
            _str(g, node_uri, GRC.assetId, node["id"])
            _str(g, node_uri, GRC.assetName, node.get("name") or node["id"])
            _str(g, node_uri, GRC.assetType, "software")
            _str(g, node_uri, GRC.softwareName, node.get("name"))
            _str(g, node_uri, GRC.softwareVersion, node.get("version"))
            # NOTE: CPE has no declared datatype property (vocabulary gap); not emitted.

        elif ntype == "CVE":
            _str(g, node_uri, GRC.cveId, node["id"])
            _str(g, node_uri, GRC.cweId, node.get("cwe"))
            if node.get("cvss_v3") is not None:
                g.add((node_uri, GRC.cvssScore,
                       Literal(Decimal(str(node["cvss_v3"])), datatype=XSD.decimal)))
            _str(g, node_uri, GRC.severity, (node.get("severity") or "").lower() or None)

        elif ntype == "ATTACKTechnique":
            _str(g, node_uri, GRC.attackTechniqueId, node.get("technique_id"))
            _str(g, node_uri, GRC.techniqueName, node.get("name"))
            _str(g, node_uri, GRC.tactic, node.get("tactic"))
            # ATT&CK ingest records no provenance fields; see AttackTechniqueShape.

        elif ntype == "NFCRMControl":
            _str(g, node_uri, GRC.controlId, node["id"])
            _str(g, node_uri, GRC.controlName, node.get("clause_description"))
            _str(g, node_uri, GRC.nfcrmClause, node.get("clause_id"))

        elif ntype == "RiskCase":
            _str(g, node_uri, GRC.riskId, node["id"])
            _str(g, node_uri, GRC.riskLevel, (node.get("risk_level") or "").lower() or None)
            if node.get("impact") is not None:
                _str(g, node_uri, GRC.impact, IMPACT_WORDS.get(int(node["impact"])))
            if node.get("likelihood") is not None:
                _str(g, node_uri, GRC.likelihood, LIKELIHOOD_WORDS.get(int(node["likelihood"])))
            if node.get("confidence") is not None:
                g.add((node_uri, GRC.confidence,
                       Literal(Decimal(str(node["confidence"])), datatype=XSD.decimal)))
            _str(g, node_uri, GRC.nfcrmClause, node.get("nfcrm_clause"))

    # Edges. MITIGATED_BY carries the RiskCase->Control link that
    # grc:RiskCaseShape checks via grc:associatedWithControl.
    relation_map = {
        "HAS_SOFTWARE": GRC.hasSoftware,
        "AFFECTED_BY_CVE": GRC.hasVulnerability,
        "EXPOSED_TO_CVE": GRC.hasVulnerability,
        "TARGETS_ASSET": GRC.targetsAsset,
        "AFFECTS_ASSET": GRC.affects,
        "USES_TECHNIQUE": GRC.usesTechnique,
        "MITIGATED_BY": GRC.associatedWithControl,
        "RELATES_TO_RISK": GRC.relatesToRisk,
    }
    for edge in arg_data.get("edges", []):
        src = GRC[edge["source"].replace(" ", "_")]
        tgt = GRC[edge["target"].replace(" ", "_")]
        prop = relation_map.get(edge["relation"])
        if prop:
            g.add((src, prop, tgt))

    return g


def load_shacl_shapes():
    """Load SHACL shapes from ontology/shapes/."""
    shapes_graph = Graph()
    shapes_dir = os.path.join(ONTOLOGY_DIR, "shapes")
    for fname in os.listdir(shapes_dir):
        if fname.endswith(".shacl.ttl"):
            path = os.path.join(shapes_dir, fname)
            shapes_graph.parse(path, format="turtle")
            print(f"  Loaded shapes: {fname} ({len(shapes_graph)} triples)")
    return shapes_graph


def main():
    t0 = time.time()

    print("Loading ARG...")
    arg_data = load_arg()
    print(f"  {arg_data['statistics']['total_nodes']} nodes, {arg_data['statistics']['total_edges']} edges")

    print("Converting to RDF...")
    data_graph = arg_to_rdf(arg_data)
    print(f"  {len(data_graph)} RDF triples")

    print("Loading SHACL shapes...")
    shapes_graph = load_shacl_shapes()

    print("Running SHACL validation...")
    # The clause set joins into the data graph so the clause-bearing shapes can
    # resolve grc:clauseId references. The ARG carries no RiskScenario or
    # ControlRecommendation nodes, so this adds no focus nodes here; it is applied
    # uniformly across validators so the harnesses cannot diverge.
    validation_graph = with_clause_context(data_graph)
    t_val = time.time()
    conforms, results_graph, results_text = validate(
        validation_graph,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
    )
    val_time = time.time() - t_val

    # Parse violations from results text
    violations = []
    violation_lines = [l.strip() for l in results_text.split("\n") if "Violation" in l or "Result" in l]

    # Count violations by querying results graph
    violation_count = 0
    violations_by_shape = {}
    for s, p, o in results_graph:
        if str(p).endswith("#resultSeverity"):
            violation_count += 1
        if str(p).endswith("#sourceShape"):
            shape_name = str(o).split("#")[-1] if "#" in str(o) else str(o).split("/")[-1]
            violations_by_shape[shape_name] = violations_by_shape.get(shape_name, 0) + 1

    # Shape coverage, measured against actual SHACL targeting rather than
    # inferred from JSON node types. A previous revision divided by node counts
    # taken from the JSON census, which could not detect that the data graph and
    # the shapes used different namespaces and so shared no focus nodes at all.
    target_classes = {
        str(o) for s, p, o in shapes_graph
        if str(p) == "http://www.w3.org/ns/shacl#targetClass"
    }
    targeted_uris = {
        str(s) for s, p, o in data_graph
        if p == RDF.type and str(o) in target_classes
    }
    total_nodes = arg_data["statistics"]["total_nodes"]
    nodes_targeted = len(targeted_uris)
    untargeted_by_type = {}
    for n in arg_data["nodes"]:
        uri = str(GRC[n["id"].replace(" ", "_")])
        if uri not in targeted_uris:
            untargeted_by_type[n["type"]] = untargeted_by_type.get(n["type"], 0) + 1

    # Distinct focus nodes that carry at least one violation.
    failing_nodes = {
        str(o) for s, p, o in results_graph
        if str(p).endswith("#focusNode")
    }
    violations_by_message = {}
    for s, p, o in results_graph:
        if str(p).endswith("#resultMessage"):
            violations_by_message[str(o)] = violations_by_message.get(str(o), 0) + 1

    total_time = time.time() - t0

    output = {
        "generated": arg_data.get("generated", ""),
        "conforms": conforms,
        "total_rdf_triples": len(data_graph),
        "total_shacl_triples": len(shapes_graph),
        "total_nodes": total_nodes,
        "nodes_targeted_by_a_shape": nodes_targeted,
        "nodes_untargeted_by_type": untargeted_by_type,
        "nodes_with_violations": len(failing_nodes),
        "nodes_targeted_and_clean": nodes_targeted - len(failing_nodes),
        "total_violations": violation_count,
        "violations_by_shape": violations_by_shape,
        "violations_by_message": violations_by_message,
        "shape_coverage_percent": round(100.0 * nodes_targeted / max(total_nodes, 1), 1),
        "conforming_share_of_targeted_percent": round(
            100.0 * (nodes_targeted - len(failing_nodes)) / max(nodes_targeted, 1), 1),
        "validation_time_seconds": round(val_time, 3),
        "total_time_seconds": round(total_time, 3),
        "results_text_excerpt": results_text[:2000],
        "node_type_coverage": {
            ntype: sum(1 for n in arg_data["nodes"] if n["type"] == ntype)
            for ntype in ("Asset", "Software", "CVE", "NFCRMControl", "RiskCase", "ATTACKTechnique")
        },
    }

    dest = os.path.join(RESULTS_DIR, "shacl_validation.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSHACL Validation Results:")
    print(f"  Conforms: {conforms}")
    print(f"  RDF triples: {len(data_graph)}")
    print(f"  Nodes targeted by a shape: {nodes_targeted}/{total_nodes} "
          f"({output['shape_coverage_percent']}%)")
    print(f"  Untargeted by type: {untargeted_by_type}")
    print(f"  Nodes with violations: {len(failing_nodes)}")
    print(f"  Violations: {violation_count}")
    print(f"  Validation time: {val_time:.3f}s")
    print(f"  -> {dest}")

    if not conforms:
        print(f"\nViolation details (first 1000 chars):")
        print(results_text[:1000])


if __name__ == "__main__":
    main()
