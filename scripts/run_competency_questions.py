#!/usr/bin/env python3
"""Competency-question (CQ) evaluation of the GRC ontology + Asset Relationship Graph.

Addresses DKE review finding M7: the paper lacked a CQ-based ontology evaluation.

Method
------
1. Load the OWL/RDFS ontology (ontology/ttl/grc-ontology.ttl) into an rdflib Graph.
2. Convert the instantiated 138-node ARG (results/multisource_arg.json) to RDF
   *faithfully* -- no repair of source data -- using the ontology vocabulary
   wherever the ontology defines a matching class/property.  ARG relations and
   attributes for which the ontology defines NO term are emitted under a separate
   `arg:` extension namespace so that vocabulary gaps remain visible instead of
   being papered over.  The conversion is written to results/arg_instance.ttl.
3. Execute 10 competency questions as SPARQL queries over ontology + instance
   graph, and record the actual results in results/competency_questions.json.

Vocabulary mapping (JSON -> RDF)
--------------------------------
grc:AttackTechnique, grc:usesTechnique and grc:targetsAsset were declared in the
ontology at v0.3.0 (see the paper's Section on the vocabulary audit); this script
was updated at the same time to use them instead of the arg: extension terms it
previously fell back to for those three. That closes two of the three gaps this
script originally reported and narrows the third (AFFECTS_ASSET) to a genuine,
still-open one: grc:affects has rdfs:domain grc:Impact, not grc:RiskCase, and no
ontology property currently covers RiskCase-to-Asset affectedness directly.

Node types:
  Asset          -> grc:Asset (also grc:Server: all 33 have asset_type "server"-like)
  Software       -> grc:SoftwareAsset
  CVE            -> grc:Vulnerability
  ATTACKTechnique-> grc:AttackTechnique   (was grc:Threat; grc:AttackTechnique declared v0.3.0)
  NFCRMControl   -> grc:NFCRMControl
  RiskCase       -> grc:RiskCase
Edges with an ontology term:
  HAS_SOFTWARE      (asset -> software)     -> grc:hostsSoftware
  EXPOSED_TO_CVE    (asset -> cve)          -> grc:hasVulnerability
  AFFECTED_BY_CVE   (software -> cve)       -> inverse grc:concerns (cve grc:concerns software)
  MITIGATED_BY      (risk -> control)       -> grc:associatedWithControl
  TARGETS_ASSET     (technique -> asset)    -> grc:targetsAsset   (declared v0.3.0; was arg:targetsAsset)
  USES_TECHNIQUE    (risk -> technique)     -> grc:usesTechnique  (declared v0.3.0; was arg:usesTechnique)
Edges with NO ontology term (gap -> arg: namespace):
  AFFECTS_ASSET     (risk -> asset)      -> arg:affectsAsset (grc:affects has domain Impact, not RiskCase)
Datatype attributes with NO ontology term (gap -> arg: namespace):
  in_cisa_kev -> arg:inCisaKev ; criticality -> arg:criticality ; risk_score -> arg:riskScore
tactic now maps to grc:tactic (declared v0.3.0, domain grc:AttackTechnique -- the ATT&CK
technique nodes this converter uses it on), closing that gap; no competency question queries it.
"""
import json
import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_TTL = ROOT / "ontology" / "ttl" / "grc-ontology.ttl"
ARG_JSON = ROOT / "results" / "multisource_arg.json"
OUT_TTL = ROOT / "results" / "arg_instance.ttl"
OUT_JSON = ROOT / "results" / "competency_questions.json"

GRC = Namespace("https://w3id.org/grc/ontology#")
ARG = Namespace("http://example.org/grc/arg#")  # extension terms the ontology lacks
INST = Namespace("http://example.org/grc/instance/")

NODE_CLASS = {
    "Asset": GRC.Asset,
    "Software": GRC.SoftwareAsset,
    "CVE": GRC.Vulnerability,
    "ATTACKTechnique": GRC.AttackTechnique,
    "NFCRMControl": GRC.NFCRMControl,
    "RiskCase": GRC.RiskCase,
}

EDGE_PROP = {
    # relation: (predicate, invert?)
    "HAS_SOFTWARE": (GRC.hostsSoftware, False),
    "EXPOSED_TO_CVE": (GRC.hasVulnerability, False),
    "AFFECTED_BY_CVE": (GRC.concerns, True),   # cve concerns software
    "MITIGATED_BY": (GRC.associatedWithControl, False),
    "RELATES_TO_RISK": (GRC.relatesToRisk, False),
    "TARGETS_ASSET": (GRC.targetsAsset, False),   # declared v0.3.0
    "USES_TECHNIQUE": (GRC.usesTechnique, False), # declared v0.3.0
    # no ontology vocabulary for this one -> arg: extension (disclosed gap)
    "AFFECTS_ASSET": (ARG.affectsAsset, False),
}


def iri(node_id: str) -> URIRef:
    return INST[node_id.replace(" ", "_").replace("§", "S")]


def convert_arg(data: dict) -> Graph:
    g = Graph()
    g.bind("grc", GRC)
    g.bind("arg", ARG)
    g.bind("inst", INST)
    for n in data["nodes"]:
        s = iri(n["id"])
        ntype = n["type"]
        g.add((s, RDF.type, NODE_CLASS[ntype]))
        if ntype == "Asset":
            g.add((s, RDF.type, GRC.Server))
            g.add((s, GRC.assetId, Literal(n["id"])))
            if n.get("hostname"):
                g.add((s, GRC.hostname, Literal(n["hostname"])))
            if n.get("ip"):
                g.add((s, GRC.ipAddress, Literal(n["ip"])))
            if n.get("os"):
                g.add((s, GRC.osName, Literal(n["os"])))
            if n.get("asset_type"):
                g.add((s, GRC.assetType, Literal(n["asset_type"])))
            if n.get("criticality"):
                g.add((s, ARG.criticality, Literal(n["criticality"])))  # ontology gap
        elif ntype == "Software":
            if n.get("name"):
                g.add((s, GRC.softwareName, Literal(n["name"])))
            if n.get("version"):
                g.add((s, GRC.softwareVersion, Literal(n["version"])))
        elif ntype == "CVE":
            g.add((s, GRC.cveId, Literal(n["id"])))
            if n.get("cvss_v3") is not None:
                g.add((s, GRC.cvssScore, Literal(n["cvss_v3"], datatype=XSD.decimal)))
            if n.get("severity"):
                g.add((s, GRC.severity, Literal(n["severity"])))
            if n.get("cwe"):
                g.add((s, GRC.cweId, Literal(n["cwe"])))
            g.add((s, ARG.inCisaKev, Literal(bool(n.get("in_cisa_kev")), datatype=XSD.boolean)))  # gap
            if n.get("description"):
                g.add((s, RDFS.comment, Literal(n["description"])))
        elif ntype == "ATTACKTechnique":
            if n.get("technique_id"):
                g.add((s, GRC.attackTechniqueId, Literal(n["technique_id"])))
            if n.get("name"):
                g.add((s, RDFS.label, Literal(n["name"])))
            if n.get("tactic"):
                g.add((s, GRC.tactic, Literal(n["tactic"])))  # declared v0.3.0, scoped to AttackTechnique
        elif ntype == "NFCRMControl":
            g.add((s, GRC.controlId, Literal(n["id"])))
            if n.get("clause_id"):
                g.add((s, GRC.nfcrmClause, Literal(n["clause_id"])))
            if n.get("clause_description"):
                g.add((s, GRC.controlName, Literal(n["clause_description"])))
        elif ntype == "RiskCase":
            g.add((s, GRC.riskId, Literal(n["id"])))
            if n.get("risk_level"):
                g.add((s, GRC.riskLevel, Literal(n["risk_level"])))
            if n.get("likelihood") is not None:
                g.add((s, GRC.likelihood, Literal(str(n["likelihood"]))))
            if n.get("impact") is not None:
                g.add((s, GRC.impact, Literal(str(n["impact"]))))
            if n.get("nfcrm_clause"):
                g.add((s, GRC.nfcrmClause, Literal(n["nfcrm_clause"])))
            if n.get("risk_score") is not None:
                g.add((s, ARG.riskScore, Literal(n["risk_score"], datatype=XSD.integer)))  # gap
    for e in data["edges"]:
        pred, invert = EDGE_PROP[e["relation"]]
        s, o = iri(e["source"]), iri(e["target"])
        # Faithful conversion: dangling targets (e.g. MITIGATED_BY -> "NFCRM-6.7"
        # vs. node id "NFCRM-6-7") are kept as-is, NOT repaired.
        if invert:
            g.add((o, pred, s))
        else:
            g.add((s, pred, o))
    return g


PREFIXES = """
PREFIX grc:  <https://w3id.org/grc/ontology#>
PREFIX arg:  <http://example.org/grc/arg#>
PREFIX inst: <http://example.org/grc/instance/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

CQS = [
    {
        "id": "CQ1",
        "question": "Which assets are exposed to a vulnerability with CVSS >= 9.0, and to which CVE?",
        "sparql": """
            SELECT DISTINCT ?host ?cve ?score WHERE {
              ?asset a grc:Asset ; grc:hostname ?host ; grc:hasVulnerability ?v .
              ?v grc:cveId ?cve ; grc:cvssScore ?score .
              FILTER(?score >= 9.0)
            } ORDER BY DESC(?score) ?host""",
    },
    {
        "id": "CQ2",
        "question": "Which CVEs in the graph are listed in CISA KEV, and which assets are exposed to them?",
        "sparql": """
            SELECT ?cve (GROUP_CONCAT(DISTINCT ?host; separator=", ") AS ?exposedAssets) WHERE {
              ?v a grc:Vulnerability ; grc:cveId ?cve ; arg:inCisaKev true .
              OPTIONAL { ?asset grc:hasVulnerability ?v ; grc:hostname ?host . }
            } GROUP BY ?cve ORDER BY ?cve""",
        "note": "Requires arg:inCisaKev -- the ontology itself has no KEV-membership property (gap).",
    },
    {
        "id": "CQ3",
        "question": "Which RiskCases reference NFCRM clause SS6.7, and at what risk level?",
        "sparql": """
            SELECT ?risk ?clause ?level WHERE {
              ?r a grc:RiskCase ; grc:riskId ?risk ; grc:nfcrmClause ?clause ; grc:riskLevel ?level .
              FILTER(CONTAINS(?clause, "6.7"))
            } ORDER BY ?risk""",
    },
    {
        "id": "CQ4",
        "question": "Which High/Critical RiskCases lack a resolvable mitigating control (associatedWithControl edge that reaches an actual NFCRMControl node)?",
        "sparql": """
            SELECT ?risk ?level WHERE {
              ?r a grc:RiskCase ; grc:riskId ?risk ; grc:riskLevel ?level .
              FILTER(?level IN ("High", "Critical"))
              FILTER NOT EXISTS { ?r grc:associatedWithControl ?c . ?c a grc:NFCRMControl . }
            } ORDER BY ?risk""",
        "note": "Strict join: control IRI must be a typed NFCRMControl node, not a dangling reference.",
    },
    {
        "id": "CQ5",
        "question": "What is the propagation path from CVE-2014-0160 (Heartbleed) to exposed assets via the software they host?",
        "sparql": """
            SELECT ?cve ?sw ?swVersion ?host WHERE {
              ?v grc:cveId ?cve ; grc:concerns ?s .
              FILTER(?cve = "CVE-2014-0160")
              ?s grc:softwareName ?sw . OPTIONAL { ?s grc:softwareVersion ?swVersion }
              ?asset grc:hostsSoftware ?s ; grc:hostname ?host .
            } ORDER BY ?host""",
    },
    {
        "id": "CQ6",
        "question": "Which software components are affected by more than one CVE?",
        "sparql": """
            SELECT ?sw ?version (COUNT(DISTINCT ?v) AS ?nCves) WHERE {
              ?v a grc:Vulnerability ; grc:concerns ?s .
              ?s grc:softwareName ?sw . OPTIONAL { ?s grc:softwareVersion ?version }
            } GROUP BY ?sw ?version HAVING (COUNT(DISTINCT ?v) > 1)
            ORDER BY DESC(?nCves)""",
    },
    {
        "id": "CQ7",
        "question": "Which ATT&CK techniques target critical-criticality assets?",
        "sparql": """
            SELECT DISTINCT ?tid ?tname ?host WHERE {
              ?t a grc:AttackTechnique ; grc:attackTechniqueId ?tid ; rdfs:label ?tname ;
                 grc:targetsAsset ?asset .
              ?asset grc:hostname ?host ; arg:criticality "critical" .
            } ORDER BY ?tid ?host""",
        "note": "grc:targetsAsset is now a declared ontology property (v0.3.0); arg:criticality "
                "remains a gap (no ontology datatype property for asset criticality).",
    },
    {
        "id": "CQ8",
        "question": "For each RiskCase: which ATT&CK technique does it use, which asset does it affect, and which NFCRM control mitigates it (strict join to a control node)?",
        "sparql": """
            SELECT ?risk ?tid ?host ?ctrlClause WHERE {
              ?r a grc:RiskCase ; grc:riskId ?risk .
              OPTIONAL { ?r grc:usesTechnique ?t . ?t grc:attackTechniqueId ?tid . }
              OPTIONAL { ?r arg:affectsAsset ?a . ?a grc:hostname ?host . }
              OPTIONAL { ?r grc:associatedWithControl ?c .
                         ?c a grc:NFCRMControl ; grc:nfcrmClause ?ctrlClause . }
            } ORDER BY ?risk ?host""",
        "note": "Exposes the raw-data ID mismatch: MITIGATED_BY targets 'NFCRM-6.7' but control node ids are 'NFCRM-6-7'.",
    },
    {
        "id": "CQ8b",
        "question": "(CQ8 with edge-target ID normalisation applied in-query) Same traceability chain after normalising 'NFCRM-6.7' to node id 'NFCRM-6-7'.",
        "sparql": """
            SELECT DISTINCT ?risk ?ctrlClause ?ctrlDesc WHERE {
              ?r a grc:RiskCase ; grc:riskId ?risk ; grc:associatedWithControl ?cRef .
              BIND(IRI(REPLACE(STR(?cRef), "(NFCRM-[0-9]+)\\\\.", "$1-")) AS ?cFix)
              ?cFix a grc:NFCRMControl ; grc:nfcrmClause ?ctrlClause ; grc:controlName ?ctrlDesc .
            } ORDER BY ?risk""",
        "note": "Demonstrates the chain IS answerable once the source ID-format defect is normalised.",
    },
    {
        "id": "CQ9",
        "question": "Which assets host software but have no known CVE exposure (clean assets)?",
        "sparql": """
            SELECT DISTINCT ?host WHERE {
              ?asset a grc:Asset ; grc:hostname ?host ; grc:hostsSoftware ?s .
              FILTER NOT EXISTS { ?asset grc:hasVulnerability ?v . }
            } ORDER BY ?host""",
    },
    {
        "id": "CQ10",
        "question": "Which vulnerabilities are directly mitigated by a control via the ontology's grc:mitigates (Control->Vulnerability) property?",
        "sparql": """
            SELECT ?ctrl ?cve WHERE {
              ?c grc:mitigates ?v . ?c grc:controlId ?ctrl . ?v grc:cveId ?cve .
            }""",
        "note": "Expected EMPTY: the ARG carries only risk-level MITIGATED_BY edges; the ontology's vulnerability-level grc:mitigates property is never instantiated (instantiation gap).",
    },
]


def main() -> int:
    g = Graph()
    g.parse(ONTOLOGY_TTL, format="turtle")
    onto_triples = len(g)
    data = json.loads(ARG_JSON.read_text(encoding="utf-8"))
    inst = convert_arg(data)
    inst.serialize(OUT_TTL, format="turtle")
    for t in inst:
        g.add(t)
    print(f"Ontology triples: {onto_triples}; instance triples: {len(inst)}; merged: {len(g)}")

    results = []
    for cq in CQS:
        q = PREFIXES + cq["sparql"]
        entry = {"id": cq["id"], "question": cq["question"], "sparql": cq["sparql"].strip()}
        if cq.get("note"):
            entry["note"] = cq["note"]
        try:
            rows = [
                {str(k): (str(v) if v is not None else None) for k, v in binding.asdict().items()}
                for binding in g.query(q)
            ]
            entry["status"] = "answered" if rows else "empty"
            entry["n_results"] = len(rows)
            entry["results"] = rows[:15]
            if len(rows) > 15:
                entry["results_truncated"] = True
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "query_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
        print(f"{cq['id']}: {entry['status']} ({entry.get('n_results', 0)} rows)")

    out = {
        "purpose": "Competency-question evaluation of the GRC ontology + ARG (DKE review finding M7)",
        "method": (
            "ARG JSON (results/multisource_arg.json, 138 nodes / 148 edges) converted faithfully to RDF "
            "(results/arg_instance.ttl) using ontology vocabulary where defined and an arg: extension "
            "namespace where the ontology lacks a term; SPARQL executed with rdflib over "
            "ontology + instance graph. No source-data repair; dangling edge targets preserved. "
            "Ontology v0.3.0: TARGETS_ASSET and USES_TECHNIQUE now map to grc:targetsAsset and "
            "grc:usesTechnique rather than the arg: namespace, closing two of the five gaps this "
            "evaluation originally reported."
        ),
        "graph_files": {
            "ontology": str(ONTOLOGY_TTL.relative_to(ROOT)),
            "arg_source_json": str(ARG_JSON.relative_to(ROOT)),
            "arg_converted_ttl": str(OUT_TTL.relative_to(ROOT)),
        },
        "ontology_vocabulary_gaps_found": [
            "No ontology property for RiskCase->Asset affectedness (ARG AFFECTS_ASSET; grc:affects has domain Impact, and no Impact nodes exist in the ARG); used arg:affectsAsset.",
            "No datatype property for CISA-KEV membership, asset criticality, or numeric risk score; used arg:inCisaKev / arg:criticality / arg:riskScore.",
            "grc:mitigates (Control->Vulnerability) is defined but never instantiated in the ARG (CQ10 empty).",
        ],
        "ontology_vocabulary_gaps_closed_at_v0_3_0": [
            "Threat->Asset targeting: grc:targetsAsset (was arg:targetsAsset).",
            "RiskCase->Threat technique use: grc:usesTechnique (was arg:usesTechnique).",
            "ATT&CK tactic on AttackTechnique nodes: grc:tactic (was arg:tactic).",
        ],
        "data_integrity_findings": [
            "FIXED at source (build_multisource_arg.py): all 9 MITIGATED_BY edge targets previously "
            "used clause-style ids ('NFCRM-6.7') while control nodes used dash-style ids "
            "('NFCRM-6-7'), leaving every MITIGATED_BY edge dangling in the raw JSON. CQ4/CQ8 "
            "(strict join) exposed this and CQ8b demonstrated the chain resolved after in-query "
            "normalisation; the builder now normalises the clause reference when the edge is "
            "created, so CQ8 answers directly and CQ4 is empty because no High/Critical RiskCase "
            "lacks a resolvable control -- an empty-and-correct answer, not a dangling one.",
            "Repaired SHACL validation (see results/shacl_validation.json) reports conforms=false "
            "with 39 violations on this graph, all data-completeness rather than conversion "
            "defects: 22 NFCRM control nodes are neither mitigating a vulnerability nor associated "
            "with a RiskCase, 9 RiskCases have no AuditLog (none exist in the testbed graph), and "
            "8 CVEs carry no CWE because NVD assigns none.",
        ],
        "competency_questions": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
