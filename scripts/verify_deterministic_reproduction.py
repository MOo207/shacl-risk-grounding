"""Independent reproduction check for the deterministic (non-LLM) pipeline claims.

Run by a fresh audit process with no prior exposure to this repository's
narrative -- given only file paths, told to verify the paper's deterministic
figures from scratch rather than trust the results/ JSON files. This script
is that check, released so a reader can re-run it rather than take the
verification on faith.

Scope, stated plainly: this reproduces the reported figures by re-running the
released validator (pyshacl) against the persisted corpus artefact. It does
NOT independently re-derive the corpus from primary CIC-IDS2018 predictions --
that would require re-running scripts/build_grc_artifact_corpus.py end to end
and is a stronger, unattempted check. A validator can report a clean number
over an artefact that was wrong at construction time (this project's own
history: the fabricated 85-RiskCase corpus validated cleanly because it was
never checked against how it was built). This script checks reproduction of
reported figures through the released tooling, not correctness of the
corpus's construction from source data.

Usage: python scripts/verify_deterministic_reproduction.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdflib import Graph, RDF, URIRef, OWL  # noqa: E402
from pyshacl import validate  # noqa: E402

from scripts.clause_context import with_clause_context  # noqa: E402


def count_ontology_terms():
    g = Graph()
    g.parse(ROOT / "ontology/ttl/grc-ontology.ttl", format="turtle")
    classes = set(g.subjects(RDF.type, OWL.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    dt_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    return len(classes), len(obj_props), len(dt_props)


def count_shapes():
    shape_pred = URIRef("http://www.w3.org/ns/shacl#NodeShape")
    counts = {}
    for f in sorted((ROOT / "ontology/shapes").glob("*.shacl.ttl")):
        g = Graph()
        g.parse(f, format="turtle")
        counts[f.name] = len(set(g.subjects(RDF.type, shape_pred)))
    return counts


def validate_corpus(with_context: bool):
    data = Graph()
    data.parse(ROOT / "results/grc_artifact_corpus.ttl", format="turtle")
    n_triples = len(data)

    shapes = Graph()
    for f in (ROOT / "ontology/shapes").glob("*.shacl.ttl"):
        shapes.parse(f, format="turtle")

    data_v = with_clause_context(data) if with_context else data

    target_pred = URIRef("http://www.w3.org/ns/shacl#targetClass")
    target_classes = {str(o) for s, p, o in shapes if p == target_pred}
    targeted = {str(s) for s, p, o in data if p == RDF.type and str(o) in target_classes}

    conforms, results_graph, _ = validate(
        data_v, shacl_graph=shapes, inference="none", abort_on_first=False)

    failing = {str(o) for s, p, o in results_graph if str(p).endswith("#focusNode")}
    msgs: dict[str, int] = {}
    for s, p, o in results_graph:
        if str(p).endswith("#resultMessage"):
            msgs[str(o)] = msgs.get(str(o), 0) + 1

    return {
        "with_clause_context": with_context,
        "triples": n_triples,
        "targeted": len(targeted),
        "clean": len(targeted) - len(failing),
        "violations": sum(msgs.values()),
        "violations_by_message": msgs,
        "conforming_share_pct": round(100 * (len(targeted) - len(failing)) / len(targeted), 1)
        if targeted else None,
    }


def main():
    n_classes, n_obj, n_dt = count_ontology_terms()
    shape_counts = count_shapes()
    n_shapes = sum(shape_counts.values())

    with_ctx = validate_corpus(with_context=True)
    without_ctx = validate_corpus(with_context=False)

    out = {
        "purpose": "Independent reproduction check of deterministic pipeline claims "
                   "(ontology/shape counts, corpus SHACL conformance). Reproduces "
                   "reported figures through the released validator against the "
                   "persisted corpus artefact -- does not independently re-derive "
                   "the corpus from primary CIC-IDS2018 data.",
        "ontology": {
            "classes": n_classes,
            "object_properties": n_obj,
            "datatype_properties": n_dt,
        },
        "shapes_by_file": shape_counts,
        "shapes_total": n_shapes,
        "corpus_validation_with_clause_context": with_ctx,
        "corpus_validation_without_clause_context": without_ctx,
        "reproducibility_note": (
            "Validating without merging the clause-individuals graph first "
            "(with_clause_context() in scripts/clause_context.py) yields a "
            "different, lower conformance figure -- both are internally "
            "consistent pyshacl runs, only one matches the pipeline as designed."
        ),
    }
    dest = ROOT / "results/deterministic_reproduction_check.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
