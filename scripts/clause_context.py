"""Load the NFCRM-1:2025 clause set into a data graph before SHACL validation.

REPRODUCIBILITY WARNING: validating results/grc_artifact_corpus.ttl (or any
corpus using the clause-bearing shapes) through pyshacl WITHOUT first calling
with_clause_context() below will run without error and report a different,
lower conformance figure -- 93.7% instead of the paper's reported 94.4% on
the released corpus -- because 4 clause-reference checks fail to resolve
their join with no clause individuals present to join against. Both numbers
look equally plausible in isolation; only 94.4% matches the pipeline as
designed and as every other script in this project actually runs it. Always
call with_clause_context(data_graph) immediately before validate().

The two clause-bearing shapes (grc:RiskScenarioShape, grc:ControlRecommendationShape)
validate clause references by joining against the grc:Clause individuals declared in
ontology/ttl/nfcrm-clauses.ttl, rather than by carrying their own inlined copy of the
26 identifiers. A SHACL SPARQL constraint queries the *data* graph, so those individuals
must be present in the graph being validated for the join to resolve.

The constraint is fail-closed by construction: with the clause set absent, every clause
reference reports a violation rather than silently passing. That direction is deliberate.
A validator wired up wrong announces itself instead of conforming vacuously, which is the
failure mode that produced the withdrawn results this constraint replaces.

TRUST BOUNDARY (not closed): the clause-bearing SHACL shapes join against
`grc:Clause` individuals in the *data* graph, because a `sh:sparql` constraint
queries the data graph, not a separate authority graph. The 26 real clause
individuals are merged into that same data graph by this module. Nothing
currently prevents an untrusted data graph from carrying its own `grc:Clause`
individual with a fabricated `grc:clauseId`, which would then satisfy the join
for that fabricated identifier -- the exact "plausible but nonexistent
citation" this constraint exists to reject. A named-graph isolation was
attempted (restricting the join to a distinct graph the data cannot write
into, via rdflib Dataset + SPARQL GRAPH) and abandoned: pyshacl's sh:sparql
execution did not reliably resolve GRAPH-scoped patterns even for a
legitimately valid case, so shipping it would have traded a narrow, disclosed
authenticity gap for a broken constraint. As deployed, the only producer of
RDF that reaches this validator is this project's own converter, which never
emits a `grc:Clause` individual, so the gap is not currently exploited. It
would be exploited by any future producer whose output is not likewise
trusted, and `scripts/verify_shape_firing.py` carries a negative control that
demonstrates the gap is real rather than asserting it is closed.

Callers should add the clause context immediately before validate(), and report their own
triple counts from the pre-context graph so the census stays comparable across revisions.
"""
import os

from rdflib import Graph

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CLAUSE_TTL = os.path.join(BASE_DIR, "ontology", "ttl", "nfcrm-clauses.ttl")


def clause_graph() -> Graph:
    """Parse the clause set on its own (26 grc:Clause individuals plus schema)."""
    g = Graph()
    g.parse(CLAUSE_TTL, format="turtle")
    return g


def with_clause_context(data_graph: Graph) -> Graph:
    """Return a new graph holding data_graph's triples plus the clause set.

    Returns a copy rather than mutating in place so a caller's own triple census,
    node counts, and serialisations are unaffected by validation context.
    """
    g = Graph()
    for triple in data_graph:
        g.add(triple)
    for triple in clause_graph():
        g.add(triple)
    for prefix, ns in data_graph.namespaces():
        g.bind(prefix, ns)
    return g


def clause_count() -> int:
    """Number of grc:Clause individuals available to the join."""
    from rdflib import RDF, URIRef
    g = clause_graph()
    clause_cls = URIRef("https://w3id.org/grc/ontology#Clause")
    return sum(1 for _ in g.subjects(RDF.type, clause_cls))
