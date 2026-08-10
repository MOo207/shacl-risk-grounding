"""Negative controls: prove each repaired shape can actually report a violation.

The withdrawn results in this project all shared one root cause -- a constraint that
reported success while validating nothing. A repair that leaves violation counts unchanged
is therefore not self-evidently a repair: "0 violations because the data is clean" and
"0 violations because the constraint never fires" produce identical output.

This script distinguishes them. For each repaired constraint it builds a minimal graph
containing a deliberate defect, validates, and asserts the violation is reported. A shape
that stays silent on input built to break it is not running.

Covers:
  1. grc:VulnerabilityMustLinkToAssetConstraint  (target was sh:targetSubjectsOf
     grc:hasVulnerability, whose subjects are Assets, while the body filters
     $this a grc:Vulnerability -- no node could satisfy both)
  2. RiskScenarioShape clause join       (was an inlined sh:in literal list)
  3. ControlRecommendationShape clause join (second copy of the same list)

Also asserts the positive direction: a well-formed instance must NOT be rejected, so the
constraints are not merely failing everything.

A fourth case documents a known, NOT-closed gap rather than asserting the clause join is
airtight: the join queries grc:Clause individuals in the data graph (see clause_context.py's
TRUST BOUNDARY note), so a data graph that carries its own forged grc:Clause individual can
authorise its own otherwise-nonexistent clause citation. That case is expected to FAIL
closure (conforms=True on a graph that should not conform) and is reported as a known
limitation, not papered over by omitting it from this suite.

Output: results/shape_firing_verification.json
"""
import json
import os
import sys

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clause_context import with_clause_context

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SHAPES_DIR = os.path.join(BASE_DIR, "ontology", "shapes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

GRC = Namespace("https://w3id.org/grc/ontology#")


def load_shapes() -> Graph:
    g = Graph()
    for fn in sorted(os.listdir(SHAPES_DIR)):
        if fn.endswith(".shacl.ttl"):
            g.parse(os.path.join(SHAPES_DIR, fn), format="turtle")
    return g


def _messages(results_graph: Graph):
    return [str(o) for _, p, o in results_graph if str(p).endswith("#resultMessage")]


def run_case(name, build_fn, expect_violation, needle, shapes):
    """Validate one hand-built graph and check the expected verdict."""
    g = Graph()
    g.bind("grc", GRC)
    build_fn(g)
    conforms, rg, _ = validate(
        with_clause_context(g), shacl_graph=shapes, inference="none", abort_on_first=False
    )
    msgs = _messages(rg)
    hit = any(needle.lower() in m.lower() for m in msgs)
    ok = (hit == expect_violation)
    return {
        "case": name,
        "expected_violation": expect_violation,
        "constraint_reported": hit,
        "passed": ok,
        "conforms": bool(conforms),
        "messages": msgs,
    }


# --- 1. Dangling vulnerability: no asset links to it -------------------------

def build_dangling_vuln(g):
    v = GRC["CVE-0000-0001"]
    g.add((v, RDF.type, GRC.Vulnerability))
    g.add((v, GRC.cveId, Literal("CVE-0000-0001")))
    g.add((v, GRC.cweId, Literal("CWE-79")))
    g.add((v, GRC.severity, Literal("high")))
    g.add((v, GRC.provenanceSource, Literal("negative-control")))
    g.add((v, GRC.provenanceTimestamp,
           Literal("2026-01-01T00:00:00", datatype=XSD.dateTime)))


def build_linked_vuln(g):
    """Same vulnerability, but an Asset points at it. Must NOT trip the orphan check."""
    build_dangling_vuln(g)
    a = GRC["SRV-negctl"]
    g.add((a, RDF.type, GRC.Asset))
    g.add((a, GRC.assetId, Literal("SRV-negctl")))
    g.add((a, GRC.assetName, Literal("negative control host")))
    g.add((a, GRC.assetType, Literal("hardware")))
    g.add((a, GRC.provenanceSource, Literal("negative-control")))
    g.add((a, GRC.provenanceTimestamp,
           Literal("2026-01-01T00:00:00", datatype=XSD.dateTime)))
    g.add((a, GRC.hasVulnerability, GRC["CVE-0000-0001"]))


# --- 2/3. Clause references: real vs plausible-but-nonexistent ---------------

def _scenario(g, clause):
    s = GRC["scenario-negctl"]
    g.add((s, RDF.type, GRC.RiskScenario))
    g.add((s, GRC.scenarioTitle, Literal("negative control")))
    g.add((s, GRC.description, Literal("negative control body")))
    g.add((s, GRC.threatVector, Literal("DDoS")))
    g.add((s, GRC.impactLevel, Literal("high")))
    g.add((s, GRC.recommendedControl, Literal("some control")))
    g.add((s, GRC.nfcrmClause, Literal(clause)))
    a = GRC["SRV-negctl"]
    g.add((a, RDF.type, GRC.Asset))
    g.add((s, GRC.affectedAsset, a))


def build_scenario_bogus_clause(g):
    # Well-formed under the retired ^SS[56]\.[0-9]+$ pattern, absent from the standard.
    _scenario(g, "SS5.999")


def build_scenario_real_clause(g):
    _scenario(g, "SS6.7")


def _recommendation(g, clause):
    r = GRC["rec-negctl"]
    g.add((r, RDF.type, GRC.ControlRecommendation))
    g.add((r, GRC.threatClass, Literal("DDoS")))
    g.add((r, GRC.justification, Literal("negative control")))
    g.add((r, GRC.controlClauseId, Literal(clause)))
    a = GRC["SRV-negctl"]
    g.add((a, RDF.type, GRC.Asset))
    g.add((r, GRC.targetAsset, a))


def build_rec_bogus_clause(g):
    _recommendation(g, "SS6.99")


def build_rec_real_clause(g):
    _recommendation(g, "SS6.7")


# --- 4. NFCRMControl clause references, long notation ------------------------

def _nfcrm_control(g, clause):
    c = GRC["NFCRM-negctl"]
    g.add((c, RDF.type, GRC.NFCRMControl))
    g.add((c, RDF.type, GRC.Control))
    g.add((c, GRC.controlId, Literal("NFCRM-negctl")))
    g.add((c, GRC.controlName, Literal("negative control")))
    g.add((c, GRC.nfcrmClause, Literal(clause)))
    g.add((c, GRC.provenanceSource, Literal("negative-control")))
    g.add((c, GRC.provenanceTimestamp,
           Literal("2026-01-01T00:00:00", datatype=XSD.dateTime)))


def build_control_bogus_clause(g):
    # Well-formed under the retired pattern, absent from the standard.
    _nfcrm_control(g, "NFCRM-1:2025-§99.99")


def build_control_real_clause(g):
    _nfcrm_control(g, "NFCRM-1:2025-§6.7")


# --- Known-gap case: forged clause authority in the data graph ---------------

def build_forged_clause_authority(g):
    """A nonexistent clause, plus a forged grc:Clause individual claiming it.

    Both sit in the same graph the join queries. Expected result: the join
    resolves against the forged individual and the constraint does NOT fire --
    documenting the trust-boundary gap described in clause_context.py, not
    demonstrating it is closed.
    """
    _scenario(g, "SS9.999")
    forged = GRC["forged-authority"]
    g.add((forged, RDF.type, GRC.Clause))
    g.add((forged, GRC.clauseId, Literal("SS9.999")))


def main():
    shapes = load_shapes()
    orphan = "must be linked to at least one asset"
    clause_scn = "is not a clause of NFCRM-1:2025"

    cases = [
        run_case("orphan_vulnerability_is_caught", build_dangling_vuln, True, orphan, shapes),
        run_case("linked_vulnerability_is_accepted", build_linked_vuln, False, orphan, shapes),
        run_case("scenario_nonexistent_clause_is_caught", build_scenario_bogus_clause, True, clause_scn, shapes),
        run_case("scenario_real_clause_is_accepted", build_scenario_real_clause, False, clause_scn, shapes),
        run_case("recommendation_nonexistent_clause_is_caught", build_rec_bogus_clause, True, clause_scn, shapes),
        run_case("recommendation_real_clause_is_accepted", build_rec_real_clause, False, clause_scn, shapes),
        run_case("nfcrm_control_nonexistent_clause_is_caught", build_control_bogus_clause, True, clause_scn, shapes),
        run_case("nfcrm_control_real_clause_is_accepted", build_control_real_clause, False, clause_scn, shapes),
    ]

    passed = sum(1 for c in cases if c["passed"])

    # Documented, NOT-closed gap: run separately from the pass/fail count above.
    # This case is EXPECTED to show the constraint failing to fire -- the point
    # is to demonstrate the trust-boundary gap is real, not to assert it is fixed.
    forged = run_case("KNOWN_GAP_forged_clause_authority_is_not_caught",
                       build_forged_clause_authority, True, clause_scn, shapes)
    gap_confirmed = not forged["passed"]  # confirmed present iff the constraint did NOT fire

    out = {
        "purpose": "Negative controls proving each repaired shape reports violations on input built to break it.",
        "shape_triples": len(shapes),
        "cases_total": len(cases),
        "cases_passed": passed,
        "all_passed": passed == len(cases),
        "cases": cases,
        "known_gaps": {
            "forged_clause_authority": {
                "description": ("The clause join queries grc:Clause individuals in the "
                                 "data graph (see clause_context.py TRUST BOUNDARY note). "
                                 "A data graph carrying its own forged grc:Clause individual "
                                 "authorises its own nonexistent clause citation."),
                "gap_confirmed_present": gap_confirmed,
                "detail": forged,
            }
        },
    }
    dest = os.path.join(RESULTS_DIR, "shape_firing_verification.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    for c in cases:
        flag = "PASS" if c["passed"] else "FAIL"
        print(f"  [{flag}] {c['case']}: expected_violation={c['expected_violation']} "
              f"reported={c['constraint_reported']}")
    print(f"\n{passed}/{len(cases)} negative controls passed -> {dest}")
    print(f"  [KNOWN GAP] forged clause authority not caught: "
          f"{'confirmed present' if gap_confirmed else 'NOT reproduced -- investigate'}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
