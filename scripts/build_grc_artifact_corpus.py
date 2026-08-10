"""Generate the GRC artefact corpus from real CIC-IDS2018 cases, and validate it.

Why this replaces the previous corpus
-------------------------------------
The figures previously reported for this corpus -- 85 RiskCases, 85 AuditLogs,
2 Controls, 172/172 SHACL conformance, 100% clause coverage, 57.6% ISO coverage
-- were not measurements.

Traced to origin (commit 21a7569, scripts/compute_grc_metrics.py, since
replaced): the script called _generate_test_events(100), which fabricated 100
dictionaries with attack_types[i % 7] and byte counts 1000 + (i*100) % 10000,
reading no dataset. Its _run_ml_inference() ran no model:

    random.seed(hash(event['event_id']))
    anomaly_score = random.uniform(0.6, 0.95)   # "Simulate ML inference"

Benign being 1 of 7 rotated classes, 100 events yielded exactly 85 RiskCases and
85 AuditLogs; 85 + 85 + 2 = 172. Conformance was checked by a Python function
whose own comment read "simplified - full validation would use pyshacl with
RDF". The artefacts were never written to disk. The successor script read those
numbers back out of its own output file and rewrote them, with identical
hardcoded fallbacks, so they propagated unchanged. The ISO figure was
literally `iso_val = round(iso_total * 0.582)`.

What this script does instead
-----------------------------
Drives the real GRCSymbolicRules engine over the 180 real CIC-IDS2018 cases in
results/unified_ablation/test_set.json -- real flow features, real asset
criticality, real paired CVEs, and the real stored XGBoost prediction and
confidence for each case. Risk signals are derived deterministically from that
case data; nothing is sampled. Every artefact produced is written to disk and
validated against ontology/shapes/ with pyshacl.

The resulting counts will not match 85/85/2. That is the point: they are
measured rather than inherited.

Usage: python scripts/build_grc_artifact_corpus.py
Output: results/grc_artifact_corpus.json
        results/grc_artifact_corpus_validation.json
"""
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(BASE_DIR))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdflib import Graph, Literal, Namespace, RDF, XSD  # noqa: E402
from pyshacl import validate  # noqa: E402

from clause_context import with_clause_context  # noqa: E402

from pipeline.inference.ml_inference_orchestrator import RiskSignal  # noqa: E402
from pipeline.post_inference.grc_symbolic_rules import GRCSymbolicRules  # noqa: E402

RESULTS_DIR = os.path.join(BASE_DIR, "results")
SHAPES_DIR = os.path.join(BASE_DIR, "ontology", "shapes")
TEST_SET = os.path.join(RESULTS_DIR, "unified_ablation", "test_set.json")

GRC = Namespace("https://w3id.org/grc/ontology#")

CRITICALITY_WEIGHT = {"Critical": 1.00, "High": 0.85, "Medium": 0.60, "Low": 0.35}

# The GRCSymbolicRules engine and NFCRM-1:2025 §6.9 use different risk-level
# vocabularies: the engine emits RiskLevel.{LOW,MEDIUM,HIGH,CRITICAL} (four
# levels, no Very Low), while §6.9 bands Likelihood x Impact into Very Low
# through Catastrophic (five levels, no "critical"). Nothing reconciled them
# because neither was ever validated. The standard is authoritative here, so the
# engine's output is mapped onto its bands and the mismatch is reported rather
# than hidden by widening the shape to accept both vocabularies.
ENGINE_TO_NFCRM_BAND = {
    "low": "low", "medium": "medium", "high": "high", "critical": "catastrophic",
}


def signals_for_case(case):
    """Derive risk signals from real case data. Deterministic; no sampling.

    Three signal sources, each grounded in a recorded value:
      * the classifier's own prediction and confidence for this case
      * the CVSS base score of the paired CVE, where the case has one
      * the asset's recorded criticality
    """
    asset = case["asset"]
    pred = case.get("xgb_predicted_class", "Benign")
    conf = float(case.get("xgb_confidence") or 0.0)
    signals = []

    if pred != "Benign":
        # Control keyword matching in the rules engine is description-driven,
        # so the description names the access-control and monitoring concepts
        # the NFCRM control catalogue is keyed on.
        signals.append(RiskSignal(
            type="anomaly", source="anomaly_detection", score=round(conf, 4),
            description=(f"Access control anomaly: classifier identified {pred} "
                         f"traffic targeting {asset['hostname']} "
                         f"({asset['asset_type']}, {asset['os']})"),
            entity_type="asset", entity_id=asset["id"], confidence=round(conf, 4),
            tags=["classification", pred.lower()],
            metadata={"case_id": case["case_id"], "predicted_class": pred,
                      "classifier": "xgboost"}))

    cve = case.get("paired_cve")
    if cve and cve.get("cvss_v3"):
        cvss = float(cve["cvss_v3"])
        signals.append(RiskSignal(
            type="risk", source="probabilistic_risk", score=round(cvss / 10.0, 4),
            description=(f"Vulnerability management exposure: {cve['id']} "
                         f"(CVSS {cvss}) affects {asset['hostname']}. "
                         f"{cve.get('description', '')}"),
            entity_type="asset", entity_id=asset["id"], confidence=1.0,
            tags=["vulnerability", "cve"],
            metadata={"case_id": case["case_id"], "cve_id": cve["id"], "cvss": cvss}))

    crit = asset.get("criticality", "Low")
    signals.append(RiskSignal(
        type="behavioral", source="graph_scoring",
        score=CRITICALITY_WEIGHT.get(crit, 0.35),
        description=(f"Asset monitoring and logging context: {asset['hostname']} "
                     f"is classified {crit} criticality in the asset inventory"),
        entity_type="asset", entity_id=asset["id"], confidence=1.0,
        tags=["criticality", crit.lower()],
        metadata={"case_id": case["case_id"], "criticality": crit}))

    return signals


# Source cases record a free-text asset_type ("server", "workstation",
# "router", ...) that does not match AssetShape's sh:in enumeration ("software",
# "hardware", "network"). Earlier revisions of this script emitted no Asset node at
# all, which sidestepped the mismatch by omission -- the resulting corpus then
# validated with zero orphan-vulnerability violations only because there was no
# asset layer for the check to run against, not because no vulnerability was
# orphaned. See Section 3.5 of the paper for the corrected figures this produces.
ASSET_TYPE_MAP = {
    "server": "hardware", "workstation": "hardware", "router": "network",
    "switch": "network", "firewall": "network", "database": "software",
}


def _map_asset_type(raw):
    return ASSET_TYPE_MAP.get((raw or "").lower(), "hardware")


def artefacts_to_rdf(risk_cases, audit_logs, controls):
    """Serialise the corpus using the ontology's own vocabulary.

    Emits one grc:Asset node per distinct asset_id referenced by a RiskCase, with
    a grc:hasVulnerability edge to every CVE that RiskCase names. Earlier revisions
    of this function omitted the asset layer entirely, which meant every CVE node
    in the corpus was orphaned by construction rather than by any property of the
    generated artefacts -- the orphan-vulnerability check was passing vacuously on
    a graph with nothing for it to validate, the same failure class reported for
    the shape library itself in Section 3.5.
    """
    g = Graph()
    g.bind("grc", GRC)

    def lit(s, p, v):
        if v not in (None, ""):
            g.add((s, p, Literal(str(v))))

    assets_emitted = set()
    for rc in risk_cases:
        asset_id = rc.get("asset_id")
        if asset_id and asset_id not in assets_emitted:
            assets_emitted.add(asset_id)
            an = GRC[asset_id]
            g.add((an, RDF.type, GRC.Asset))
            lit(an, GRC.assetId, asset_id)
            lit(an, GRC.assetName, rc.get("asset_hostname") or asset_id)
            lit(an, GRC.assetType, _map_asset_type(rc.get("asset_type")))
            lit(an, GRC.hostname, rc.get("asset_hostname"))
            lit(an, GRC.osName, rc.get("asset_os"))
            lit(an, GRC.provenanceSource, "unified_ablation/test_set.json")
            g.add((an, GRC.provenanceTimestamp,
                   Literal(rc["provenance_timestamp"], datatype=XSD.dateTime)))
        for cve in rc.get("related_cves", []):
            if asset_id and cve.get("id"):
                g.add((GRC[asset_id], GRC.hasVulnerability, GRC[cve["id"]]))

    for rc in risk_cases:
        n = GRC[rc["risk_id"]]
        g.add((n, RDF.type, GRC.RiskCase))
        lit(n, GRC.riskId, rc["risk_id"])
        lit(n, GRC.riskLevel, rc["risk_level_nfcrm_band"])
        lit(n, GRC.likelihood, rc["likelihood"])
        lit(n, GRC.impact, rc["impact"])
        # Decimal, not float: rdflib retains the Python value behind the literal
        # and pyshacl's sh:datatype check inspects it, so Literal(0.79,
        # datatype=XSD.decimal) fails xsd:decimal despite an identical lexical form.
        g.add((n, GRC.confidence,
               Literal(Decimal(str(round(float(rc["confidence"]), 4))),
                       datatype=XSD.decimal)))
        lit(n, GRC.provenanceSource, rc["provenance_source"])
        g.add((n, GRC.provenanceTimestamp,
               Literal(rc["provenance_timestamp"], datatype=XSD.dateTime)))
        for cd in rc["control_details"]:
            g.add((n, GRC.associatedWithControl, GRC[cd["control_id"]]))
        if rc.get("audit_id"):
            g.add((n, GRC.auditedBy, GRC[rc["audit_id"]]))
        for cve in rc.get("related_cves", []):
            v = GRC[cve["id"]]
            g.add((v, RDF.type, GRC.Vulnerability))
            lit(v, GRC.cveId, cve["id"])
            lit(v, GRC.cweId, cve.get("cwe"))
            if cve.get("cvss_v3") is not None:
                g.add((v, GRC.cvssScore,
                       Literal(Decimal(str(round(float(cve["cvss_v3"]), 1))),
                               datatype=XSD.decimal)))
            lit(v, GRC.severity, (cve.get("severity") or "").lower() or None)
            lit(v, GRC.provenanceSource, "NVD")
            g.add((v, GRC.provenanceTimestamp,
                   Literal(rc["provenance_timestamp"], datatype=XSD.dateTime)))
            g.add((v, GRC.relatesToRisk, n))

    for al in audit_logs:
        n = GRC[al["audit_id"]]
        g.add((n, RDF.type, GRC.AuditLog))
        lit(n, GRC.auditor, al["auditor"])
        g.add((n, GRC.auditTimestamp,
               Literal(al["audit_timestamp"], datatype=XSD.dateTime)))
        lit(n, GRC.provenanceSource, al["provenance_source"])
        g.add((n, GRC.provenanceTimestamp,
               Literal(al["provenance_timestamp"], datatype=XSD.dateTime)))

    for c in controls:
        n = GRC[c["control_id"]]
        g.add((n, RDF.type, GRC.Control))
        if c.get("nfcrm_clause"):
            g.add((n, RDF.type, GRC.NFCRMControl))
            lit(n, GRC.nfcrmClause, c["nfcrm_clause"])
        if c.get("iso27005_clause"):
            g.add((n, RDF.type, GRC.ISO27005Control))
            lit(n, GRC.iso27005Clause, c["iso27005_clause"])
        lit(n, GRC.controlId, c["control_id"])
        lit(n, GRC.controlName, c["control_name"])
        lit(n, GRC.provenanceSource, c["provenance_source"])
        g.add((n, GRC.provenanceTimestamp,
               Literal(c["provenance_timestamp"], datatype=XSD.dateTime)))
    return g


def main():
    with open(TEST_SET, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    nvd_cache = {}
    for fname in ("nvd_arg_cves.json", "nvd_kev_pool.json"):
        path = os.path.join(BASE_DIR, "data", "external", fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                nvd_cache.update(json.load(f).get("cves", {}))

    rules = GRCSymbolicRules()
    stamp = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None).isoformat()

    risk_cases, audit_logs, controls_seen = [], [], {}
    no_signal_cases, no_control_cases = [], []

    for case in cases:
        signals = signals_for_case(case)
        if not signals:
            no_signal_cases.append(case["case_id"])
            continue

        rc = rules.create_risk_case(signals, event_context={
            "case_id": case["case_id"],
            "attack_type": case.get("xgb_predicted_class"),
            "asset_criticality": case["asset"].get("criticality"),
        })
        al = rules.create_audit_log(
            risk_case_id=rc.riskId, event_type="risk_assessment",
            auditor="intersymbolic_pipeline",
            description=(f"Risk assessment for case {case['case_id']} on asset "
                         f"{case['asset']['id']}"),
            details={"case_id": case["case_id"],
                     "predicted_class": case.get("xgb_predicted_class"),
                     "true_class": case.get("true_attack_class"),
                     "classifier_confidence": case.get("xgb_confidence")})

        control_details = []
        for cid in rc.associatedControls:
            ctl = (rules.NFCRM_CONTROLS.get(cid) or rules.ISO27005_CONTROLS.get(cid))
            if not ctl:
                continue
            rec = {
                "control_id": ctl.controlId, "control_name": ctl.controlName,
                "provenance_source": ctl.provenanceSource,
                "nfcrm_clause": ctl.nfcrmClause, "iso27005_clause": ctl.iso27005Clause,
                "severity": ctl.severity, "description": ctl.description,
                "provenance_timestamp": stamp,
            }
            controls_seen[ctl.controlId] = rec
            control_details.append(rec)
        if not control_details:
            no_control_cases.append(case["case_id"])

        cves = []
        if case.get("paired_cve"):
            # The test set's paired-CVE records carry id/CVSS/description but
            # empty cwe and severity. NVD is authoritative for those, so they are
            # filled from the fetched cache rather than left blank or invented.
            cve = dict(case["paired_cve"])
            auth = nvd_cache.get(cve["id"], {})
            if auth:
                cve["cwe"] = auth.get("cwe") or cve.get("cwe", "")
                cve["severity"] = auth.get("severity") or cve.get("severity", "")
                cve["cvss_v3"] = auth.get("cvss_score", cve.get("cvss_v3"))
                cve["cvss_version"] = auth.get("cvss_version", "")
            cves.append(cve)

        risk_cases.append({
            "risk_id": rc.riskId, "case_id": case["case_id"],
            "risk_level_engine": rc.riskLevel.value,
            "risk_level_nfcrm_band": ENGINE_TO_NFCRM_BAND.get(
                rc.riskLevel.value, rc.riskLevel.value),
            "likelihood": rc.likelihood.value,
            "impact": rc.impact.value, "confidence": rc.confidence,
            "associated_controls": rc.associatedControls,
            "control_details": control_details,
            "audit_id": al.auditId,
            "related_cves": cves,
            "asset_id": case["asset"]["id"],
            "asset_hostname": case["asset"].get("hostname"),
            "asset_type": case["asset"].get("asset_type"),
            "asset_os": case["asset"].get("os"),
            "asset_criticality": case["asset"].get("criticality"),
            "predicted_class": case.get("xgb_predicted_class"),
            "true_class": case.get("true_attack_class"),
            "ground_truth_risk_level": case.get("ground_truth_risk_level"),
            "signal_count": len(signals),
            "provenance_source": rc.provenanceSource,
            "provenance_timestamp": stamp,
        })
        audit_logs.append({
            "audit_id": al.auditId, "risk_case_id": rc.riskId,
            "audit_timestamp": stamp, "auditor": al.auditor,
            "provenance_source": al.provenanceSource,
            "provenance_timestamp": stamp,
            "event_type": al.eventType, "description": al.description,
            "details": al.details,
        })

    controls = sorted(controls_seen.values(), key=lambda c: c["control_id"])

    # ---- clause coverage, measured from the artefacts themselves ----
    with_nfcrm = [r for r in risk_cases
                  if any(c.get("nfcrm_clause") for c in r["control_details"])]
    with_iso = [r for r in risk_cases
                if any(c.get("iso27005_clause") for c in r["control_details"])]
    nfcrm_clauses = sorted({c["nfcrm_clause"] for c in controls if c.get("nfcrm_clause")})
    iso_clauses = sorted({c["iso27005_clause"] for c in controls if c.get("iso27005_clause")})

    corpus = {
        "purpose": "GRC artefact corpus generated by the real GRCSymbolicRules "
                   "engine over the 180 real CIC-IDS2018 cases of "
                   "results/unified_ablation/test_set.json. Supersedes the "
                   "previously reported 85/85/2 corpus, which was produced from "
                   "fabricated events with a seeded RNG standing in for ML "
                   "inference and was never written to disk.",
        "generated": stamp,
        "source_cases": len(cases),
        "signal_derivation": "Deterministic from recorded case data: stored XGBoost "
                             "prediction and confidence, paired-CVE CVSS base score, "
                             "and asset criticality weight. No sampling.",
        "counts": {
            "risk_cases": len(risk_cases),
            "audit_logs": len(audit_logs),
            "distinct_controls": len(controls),
            "total_artefacts": len(risk_cases) + len(audit_logs) + len(controls),
            "cases_yielding_no_signal": len(no_signal_cases),
            "cases_yielding_no_control": len(no_control_cases),
        },
        "clause_coverage": {
            "nfcrm_riskcases_with_clause": len(with_nfcrm),
            "nfcrm_coverage_percent": round(100.0 * len(with_nfcrm) / max(len(risk_cases), 1), 1),
            "iso27005_riskcases_with_clause": len(with_iso),
            "iso27005_coverage_percent": round(100.0 * len(with_iso) / max(len(risk_cases), 1), 1),
            "distinct_nfcrm_clauses": nfcrm_clauses,
            "distinct_iso27005_clauses": iso_clauses,
        },
        "risk_level_distribution_engine": dict(
            Counter(r["risk_level_engine"] for r in risk_cases)),
        "risk_level_distribution_nfcrm_band": dict(
            Counter(r["risk_level_nfcrm_band"] for r in risk_cases)),
        "vocabulary_note": (
            "GRCSymbolicRules emits four levels (low/medium/high/critical); "
            "NFCRM-1:2025 §6.9 bands are Very Low/Low/Medium/High/Catastrophic. "
            "The engine's 'critical' is mapped to the standard's 'catastrophic'; "
            "the engine never emits 'very low'. The mismatch survived because "
            "neither vocabulary was ever SHACL-validated."),
        "controls": controls,
        "risk_cases": risk_cases,
        "audit_logs": audit_logs,
    }

    dest = os.path.join(RESULTS_DIR, "grc_artifact_corpus.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    # ---- real SHACL validation over the persisted corpus ----
    shapes = Graph()
    for fn in sorted(os.listdir(SHAPES_DIR)):
        if fn.endswith(".shacl.ttl"):
            shapes.parse(os.path.join(SHAPES_DIR, fn), format="turtle")

    data = artefacts_to_rdf(risk_cases, audit_logs, controls)
    conforms, rg, _ = validate(with_clause_context(data), shacl_graph=shapes, inference="none",
                               abort_on_first=False)
    msgs = Counter(str(o) for s, p, o in rg if str(p).endswith("#resultMessage"))
    failing = {str(o) for s, p, o in rg if str(p).endswith("#focusNode")}

    target_classes = {str(o) for s, p, o in shapes
                      if str(p) == "http://www.w3.org/ns/shacl#targetClass"}
    targeted = {str(s) for s, p, o in data
                if p == RDF.type and str(o) in target_classes}

    ttl_dest = os.path.join(RESULTS_DIR, "grc_artifact_corpus.ttl")
    data.serialize(destination=ttl_dest, format="turtle")

    total_artefacts = len(risk_cases) + len(audit_logs) + len(controls)
    validation = {
        "purpose": "Real pyshacl validation of the persisted GRC artefact corpus "
                   "against ontology/shapes/. Replaces a Python stand-in whose own "
                   "comment read 'simplified - full validation would use pyshacl "
                   "with RDF' and which validated artefacts built from the same "
                   "template that produced them.",
        "corpus_file": "results/grc_artifact_corpus.json",
        "rdf_file": "results/grc_artifact_corpus.ttl",
        "rdf_triples": len(data),
        "shape_triples": len(shapes),
        "total_artefacts": total_artefacts,
        "artefacts_targeted_by_a_shape": len(targeted),
        "conforms": bool(conforms),
        "total_violations": sum(msgs.values()),
        "artefacts_with_violations": len(failing),
        "artefacts_clean": len(targeted) - len(failing),
        "conforming_share_of_targeted_percent": round(
            100.0 * (len(targeted) - len(failing)) / max(len(targeted), 1), 1),
        "violations_by_message": dict(msgs.most_common()),
    }
    vdest = os.path.join(RESULTS_DIR, "grc_artifact_corpus_validation.json")
    with open(vdest, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    c = corpus["counts"]
    cc = corpus["clause_coverage"]
    print(f"cases in                : {len(cases)}")
    print(f"RiskCases               : {c['risk_cases']}")
    print(f"AuditLogs               : {c['audit_logs']}")
    print(f"distinct Controls       : {c['distinct_controls']}")
    print(f"total artefacts         : {c['total_artefacts']}")
    print(f"cases with no control   : {c['cases_yielding_no_control']}")
    print(f"NFCRM clause coverage   : {cc['nfcrm_riskcases_with_clause']}"
          f"/{c['risk_cases']} = {cc['nfcrm_coverage_percent']}%")
    print(f"ISO 27005 clause cover  : {cc['iso27005_riskcases_with_clause']}"
          f"/{c['risk_cases']} = {cc['iso27005_coverage_percent']}%")
    print(f"risk levels (engine)    : {corpus['risk_level_distribution_engine']}")
    print(f"risk levels (NFCRM band): {corpus['risk_level_distribution_nfcrm_band']}")
    print(f"\nSHACL conforms          : {validation['conforms']}")
    print(f"  triples               : {validation['rdf_triples']}")
    print(f"  targeted artefacts    : {validation['artefacts_targeted_by_a_shape']}")
    print(f"  violations            : {validation['total_violations']} "
          f"over {validation['artefacts_with_violations']} artefacts")
    for m, n in list(validation["violations_by_message"].items())[:8]:
        print(f"      {n:5d}  {m[:84]}")
    print(f"\n  -> {dest}\n  -> {vdest}\n  -> {ttl_dest}")


if __name__ == "__main__":
    main()
