# GRC Ontology and SHACL Shapes

This directory contains the ontology and SHACL validation rules for the INTERSYMBOLIC-GRC framework.

## Directory Structure

```
ontology/
├── README.md                    # This file
├── ttl/                         # OWL/RDF ontology files (Turtle format)
│   └── grc-ontology.ttl        # Core GRC ontology with classes and properties
├── shapes/                      # SHACL shapes for validation
│   ├── technical-entities.shacl.ttl    # SHACL shapes for Assets, Vulnerabilities
│   └── grc-entities.shacl.ttl          # SHACL shapes for Controls, RiskCases, AuditLogs
├── load-neo4j.sh              # Script to load SHACL shapes into Neo4j
└── example-data/              # Example RDF data for testing (optional)
```

## Overview

### Core Ontology (grc-ontology.ttl)

Defines OWL classes and properties representing:

**Technical Entities:**
- `Asset` (base class)
  - `SoftwareAsset`
  - `HardwareAsset`
  - `NetworkAsset`
- `Vulnerability`
- `Control` (base class)
  - `NFCRMControl` (aligned with NFCRM-1:2025)
  - `ISO27005Control` (aligned with ISO/IEC 27005)

**GRC Entities:**
- `RiskCase`
- `AuditLog`
- `ContinuityPlan`

**Event and Context:**
- `SecurityEvent`
- `Threat`

**Relationships (Object Properties):**
- `hasVulnerability` (Asset → Vulnerability)
- `hasControl` (Asset → Control)
- `mitigates` (Control → Vulnerability)
- `relatesToRisk` (Vulnerability → RiskCase)
- `triggersEvent` (Vulnerability → SecurityEvent)
- `exploits` (Threat → Vulnerability)
- `auditedBy` (RiskCase → AuditLog)
- `linkedToContinuityPlan` (RiskCase → ContinuityPlan)
- `associatedWithControl` (RiskCase → Control)

**Attributes (Data Properties):**
- Asset: `assetId`, `assetName`, `assetType`, `provenanceSource`, `provenanceTimestamp`
- Vulnerability: `cveId`, `cweId`, `cvssScore`, `severity`, `provenanceSource`, `provenanceTimestamp`
- Control: `controlId`, `controlName`, `nfcrmClause`, `iso27005Clause`, `provenanceSource`, `provenanceTimestamp`
- RiskCase: `riskId`, `riskLevel`, `likelihood`, `impact`, `confidence`, `provenanceSource`, `provenanceTimestamp`
- AuditLog: `auditTimestamp`, `auditor`, `provenanceSource`, `provenanceTimestamp`

### SHACL Validation Rules

**technical-entities.shacl.ttl:**
- Validates Asset properties (assetId, assetName, assetType, provenance)
- Validates Vulnerability properties (cveId format, cweId format, cvssScore range, severity values)
- Constraint: Every Vulnerability must be linked to at least one Asset

**grc-entities.shacl.ttl:**
- Validates Control properties (controlId, controlName, standards clauses)
- Validates NFCRMControl (clause format pattern: NFCRM-1:2025-NN[.NN])
- Validates ISO27005Control (clause format pattern: ISO/IEC 27005:NN[.NN])
- Validates RiskCase properties (riskId, riskLevel, likelihood, impact, confidence)
- Constraint: Every Control must either mitigate a Vulnerability or be associated with a RiskCase
- Constraint: Every RiskCase must be linked to at least one Vulnerability or SecurityEvent
- Constraint: Every RiskCase must have at least one associated AuditLog
- Validates AuditLog properties (auditTimestamp, auditor, provenance)

## NFCRM-1:2025 Alignment

Controls mapped to NFCRM-1:2025 clauses:
- Format: `NFCRM-1:2025-<clause-number>`
- Example: `NFCRM-1:2025-6.1` (Access Control)
- Provides direct traceability from technical controls to compliance requirements

## ISO/IEC 27005 Alignment

Controls mapped to ISO/IEC 27005 clauses:
- Format: `ISO/IEC 27005:<clause-number>`
- Example: `ISO/IEC 27005:6.2` (Risk Assessment)
- Aligns with international risk management standards

## Loading into Neo4j

### Prerequisites

1. Neo4j 4.x or 5.x installed and running
2. neosemantics plugin installed and configured
3. RDF data files ready for ingestion

### Using neosemantics Plugin

The neosemantics plugin provides procedures to:
- Load RDF data into Neo4j as property graphs
- Apply SHACL validation
- Query using Cypher with RDF semantics

### Load Script

See `load-neo4j.sh` for an example script to load ontology and SHACL shapes into Neo4j.

## Usage

### 1. Define Your Data

Create RDF data (Turtle format) representing your assets, vulnerabilities, controls, and risk cases.

Example:
```turtle
@prefix grc: <https://w3id.org/grc/ontology#> .

grc:asset-001 a grc:SoftwareAsset ;
    grc:assetId "asset-001" ;
    grc:assetName "Web Server" ;
    grc:assetType "software" ;
    grc:hasVulnerability grc:vuln-CVE-2024-1234 ;
    grc:hasControl grc:ctrl-NFCRM-6.1 ;
    grc:provenanceSource "CMDB" ;
    grc:provenanceTimestamp "2026-02-13T10:00:00Z"^^xsd:dateTime .

grc:vuln-CVE-2024-1234 a grc:Vulnerability ;
    grc:cveId "CVE-2024-1234" ;
    grc:cweId "CWE-79" ;
    grc:cvssScore 7.5 ;
    grc:severity "high" ;
    grc:relatesToRisk grc:risk-case-001 ;
    grc:provenanceSource "NVD" ;
    grc:provenanceTimestamp "2026-02-13T10:00:00Z"^^xsd:dateTime .
```

### 2. Validate with SHACL

Load your data into Neo4j, then apply SHACL validation:

```cypher
CALL n10s.validation.shacl.validate('{validationReport:true}')
YIELD report
RETURN report
```

### 3. Query the ARG

Use Cypher to query relationships and risk propagation:

```cypher
MATCH (a:Asset)-[:HAS_VULNERABILITY]->(v:Vulnerability)
MATCH (v)-[:RELATES_TO_RISK]->(r:RiskCase)
MATCH (r)-[:AUDITED_BY]->(al:AuditLog)
RETURN a.assetId AS asset, v.cveId AS vulnerability, r.riskId AS risk, al.auditor AS auditor
```

## Compliance Traceability

The ontology provides explicit traceability from:
1. Technical entities (Asset, Vulnerability) → Control (via mitigates/hasControl)
2. Control → Standard clauses (via nfcrmClause/iso27005Clause)
3. Vulnerability → RiskCase (via relatesToRisk)
4. RiskCase → AuditLog (via auditedBy)

This ensures full GRC traceability required by NFCRM-1:2025 and ISO/IEC 27005.

## Provenance Tracking

All entities require provenance metadata:
- `provenanceSource`: Data origin (CMDB, NVD, manual entry, etc.)
- `provenanceTimestamp`: Data ingestion/creation timestamp
- `confidence` (for RiskCase): Automated assessment confidence (0.0-1.0)

This supports auditability and human-in-the-loop review.

## References

- [NFCRM-1:2025](https://www.nca.gov.sa/en/national-cybersecurity-authority/standards-and-regulations/national-framework-for-cybersecurity-risk-management-nfcrm)
- [ISO/IEC 27005:2022](https://www.iso.org/standard/75281.html)
- [SHACL Specification](https://www.w3.org/TR/shacl/)
- [neosemantics Plugin](https://github.com/jbarrasa/neosemantics)

## License

MIT License - see LICENSE file in project root.
