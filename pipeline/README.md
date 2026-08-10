# Data Ingestion Pipeline

This pipeline ingests multi-source cybersecurity event data into the Asset Relationship Graph (ARG) with SHACL validation and data quality checks.

## Overview

The Data Ingestion Pipeline performs end-to-end data ingestion:

1. **Load**: Raw dataset (CSE-CIC-IDS2018)
2. **Parse**: Convert flow records to asset-centric graph entities
3. **Validate**: Check data quality (>90% threshold)
4. **Ingest**: Load entities into Neo4j ARG
5. **Validate**: SHACL shape validation (optional)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python -m pipeline.orchestrator \
  --dataset /path/to/cse-cic-ids2018 \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password your_password
```

### With SHACL Validation (Default)

```bash
python -m pipeline.orchestrator \
  --dataset /path/to/cse-cic-ids2018
```

### Without SHACL Validation

```bash
python -m pipeline.orchestrator \
  --dataset /path/to/cse-cic-ids2018 \
  --no-shacl
```

## Components

### Loaders (`pipeline/loaders/`)

- `CSECICIDS2018Loader`: Loads CSE-CIC-IDS2018 AWS network traffic dataset
- Supports all attack scenarios (Botnet, DDoS, DoS, Infiltration, Web Attacks)
- Handles multiple CSV files and encodings

### Parsers (`pipeline/parsers/`)

- `FlowParser`: Converts network flow records to graph entities
  - Flow Record → NetworkAsset (software/hardware/network)
  - Source/Dest IP → Asset nodes
  - Flow → NetworkConnection relationships
  - Attack label → ThreatEvent nodes
  - Protocols/Ports → SoftwareComponent nodes

### Validators (`pipeline/validators/`)

- `SHACLValidator`: Validates graph against SHACL shapes using neosemantics
- `DataQualityValidator`: Checks data quality (completeness, consistency, accuracy, uniqueness)

### Ingestion API (`api/ingestion.py`)

- `ARGIngestionAPI`: Cypher-based API for ingesting entities into Neo4j
- Supports assets, vulnerabilities, controls, risk cases, audit logs
- Batch ingestion support

## Data Quality Metrics

The pipeline validates:
- **Completeness**: Required fields are present
- **Consistency**: Relationships are valid, timestamps in range
- **Accuracy**: IP addresses valid, ports in range
- **Uniqueness**: No duplicate entities

**Threshold**: 90% quality score required (warnings below, errors in strict mode)

## Output

### Ingestion Report

```json
{
  "status": "success",
  "duration_seconds": 45.3,
  "dataset_path": "/path/to/dataset",
  "raw_data_stats": {
    "total_flow_records": 50000
  },
  "parsed_entities": {
    "assets": 150,
    "connections": 50000,
    "threat_events": 2500,
    "software_components": 45
  },
  "quality_report": {
    "overall_score": 95.2,
    "total_entities": 52695,
    "valid_entities": 50160,
    "total_issues": 2535,
    "meets_threshold": true
  },
  "ingestion_stats": {
    "assets_ingested": 150,
    "connections_ingested": 50000,
    "threats_ingested": 2500,
    "components_ingested": 45,
    "failures": 0
  },
  "shacl_report": {
    "conformant": true,
    "violation_count": 0
  }
}
```

## Entity Types

### Asset

```python
{
  "assetId": "asset-192-168-1-100",
  "assetName": "Asset 192.168.1.100",
  "assetType": "software",
  "ipAddress": "192.168.1.100",
  "isInternal": true,
  "role": "source",
  "provenanceSource": "CSE-CIC-IDS2018"
}
```

### NetworkConnection

```python
{
  "connectionId": "conn-12345",
  "sourceAssetId": "asset-192-168-1-100",
  "destinationAssetId": "asset-10-0-0-50",
  "sourcePort": 54321,
  "destinationPort": 443,
  "protocol": "TCP",
  "timestamp": "2024-02-14T10:00:00",
  "isAttack": false
}
```

### ThreatEvent

```python
{
  "threatId": "threat-67890",
  "threatType": "botnet",
  "sourceAssetId": "asset-203-0-113-10",
  "destinationAssetId": "asset-192-168-1-100",
  "timestamp": "2024-02-14T10:00:01",
  "confidence": 0.8
}
```

## Neo4j Setup

### Prerequisites

1. Install Neo4j
2. Install neosemantics plugin (for SHACL validation)

```bash
# Using Docker Compose
docker-compose up -d neo4j
```

### Load SHACL Shapes

```bash
# From project root
./scripts/load-ontology.sh
```

### Initialize Database

```bash
./scripts/init-neo4j.sh
```

## Troubleshooting

### Connection Failed

```
Failed to connect to Neo4j: Failed to establish connection
```

**Solution**: Check Neo4j is running and credentials are correct.

```bash
# Verify Neo4j status
docker ps | grep neo4j

# Test connection
cypher-shell -u neo4j -p YOUR_PASSWORD_HERE
```

### No Data Loaded

```
No data loaded from any scenario
```

**Solution**: Verify dataset path is correct and CSV files exist.

```bash
# Check dataset structure
ls -la /path/to/cse-cic-ids2018/

# Should see directories like:
# Botnet/
# DDOS/
# DoS/
# Infiltration/
# WebAttacks/
```

### SHACL Validation Failed

```
neosemantics plugin not available
```

**Solution**: Install neosemantics plugin.

```bash
# Download neosemantics jar
wget https://github.com/jbarrasa/neosemantics/releases/download/4.x.x/neosemantics-4.x.x.jar

# Copy to Neo4j plugins directory
cp neosemantics-*.jar /var/lib/neo4j/plugins/

# Restart Neo4j
docker-compose restart neo4j
```

### Data Quality Below Threshold

```
Data quality score (85.5%) below threshold (90%)
```

**Solution**: Review quality issues and clean data.

```python
# Quality report includes specific issues
issues = report['quality_report']['issues']

# Example fix: Remove duplicate records
df.drop_duplicates(inplace=True)

# Re-run pipeline
```

## Development

### Adding New Loaders

```python
# pipeline/loaders/custom_loader.py
class CustomLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load(self) -> pd.DataFrame:
        # Load your data
        return df
```

### Adding New Validators

```python
# pipeline/validators/custom_validator.py
class CustomValidator:
    def validate(self, entities: List[Dict]) -> Dict[str, Any]:
        # Your validation logic
        return report
```

### Running Tests

```bash
# Run loader tests
python -m pytest pipeline/loaders/test_*.py

# Run parser tests
python -m pytest pipeline/parsers/test_*.py

# Run validator tests
python -m pytest pipeline/validators/test_*.py
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.
