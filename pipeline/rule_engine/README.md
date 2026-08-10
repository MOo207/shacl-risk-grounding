# Pure Rule Engine

Pure-rule based risk assessment engine that encodes expert knowledge into configurable rules without ML models.

## Overview

The Pure Rule Engine implements a **neuro-symbolic baseline** for INTERSYMBOLIC-GRC risk assessment. It provides:

- **100% Explainability**: Every risk decision is traceable to specific rules
- **Direct Compliance Mapping**: Rules map to NFCRM-1:2025 and ISO/IEC 27005 requirements
- **Rule-Based Reasoning**: No black-box ML models, only transparent rule evaluation
- **Expert Knowledge Integration**: Encodes security best practices and compliance standards

## Architecture

### Rule Categories

The engine supports **5 rule categories**:

1. **Event-Based** (`event_based`): Triggered by specific events (vulnerabilities, logins)
2. **Asset-Based** (`asset_based`): Assesses asset characteristics (firewall, encryption)
3. **Relationship-Based** (`relationship_based`): Analyzes entity relationships (external connections)
4. **Compliance-Based** (`compliance_based`): Validates standard compliance (NFCRM, ISO)
5. **Time-Based** (`time_based`): Detects temporal anomalies (business hours)

### Scoring System

The engine uses a **multi-tier scoring formula**:

```
Final Score = Baseline Score (0-50) + Risk Modifier (-50 to +50) + Evidence Weight (0-20)
```

**Score Levels**:
- **EXCELLENT** (0-19): No significant risks detected
- **GOOD** (20-39): Low risk level
- **FAIR** (40-59): Medium risk level
- **POOR** (60-79): High risk level
- **CRITICAL** (80-100): Severe risks detected

**Components**:
- **Baseline Score (0-50)**: Sum of activated rule scores
- **Risk Modifier (-50 to +50)**: Adjusts score based on mitigating/positive factors
- **Evidence Weight (0-20)**: Weights rules by priority (higher priority = higher evidence)

## Installation

### Prerequisites

- Python 3.9+
- Neo4j (for integration with INTERSYMBOLIC-GRC ARG)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Neo4j is running
docker-compose up -d neo4j
```

## Usage

### Basic Usage

```python
from pipeline.rule_engine.pure_rule_engine import PureRuleEngine

# Initialize engine
engine = PureRuleEngine(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="YOUR_NEO4J_PASSWORD"
)

# Evaluate event
event = {
    "event_id": "EVT-001",
    "event_type": "VULNERABILITY_DETECTED",
    "has_critical_cve": True,
    "cve_severity": "CRITICAL",
    "cve_id": "CVE-2026-1234"
}

result = engine.evaluate_event(event)
print(f"Final Score: {result['final_score']}")
print(f"Score Level: {result['score_level']}")
print(f"Compliance Gaps: {result['compliance_gaps']}")
print(f"Recommendations: {result['recommendations']}")

# Close connection
engine.close()
```

### Rule Configuration

Rules are defined in JSON format:

```json
{
  "rules": [
    {
      "rule_id": "EVT-001",
      "category": "event_based",
      "priority": 1,
      "description": "Critical vulnerability CVE triggered",
      "conditions": {
        "has_critical_cve": true,
        "cve_severity": "CRITICAL"
      },
      "score": 50,
      "explanation": "Critical vulnerability detected with high severity rating",
      "compliance_mapping": ["NFCRM-1:2025-NN1.1"]
    }
  ]
}
```

**Rule Fields**:
- `rule_id`: Unique identifier (e.g., "EVT-001", "AST-002")
- `category`: Rule category (`event_based`, `asset_based`, `relationship_based`, `compliance_based`, `time_based`)
- `priority`: Rule priority (1-10, higher = more important)
- `description`: Human-readable description
- `conditions`: Conditions to evaluate (dict with key-value pairs or lambda function)
- `score`: Base score (0-50)
- `explanation`: Why this rule was triggered
- `compliance_mapping`: List of compliance requirements (NFCRM, ISO)

### Condition Evaluation

Conditions can be **simple key-value pairs** or **lambda functions**:

```python
# Simple condition
{
  "has_critical_cve": True,
  "cve_severity": "CRITICAL"
}

# Lambda function (time-based rules)
{
  "login_hour": lambda x: x < 6 or x > 22,
  "login_frequency": "UNUSUAL"
}
```

### Evaluating Event Data

Event data is evaluated against all rules:

```python
event = {
    "event_id": "EVT-002",
    "event_type": "LOGIN_EVENT",
    "login_hour": 3,  # 3 AM
    "login_frequency": "UNUSUAL"
}

result = engine.evaluate_event(event)
```

**Result Structure**:
```python
{
    "event_id": "EVT-002",
    "event_type": "LOGIN_EVENT",
    "activated_rules": [
        {
            "rule_id": "TMB-001",
            "rule_name": "TMB-001",
            "category": "time_based",
            "description": "Suspicious login activity outside business hours",
            "priority": 1,
            "score": 25,
            "explanation": "Login activity outside typical business hours",
            "compliance_mapping": ["NFCRM-1:2025-NN10.1"],
            "activated_at": "2026-02-16T10:30:00"
        }
    ],
    "baseline_score": 25,
    "risk_modifier": 0,
    "evidence_weight": 2,
    "final_score": 27,
    "score_level": "GOOD",
    "compliance_percentage": 100,
    "compliance_gaps": [],
    "recommendations": [
        "Review compliance requirement: NFCRM-1:2025-NN10.1"
    ],
    "timestamp": "2026-02-16T10:30:00"
}
```

## API Reference

### PureRuleEngine Class

#### Constructor

```python
PureRuleEngine(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: Optional[str] = None,
    rules_config_path: str = "rules/rule_definitions.json"
)
```

**Parameters**:
- `neo4j_uri`: Neo4j connection URI
- `neo4j_user`: Neo4j username
- `neo4j_password`: Neo4j password (or use `NEO4J_PASSWORD` env var)
- `rules_config_path`: Path to JSON rules configuration

#### Methods

##### `evaluate_event(event_data: Dict[str, Any]) -> Dict[str, Any]`

Evaluate an event against all rules.

**Parameters**:
- `event_data`: Event data dictionary

**Returns**:
- Risk assessment result with scores, compliance gaps, and recommendations

##### `get_rule_activation_log() -> List[Dict[str, Any]]`

Get rule activation log.

**Returns**:
- List of rule activation records

##### `reset_activation_log()`

Reset rule activation log.

##### `close()`

Close Neo4j connection.

## Examples

### Example 1: Critical Vulnerability Detection

```python
from pipeline.rule_engine.pure_rule_engine import PureRuleEngine

engine = PureRuleEngine()

event = {
    "event_id": "CVE-2026-1234",
    "event_type": "VULNERABILITY_DETECTED",
    "has_critical_cve": True,
    "cve_severity": "CRITICAL",
    "cve_id": "CVE-2026-1234",
    "cve_cvss": 9.8
}

result = engine.evaluate_event(event)

print(f"Score: {result['final_score']}")
print(f"Level: {result['score_level']}")
print(f"Activated Rules: {len(result['activated_rules'])}")
print(f"Compliance Gaps: {result['compliance_gaps']}")
print(f"Recommendations:")
for rec in result['recommendations']:
    print(f"  - {rec}")
```

**Output**:
```
Score: 100
Level: CRITICAL
Activated Rules: 1
Compliance Gaps: ['Critical vulnerability detected with high severity rating (Compliance: NFCRM-1:2025-NN1.1)']
Recommendations:
  - Review compliance requirement: NFCRM-1:2025-NN1.1
  - Apply security patches for detected vulnerabilities
```

### Example 2: Multiple Risk Factors

```python
event = {
    "event_id": "EVT-003",
    "event_type": "COMPLEX_RISK",
    "has_critical_cve": True,
    "cve_severity": "CRITICAL",
    "is_internet_facing": True,
    "firewall_configured": False,
    "contains_pii": True,
    "data_encrypted": False,
    "has_privileged_account": True,
    "mfa_required": True,
    "mfa_enabled": False,
    "has_mitigating_controls": True,
    "security_training_completed": True
}

result = engine.evaluate_event(event)

print(f"Final Score: {result['final_score']}")
print(f"Risk Modifier: {result['risk_modifier']}")
print(f"Evidence Weight: {result['evidence_weight']}")
```

**Output**:
```
Final Score: 117
Risk Modifier: 15
Evidence Weight: 10
```

### Example 3: Time-Based Anomaly Detection

```python
event = {
    "event_id": "EVT-004",
    "event_type": "LOGIN_EVENT",
    "login_hour": 3,  # 3 AM
    "login_frequency": "UNUSUAL",
    "source_ip": "203.0.113.42"
}

result = engine.evaluate_event(event)

print(f"Score: {result['final_score']}")
print(f"Compliance Gaps: {result['compliance_gaps']}")
```

## Testing

Run test suite:

```bash
pytest tests/test_pure_rule_engine.py -v
```

Test coverage includes:
- Rule evaluation
- Score calculation
- Compliance mapping
- Event integration
- Edge cases
- Integration tests

## Integration with INTERSYMBOLIC-GRC

The Pure Rule Engine integrates with the INTERSYMBOLIC-GRC pipeline:

### Pre-Inference Pipeline

```python
from pipeline.pre_inference.pre_inference_pipeline import PreInferencePipeline
from pipeline.rule_engine.pure_rule_engine import PureRuleEngine

# Initialize engines
pre_inference = PreInferencePipeline(neo4j_uri, neo4j_user, neo4j_password)
rule_engine = PureRuleEngine(neo4j_uri, neo4j_user, neo4j_password)

# Event flow
event_data = pre_inference.process_event(raw_event)  # Apply symbolic rules
risk_result = rule_engine.evaluate_event(event_data)  # Apply pure-rule assessment
```

### GRC Artifact Generation

The risk result is used to generate compliance artifacts:

```python
from pipeline.post_inference.grc_artifact_generator import GRCArtifactGenerator

generator = GRCArtifactGenerator()
artifact = generator.generate_risk_register(result, event_data)
```

## Performance Considerations

### Scalability

- **Rule Evaluation**: O(n) where n = number of rules
- **Memory**: Minimal (rules loaded once at initialization)
- **Latency**: <10ms per event (based on rule count)

### Optimization Tips

1. **Rule Categorization**: Organize rules by category for efficient filtering
2. **Condition Caching**: Lambda functions cached for repeated evaluations
3. **Batch Processing**: Process multiple events in batch for throughput

## Limitations

1. **Expert Bottleneck**: Rules must be defined by security experts
2. **Coverage Gaps**: May not catch novel attack patterns without rules
3. **Static Nature**: Rules require periodic review and updates
4. **False Positive Overhead**: Potential for high false positive rates

## Best Practices

1. **Rule Definition**: Follow naming conventions (EVT-xxx, AST-xxx, etc.)
2. **Prioritization**: Assign priority based on risk impact (1-10)
3. **Compliance Mapping**: Always map rules to specific NFCRM/ISO clauses
4. **Documentation**: Provide clear explanations for each rule
5. **Testing**: Comprehensive test coverage for all rules

## Comparison with ML Baseline

| Feature | Pure Rule Engine | ML Baseline |
|---------|-----------------|-------------|
| Explainability | 100% (rule-based) | Low (black-box) |
| Compliance Mapping | Direct (NFCRM/ISO) | Indirect |
| Expert Knowledge | Explicit | Implicit (training data) |
| Adaptability | High (rule updates) | Low (model retraining) |
| False Positives | High (expert defined) | Variable |
| False Negatives | High (no coverage) | Variable |

## Troubleshooting

### Common Issues

**Issue**: Rules not activating
- **Solution**: Check rule conditions match event data
- **Verify**: Print event_data to ensure keys match

**Issue**: Scores too high/low
- **Solution**: Adjust rule scores and modifiers
- **Check**: Rule priority and weight calculation

**Issue**: Neo4j connection error
- **Solution**: Ensure Neo4j is running
- **Verify**: Connection URI and credentials

## Future Enhancements

1. **Rule Templates**: Pre-built rule sets for common use cases
2. **Rule Versioning**: Track rule changes over time
3. **Rule Mining**: Automatically extract rules from expert documentation
4. **Dynamic Rules**: Rules that change based on context
5. **Performance Optimization**: Parallel rule evaluation

## References

- Research: `research/011-pure-rule-baseline-implementation-strategies.md`
- INTERSYMBOLIC-GRC Pipeline: `pipeline/intersymbolic_pipeline.md`
- SHACL Validation: `ontology/shacl_validation.md`

## License

Same as INTERSYMBOLIC-GRC project.

## Authors

- @forge (Implementation)
- @sage (Research and Design)
