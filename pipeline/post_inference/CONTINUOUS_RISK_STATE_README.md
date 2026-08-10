# Continuous Risk State Manager

**Module:** `pipeline.post_inference.continuous_risk_state_manager`

**Purpose:** Track and update asset risk states over time with temporal aggregation, change detection, and alert generation.

## Overview

The Continuous Risk State Manager provides real-time risk state tracking for cybersecurity assets with:

- **Asset risk state tracking** - Maintain current and historical risk levels for each asset
- **Temporal risk aggregation** - Aggregate risk metrics over configurable time windows (1h, 6h, 24h, 7d, 30d)
- **Risk change detection** - Identify significant increases or decreases in asset risk
- **Alert generation** - Generate alerts for risk changes based on severity thresholds
- **JSON export** - Export risk states, alerts, and temporal aggregations

## Key Components

### 1. AssetRiskState

Tracks risk state for a specific asset:

```python
from pipeline.post_inference.continuous_risk_state_manager import AssetRiskState, RiskLevel

state = AssetRiskState(
    assetId="web-server-1",
    currentRiskLevel=RiskLevel.MEDIUM,
    currentScore=0.5,
    confidence=0.8,
    lastUpdated=datetime.now()
)

# Update risk state
state.update(
    riskLevel=RiskLevel.HIGH,
    score=0.7,
    confidence=0.9,
    riskCaseId="RC_001"
)
```

### 2. ContinuousRiskStateManager

Main manager class that orchestrates risk state tracking:

```python
from pipeline.post_inference.continuous_risk_state_manager import ContinuousRiskStateManager

# Initialize manager
manager = ContinuousRiskStateManager(output_dir="./risk_states")

# Add asset
manager.add_asset("web-server-1")

# Update risk state
alert = manager.update_asset_risk(
    assetId="web-server-1",
    riskLevel=RiskLevel.HIGH,
    score=0.7,
    confidence=0.9,
    riskCaseId="RC_001"
)

# Get asset state
state = manager.get_asset_state("web-server-1")

# Aggregate risk over time window
aggregation = manager.aggregate_risk_by_window("web-server-1", TemporalWindow.HOUR_24)
```

### 3. RiskChangeAlert

Alert generated for significant risk changes:

```python
alert.changeDirection  # "increase" or "decrease"
alert.riskLevel  # Current risk level
alert.previousLevel  # Previous risk level
alert.severity  # "info", "warning", "moderate", "high", "critical"
```

### 4. TemporalRiskAggregation

Aggregated risk metrics over time windows:

```python
aggregation = manager.aggregate_risk_by_window("web-server-1", TemporalWindow.DAY_7)
aggregation.averageScore  # Average risk score
aggregation.criticalCount  # Number of critical risk cases
aggregation.highCount  # Number of high risk cases
```

## Usage Patterns

### Real-time Risk Tracking

```python
# Initialize manager
manager = ContinuousRiskStateManager()

# Add assets
manager.add_asset("server-1")
manager.add_asset("server-2")

# Simulate continuous monitoring
for _ in range(10):
    risk_level = RiskLevel.MEDIUM
    risk_score = 0.5 + (0.05 * random.random())

    alert = manager.update_asset_risk(
        assetId="server-1",
        riskLevel=risk_level,
        score=risk_score,
        confidence=0.8,
        riskCaseId=f"RC_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    if alert:
        print(f"Alert: {alert.alertId} - {alert.description}")
```

### Temporal Risk Analysis

```python
# Aggregate risk over different windows
windows = [TemporalWindow.HOUR_1, TemporalWindow.HOUR_6, TemporalWindow.HOUR_24]

for window in windows:
    aggregations = manager.aggregate_all_assets_by_window(window)

    print(f"\n{window.value} Aggregation:")
    for asset_id, agg in aggregations.items():
        print(f"  {asset_id}: avg={agg.averageScore:.2f}, "
              f"critical={agg.criticalCount}, high={agg.highCount}")
```

### Alert Monitoring

```python
# Get all active alerts
all_alerts = manager.get_active_alerts()

for alert in all_alerts:
    print(f"[{alert.severity}] {alert.assetId}: "
          f"{alert.previousLevel.value} → {alert.riskLevel.value}")

# Get high severity alerts only
high_alerts = manager.get_active_alerts(RiskChangeSeverity.HIGH)

# Get recent alerts (last 7 days)
recent_alerts = manager.get_alert_history(days=7)
```

### Export & Reporting

```python
# Export all data to files
file_paths = manager.save_to_files()
print(f"Saved to: {file_paths}")

# Export to JSON strings
asset_states_json = manager.export_asset_states_to_json()
alerts_json = manager.export_alerts_to_json()
aggregations_json = manager.export_aggregations_by_window()

# Generate risk summary
summary = manager.generate_risk_summary()
print(f"Total assets: {summary['total_assets']}")
print(f"Total risk cases: {summary['total_risk_cases']}")
```

## Integration with ML Pipeline

The Continuous Risk State Manager integrates with the ML inference orchestrator:

```python
from pipeline.inference.ml_inference_orchestrator import MLInferenceOrchestrator
from pipeline.post_inference.continuous_risk_state_manager import ContinuousRiskStateManager

# Initialize orchestrator and risk manager
orchestrator = MLInferenceOrchestrator()
risk_manager = ContinuousRiskStateManager()

# Process inference result
result = orchestrator.process_event_stream(events)

# Update risk state for each asset
for risk_case_id, risk_signals in result.risk_cases.items():
    for signal in risk_signals:
        asset_id = signal.assetId  # Extract from signal
        alert = risk_manager.update_asset_risk(
            assetId=asset_id,
            riskLevel=risk_case.riskLevel,
            score=risk_case.currentScore,
            confidence=risk_case.confidence,
            riskCaseId=risk_case_id
        )

        if alert:
            # Send alert to alerting system
            send_alert(alert)

# Save risk state history
risk_manager.save_to_files()
```

## Risk Level Hierarchy

```
CRITICAL (0.8-1.0)  ← highest
HIGH (0.6-0.8)
MEDIUM (0.4-0.6)
LOW (0.0-0.4)
```

## Change Direction

- **INCREASE**: Risk level or score increased (e.g., LOW → MEDIUM)
- **DECREASE**: Risk level or score decreased (e.g., HIGH → MEDIUM)
- **STABLE**: No significant change

## Alert Severity Thresholds

| Change Direction | Previous Level | New Level | Severity |
|------------------|----------------|-----------|----------|
| INCREASE         | LOW            | MEDIUM    | INFO     |
| INCREASE         | MEDIUM         | HIGH      | MODERATE |
| INCREASE         | HIGH           | CRITICAL  | HIGH     |
| DECREASE         | HIGH           | MEDIUM    | WARNING  |
| DECREASE         | CRITICAL       | HIGH      | MODERATE |

## Time Windows

- **1h**: Last hour
- **6h**: Last 6 hours
- **24h**: Last 24 hours
- **7d**: Last 7 days
- **30d**: Last 30 days

## Testing

Run the test suite:

```bash
pytest tests/test_continuous_risk_state_manager.py -v
```

Test Coverage:
- Asset state initialization and updates
- Risk case ID tracking
- Risk distribution calculation
- Alert generation and severity
- Temporal aggregation
- JSON export
- File saving
- Singleton creation

## Output Files

When `save_to_files()` is called, the following files are created:

- `asset_states.json` - Current and historical asset risk states
- `alerts.json` - All generated risk change alerts
- `aggregations.json` - Temporal aggregations for all windows

## Future Enhancements

- [ ] Neo4j integration for persistent storage
- [ ] Real-time WebSocket alerts
- [ ] Risk trend visualization
- [ ] Machine learning for change prediction
- [ ] Integration with SIEM for alerts
- [ ] REST API for risk state queries

## Dependencies

- `dataclasses`
- `datetime`
- `json`
- `logging`
- `pathlib`
- `collections.defaultdict`

## Related Modules

- `grc_symbolic_rules.py` - GRC entity mapping (RiskCase, AuditLog)
- `grc_artifact_generator.py` - GRC artifact generation
- `ml_inference_orchestrator.py` - ML inference pipeline integration
