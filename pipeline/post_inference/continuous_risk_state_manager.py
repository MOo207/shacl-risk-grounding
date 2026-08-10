"""
Continuous Risk State Manager for INTERSYMBOLIC-GRC

Tracks and updates asset risk states over time with temporal aggregation,
change detection, and alert generation for significant risk changes.

Features:
- Asset risk state tracking with historical timestamps
- Temporal risk aggregation (1h, 6h, 24h, 7d, 30d windows)
- Risk change detection (increases/decreases)
- Alert generation for significant changes
- Integration with GRC symbolic rules and artifact generation
- JSON export of risk state history
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict

from .grc_symbolic_rules import RiskCase, RiskLevel
from .grc_artifact_generator import GRCArtifactGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChangeDirection(Enum):
    """Direction of risk change"""
    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"


class RiskChangeSeverity(Enum):
    """Severity of risk change"""
    INFO = "info"
    WARNING = "warning"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TemporalWindow(Enum):
    """Temporal aggregation windows"""
    HOUR_1 = "1h"
    HOUR_6 = "6h"
    HOUR_24 = "24h"
    DAY_7 = "7d"
    DAY_30 = "30d"


@dataclass
class AssetRiskState:
    """Risk state for a specific asset"""
    assetId: str
    currentRiskLevel: RiskLevel
    currentScore: float
    confidence: float
    lastUpdated: datetime
    totalRiskCases: int = 0
    openRiskCases: int = 0
    criticalRiskCases: int = 0
    highRiskCases: int = 0
    mediumRiskCases: int = 0
    lowRiskCases: int = 0
    riskCaseIds: Set[str] = field(default_factory=set)
    historicalStates: List[Dict[str, Any]] = field(default_factory=list)

    def add_history_state(self, riskLevel: RiskLevel, score: float) -> None:
        """Add historical state for temporal aggregation"""
        self.historicalStates.append({
            'timestamp': self.lastUpdated.isoformat(),
            'risk_level': riskLevel.value,
            'score': score
        })
        # Keep only last 100 states
        if len(self.historicalStates) > 100:
            self.historicalStates = self.historicalStates[-100:]

    def update(
        self,
        riskLevel: RiskLevel,
        score: float,
        confidence: float,
        riskCaseId: str
    ) -> None:
        """Update asset risk state"""
        # Store previous state
        prev_level = self.currentRiskLevel
        prev_score = self.currentScore

        # Update current state
        self.currentRiskLevel = riskLevel
        self.currentScore = score
        self.confidence = confidence
        self.lastUpdated = datetime.now()
        self.totalRiskCases += 1

        # Update RiskCase counts
        if riskCaseId not in self.riskCaseIds:
            self.riskCaseIds.add(riskCaseId)
            self.openRiskCases += 1
            if riskLevel == RiskLevel.CRITICAL:
                self.criticalRiskCases += 1
            elif riskLevel == RiskLevel.HIGH:
                self.highRiskCases += 1
            elif riskLevel == RiskLevel.MEDIUM:
                self.mediumRiskCases += 1
            elif riskLevel == RiskLevel.LOW:
                self.lowRiskCases += 1

        # Add historical state
        self.add_history_state(riskLevel, score)

    def remove_risk_case(self, riskCaseId: str) -> None:
        """Remove RiskCase from asset"""
        if riskCaseId in self.riskCaseIds:
            self.riskCaseIds.remove(riskCaseId)
            self.openRiskCases -= 1

    def get_riskDistribution(self) -> Dict[str, int]:
        """Get RiskCase distribution"""
        return {
            'critical': self.criticalRiskCases,
            'high': self.highRiskCases,
            'medium': self.mediumRiskCases,
            'low': self.lowRiskCases,
            'open': self.openRiskCases,
            'total': self.totalRiskCases
        }



@dataclass
class RiskChangeAlert:
    """Alert for risk change"""
    alertId: str
    assetId: str
    changeDirection: ChangeDirection
    riskLevel: RiskLevel
    previousLevel: RiskLevel
    previousScore: float
    currentScore: float
    confidence: float
    severity: RiskChangeSeverity
    triggeredAt: datetime
    riskCaseId: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alertId': self.alertId,
            'assetId': self.assetId,
            'changeDirection': self.changeDirection.value,
            'riskLevel': self.riskLevel.value,
            'previousLevel': self.previousLevel.value,
            'previousScore': self.previousScore,
            'currentScore': self.currentScore,
            'confidence': self.confidence,
            'severity': self.severity.value,
            'triggeredAt': self.triggeredAt.isoformat(),
            'riskCaseId': self.riskCaseId,
            'description': self.description
        }


@dataclass
class TemporalRiskAggregation:
    """Temporal risk aggregation for an asset"""
    assetId: str
    window: TemporalWindow
    averageScore: float
    minScore: float
    maxScore: float
    totalRiskCases: int
    criticalCount: int
    highCount: int
    mediumCount: int
    lowCount: int
    startTimestamp: datetime
    endTimestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'assetId': self.assetId,
            'window': self.window.value,
            'averageScore': self.averageScore,
            'minScore': self.minScore,
            'maxScore': self.maxScore,
            'totalRiskCases': self.totalRiskCases,
            'criticalCount': self.criticalCount,
            'highCount': self.highCount,
            'mediumCount': self.mediumCount,
            'lowCount': self.lowCount,
            'startTimestamp': self.startTimestamp.isoformat(),
            'endTimestamp': self.endTimestamp.isoformat()
        }


class ContinuousRiskStateManager:
    """
    Manager for continuous risk state updates and temporal aggregation

    Provides:
    - Asset risk state tracking
    - Temporal risk aggregation
    - Risk change detection and alert generation
    - JSON export of risk state history
    """

    def __init__(
        self,
        artifact_generator: Optional[GRCArtifactGenerator] = None,
        output_dir: str = "./risk_states"
    ):
        """
        Initialize continuous risk state manager

        Args:
            artifact_generator: GRC artifact generator for integration
            output_dir: Output directory for risk state exports
        """
        self.artifact_generator = artifact_generator or GRCArtifactGenerator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Asset risk state storage
        self.asset_states: Dict[str, AssetRiskState] = {}
        self.alerts: List[RiskChangeAlert] = []

        # Risk change thresholds
        self.change_thresholds = {
            ChangeDirection.INCREASE: {
                RiskLevel.MEDIUM: RiskChangeSeverity.INFO,
                RiskLevel.HIGH: RiskChangeSeverity.MODERATE,
                RiskLevel.CRITICAL: RiskChangeSeverity.HIGH
            },
            ChangeDirection.DECREASE: {
                RiskLevel.HIGH: RiskChangeSeverity.WARNING,
                RiskLevel.CRITICAL: RiskChangeSeverity.MODERATE
            }
        }

        logger.info("Continuous risk state manager initialized")

    def add_asset(
        self,
        assetId: str,
        initialRiskLevel: RiskLevel = RiskLevel.LOW,
        initialScore: float = 0.3
    ) -> None:
        """
        Add a new asset to risk tracking

        Args:
            assetId: Asset identifier
            initialRiskLevel: Initial risk level
            initialScore: Initial risk score (0.0-1.0)
        """
        if assetId in self.asset_states:
            logger.warning(f"Asset {assetId} already exists, skipping creation")
            return

        self.asset_states[assetId] = AssetRiskState(
            assetId=assetId,
            currentRiskLevel=initialRiskLevel,
            currentScore=initialScore,
            confidence=0.5,
            lastUpdated=datetime.now()
        )

        logger.info(f"Added asset {assetId} with initial risk level {initialRiskLevel.value}")

    def update_asset_risk(
        self,
        assetId: str,
        riskLevel: RiskLevel,
        score: float = 0.5,
        confidence: float = 0.8,
        riskCaseId: str = ""
    ) -> Optional[RiskChangeAlert]:
        """
        Update asset risk state

        Args:
            assetId: Asset identifier
            riskLevel: New risk level
            score: New risk score
            confidence: Confidence in this assessment
            riskCaseId: Associated RiskCase ID

        Returns:
            Alert if risk change is significant, else None
        """
        if assetId not in self.asset_states:
            self.add_asset(assetId, initialRiskLevel=riskLevel, initialScore=score)
            return None

        asset_state = self.asset_states[assetId]

        # Store previous values
        prev_level = asset_state.currentRiskLevel
        prev_score = asset_state.currentScore

        # Update state (openRiskCases incremented inside update())
        asset_state.update(riskLevel, score, confidence, riskCaseId)

        # Detect risk change
        if prev_level != riskLevel or abs(prev_score - score) > 0.15:
            change_direction = self._determine_change_direction(prev_level, riskLevel)
            return self._generate_alert(
                asset_state, prev_level, prev_score, riskLevel, score, change_direction, riskCaseId
            )

        return None

    def _determine_change_direction(
        self,
        prev_level: RiskLevel,
        new_level: RiskLevel
    ) -> ChangeDirection:
        """
        Determine change direction based on risk levels

        Args:
            prev_level: Previous risk level
            new_level: New risk level

        Returns:
            ChangeDirection enum value
        """
        level_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

        if level_order.index(prev_level) < level_order.index(new_level):
            return ChangeDirection.INCREASE
        elif level_order.index(prev_level) > level_order.index(new_level):
            return ChangeDirection.DECREASE
        else:
            return ChangeDirection.STABLE

    def _generate_alert(
        self,
        asset_state: AssetRiskState,
        prev_level: RiskLevel,
        prev_score: float,
        new_level: RiskLevel,
        new_score: float,
        change_direction: ChangeDirection,
        riskCaseId: str
    ) -> RiskChangeAlert:
        """
        Generate a risk change alert

        Args:
            asset_state: Asset risk state
            prev_level: Previous risk level
            prev_score: Previous risk score
            new_level: New risk level
            new_score: New risk score
            change_direction: Change direction
            riskCaseId: Associated RiskCase ID

        Returns:
            RiskChangeAlert instance
        """
        # Determine severity
        severity = self.change_thresholds.get(change_direction, {}).get(new_level, RiskChangeSeverity.INFO)

        # Generate description
        direction_str = "increased to" if change_direction == ChangeDirection.INCREASE else "decreased to"
        description = (
            f"Asset {asset_state.assetId} risk {direction_str} {new_level.value} "
            f"(score: {new_score:.2f}, confidence: {asset_state.confidence:.2f}), "
            f"previously {prev_level.value} with score {prev_score:.2f}"
        )

        # Create alert
        alert = RiskChangeAlert(
            alertId=self._generate_alert_id(),
            assetId=asset_state.assetId,
            changeDirection=change_direction,
            riskLevel=new_level,
            previousLevel=prev_level,
            previousScore=prev_score,
            currentScore=new_score,
            confidence=asset_state.confidence,
            severity=severity,
            triggeredAt=datetime.now(),
            riskCaseId=riskCaseId,
            description=description
        )

        self.alerts.append(alert)
        logger.warning(f"Risk change alert generated: {alert.alertId} - {description}")

        return alert

    def _generate_alert_id(self) -> str:
        """Generate a unique alert ID"""
        # A sequence counter, not a timestamp alone. The previous "random suffix"
        # was "".join(str(i) for i in range(1000, 9999)) -- a ~35KB constant
        # identical on every call, so alerts raised in the same second collided.
        self._alert_seq = getattr(self, "_alert_seq", 0) + 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ARC_{timestamp}_{self._alert_seq:06d}"

    def remove_risk_case(
        self,
        assetId: str,
        riskCaseId: str
    ) -> None:
        """
        Remove a RiskCase from an asset

        Args:
            assetId: Asset identifier
            riskCaseId: Risk case ID to remove
        """
        if assetId in self.asset_states:
            asset_state = self.asset_states[assetId]
            asset_state.remove_risk_case(riskCaseId)  # already decrements openRiskCases

    def get_asset_state(self, assetId: str) -> Optional[AssetRiskState]:
        """
        Get asset risk state

        Args:
            assetId: Asset identifier

        Returns:
            AssetRiskState or None
        """
        return self.asset_states.get(assetId)

    def get_all_assets(self) -> List[str]:
        """
        Get all tracked asset IDs

        Returns:
            List of asset IDs
        """
        return list(self.asset_states.keys())

    def aggregate_risk_by_window(
        self,
        assetId: str,
        window: TemporalWindow,
        min_time: Optional[datetime] = None
    ) -> Optional[TemporalRiskAggregation]:
        """
        Aggregate risk over a temporal window

        Args:
            assetId: Asset identifier
            window: Temporal window size
            min_time: Minimum timestamp (optional)

        Returns:
            TemporalRiskAggregation or None
        """
        if assetId not in self.asset_states:
            return None

        asset_state = self.asset_states[assetId]

        # Calculate time window
        if min_time is None:
            min_time = datetime.now() - timedelta(
                days={
                    TemporalWindow.HOUR_1: 1,
                    TemporalWindow.HOUR_6: 6,
                    TemporalWindow.HOUR_24: 24,
                    TemporalWindow.DAY_7: 7,
                    TemporalWindow.DAY_30: 30
                }[window]
            )

        # Filter historical states
        relevant_states = [
            state for state in asset_state.historicalStates
            if datetime.fromisoformat(state['timestamp']) >= min_time
        ]

        if not relevant_states:
            logger.warning(f"No historical states found for asset {assetId} in {window.value} window")
            return None

        # Calculate statistics
        scores = [state['score'] for state in relevant_states]
        risk_levels = [state['risk_level'] for state in relevant_states]

        return TemporalRiskAggregation(
            assetId=assetId,
            window=window,
            averageScore=sum(scores) / len(scores),
            minScore=min(scores),
            maxScore=max(scores),
            totalRiskCases=len(relevant_states),
            criticalCount=risk_levels.count('critical'),
            highCount=risk_levels.count('high'),
            mediumCount=risk_levels.count('medium'),
            lowCount=risk_levels.count('low'),
            startTimestamp=min_time,
            endTimestamp=datetime.now()
        )

    def aggregate_all_assets_by_window(
        self,
        window: TemporalWindow
    ) -> Dict[str, TemporalRiskAggregation]:
        """
        Aggregate risk for all assets over a temporal window

        Args:
            window: Temporal window size

        Returns:
            Dictionary of asset ID to TemporalRiskAggregation
        """
        aggregations = {}

        for assetId in self.asset_states:
            aggregation = self.aggregate_risk_by_window(assetId, window)
            if aggregation:
                aggregations[assetId] = aggregation

        return aggregations

    def get_active_alerts(self, severity: Optional[RiskChangeSeverity] = None) -> List[RiskChangeAlert]:
        """
        Get active alerts

        Args:
            severity: Filter by severity (optional)

        Returns:
            List of alerts
        """
        if severity:
            return [alert for alert in self.alerts if alert.severity == severity]
        return self.alerts.copy()

    def get_alert_history(self, days: int = 30) -> List[RiskChangeAlert]:
        """
        Get alert history for past N days

        Args:
            days: Number of days to look back

        Returns:
            List of alerts
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [
            alert for alert in self.alerts
            if alert.triggeredAt >= cutoff
        ]

    def export_asset_states_to_json(self) -> str:
        """
        Export all asset states to JSON

        Returns:
            JSON string
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_assets': len(self.asset_states),
            'total_alerts': len(self.alerts),
            'assets': []
        }

        for asset_id, asset_state in self.asset_states.items():
            export_data['assets'].append({
                'assetId': asset_state.assetId,
                'currentRiskLevel': asset_state.currentRiskLevel.value,
                'currentScore': asset_state.currentScore,
                'confidence': asset_state.confidence,
                'lastUpdated': asset_state.lastUpdated.isoformat(),
                'riskDistribution': asset_state.get_riskDistribution(),
                'historicalStates': asset_state.historicalStates,
                'openRiskCases': asset_state.openRiskCases
            })

        return json.dumps(export_data, indent=2)

    def export_alerts_to_json(self) -> str:
        """
        Export all alerts to JSON

        Returns:
            JSON string
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_alerts': len(self.alerts),
            'alerts': [alert.to_dict() for alert in self.alerts]
        }

        return json.dumps(export_data, indent=2)

    def export_aggregations_by_window(self) -> Dict[str, Any]:
        """
        Export temporal aggregations for all windows

        Returns:
            Dictionary with aggregations for each window
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat()
        }

        for window in TemporalWindow:
            aggregations = self.aggregate_all_assets_by_window(window)
            export_data[window.value] = [agg.to_dict() for agg in aggregations.values()]

        return export_data

    def save_to_files(self) -> Dict[str, str]:
        """
        Save asset states and alerts to files

        Returns:
            Dictionary of file paths
        """
        file_paths = {}

        # Save asset states
        states_json = self.export_asset_states_to_json()
        states_file = self.output_dir / "asset_states.json"
        with open(states_file, 'w', encoding='utf-8') as f:
            f.write(states_json)
        file_paths['asset_states'] = str(states_file)

        # Save alerts
        alerts_json = self.export_alerts_to_json()
        alerts_file = self.output_dir / "alerts.json"
        with open(alerts_file, 'w', encoding='utf-8') as f:
            f.write(alerts_json)
        file_paths['alerts'] = str(alerts_file)

        # Save aggregations
        aggregations_json = json.dumps(self.export_aggregations_by_window(), indent=2)
        aggregations_file = self.output_dir / "aggregations.json"
        with open(aggregations_file, 'w', encoding='utf-8') as f:
            f.write(aggregations_json)
        file_paths['aggregations'] = str(aggregations_file)

        logger.info(f"Risk state exports saved to {self.output_dir}")
        return file_paths

    def generate_risk_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk summary

        Returns:
            Risk summary dictionary
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_assets': len(self.asset_states),
            'total_risk_cases': sum(state.totalRiskCases for state in self.asset_states.values()),
            'open_risk_cases': sum(state.openRiskCases for state in self.asset_states.values()),
            'asset_states': {}
        }

        for asset_id, asset_state in self.asset_states.items():
            summary['asset_states'][asset_id] = {
                'current_risk_level': asset_state.currentRiskLevel.value,
                'current_score': asset_state.currentScore,
                'confidence': asset_state.confidence,
                'open_cases': asset_state.openRiskCases,
                'risk_distribution': asset_state.get_riskDistribution()
            }

        return summary


# ============================================================================
# Singleton Instance
# ============================================================================

def create_continuous_risk_state_manager(
    artifact_generator: Optional[GRCArtifactGenerator] = None,
    output_dir: str = "./risk_states"
) -> ContinuousRiskStateManager:
    """
    Create continuous risk state manager

    Args:
        artifact_generator: GRC artifact generator
        output_dir: Output directory for exports

    Returns:
        ContinuousRiskStateManager instance
    """
    return ContinuousRiskStateManager(artifact_generator, output_dir)
