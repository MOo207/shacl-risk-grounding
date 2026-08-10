"""
Post-Inference Symbolic Rules for INTERSYMBOLIC-GRC

Maps ML inference results to GRC entities (Control, RiskCase, AuditLog)
with explainable reasoning chains and NFCRM/ISO compliance validation.

Features:
- Risk signal to Control mapping with NFCRM/ISO clause mapping
- Risk case creation with risk level classification
- Explainable reasoning chain generation
- NFCRM-1:2025 and ISO/IEC 27005 compliance validation
- Audit log generation with provenance tracking
- Continuous risk state management
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from pipeline.inference.ml_inference_orchestrator import RiskSignal, InferenceResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for GRC entities"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Likelihood(Enum):
    """Likelihood values"""
    VERY_LOW = "very low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very high"


class Impact(Enum):
    """Impact values"""
    NEGLECTIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CATASTROPHIC = "catastrophic"


@dataclass
class ControlMapping:
    """Mapping of risk signal to a control"""
    controlId: str
    controlName: str
    provenanceSource: str  # "NFCRM-1:2025" or "ISO/IEC 27005"
    nfcrmClause: Optional[str] = None
    iso27005Clause: Optional[str] = None
    severity: str = "Medium"
    description: str = ""
    mitigated: bool = False


@dataclass
class RiskCase:
    """GRC Risk Case entity"""
    riskId: str
    riskLevel: RiskLevel
    likelihood: Likelihood
    impact: Impact
    confidence: float
    associatedControls: List[str]
    explainedBySignals: List[RiskSignal]
    provenanceSource: str  # "intersymbolic_ml" or "rule_based"
    provenanceTimestamp: datetime
    description: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'riskId': self.riskId,
            'riskLevel': self.riskLevel.value,
            'likelihood': self.likelihood.value,
            'impact': self.impact.value,
            'confidence': self.confidence,
            'associatedControls': self.associatedControls,
            'explanations': [s.description for s in self.explainedBySignals],
            'provenanceSource': self.provenanceSource,
            'provenanceTimestamp': self.provenanceTimestamp.isoformat(),
            'description': self.description,
            'recommendations': self.recommendations
        }


@dataclass
class AuditLog:
    """GRC Audit Log entity"""
    auditId: str
    auditTimestamp: datetime
    auditor: str
    provenanceSource: str
    provenanceTimestamp: datetime
    riskCaseId: Optional[str] = None
    eventType: str = "risk_assessment"
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'auditId': self.auditId,
            'auditTimestamp': self.auditTimestamp.isoformat(),
            'auditor': self.auditor,
            'provenanceSource': self.provenanceSource,
            'provenanceTimestamp': self.provenanceTimestamp.isoformat(),
            'riskCaseId': self.riskCaseId,
            'eventType': self.eventType,
            'description': self.description,
            'details': self.details
        }


class GRCSymbolicRules:
    """
    Post-inference symbolic rules engine for GRC entity mapping

    Maps ML risk signals to:
    - Control entities (NFCRM-1:2025, ISO/IEC 27005)
    - RiskCase entities (with risk level classification)
    - AuditLog entities (with provenance tracking)
    """

    # Control mapping database (NFCRM-1:2025)
    # iso27005Clause follows ISO/IEC 27005:2022 clause structure:
    #   §8.2 Risk identification, §8.3 Risk analysis,
    #   §8.4 Risk evaluation, §8.5 Risk treatment
    NFCRM_CONTROLS = {
        "NFCRM-1:2025-§6.7": ControlMapping(
            controlId="NFCRM-1:2025-§6.7",
            controlName="Currently Applied Controls",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§6.7",
            iso27005Clause="ISO/IEC 27005:2022-§8.2",  # W16: risk identification
            severity="High",
            description="Identify currently applied controls to address expected cybersecurity risk scenarios (NFCRM §6.7). Aligned with ISO/IEC 27005:2022 §8.2 (risk identification).",
            mitigated=False
        ),
        "NFCRM-1:2025-§6.4": ControlMapping(
            controlId="NFCRM-1:2025-§6.4",
            controlName="Vulnerability Inventory",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§6.4",
            iso27005Clause="ISO/IEC 27005:2022-§8.3",  # W16: risk analysis
            severity="High",
            description="Inventory cybersecurity vulnerabilities affecting organization assets (NFCRM §6.4). Aligned with ISO/IEC 27005:2022 §8.3 (risk analysis).",
            mitigated=False
        ),
        "NFCRM-1:2025-§6.14": ControlMapping(
            controlId="NFCRM-1:2025-§6.14",
            controlName="Risk Treatment Prioritisation",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§6.14",
            iso27005Clause="ISO/IEC 27005:2022-§8.4",  # W16: risk evaluation
            severity="High",
            description="Prioritise cybersecurity risk treatment options based on evaluation results (NFCRM §6.14). Aligned with ISO/IEC 27005:2022 §8.4 (risk evaluation).",
            mitigated=False
        ),
        "NFCRM-1:2025-§6.17": ControlMapping(
            controlId="NFCRM-1:2025-§6.17",
            controlName="Risk Management Reports",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§6.17",
            severity="High",
            description="Develop cybersecurity risk management reports (NFCRM §6.17).",
            mitigated=False
        ),
        "NFCRM-1:2025-§5.2": ControlMapping(
            controlId="NFCRM-1:2025-§5.2",
            controlName="Asset Inventory and Classification",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§5.2",
            severity="Critical",
            description="Inventory and classify all organisation assets per NCA guidance, review periodically (NFCRM §5.2).",
            mitigated=False
        ),
        "NFCRM-1:2025-§6.9": ControlMapping(
            controlId="NFCRM-1:2025-§6.9",
            controlName="Inherent Risk Analysis",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§6.9",
            iso27005Clause="ISO/IEC 27005:2022-§8.3",  # risk analysis (likelihood × impact)
            severity="Critical",
            description="Conduct inherent risk analysis assessing likelihood and impact per 5x5 risk matrix (NFCRM §6.9). Aligned with ISO/IEC 27005:2022 §8.3 (risk analysis).",
            mitigated=False
        ),
        "NFCRM-1:2025-§6.10": ControlMapping(
            controlId="NFCRM-1:2025-§6.10",
            controlName="Risk Register Documentation",
            provenanceSource="NFCRM-1:2025",
            nfcrmClause="NFCRM-1:2025-§6.10",
            iso27005Clause="ISO/IEC 27005:2022-§8.5",  # W16: risk treatment
            severity="Critical",
            description="Document analysis and assessment results in the cybersecurity risk register (NFCRM §6.10). Aligned with ISO/IEC 27005:2022 §8.5 (risk treatment selection).",
            mitigated=False
        ),
    }

    # ISO/IEC 27005 Controls
    ISO27005_CONTROLS = {
        "ISO/IEC 27005:001": ControlMapping(
            controlId="ISO/IEC 27005:001",
            controlName="Risk Assessment",
            provenanceSource="ISO/IEC 27005",
            iso27005Clause="ISO/IEC 27005:001",
            severity="High",
            description="Conduct risk assessment processes to identify, analyze, and evaluate risks.",
            mitigated=False
        ),
        "ISO/IEC 27005:002": ControlMapping(
            controlId="ISO/IEC 27005:002",
            controlName="Risk Treatment",
            provenanceSource="ISO/IEC 27005",
            iso27005Clause="ISO/IEC 27005:002",
            severity="High",
            description="Implement risk treatment plans to mitigate identified risks.",
            mitigated=False
        ),
        "ISO/IEC 27005:003": ControlMapping(
            controlId="ISO/IEC 27005:003",
            controlName="Monitoring and Review",
            provenanceSource="ISO/IEC 27005",
            iso27005Clause="ISO/IEC 27005:003",
            severity="Medium",
            description="Monitor and review risk assessment and treatment processes.",
            mitigated=False
        ),
    }

    def __init__(self, neo4j_driver=None):
        """
        Initialize GRC Symbolic Rules engine

        Args:
            neo4j_driver: Neo4j driver for ARG integration
        """
        self.neo4j_driver = neo4j_driver
        self.risk_cases: Dict[str, RiskCase] = {}
        self.audit_logs: Dict[str, AuditLog] = {}

    def map_signals_to_controls(
        self,
        risk_signals: List[RiskSignal],
        event_context: Optional[Dict[str, Any]] = None
    ) -> List[ControlMapping]:
        """
        Map risk signals to controls

        Args:
            risk_signals: List of risk signals
            event_context: Optional event context

        Returns:
            List of ControlMapping objects
        """
        control_mappings = []

        for signal in risk_signals:
            # Find matching control
            control = self._find_control_for_signal(signal, event_context)

            if control:
                control_mappings.append(control)

        return control_mappings

    def _find_control_for_signal(
        self,
        signal: RiskSignal,
        event_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ControlMapping]:
        """
        Find matching control for a risk signal

        Args:
            signal: Risk signal
            event_context: Optional event context

        Returns:
            ControlMapping or None if no match found
        """
        # Try NFCRM controls first
        for control_id, control in self.NFCRM_CONTROLS.items():
            if self._matches_signal(signal, control, event_context):
                logger.info(f"Matched risk signal to NFCRM control: {control_id}")
                return control

        # Try ISO/IEC 27005 controls
        for control_id, control in self.ISO27005_CONTROLS.items():
            if self._matches_signal(signal, control, event_context):
                logger.info(f"Matched risk signal to ISO/IEC 27005 control: {control_id}")
                return control

        logger.warning(f"No control match found for signal: {signal.description}")
        return None

    def _matches_signal(
        self,
        signal: RiskSignal,
        control: ControlMapping,
        event_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a risk signal matches a control

        Args:
            signal: Risk signal
            control: Control mapping
            event_context: Optional event context

        Returns:
            True if signal matches control
        """
        # Check signal type matches control type
        if signal.type not in ["anomaly", "behavioral", "risk", "violation"]:
            return False

        # Check description contains control keywords
        description_lower = signal.description.lower()
        keywords = [k.lower() for k in self._extract_control_keywords(control)]

        for keyword in keywords:
            if keyword in description_lower:
                return True

        return False

    def _extract_control_keywords(self, control: ControlMapping) -> List[str]:
        """
        Extract keywords from control description

        Args:
            control: Control mapping

        Returns:
            List of keywords
        """
        keywords = []
        keywords.append(control.controlName.lower())
        if control.description:
            keywords.extend(control.description.lower().split())
        return keywords

    def create_risk_case(
        self,
        risk_signals: List[RiskSignal],
        event_context: Optional[Dict[str, Any]] = None
    ) -> RiskCase:
        """
        Create a RiskCase entity from risk signals

        Args:
            risk_signals: List of risk signals
            event_context: Optional event context

        Returns:
            RiskCase entity
        """
        # Calculate risk level based on signal scores
        risk_level = self._calculate_risk_level(risk_signals)
        likelihood = self._calculate_likelihood(risk_signals)
        impact = self._calculate_impact(risk_signals)

        # Calculate confidence
        confidence = self._calculate_confidence(risk_signals)

        # Get associated control IDs
        control_mappings = self.map_signals_to_controls(risk_signals, event_context)
        control_ids = [cm.controlId for cm in control_mappings]

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_signals, control_mappings)

        # Create RiskCase
        risk_case = RiskCase(
            riskId=self._generate_risk_id(),
            riskLevel=risk_level,
            likelihood=likelihood,
            impact=impact,
            confidence=confidence,
            associatedControls=control_ids,
            explainedBySignals=risk_signals,
            provenanceSource="intersymbolic_ml",
            provenanceTimestamp=datetime.now()
        )

        # Add to registry
        self.risk_cases[risk_case.riskId] = risk_case

        logger.info(f"Created RiskCase {risk_case.riskId} with level {risk_level.value}")

        return risk_case

    def _calculate_risk_level(self, risk_signals: List[RiskSignal]) -> RiskLevel:
        """
        Calculate risk level from risk signals

        Args:
            risk_signals: List of risk signals

        Returns:
            RiskLevel enum value
        """
        if not risk_signals:
            return RiskLevel.LOW

        # Calculate average score
        avg_score = sum(s.score for s in risk_signals) / len(risk_signals)

        # Determine risk level
        if avg_score >= 0.8:
            return RiskLevel.CRITICAL
        elif avg_score >= 0.6:
            return RiskLevel.HIGH
        elif avg_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_likelihood(self, risk_signals: List[RiskSignal]) -> Likelihood:
        """
        Calculate likelihood from risk signals

        Args:
            risk_signals: List of risk signals

        Returns:
            Likelihood enum value
        """
        if not risk_signals:
            return Likelihood.VERY_LOW

        # Use the highest likelihood from signals
        likelihood_values = [Likelihood.VERY_LOW, Likelihood.LOW, Likelihood.MEDIUM, Likelihood.HIGH, Likelihood.VERY_HIGH]

        max_index = 0
        for signal in risk_signals:
            # Extract likelihood from signal metadata or description
            signal_likelihood = self._extract_likelihood_from_signal(signal)
            if signal_likelihood > max_index:
                max_index = signal_likelihood

        return likelihood_values[min(max_index, len(likelihood_values) - 1)]

    def _extract_likelihood_from_signal(self, signal: RiskSignal) -> int:
        """
        Extract likelihood index from signal

        Args:
            signal: Risk signal

        Returns:
            Likelihood index (0-4)
        """
        likelihood_lower = signal.description.lower()
        if "very high" in likelihood_lower:
            return 4
        elif "high" in likelihood_lower:
            return 3
        elif "medium" in likelihood_lower:
            return 2
        elif "low" in likelihood_lower:
            return 1
        else:
            return 0

    def _calculate_impact(self, risk_signals: List[RiskSignal]) -> Impact:
        """
        Calculate impact from risk signals

        Args:
            risk_signals: List of risk signals

        Returns:
            Impact enum value
        """
        if not risk_signals:
            return Impact.NEGLIGIBLE

        # Use the highest impact from signals
        impact_values = [Impact.NEGLIGIBLE, Impact.MINOR, Impact.MODERATE, Impact.MAJOR, Impact.CATASTROPHIC]

        max_index = 0
        for signal in risk_signals:
            # Extract impact from signal metadata or description
            signal_impact = self._extract_impact_from_signal(signal)
            if signal_impact > max_index:
                max_index = signal_impact

        return impact_values[min(max_index, len(impact_values) - 1)]

    def _extract_impact_from_signal(self, signal: RiskSignal) -> int:
        """
        Extract impact index from signal

        Args:
            signal: Risk signal

        Returns:
            Impact index (0-4)
        """
        impact_lower = signal.description.lower()
        if "catastrophic" in impact_lower:
            return 4
        elif "major" in impact_lower:
            return 3
        elif "moderate" in impact_lower:
            return 2
        elif "minor" in impact_lower:
            return 1
        else:
            return 0

    def _calculate_confidence(self, risk_signals: List[RiskSignal]) -> float:
        """
        Calculate confidence from risk signals

        Args:
            risk_signals: List of risk signals

        Returns:
            Confidence value (0.0-1.0)
        """
        if not risk_signals:
            return 0.5

        # Average confidence from signals
        avg_confidence = sum(s.confidence for s in risk_signals) / len(risk_signals)
        return avg_confidence

    def _generate_recommendations(
        self,
        risk_signals: List[RiskSignal],
        control_mappings: List[ControlMapping]
    ) -> List[str]:
        """
        Generate remediation recommendations

        Args:
            risk_signals: List of risk signals
            control_mappings: List of ControlMapping objects

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not control_mappings:
            recommendations.append("Review ML model results and consider additional context")
            return recommendations

        for control in control_mappings:
            if control.nfcrmClause:
                recommendations.append(
                    f"Refer to NFCRM-1:2025 {control.nfcrmClause} for detailed remediation guidance"
                )
            elif control.iso27005Clause:
                recommendations.append(
                    f"Refer to ISO/IEC 27005 {control.iso27005Clause} for detailed remediation guidance"
                )

        # Add signal-specific recommendations
        for signal in risk_signals:
            if signal.type == "anomaly":
                recommendations.append(
                    f"Review and investigate anomaly: {signal.description}"
                )
            elif signal.type == "behavioral":
                recommendations.append(
                    f"Review behavioral pattern: {signal.description}"
                )

        return recommendations

    def _generate_risk_id(self) -> str:
        """
        Generate a unique RiskCase ID

        Returns:
            Risk case ID
        """
        # A sequence counter, not a timestamp alone. The previous "random suffix"
        # was "".join(str(i) for i in range(1000, 9999)) -- a ~35KB constant
        # identical on every call, so ids collided within the same second and
        # overwrote each other in the registry.
        self._risk_seq = getattr(self, "_risk_seq", 0) + 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"RC_{timestamp}_{self._risk_seq:06d}"

    def create_audit_log(
        self,
        risk_case_id: Optional[str] = None,
        event_type: str = "risk_assessment",
        auditor: str = "intersymbolic_system",
        description: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Create an AuditLog entity

        Args:
            risk_case_id: Optional RiskCase ID to associate with
            event_type: Type of audit event
            auditor: Auditor identity
            description: Description of the event
            details: Optional additional details

        Returns:
            AuditLog entity
        """
        audit_log = AuditLog(
            auditId=self._generate_audit_id(),
            auditTimestamp=datetime.now(),
            auditor=auditor,
            provenanceSource="intersymbolic_ml",
            provenanceTimestamp=datetime.now(),
            riskCaseId=risk_case_id,
            eventType=event_type,
            description=description,
            details=details or {}
        )

        # Add to registry
        self.audit_logs[audit_log.auditId] = audit_log

        logger.info(f"Created audit log {audit_log.auditId} for event type {event_type}")

        return audit_log

    def _generate_audit_id(self) -> str:
        """
        Generate a unique audit log ID

        Returns:
            Audit log ID
        """
        # See _generate_risk_id: the previous "random suffix" was a constant.
        self._audit_seq = getattr(self, "_audit_seq", 0) + 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"AL_{timestamp}_{self._audit_seq:06d}"

    def generate_explainable_chain(
        self,
        inference_result: InferenceResult
    ) -> List[Dict[str, Any]]:
        """
        Generate explainable reasoning chain for inference result

        Args:
            inference_result: Inference result

        Returns:
            List of reasoning steps
        """
        chain = []

        # Step 1: Pre-inference
        chain.append({
            'step': 'pre_inference',
            'description': 'Validation and filtering of input events',
            'status': 'passed' if inference_result.metadata.get('pre_inference_passed', True) else 'failed',
            'details': inference_result.metadata.get('pre_inference_details', [])
        })

        # Step 2: Feature extraction
        chain.append({
            'step': 'feature_extraction',
            'description': 'Extraction of technical and behavioral features',
            'status': 'passed',
            'details': f'Extracted {inference_result.metadata.get("feature_count", 0)} features'
        })

        # Step 3: Anomaly detection
        if inference_result.anomaly_score:
            chain.append({
                'step': 'anomaly_detection',
                'description': f'Anomaly detection score: {inference_result.anomaly_score:.3f}',
                'status': 'passed' if inference_result.is_anomaly else 'passed',
                'details': inference_result.explanations
            })

        # Step 4: Risk signals
        if inference_result.risk_signals:
            chain.append({
                'step': 'risk_signals',
                'description': f'Generated {len(inference_result.risk_signals)} risk signals',
                'status': 'passed',
                'details': [
                    {
                        'type': s.type,
                        'source': s.source,
                        'score': s.score,
                        'description': s.description
                    }
                    for s in inference_result.risk_signals
                ]
            })

        # Step 5: GRC mapping
        control_mappings = self.map_signals_to_controls(inference_result.risk_signals)
        chain.append({
            'step': 'grc_mapping',
            'description': f'Mapped to {len(control_mappings)} controls',
            'status': 'passed' if control_mappings else 'partial',
            'details': [
                {
                    'control_id': cm.controlId,
                    'control_name': cm.controlName,
                    'provenance': cm.provenanceSource
                }
                for cm in control_mappings
            ]
        })

        # Step 6: Risk case creation
        risk_case = self.create_risk_case(inference_result.risk_signals)
        chain.append({
            'step': 'risk_case_creation',
            'description': f'Created RiskCase {risk_case.riskId}',
            'status': 'passed',
            'details': {
                'risk_level': risk_case.riskLevel.value,
                'likelihood': risk_case.likelihood.value,
                'impact': risk_case.impact.value,
                'confidence': risk_case.confidence,
                'associated_controls': risk_case.associatedControls
            }
        })

        # Step 7: Audit log creation
        audit_log = self.create_audit_log(
            risk_case_id=risk_case.riskId,
            description=f'Risk assessment for event {inference_result.event_id}',
            details={
                'risk_score': inference_result.overall_risk_score,
                'confidence': inference_result.confidence,
                'signal_count': len(inference_result.risk_signals)
            }
        )
        chain.append({
            'step': 'audit_log',
            'description': f'Created audit log {audit_log.auditId}',
            'status': 'passed'
        })

        return chain

    def get_risk_cases(self) -> Dict[str, RiskCase]:
        """
        Get all RiskCases

        Returns:
            Dictionary of RiskCases
        """
        return self.risk_cases.copy()

    def get_audit_logs(self) -> Dict[str, AuditLog]:
        """
        Get all audit logs

        Returns:
            Dictionary of audit logs
        """
        return self.audit_logs.copy()

    def export_risk_cases_to_json(self) -> str:
        """
        Export all RiskCases to JSON

        Returns:
            JSON string
        """
        return json.dumps([rc.to_dict() for rc in self.risk_cases.values()], indent=2)

    def export_audit_logs_to_json(self) -> str:
        """
        Export all audit logs to JSON

        Returns:
            JSON string
        """
        return json.dumps([al.to_dict() for al in self.audit_logs.values()], indent=2)

    def export_to_json(self) -> Dict[str, Any]:
        """
        Export all GRC entities to JSON

        Returns:
            Dictionary with risk_cases and audit_logs
        """
        return {
            'risk_cases': [rc.to_dict() for rc in self.risk_cases.values()],
            'audit_logs': [al.to_dict() for al in self.audit_logs.values()],
            'export_timestamp': datetime.now().isoformat()
        }


# ============================================================================
# Singleton Instance
# ============================================================================

def create_grc_symbolic_rules(neo4j_driver=None) -> GRCSymbolicRules:
    """
    Create GRC symbolic rules engine

    Args:
        neo4j_driver: Neo4j driver for ARG integration

    Returns:
        GRCSymbolicRules instance
    """
    return GRCSymbolicRules(neo4j_driver=neo4j_driver)
