"""
GRC Artifact Generation Engine for INTERSYMBOLIC-GRC

Generates GRC artifacts for:
- Risk Register entries (JSON, CSV formats)
- Audit logs (traceable to technical events)
- Continuity plan linkages
- Control-related evidence reports
- Compliance assessment reports

Features:
- Multi-format export (JSON, CSV, Markdown)
- Audit trail integration with session management
- Control-to-evidence mapping
- Compliance status calculation
- Historical tracking and trend analysis
"""

import json
import csv
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .grc_symbolic_rules import GRCSymbolicRules, RiskCase, AuditLog, ControlMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export format types"""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


class ArtifactCategory(Enum):
    """Artifact categories"""
    RISK_REGISTER = "risk_register"
    AUDIT_LOGS = "audit_logs"
    COMPLIANCE_REPORTS = "compliance_reports"
    CONTINUITY_PLANS = "continuity_plans"
    EVIDENCE_REPORTS = "evidence_reports"


@dataclass
class ArtifactExportOptions:
    """Export options"""
    format: ExportFormat = ExportFormat.JSON
    include_metadata: bool = True
    include_recommendations: bool = True
    include_history: bool = True
    output_dir: str = "./exports"


class GRCArtifactGenerator:
    """
    GRC artifact generation engine

    Generates comprehensive GRC artifacts for risk assessment results
    with multiple export formats and detailed metadata.
    """

    def __init__(
        self,
        grc_rules: Optional[GRCSymbolicRules] = None,
        output_dir: str = "./exports"
    ):
        """
        Initialize GRC artifact generator

        Args:
            grc_rules: GRC symbolic rules engine
            output_dir: Output directory for exports
        """
        self.grc_rules = grc_rules or GRCSymbolicRules()
        self.output_dir = Path(output_dir)
        self.session_metadata: Dict[str, Any] = {}
        self.artifact_history: List[Dict[str, Any]] = []

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """
        Set session metadata for artifact generation

        Args:
            session_id: Session identifier
            metadata: Session metadata
        """
        self.session_metadata[session_id] = metadata
        logger.info(f"Set session metadata for {session_id}")

    def generate_risk_register_entry(
        self,
        risk_case: RiskCase,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a risk register entry for a RiskCase

        Args:
            risk_case: Risk case entity
            session_id: Optional session identifier

        Returns:
            Risk register entry dictionary
        """
        entry = {
            'risk_id': risk_case.riskId,
            'risk_level': risk_case.riskLevel.value,
            'likelihood': risk_case.likelihood.value,
            'impact': risk_case.impact.value,
            'confidence': risk_case.confidence,
            'associated_controls': risk_case.associatedControls,
            'explanations': risk_case.explainedBySignals,
            'provenance_source': risk_case.provenanceSource,
            'provenance_timestamp': risk_case.provenanceTimestamp.isoformat(),
            'recommendations': risk_case.recommendations,
            'description': risk_case.description,
            'session_id': session_id or "default",
            'export_timestamp': datetime.now().isoformat(),
            'status': 'open'
        }

        # Add control details
        control_details = []
        for control_id in risk_case.associatedControls:
            control = self.grc_rules.NFCRM_CONTROLS.get(control_id) or self.grc_rules.ISO27005_CONTROLS.get(control_id)
            if control:
                control_details.append({
                    'control_id': control.controlId,
                    'control_name': control.controlName,
                    'provenance_source': control.provenanceSource,
                    'nfcrm_clause': control.nfcrmClause,
                    'iso27005_clause': control.iso27005Clause,
                    'severity': control.severity,
                    'description': control.description,
                    'mitigated': control.mitigated
                })

        entry['control_details'] = control_details

        return entry

    def generate_audit_log_entry(
        self,
        audit_log: AuditLog,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an audit log entry

        Args:
            audit_log: Audit log entity
            session_id: Optional session identifier

        Returns:
            Audit log entry dictionary
        """
        entry = {
            'audit_id': audit_log.auditId,
            'audit_timestamp': audit_log.auditTimestamp.isoformat(),
            'auditor': audit_log.auditor,
            'provenance_source': audit_log.provenanceSource,
            'provenance_timestamp': audit_log.provenanceTimestamp.isoformat(),
            'risk_case_id': audit_log.riskCaseId,
            'event_type': audit_log.eventType,
            'description': audit_log.description,
            'details': audit_log.details,
            'session_id': session_id or "default",
            'export_timestamp': datetime.now().isoformat()
        }

        return entry

    def generate_compliance_report(
        self,
        risk_cases: List[RiskCase],
        standards: List[str] = ["NFCRM-1:2025", "ISO/IEC 27005"]
    ) -> Dict[str, Any]:
        """
        Generate a compliance assessment report

        Args:
            risk_cases: List of RiskCases
            standards: List of compliance standards to assess against

        Returns:
            Compliance report dictionary
        """
        report = {
            'report_id': f"CP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'standards': standards,
            'total_risk_cases': len(risk_cases),
            'risk_cases': [],
            'summary': {
                'total_risks': 0,
                'critical_risks': 0,
                'high_risks': 0,
                'medium_risks': 0,
                'low_risks': 0,
                'overall_compliance_score': 0.0,
                'compliant_controls': 0,
                'non_compliant_controls': 0
            }
        }

        # Calculate compliance for each standard
        for standard in standards:
            standard_report = {
                'standard': standard,
                'compliant_controls': [],
                'non_compliant_controls': [],
                'total_controls': 0,
                'compliance_rate': 0.0,
                'controls': []
            }

            # Count controls for this standard
            controls_count = 0
            compliant_count = 0

            for risk_case in risk_cases:
                for control_id in risk_case.associatedControls:
                    control = self.grc_rules.NFCRM_CONTROLS.get(control_id)

                    if control and control.provenanceSource == standard:
                        controls_count += 1

                        # Simple compliance check: control must be associated with a RiskCase
                        # and have a mitigation plan
                        if control.mitigated:
                            compliant_count += 1
                            standard_report['compliant_controls'].append(control_id)
                        else:
                            standard_report['non_compliant_controls'].append(control_id)

                        standard_report['controls'].append({
                            'control_id': control.controlId,
                            'control_name': control.controlName,
                            'provenance_source': control.provenanceSource,
                            'nfcrm_clause': control.nfcrmClause,
                            'severity': control.severity,
                            'mitigated': control.mitigated
                        })

            # Calculate compliance rate
            if controls_count > 0:
                standard_report['compliance_rate'] = (compliant_count / controls_count) * 100
            else:
                standard_report['compliance_rate'] = 0.0

            report['risk_cases'].append(standard_report)

        # Calculate overall compliance score
        if report['risk_cases']:
            report['summary']['total_risks'] = len(risk_cases)
            report['summary']['critical_risks'] = sum(1 for rc in risk_cases if rc.riskLevel.value == 'critical')
            report['summary']['high_risks'] = sum(1 for rc in risk_cases if rc.riskLevel.value == 'high')
            report['summary']['medium_risks'] = sum(1 for rc in risk_cases if rc.riskLevel.value == 'medium')
            report['summary']['low_risks'] = sum(1 for rc in risk_cases if rc.riskLevel.value == 'low')

            # Calculate overall compliance score
            total_controls = sum(sr['controls'] for sr in report['risk_cases'])
            compliant_controls = sum(sr['compliant_controls'] for sr in report['risk_cases'])
            report['summary']['compliant_controls'] = compliant_controls
            report['summary']['non_compliant_controls'] = total_controls - compliant_controls

            if total_controls > 0:
                report['summary']['overall_compliance_score'] = (compliant_controls / total_controls) * 100
            else:
                report['summary']['overall_compliance_score'] = 0.0

        return report

    def generate_continuity_plan_linkage(
        self,
        risk_cases: List[RiskCase],
        continuity_plan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate continuity plan linkages for RiskCases

        Args:
            risk_cases: List of RiskCases
            continuity_plan_id: Optional continuity plan ID

        Returns:
            Continuity plan linkage dictionary
        """
        linkage = {
            'continuity_plan_id': continuity_plan_id or f"CP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'risk_case_linkages': [],
            'summary': {
                'total_risk_cases': len(risk_cases),
                'total_controls': 0,
                'critical_risks_linked': 0,
                'high_risks_linked': 0
            }
        }

        # Link each RiskCase to continuity plan actions
        for risk_case in risk_cases:
            linkage['risk_case_linkages'].append({
                'risk_case_id': risk_case.riskId,
                'risk_level': risk_case.riskLevel.value,
                'associated_controls': risk_case.associatedControls,
                'generated_actions': [
                    f"Review {cm.controlName} for mitigation planning"
                    for cm in self.grc_rules.map_signals_to_controls(risk_case.explainedBySignals)
                ] + [f"Update risk register for {risk_case.riskLevel.value} risk"],
                'timeline': "immediate" if risk_case.riskLevel.value == 'critical' else "30 days"
            })

            linkage['summary']['total_controls'] += len(risk_case.associatedControls)

            if risk_case.riskLevel.value == 'critical':
                linkage['summary']['critical_risks_linked'] += 1
            elif risk_case.riskLevel.value == 'high':
                linkage['summary']['high_risks_linked'] += 1

        return linkage

    def generate_evidence_report(
        self,
        risk_case: RiskCase,
        evidence_type: str = "compliance_evidence"
    ) -> Dict[str, Any]:
        """
        Generate an evidence report for a RiskCase

        Args:
            risk_case: Risk case entity
            evidence_type: Type of evidence (compliance_evidence, technical_evidence, audit_evidence)

        Returns:
            Evidence report dictionary
        """
        report = {
            'evidence_id': f"EVT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'evidence_type': evidence_type,
            'risk_case_id': risk_case.riskId,
            'generated_at': datetime.now().isoformat(),
            'risk_details': {
                'risk_level': risk_case.riskLevel.value,
                'likelihood': risk_case.likelihood.value,
                'impact': risk_case.impact.value,
                'confidence': risk_case.confidence
            },
            'controls_under_review': [],
            'evidence_sources': [],
            'assessment_results': []
        }

        # Add control details
        control_mappings = self.grc_rules.map_signals_to_controls(risk_case.explainedBySignals)
        for control in control_mappings:
            report['controls_under_review'].append({
                'control_id': control.controlId,
                'control_name': control.controlName,
                'provenance_source': control.provenanceSource,
                'nfcrm_clause': control.nfcrmClause,
                'severity': control.severity,
                'current_status': 'under_assessment'
            })

        # Add evidence sources
        report['evidence_sources'] = [
            'ML inference results',
            f'Risk signals: {len(risk_case.explainedBySignals)} signals',
            f'Associated controls: {len(risk_case.associatedControls)} controls',
            f'Provenance: {risk_case.provenanceSource}'
        ]

        # Add assessment results
        report['assessment_results'] = [
            {
                'component': 'risk_classification',
                'result': f'Risk classified as {risk_case.riskLevel.value}',
                'confidence': risk_case.confidence
            },
            {
                'component': 'control_mapping',
                'result': f'Linked to {len(risk_case.associatedControls)} controls',
                'confidence': 0.9
            },
            {
                'component': 'recommendations',
                'result': f'Generated {len(risk_case.recommendations)} recommendations',
                'confidence': 0.8
            }
        ]

        return report

    def export_to_format(
        self,
        artifact: Dict[str, Any],
        format_type: ExportFormat = ExportFormat.JSON,
        filename: Optional[str] = None
    ) -> str:
        """
        Export artifact to specified format

        Args:
            artifact: Artifact dictionary
            format_type: Export format
            filename: Optional custom filename

        Returns:
            Exported content string
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"artifact_{timestamp}"

        if format_type == ExportFormat.JSON:
            content = json.dumps(artifact, indent=2, default=str)
        elif format_type == ExportFormat.CSV:
            content = self._export_to_csv(artifact)
        elif format_type == ExportFormat.MARKDOWN:
            content = self._export_to_markdown(artifact)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Save to file
        file_path = self.output_dir / f"{filename}.{format_type.value}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Exported artifact to {file_path}")

        # Record in history
        self.artifact_history.append({
            'artifact_id': filename,
            'format': format_type.value,
            'path': str(file_path),
            'generated_at': datetime.now().isoformat()
        })

        return content

    def _export_to_csv(self, artifact: Dict[str, Any]) -> str:
        """
        Export artifact to CSV format

        Args:
            artifact: Artifact dictionary

        Returns:
            CSV string
        """
        # Extract relevant fields
        if isinstance(artifact, list):
            # Handle list of artifacts
            rows = []
            for item in artifact:
                row = {}
                for key, value in item.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value, default=str)
                    else:
                        row[key] = str(value)
                rows.append(row)
            return csv.DictToString(rows)
        else:
            # Handle single artifact
            rows = [artifact]
            return csv.DictToString(rows)

    def _export_to_markdown(self, artifact: Dict[str, Any]) -> str:
        """
        Export artifact to Markdown format

        Args:
            artifact: Artifact dictionary

        Returns:
            Markdown string
        """
        md_lines = []

        # Title
        md_lines.append(f"# Artifact Export\n")
        md_lines.append(f"**Generated:** {datetime.now().isoformat()}\n\n")

        # Summary
        if 'summary' in artifact:
            md_lines.append("## Summary\n")
            md_lines.append(self._dict_to_markdown(artifact['summary']))
            md_lines.append("\n")

        # Details
        md_lines.append("## Details\n")
        md_lines.append(self._dict_to_markdown(artifact))
        md_lines.append("\n")

        return "\n".join(md_lines)

    def _dict_to_markdown(self, data: Dict[str, Any], indent: int = 0) -> str:
        """
        Convert dictionary to Markdown format

        Args:
            data: Dictionary data
            indent: Indentation level

        Returns:
            Markdown string
        """
        md_lines = []

        for key, value in data.items():
            if isinstance(value, dict):
                md_lines.append(f"{'  ' * indent}### {key.capitalize()}\n")
                md_lines.append(self._dict_to_markdown(value, indent + 1))
            elif isinstance(value, list):
                if len(value) == 0:
                    md_lines.append(f"{'  ' * indent}- {key}: (empty)\n")
                else:
                    md_lines.append(f"{'  ' * indent}- {key}:\n")
                    for item in value:
                        if isinstance(item, dict):
                            md_lines.append(self._dict_to_markdown(item, indent + 2))
                        else:
                            md_lines.append(f"{'  ' * (indent + 2)}- {item}\n")
            else:
                md_lines.append(f"{'  ' * indent}- **{key}:** {value}\n")

        return "".join(md_lines)

    def export_all_artifacts(
        self,
        risk_cases: List[RiskCase],
        standards: List[str] = ["NFCRM-1:2025", "ISO/IEC 27005"],
        session_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export all GRC artifacts in multiple formats

        Args:
            risk_cases: List of RiskCases
            standards: List of compliance standards
            session_id: Optional session identifier

        Returns:
            Dictionary of exported file paths
        """
        exports = {}

        # Generate and export risk register
        risk_register = []
        for risk_case in risk_cases:
            entry = self.generate_risk_register_entry(risk_case, session_id)
            risk_register.append(entry)

        exports['risk_register_json'] = self.export_to_format(
            risk_register,
            ExportFormat.JSON,
            'risk_register'
        )
        exports['risk_register_csv'] = self.export_to_format(
            risk_register,
            ExportFormat.CSV,
            'risk_register'
        )

        # Generate and export compliance report
        compliance_report = self.generate_compliance_report(risk_cases, standards)
        exports['compliance_report_json'] = self.export_to_format(
            compliance_report,
            ExportFormat.JSON,
            'compliance_report'
        )
        exports['compliance_report_md'] = self.export_to_format(
            compliance_report,
            ExportFormat.MARKDOWN,
            'compliance_report'
        )

        # Generate and export continuity plan linkage
        continuity_plan = self.generate_continuity_plan_linkage(risk_cases)
        exports['continuity_plan_json'] = self.export_to_format(
            continuity_plan,
            ExportFormat.JSON,
            'continuity_plan'
        )

        # Generate evidence reports
        for risk_case in risk_cases:
            evidence_report = self.generate_evidence_report(risk_case)
            filename = f"evidence_{risk_case.riskId}"
            exports[f'{filename}_json'] = self.export_to_format(
                evidence_report,
                ExportFormat.JSON,
                filename
            )

        return exports

    def get_export_history(self) -> List[Dict[str, Any]]:
        """
        Get export history

        Returns:
            List of export records
        """
        return self.artifact_history.copy()


# ============================================================================
# Session Metadata Helpers
# ============================================================================

class SessionMetadataHelper:
    """Helper class for managing session metadata"""

    def __init__(self, artifact_generator: GRCArtifactGenerator):
        """
        Initialize session metadata helper

        Args:
            artifact_generator: GRC artifact generator instance
        """
        self.artifact_generator = artifact_generator

    def set_session_context(
        self,
        session_id: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Set session context

        Args:
            session_id: Session identifier
            context: Session context dictionary
        """
        self.artifact_generator.set_session_metadata(session_id, context)

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session context

        Args:
            session_id: Session identifier

        Returns:
            Session context or None
        """
        return self.artifact_generator.session_metadata.get(session_id)

    def clear_session_context(self, session_id: str) -> None:
        """
        Clear session context

        Args:
            session_id: Session identifier
        """
        if session_id in self.artifact_generator.session_metadata:
            del self.artifact_generator.session_metadata[session_id]


# ============================================================================
# Singleton Instance
# ============================================================================

def create_grc_artifact_generator(
    grc_rules: Optional[GRCSymbolicRules] = None,
    output_dir: str = "./exports"
) -> GRCArtifactGenerator:
    """
    Create GRC artifact generator

    Args:
        grc_rules: GRC symbolic rules engine
        output_dir: Output directory for exports

    Returns:
        GRCArtifactGenerator instance
    """
    return GRCArtifactGenerator(grc_rules, output_dir)
