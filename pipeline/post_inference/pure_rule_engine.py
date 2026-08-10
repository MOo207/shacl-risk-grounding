"""
Pure Rule Engine for Risk Assessment

Implements expert-rule-based risk assessment without ML models.
Based on research/011-pure-rule-baseline-implementation-strategies.md
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

# Optional imports - these are part of the larger INTERSYMBOLIC-GRC pipeline
try:
    from pipeline.post_inference.grc_symbolic_rules import GRCMapper
    GRC_MAPPER_AVAILABLE = True
except ImportError:
    GRC_MAPPER_AVAILABLE = False
    GRCMapper = None


class RuleCategory(Enum):
    """Rule categories for categorization and filtering"""
    EVENT_BASED = "event_based"
    ASSET_BASED = "asset_based"
    RELATIONSHIP_BASED = "relationship_based"
    COMPLIANCE_BASED = "compliance_based"
    TIME_BASED = "time_based"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RuleCondition:
    """A single condition in a rule"""
    field: str
    operator: str
    value: Any
    description: str = ""


@dataclass
class Rule:
    """A single expert rule definition"""
    rule_id: str
    name: str
    category: RuleCategory
    priority: int  # 1-100, higher is more critical
    description: str
    conditions: List[RuleCondition]
    score_modifier: int  # -50 to +50, negative is worse
    explanation_template: str
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category.value,
            "priority": self.priority,
            "description": self.description,
            "conditions": [
                {
                    "field": c.field,
                    "operator": c.operator,
                    "value": c.value,
                    "description": c.description
                }
                for c in self.conditions
            ],
            "score_modifier": self.score_modifier,
            "explanation_template": self.explanation_template,
            "enabled": self.enabled
        }


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a rule"""
    rule: Rule
    matched: bool
    score_earned: int  # 0 to 20 (evidence weight)
    explanation: str
    evidence_items: List[str]
    conditions_met: List[RuleCondition]
    conditions_not_met: List[RuleCondition]


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment result"""
    risk_score: float  # 0 to 100
    risk_level: RiskLevel
    rules_evaluated: List[RuleEvaluationResult]
    total_score_modifier: int  # -50 to +50
    total_evidence_weight: int  # 0 to 20
    compliance_mapping: Optional[Dict[str, Any]] = None
    explainable_chain: List[str] = field(default_factory=list)


class PureRuleEngine:
    """
    Pure rule-based risk assessment engine.

    Rules are encoded by domain experts and evaluate events/asset/relationship
    data against compliance frameworks (NFCRM-1:2025, ISO/IEC 27005).
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the pure rule engine.

        Args:
            config_file: Path to JSON configuration file containing rules
        """
        self.rules: List[Rule] = []

        # Initialize optional dependencies if available
        self.grc_mapper = GRCMapper() if GRC_MAPPER_AVAILABLE else None

        # Load rules (default or from file)
        self.load_rules(config_file)

    def load_rules(self, config_file: Optional[str] = None) -> None:
        """
        Load rules from file or use default rules.

        Args:
            config_file: Optional path to JSON configuration file
        """
        if config_file:
            with open(config_file, 'r') as f:
                rules_data = json.load(f)
                for rule_data in rules_data:
                    self.rules.append(self._create_rule_from_dict(rule_data))
        else:
            # Load default rules
            self.rules = self._get_default_rules()

    def _create_rule_from_dict(self, data: Dict[str, Any]) -> Rule:
        """Create Rule object from dictionary data"""
        return Rule(
            rule_id=data["rule_id"],
            name=data["name"],
            category=RuleCategory(data["category"]),
            priority=data["priority"],
            description=data["description"],
            conditions=[
                RuleCondition(
                    field=c["field"],
                    operator=c["operator"],
                    value=c["value"],
                    description=c.get("description", "")
                )
                for c in data["conditions"]
            ],
            score_modifier=data["score_modifier"],
            explanation_template=data["explanation_template"],
            enabled=data.get("enabled", True)
        )

    def _get_default_rules(self) -> List[Rule]:
        """
        Get default expert rules for risk assessment.

        These rules are based on common cybersecurity patterns and
        compliance requirements for NFCRM-1:2025 and ISO/IEC 27005.
        """
        rules = [
            # EVENT-BASED RULES
            Rule(
                rule_id="EVT-001",
                name="Critical CVE in Asset",
                category=RuleCategory.EVENT_BASED,
                priority=100,
                description="Asset has a critical CVE with no patch applied",
                conditions=[
                    RuleCondition(field="cve_severity", operator="==", value="critical"),
                    RuleCondition(field="patch_status", operator="==", value="unpatched")
                ],
                score_modifier=-50,
                explanation_template="Asset {asset_id} has Critical CVE {cve_id} but no patch applied. This is a severe risk."
            ),

            Rule(
                rule_id="EVT-002",
                name="Privileged Account No MFA",
                category=RuleCategory.EVENT_BASED,
                priority=90,
                description="Privileged account lacks multi-factor authentication",
                conditions=[
                    RuleCondition(field="account_type", operator="==", value="privileged"),
                    RuleCondition(field="mfa_enabled", operator="==", value=False)
                ],
                score_modifier=-40,
                explanation_template="Privileged account {account_id} has no MFA enabled. This is a critical vulnerability."
            ),

            Rule(
                rule_id="EVT-003",
                name="Internet-facing Asset No Firewall",
                category=RuleCategory.EVENT_BASED,
                priority=85,
                description="Asset exposed to internet without firewall rule",
                conditions=[
                    RuleCondition(field="internet_facing", operator="==", value=True),
                    RuleCondition(field="firewall_rules", operator="==", value=0)
                ],
                score_modifier=-35,
                explanation_template="Asset {asset_id} is internet-facing with no firewall rules. This increases attack surface."
            ),

            Rule(
                rule_id="EVT-004",
                name="PII Data Without Encryption",
                category=RuleCategory.EVENT_BASED,
                priority=95,
                description="Asset stores PII data without encryption at rest",
                conditions=[
                    RuleCondition(field="data_classification", operator="==", value="PII"),
                    RuleCondition(field="encryption_at_rest", operator="==", value=False)
                ],
                score_modifier=-45,
                explanation_template="Asset {asset_id} stores PII data without encryption at rest. This is a severe compliance issue."
            ),

            Rule(
                rule_id="EVT-005",
                name="Anomalous Traffic Pattern",
                category=RuleCategory.EVENT_BASED,
                priority=70,
                description="Unusual network traffic pattern detected",
                conditions=[
                    RuleCondition(field="traffic_anomaly", operator="==", value=True),
                    RuleCondition(field="anomaly_score", operator=">", value=0.8)
                ],
                score_modifier=-20,
                explanation_template="Asset {asset_id} shows anomalous traffic pattern with score {anomaly_score}. Review is required."
            ),

            # ASSET-BASED RULES
            Rule(
                rule_id="AST-001",
                name="Outdated Software Version",
                category=RuleCategory.ASSET_BASED,
                priority=80,
                description="Software version exceeds EOL date",
                conditions=[
                    RuleCondition(field="software_name", operator="in", value=["Windows", "Linux", "RHEL"]),
                    RuleCondition(field="version", operator=">", value="EOL"),
                    RuleCondition(field="patch_available", operator="==", value=False)
                ],
                score_modifier=-30,
                explanation_template="Asset {asset_id} uses outdated software {software_name} {version} which is past EOL with no patch available."
            ),

            Rule(
                rule_id="AST-002",
                name="Missing Security Controls",
                category=RuleCategory.ASSET_BASED,
                priority=75,
                description="Asset missing required security controls",
                conditions=[
                    RuleCondition(field="security_control_count", operator="<", value=3),
                    RuleCondition(field="required_controls", operator="==", value=True)
                ],
                score_modifier=-25,
                explanation_template="Asset {asset_id} is missing required security controls. Only {security_control_count} controls configured."
            ),

            # RELATIONSHIP-BASED RULES
            Rule(
                rule_id="REL-001",
                name="Zero-day Vulnerability Chain",
                category=RuleCategory.RELATIONSHIP_BASED,
                priority=95,
                description="Multiple assets share zero-day vulnerability",
                conditions=[
                    RuleCondition(field="vulnerability_type", operator="==", value="zero-day"),
                    RuleCondition(field="affected_assets_count", operator=">", value=5)
                ],
                score_modifier=-50,
                explanation_template="Zero-day vulnerability {vulnerability_id} affects {affected_assets_count} assets, creating widespread risk."
            ),

            Rule(
                rule_id="REL-002",
                name="Critical Asset Exposed to Untrusted Network",
                category=RuleCategory.RELATIONSHIP_BASED,
                priority=90,
                description="Critical asset connected to untrusted network",
                conditions=[
                    RuleCondition(field="asset_criticality", operator="==", value="critical"),
                    RuleCondition(field="network_trust", operator="==", value="untrusted"),
                    RuleCondition(field="network_category", operator="==", value="internet")
                ],
                score_modifier=-45,
                explanation_template="Critical asset {asset_id} is connected to untrusted network {network_name}. This is a severe risk."
            ),

            # COMPLIANCE-BASED RULES
            Rule(
                rule_id="CMP-001",
                name="NFCRM-1:2025 Compliance Gap",
                category=RuleCategory.COMPLIANCE_BASED,
                priority=100,
                description="Control from NFCRM-1:2025 not implemented",
                conditions=[
                    RuleCondition(field="control_id", operator="in", value=["NFCRM-1:2025-§6.7", "NFCRM-1:2025-§6.9"]),
                    RuleCondition(field="implementation_status", operator="==", value="not_implemented")
                ],
                score_modifier=-50,
                explanation_template="NFCRM-1:2025 control {control_id} is not implemented. This is a critical compliance gap."
            ),

            Rule(
                rule_id="CMP-002",
                name="ISO/IEC 27005 Risk Assessment Missing",
                category=RuleCategory.COMPLIANCE_BASED,
                priority=85,
                description="No risk assessment completed for asset",
                conditions=[
                    RuleCondition(field="standard", operator="==", value=["ISO/IEC 27005", "NFCRM-1:2025"]),
                    RuleCondition(field="risk_assessment_complete", operator="==", value=False)
                ],
                score_modifier=-35,
                explanation_template="ISO/IEC 27005 risk assessment is missing for asset {asset_id}. Compliance requires regular assessments."
            ),

            # TIME-BASED RULES
            Rule(
                rule_id="TIM-001",
                name="Patch Not Applied in Time Window",
                category=RuleCategory.TIME_BASED,
                priority=80,
                description="Critical patch available but not applied after 30 days",
                conditions=[
                    RuleCondition(field="patch_available", operator="==", value=True),
                    RuleCondition(field="days_since_patch_release", operator=">", value=30),
                    RuleCondition(field="patch_applied", operator="==", value=False)
                ],
                score_modifier=-30,
                explanation_template="Critical patch {patch_id} was released {days_since_patch_release} days ago but has not been applied."
            ),

            Rule(
                rule_id="TIM-002",
                name="Access Rights Not Reviewed in 90 Days",
                category=RuleCategory.TIME_BASED,
                priority=70,
                description="User access rights not reviewed for 90+ days",
                conditions=[
                    RuleCondition(field="last_access_review_date", operator="<", value="-90"),
                    RuleCondition(field="access_type", operator="==", value="privileged")
                ],
                score_modifier=-20,
                explanation_template="Privileged access rights for user {user_id} have not been reviewed for {review_days} days. Periodic review is required."
            ),
        ]

        return rules

    def evaluate_rule(self, rule: Rule, data: Dict[str, Any]) -> RuleEvaluationResult:
        """
        Evaluate a single rule against event/asset data.

        Args:
            rule: Rule to evaluate
            data: Event/asset data to evaluate against

        Returns:
            RuleEvaluationResult with match status and details
        """
        conditions_met = []
        conditions_not_met = []

        for condition in rule.conditions:
            if self._condition_met(condition, data):
                conditions_met.append(condition)
            else:
                conditions_not_met.append(condition)

        matched = len(conditions_met) == len(rule.conditions) and rule.enabled

        # Calculate evidence weight (0-20) based on number of conditions met
        if matched:
            evidence_weight = min(len(conditions_met), 20)  # Cap at 20
        else:
            evidence_weight = 0

        # Generate explanation
        explanation = rule.explanation_template.format(
            asset_id=data.get("asset_id", "unknown"),
            account_id=data.get("account_id", "unknown"),
            cve_id=data.get("cve_id", "unknown"),
            vulnerability_id=data.get("vulnerability_id", "unknown"),
            anomaly_score=data.get("anomaly_score", "N/A"),
            software_name=data.get("software_name", "unknown"),
            version=data.get("version", "unknown"),
            asset_criticality=data.get("asset_criticality", "unknown"),
            network_name=data.get("network_name", "unknown"),
            control_id=data.get("control_id", "unknown"),
            user_id=data.get("user_id", "unknown"),
            review_days=data.get("review_days", "N/A"),
            days_since_patch_release=data.get("days_since_patch_release", "N/A"),
            patch_id=data.get("patch_id", "unknown"),
            network_category=data.get("network_category", "unknown"),
            data_classification=data.get("data_classification", "unknown"),
            patch_status=data.get("patch_status", "unknown"),
            mfa_enabled=data.get("mfa_enabled", False),
            firewall_rules=data.get("firewall_rules", 0),
            encryption_at_rest=data.get("encryption_at_rest", False),
            security_control_count=data.get("security_control_count", 0),
            required_controls=data.get("required_controls", False),
            affected_assets_count=data.get("affected_assets_count", 0),
            vulnerability_type=data.get("vulnerability_type", "unknown")
        )

        return RuleEvaluationResult(
            rule=rule,
            matched=matched,
            score_earned=evidence_weight,
            explanation=explanation,
            evidence_items=[condition.field for condition in conditions_met],
            conditions_met=conditions_met,
            conditions_not_met=conditions_not_met
        )

    def _condition_met(self, condition: RuleCondition, data: Dict[str, Any]) -> bool:
        """
        Evaluate a single condition against data.

        Args:
            condition: Condition to evaluate
            data: Event/asset data

        Returns:
            True if condition is met, False otherwise
        """
        value = data.get(condition.field)

        if value is None:
            return False

        # Handle "in" operator for list values
        if condition.operator == "in":
            return value in condition.value

        # Handle comparison operators
        if condition.operator == "==":
            return value == condition.value
        elif condition.operator == "!=":
            return value != condition.value
        elif condition.operator == ">":
            return value > condition.value
        elif condition.operator == ">=":
            return value >= condition.value
        elif condition.operator == "<":
            return value < condition.value
        elif condition.operator == "<=":
            return value <= condition.value

        # Unknown operator
        return False

    def assess_risk(self, data: Dict[str, Any]) -> RiskAssessmentResult:
        """
        Perform complete risk assessment using all rules.

        Args:
            data: Event/asset data to assess

        Returns:
            RiskAssessmentResult with risk score and level
        """
        # Filter enabled rules
        active_rules = [r for r in self.rules if r.enabled]

        results = []
        total_score_modifier = 0
        total_evidence_weight = 0

        for rule in active_rules:
            result = self.evaluate_rule(rule, data)
            results.append(result)

            if result.matched:
                total_score_modifier += rule.score_modifier
                total_evidence_weight += result.score_earned

        # Calculate risk score (0-100)
        # Baseline: 50, adjusted by score_modifier (-50 to +50) and evidence_weight (0-20)
        risk_score = 50 + total_score_modifier + (total_evidence_weight / 5)  # evidence_weight / 5 adds 0-4 to score

        # Clamp risk score to 0-100 range
        risk_score = max(0, min(100, risk_score))

        # Determine risk level
        if risk_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 40:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Generate explainable chain
        explainable_chain = []
        for result in results:
            if result.matched:
                explainable_chain.append({
                    "rule_id": result.rule.rule_id,
                    "rule_name": result.rule.name,
                    "explanation": result.explanation,
                    "score_modifier": result.rule.score_modifier
                })

        # Create compliance mapping
        compliance_mapping = self._create_compliance_mapping(results)

        return RiskAssessmentResult(
            risk_score=risk_score,
            risk_level=risk_level,
            rules_evaluated=results,
            total_score_modifier=total_score_modifier,
            total_evidence_weight=total_evidence_weight,
            compliance_mapping=compliance_mapping,
            explainable_chain=explainable_chain
        )

    def _create_compliance_mapping(self, results: List[RuleEvaluationResult]) -> Dict[str, Any]:
        """
        Create compliance mapping based on evaluated rules.

        Args:
            results: List of rule evaluation results

        Returns:
            Compliance mapping dictionary
        """
        # Count violations by category
        violations_by_category = {}
        for result in results:
            if result.matched:
                category = result.rule.category.value
                violations_by_category[category] = violations_by_category.get(category, 0) + 1

        # Map to NFCRM-1:2025 and ISO/IEC 27005 controls
        compliance_mapping = {
            "nfcrm_compliance": {
                "compliant": True,
                "violations": violations_by_category.get("compliance_based", 0),
                "controls": [],
                "requirements_met": []
            },
            "iso27005_compliance": {
                "compliant": True,
                "violations": violations_by_category.get("compliance_based", 0),
                "controls": [],
                "risk_assessed": True
            },
            "overall_compliance": {
                "status": "compliant" if violations_by_category.get("compliance_based", 0) == 0 else "non_compliant",
                "total_violations": violations_by_category.get("compliance_based", 0)
            }
        }

        # Add control IDs from matched rules
        for result in results:
            if result.matched and result.rule.category == RuleCategory.COMPLIANCE_BASED:
                compliance_mapping["nfcrm_compliance"]["controls"].append({
                    "control_id": result.rule.rule_id,
                    "status": "non_compliant",
                    "requirement": result.rule.description
                })
                compliance_mapping["nfcrm_compliance"]["requirements_met"].append(False)

        return compliance_mapping

    def batch_assess(self, data_list: List[Dict[str, Any]]) -> List[RiskAssessmentResult]:
        """
        Assess multiple events/asset data.

        Args:
            data_list: List of event/asset data to assess

        Returns:
            List of RiskAssessmentResult objects
        """
        return [self.assess_risk(data) for data in data_list]

    def get_rules_by_category(self, category: RuleCategory) -> List[Rule]:
        """
        Get all rules of a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of rules in the category
        """
        return [rule for rule in self.rules if rule.category == category]

    def get_high_priority_rules(self, threshold: int = 80) -> List[Rule]:
        """
        Get high-priority rules.

        Args:
            threshold: Minimum priority score

        Returns:
            List of high-priority rules
        """
        return [rule for rule in self.rules if rule.priority >= threshold and rule.enabled]

    def export_rules(self, output_file: str):
        """
        Export all rules to JSON file.

        Args:
            output_file: Path to output file
        """
        rules_data = [rule.to_dict() for rule in self.rules]
        with open(output_file, 'w') as f:
            json.dump(rules_data, f, indent=2, default=str)

    def save_rule_sets(self, base_dir: str = "rules"):
        """
        Save different rule sets to JSON files.

        Args:
            base_dir: Base directory for rule files
        """
        import os
        os.makedirs(base_dir, exist_ok=True)

        # Save all rules
        self.export_rules(f"{base_dir}/all_rules.json")

        # Save by category
        for category in RuleCategory:
            rules = self.get_rules_by_category(category)
            category_rules = [rule.to_dict() for rule in rules]
            with open(f"{base_dir}/{category.value}_rules.json", 'w') as f:
                json.dump(category_rules, f, indent=2, default=str)


if __name__ == "__main__":
    # Test the pure rule engine
    engine = PureRuleEngine()

    # Test data: Critical CVE with no patch
    test_data = {
        "asset_id": "SRV-001",
        "cve_id": "CVE-2026-1234",
        "cve_severity": "critical",
        "patch_status": "unpatched",
        "patch_id": "Patch-XYZ-001",
        "days_since_patch_release": 45
    }

    result = engine.assess_risk(test_data)

    print("=" * 80)
    print("PURE RULE ENGINE - RISK ASSESSMENT RESULT")
    print("=" * 80)
    print(f"Risk Score: {result.risk_score:.1f}/100")
    print(f"Risk Level: {result.risk_level.value.upper()}")
    print(f"Total Score Modifier: {result.total_score_modifier}")
    print(f"Total Evidence Weight: {result.total_evidence_weight}")
    print()

    print("EXPLAINABLE CHAIN:")
    for i, chain_item in enumerate(result.explainable_chain, 1):
        print(f"{i}. {chain_item['rule_name']}")
        print(f"   Score Modifier: {chain_item['score_modifier']}")
        print(f"   Explanation: {chain_item['explanation']}")
        print()

    print("COMPLIANCE MAPPING:")
    print(json.dumps(result.compliance_mapping, indent=2, default=str))

    # Export rules
    print("\nExporting rules...")
    engine.save_rule_sets()
    print(f"Rules saved to rules/ directory")
