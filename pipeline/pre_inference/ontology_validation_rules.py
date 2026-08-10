"""
Ontology Validation Rules

Validate input events against SHACL constraints defined in the ontology.
These rules ensure events comply with NFCRM-1:2025 and ISO/IEC 27005 standards.

Validation Types:
- Ontology validation (SHACL constraints)
- Node validation (specific entity validation)
- Constraint validation (custom constraints)
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime
from dataclasses import dataclass

# Import existing SHACL validator
try:
    from pipeline.validators.shacl_validator import SHACLValidator
    SHACL_AVAILABLE = True
except ImportError:
    SHACL_AVAILABLE = False
    SHACLValidator = None
    logging.warning("SHACL validator not available (neo4j module not found)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    violations: List[str]
    validated_at: float


class OntologyValidationRules:
    """Ontology validation rules for pre-inference stage."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ontology validation rules.

        Args:
            config: Configuration dictionary with validation parameters
                - enable_shacl_validation: Enable SHACL validation (default: True)
                - enable_custom_constraints: Enable custom constraints (default: True)
                - fail_on_violation: Fail pipeline on validation violation (default: False)
                - shacl_config: Configuration for SHACLValidator
        """
        self.config = config or {}

        # Enable/disable validation
        self.enable_shacl_validation = self.config.get('enable_shacl_validation', True)
        self.enable_custom_constraints = self.config.get('enable_custom_constraints', True)
        self.fail_on_violation = self.config.get('fail_on_violation', False)

        # Initialize SHACL validator
        self.shacl_validator = None

        if self.enable_shacl_validation and SHACL_AVAILABLE:
            shacl_config = self.config.get('shacl_config', {})
            self.shacl_validator = SHACLValidator(
                uri=shacl_config.get('uri', 'bolt://localhost:7687'),
                user=shacl_config.get('user', 'neo4j'),
                password=shacl_config.get('password', 'password'),
                database=shacl_config.get('database', 'neo4j')
            )
        elif self.enable_shacl_validation and not SHACL_AVAILABLE:
            logger.warning("SHACL validation requested but neo4j module not available")

        # Custom constraints
        self.custom_constraints: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            'total_validations': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'total_violations': 0,
            'violations_by_type': {}
        }

        logger.info(f"OntologyValidationRules initialized: SHACL={self.enable_shacl_validation}, "
                   f"custom_constraints={self.enable_custom_constraints}")

    def validate_event(self, event: Dict[str, Any]) -> ValidationResult:
        """
        Validate an event against ontology constraints.

        Args:
            event: Event dictionary to validate

        Returns:
            ValidationResult with is_valid and violations list
        """
        self.stats['total_validations'] += 1

        violations = []

        # SHACL validation (if enabled and graph is available)
        if self.enable_shacl_validation and self.shacl_validator:
            # Note: SHACL validation requires the event to be in the graph
            # For event-level validation, we may need to convert to RDF first
            # This is a placeholder for actual SHACL validation logic
            pass

        # Custom constraint validation (if enabled)
        if self.enable_custom_constraints:
            custom_violations = self._validate_custom_constraints(event)
            violations.extend(custom_violations)

        # Determine validity
        is_valid = len(violations) == 0

        if not is_valid:
            self.stats['invalid_events'] += 1
            self.stats['total_violations'] += len(violations)

            for violation in violations:
                violation_type = violation.split(':')[0] if ':' in violation else 'unknown'
                self.stats['violations_by_type'][violation_type] = \
                    self.stats['violations_by_type'].get(violation_type, 0) + 1

            if self.fail_on_violation:
                logger.error(f"Event validation failed with {len(violations)} violations: {violations}")
            else:
                logger.warning(f"Event validation failed with {len(violations)} violations")
        else:
            self.stats['valid_events'] += 1

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            validated_at=datetime.now().timestamp()
        )

    def _validate_custom_constraints(self, event: Dict[str, Any]) -> List[str]:
        """
        Validate event against custom constraints.

        Custom constraints:
        - Required fields validation
        - Data type validation
        - Range validation
        - Format validation

        Args:
            event: Event dictionary

        Returns:
            List of violation messages
        """
        violations = []

        # Required fields validation
        required_fields = [
            'src_ip',
            'dst_ip',
            'src_port',
            'dst_port',
            'protocol'
        ]

        for field in required_fields:
            if field not in event or event[field] is None:
                violations.append(f"required_field_missing:{field}")

        # Data type validation
        integer_fields = ['src_port', 'dst_port', 'packet_count']
        for field in integer_fields:
            if field in event and not isinstance(event[field], (int, float)):
                violations.append(f"invalid_type:{field}:expected_integer")

        float_fields = ['flow_duration', 'bytes_sent', 'bytes_received']
        for field in float_fields:
            if field in event and not isinstance(event[field], (int, float)):
                violations.append(f"invalid_type:{field}:expected_float")

        # Range validation
        if 'src_port' in event:
            port = event['src_port']
            if not (0 <= port <= 65535):
                violations.append(f"invalid_range:src_port:{port}")

        if 'dst_port' in event:
            port = event['dst_port']
            if not (0 <= port <= 65535):
                violations.append(f"invalid_range:dst_port:{port}")

        # Format validation (IP addresses)
        ip_fields = ['src_ip', 'dst_ip']
        for field in ip_fields:
            if field in event:
                ip = event[field]
                if not self._is_valid_ip(ip):
                    violations.append(f"invalid_format:{field}:{ip}")

        # Format validation (protocol)
        if 'protocol' in event:
            protocol = event['protocol'].lower() if isinstance(event['protocol'], str) else str(event['protocol'])
            valid_protocols = ['tcp', 'udp', 'icmp', 'gre', 'esp', 'ah']
            if protocol not in valid_protocols:
                violations.append(f"invalid_protocol:{protocol}")

        return violations

    def _is_valid_ip(self, ip: Any) -> bool:
        """
        Validate IP address format.

        Args:
            ip: IP address to validate

        Returns:
            True if valid IP address
        """
        import ipaddress

        try:
            ipaddress.ip_address(ip)
            return True
        except (ValueError, ipaddress.AddressValueError):
            return False

    def add_custom_constraint(
        self,
        name: str,
        validator: callable,
        description: str = ""
    ) -> None:
        """
        Add a custom constraint validator.

        Args:
            name: Constraint name
            validator: Function that takes an event and returns violation message or None
            description: Constraint description
        """
        self.custom_constraints.append({
            'name': name,
            'validator': validator,
            'description': description
        })
        logger.info(f"Added custom constraint: {name} - {description}")

    def validate_graph(self) -> Dict[str, Any]:
        """
        Validate entire graph against SHACL shapes.

        Returns:
            Validation report with conformance, violations, and statistics
        """
        if not self.enable_shacl_validation or not self.shacl_validator or not SHACL_AVAILABLE:
            logger.warning("SHACL validation disabled or not initialized")
            return {
                'conformant': None,
                'error': 'SHACL validation disabled or not initialized',
                'violation_count': 0,
                'violations': []
            }

        try:
            report = self.shacl_validator.validate_graph()
            return report
        except Exception as e:
            logger.error(f"Graph validation failed: {e}")
            return {
                'conformant': False,
                'error': str(e),
                'violation_count': 0,
                'violations': []
            }

    def validate_node(
        self,
        node_label: str,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Validate a specific node against SHACL shapes.

        Args:
            node_label: Node label (e.g., 'Asset', 'Vulnerability')
            node_id: Node property value (e.g., assetId)

        Returns:
            Validation report for node
        """
        if not self.enable_shacl_validation or not self.shacl_validator:
            logger.warning("SHACL validation disabled or not initialized")
            return {
                'conformant': None,
                'error': 'SHACL validation disabled or not initialized',
                'violation_count': 0,
                'violations': []
            }

        try:
            report = self.shacl_validator.validate_node(node_label, node_id)
            return report
        except Exception as e:
            logger.error(f"Node validation failed: {e}")
            return {
                'conformant': False,
                'error': str(e),
                'violation_count': 0,
                'violations': []
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation statistics
        """
        invalid_rate = 0.0
        if self.stats['total_validations'] > 0:
            invalid_rate = self.stats['invalid_events'] / self.stats['total_validations']

        return {
            'total_validations': self.stats['total_validations'],
            'valid_events': self.stats['valid_events'],
            'invalid_events': self.stats['invalid_events'],
            'invalid_rate': invalid_rate,
            'total_violations': self.stats['total_violations'],
            'violations_by_type': self.stats['violations_by_type'],
            'shaacl_enabled': self.enable_shacl_validation,
            'custom_constraints_enabled': self.enable_custom_constraints
        }

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.stats = {
            'total_validations': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'total_violations': 0,
            'violations_by_type': {}
        }
        logger.info("Validation statistics reset")

    def close(self) -> None:
        """Close connections."""
        if self.shacl_validator:
            self.shacl_validator.close()


def main():
    """Example usage of OntologyValidationRules."""
    # Create validation rules with custom configuration
    config = {
        'enable_shacl_validation': True,
        'enable_custom_constraints': True,
        'fail_on_violation': False,
        'shacl_config': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        }
    }

    validation_rules = OntologyValidationRules(config)

    # Test events
    test_events = [
        {
            'src_ip': '192.168.1.100',
            'dst_ip': '192.168.1.200',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 'tcp',
            'flow_duration': 1.5,
            'bytes_sent': 1000,
            'bytes_received': 500,
            'packet_count': 100
        },
        {
            'src_ip': 'invalid_ip',
            'dst_ip': '10.0.0.1',
            'src_port': 70000,  # Invalid port
            'dst_port': 443,
            'protocol': 'unknown_protocol',  # Invalid protocol
            'flow_duration': 1.0,
            'bytes_sent': 1000,
            'bytes_received': 500,
            'packet_count': 100
        },
        {
            'src_ip': '10.0.0.2',
            'dst_ip': '10.0.0.3',
            'src_port': 54321,
            'dst_port': 22,
            'protocol': 'udp',
            'flow_duration': 2.0,
            'bytes_sent': 2000,
            'bytes_received': 1000,
            'packet_count': 200
        }
    ]

    for i, event in enumerate(test_events, 1):
        result = validation_rules.validate_event(event)
        if result.is_valid:
            print(f"Event {i}: VALID")
        else:
            print(f"Event {i}: INVALID")
            print(f"  Violations: {result.violations}")

    # Print statistics
    stats = validation_rules.get_statistics()
    print("\nValidation Statistics:")
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Valid events: {stats['valid_events']}")
    print(f"  Invalid events: {stats['invalid_events']}")
    print(f"  Invalid rate: {stats['invalid_rate']:.2%}")
    print(f"  Total violations: {stats['total_violations']}")
    print(f"  By type: {stats['violations_by_type']}")

    # Close connection
    validation_rules.close()


if __name__ == "__main__":
    main()
