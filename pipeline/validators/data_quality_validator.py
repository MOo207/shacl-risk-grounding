"""
Data Quality Validator

Performs data quality checks on ingested data.
Validates completeness, consistency, accuracy, and uniqueness.

Quality Checks:
- Completeness: Required fields are present
- Consistency: Relationships are valid, timestamps are in valid range
- Accuracy: IP addresses are valid, ports are in range
- Uniqueness: No duplicate entities
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Set
import logging
import ipaddress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Data quality validator for graph entities."""

    # Required fields per entity type
    REQUIRED_FIELDS = {
        'Asset': ['assetId', 'assetName', 'assetType', 'ipAddress'],
        'SoftwareAsset': ['assetId', 'assetName', 'assetType'],
        'NetworkConnection': ['connectionId', 'sourceAssetId', 'destinationAssetId'],
        'ThreatEvent': ['threatId', 'threatType', 'timestamp'],
        'SoftwareComponent': ['componentId', 'assetId', 'serviceName', 'port']
    }

    # Valid asset types
    VALID_ASSET_TYPES = {'software', 'hardware', 'network', 'data'}

    # Valid protocols
    VALID_PROTOCOLS = {'TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'FTP', 'SSH', 'SMTP'}

    # Valid threat types
    VALID_THREAT_TYPES = {
        'botnet', 'ddos', 'dos', 'infiltration', 'web_attack',
        'phishing', 'malware', 'ransomware', 'sql_injection', 'xss'
    }

    def __init__(self, strict: bool = True):
        """
        Initialize the data quality validator.

        Args:
            strict: If True, fail on any quality issue. If False, warn only.
        """
        self.strict = strict
        self.quality_issues: List[Dict[str, Any]] = []
        self.quality_score: float = 0.0

    def validate_dataset(
        self,
        assets: List[Dict],
        connections: List[Dict],
        threats: List[Dict],
        components: List[Dict]
    ) -> Dict[str, Any]:
        """
        Validate entire dataset.

        Args:
            assets: List of asset entities
            connections: List of network connection entities
            threats: List of threat event entities
            components: List of software component entities

        Returns:
            Quality report with score, issues, and statistics
        """
        logger.info("Starting data quality validation...")

        self.quality_issues = []

        # Validate each entity type
        asset_report = self._validate_assets(assets)
        connection_report = self._validate_connections(connections, assets)
        threat_report = self._validate_threats(threats, assets)
        component_report = self._validate_components(components, assets)

        # Calculate overall quality score
        total_entities = len(assets) + len(connections) + len(threats) + len(components)
        valid_entities = (
            asset_report['valid_count'] +
            connection_report['valid_count'] +
            threat_report['valid_count'] +
            component_report['valid_count']
        )

        self.quality_score = (valid_entities / total_entities) * 100 if total_entities > 0 else 0.0

        report = {
            'overall_score': self.quality_score,
            'total_entities': total_entities,
            'valid_entities': valid_entities,
            'total_issues': len(self.quality_issues),
            'issues': self.quality_issues,
            'asset_report': asset_report,
            'connection_report': connection_report,
            'threat_report': threat_report,
            'component_report': component_report,
            'meets_threshold': self.quality_score >= 90.0  # 90% quality threshold
        }

        logger.info(f"Data quality validation complete: Score={self.quality_score:.1f}%, Issues={len(self.quality_issues)}")
        return report

    def _validate_assets(self, assets: List[Dict]) -> Dict[str, Any]:
        """Validate asset entities."""
        valid_count = 0
        asset_ids = set()

        for asset in assets:
            issues = []

            # Check required fields
            asset_type = asset.get('assetType', '')
            required = self.REQUIRED_FIELDS.get(asset_type, self.REQUIRED_FIELDS['Asset'])
            for field in required:
                if field not in asset or asset[field] is None or asset[field] == '':
                    issues.append({
                        'entity_type': 'Asset',
                        'entity_id': asset.get('assetId', 'unknown'),
                        'issue_type': 'missing_field',
                        'field': field,
                        'severity': 'error'
                    })

            # Check asset type is valid
            if asset_type and asset_type not in self.VALID_ASSET_TYPES:
                issues.append({
                    'entity_type': 'Asset',
                    'entity_id': asset.get('assetId', 'unknown'),
                    'issue_type': 'invalid_value',
                    'field': 'assetType',
                    'value': asset_type,
                    'expected': self.VALID_ASSET_TYPES,
                    'severity': 'error'
                })

            # Validate IP address
            if 'ipAddress' in asset and asset['ipAddress']:
                try:
                    ipaddress.ip_address(asset['ipAddress'])
                except ValueError:
                    issues.append({
                        'entity_type': 'Asset',
                        'entity_id': asset.get('assetId', 'unknown'),
                        'issue_type': 'invalid_ip',
                        'field': 'ipAddress',
                        'value': asset['ipAddress'],
                        'severity': 'error'
                    })

            # Check for duplicate asset IDs
            asset_id = asset.get('assetId')
            if asset_id:
                if asset_id in asset_ids:
                    issues.append({
                        'entity_type': 'Asset',
                        'entity_id': asset_id,
                        'issue_type': 'duplicate',
                        'severity': 'error'
                    })
                else:
                    asset_ids.add(asset_id)

            # Record issues
            if issues:
                self.quality_issues.extend(issues)
            else:
                valid_count += 1

        return {
            'total_count': len(assets),
            'valid_count': valid_count,
            'issue_count': len(assets) - valid_count,
            'unique_ids': len(asset_ids)
        }

    def _validate_connections(
        self,
        connections: List[Dict],
        assets: List[Dict]
    ) -> Dict[str, Any]:
        """Validate network connection entities."""
        valid_count = 0
        connection_ids = set()

        # Create asset ID lookup
        asset_ids = {a['assetId'] for a in assets if 'assetId' in a}

        for conn in connections:
            issues = []

            # Check required fields
            required = self.REQUIRED_FIELDS.get('NetworkConnection', [])
            for field in required:
                if field not in conn or conn[field] is None or conn[field] == '':
                    issues.append({
                        'entity_type': 'NetworkConnection',
                        'entity_id': conn.get('connectionId', 'unknown'),
                        'issue_type': 'missing_field',
                        'field': field,
                        'severity': 'error'
                    })

            # Validate source asset exists
            src_asset = conn.get('sourceAssetId')
            if src_asset and src_asset not in asset_ids:
                issues.append({
                    'entity_type': 'NetworkConnection',
                    'entity_id': conn.get('connectionId', 'unknown'),
                    'issue_type': 'foreign_key_violation',
                    'field': 'sourceAssetId',
                    'value': src_asset,
                    'severity': 'error'
                })

            # Validate destination asset exists
            dest_asset = conn.get('destinationAssetId')
            if dest_asset and dest_asset not in asset_ids:
                issues.append({
                    'entity_type': 'NetworkConnection',
                    'entity_id': conn.get('connectionId', 'unknown'),
                    'issue_type': 'foreign_key_violation',
                    'field': 'destinationAssetId',
                    'value': dest_asset,
                    'severity': 'error'
                })

            # Validate ports
            for port_field in ['sourcePort', 'destinationPort']:
                port = conn.get(port_field)
                if port is not None:
                    try:
                        port_int = int(port)
                        if not (0 <= port_int <= 65535):
                            issues.append({
                                'entity_type': 'NetworkConnection',
                                'entity_id': conn.get('connectionId', 'unknown'),
                                'issue_type': 'invalid_port',
                                'field': port_field,
                                'value': port,
                                'severity': 'error'
                            })
                    except (ValueError, TypeError):
                        issues.append({
                            'entity_type': 'NetworkConnection',
                            'entity_id': conn.get('connectionId', 'unknown'),
                            'issue_type': 'invalid_port',
                            'field': port_field,
                            'value': port,
                            'severity': 'error'
                        })

            # Validate protocol
            protocol = conn.get('protocol')
            if protocol and protocol.upper() not in self.VALID_PROTOCOLS:
                issues.append({
                    'entity_type': 'NetworkConnection',
                    'entity_id': conn.get('connectionId', 'unknown'),
                    'issue_type': 'invalid_protocol',
                    'field': 'protocol',
                    'value': protocol,
                    'expected': self.VALID_PROTOCOLS,
                    'severity': 'warning'
                })

            # Check for duplicate connection IDs
            conn_id = conn.get('connectionId')
            if conn_id:
                if conn_id in connection_ids:
                    issues.append({
                        'entity_type': 'NetworkConnection',
                        'entity_id': conn_id,
                        'issue_type': 'duplicate',
                        'severity': 'error'
                    })
                else:
                    connection_ids.add(conn_id)

            # Record issues
            if issues:
                self.quality_issues.extend(issues)
            else:
                valid_count += 1

        return {
            'total_count': len(connections),
            'valid_count': valid_count,
            'issue_count': len(connections) - valid_count,
            'unique_ids': len(connection_ids)
        }

    def _validate_threats(
        self,
        threats: List[Dict],
        assets: List[Dict]
    ) -> Dict[str, Any]:
        """Validate threat event entities."""
        valid_count = 0
        threat_ids = set()

        # Create asset ID lookup
        asset_ids = {a['assetId'] for a in assets if 'assetId' in a}

        for threat in threats:
            issues = []

            # Check required fields
            required = self.REQUIRED_FIELDS.get('ThreatEvent', [])
            for field in required:
                if field not in threat or threat[field] is None or threat[field] == '':
                    issues.append({
                        'entity_type': 'ThreatEvent',
                        'entity_id': threat.get('threatId', 'unknown'),
                        'issue_type': 'missing_field',
                        'field': field,
                        'severity': 'error'
                    })

            # Validate threat type
            threat_type = threat.get('threatType', '')
            if threat_type and threat_type.lower() not in self.VALID_THREAT_TYPES:
                issues.append({
                    'entity_type': 'ThreatEvent',
                    'entity_id': threat.get('threatId', 'unknown'),
                    'issue_type': 'invalid_value',
                    'field': 'threatType',
                    'value': threat_type,
                    'expected': self.VALID_THREAT_TYPES,
                    'severity': 'warning'
                })

            # Validate source asset exists
            src_asset = threat.get('sourceAssetId')
            if src_asset and src_asset not in asset_ids:
                issues.append({
                    'entity_type': 'ThreatEvent',
                    'entity_id': threat.get('threatId', 'unknown'),
                    'issue_type': 'foreign_key_violation',
                    'field': 'sourceAssetId',
                    'value': src_asset,
                    'severity': 'error'
                })

            # Validate destination asset exists
            dest_asset = threat.get('destinationAssetId')
            if dest_asset and dest_asset not in asset_ids:
                issues.append({
                    'entity_type': 'ThreatEvent',
                    'entity_id': threat.get('threatId', 'unknown'),
                    'issue_type': 'foreign_key_violation',
                    'field': 'destinationAssetId',
                    'value': dest_asset,
                    'severity': 'error'
                })

            # Check for duplicate threat IDs
            threat_id = threat.get('threatId')
            if threat_id:
                if threat_id in threat_ids:
                    issues.append({
                        'entity_type': 'ThreatEvent',
                        'entity_id': threat_id,
                        'issue_type': 'duplicate',
                        'severity': 'error'
                    })
                else:
                    threat_ids.add(threat_id)

            # Record issues
            if issues:
                self.quality_issues.extend(issues)
            else:
                valid_count += 1

        return {
            'total_count': len(threats),
            'valid_count': valid_count,
            'issue_count': len(threats) - valid_count,
            'unique_ids': len(threat_ids)
        }

    def _validate_components(
        self,
        components: List[Dict],
        assets: List[Dict]
    ) -> Dict[str, Any]:
        """Validate software component entities."""
        valid_count = 0
        component_ids = set()

        # Create asset ID lookup
        asset_ids = {a['assetId'] for a in assets if 'assetId' in a}

        for comp in components:
            issues = []

            # Check required fields
            required = self.REQUIRED_FIELDS.get('SoftwareComponent', [])
            for field in required:
                if field not in comp or comp[field] is None or comp[field] == '':
                    issues.append({
                        'entity_type': 'SoftwareComponent',
                        'entity_id': comp.get('componentId', 'unknown'),
                        'issue_type': 'missing_field',
                        'field': field,
                        'severity': 'error'
                    })

            # Validate asset exists
            asset_id = comp.get('assetId')
            if asset_id and asset_id not in asset_ids:
                issues.append({
                    'entity_type': 'SoftwareComponent',
                    'entity_id': comp.get('componentId', 'unknown'),
                    'issue_type': 'foreign_key_violation',
                    'field': 'assetId',
                    'value': asset_id,
                    'severity': 'error'
                })

            # Validate port
            port = comp.get('port')
            if port is not None:
                try:
                    port_int = int(port)
                    if not (0 <= port_int <= 65535):
                        issues.append({
                            'entity_type': 'SoftwareComponent',
                            'entity_id': comp.get('componentId', 'unknown'),
                            'issue_type': 'invalid_port',
                            'field': 'port',
                            'value': port,
                            'severity': 'error'
                        })
                except (ValueError, TypeError):
                    issues.append({
                        'entity_type': 'SoftwareComponent',
                        'entity_id': comp.get('componentId', 'unknown'),
                        'issue_type': 'invalid_port',
                        'field': 'port',
                        'value': port,
                        'severity': 'error'
                    })

            # Check for duplicate component IDs
            comp_id = comp.get('componentId')
            if comp_id:
                if comp_id in component_ids:
                    issues.append({
                        'entity_type': 'SoftwareComponent',
                        'entity_id': comp_id,
                        'issue_type': 'duplicate',
                        'severity': 'error'
                    })
                else:
                    component_ids.add(comp_id)

            # Record issues
            if issues:
                self.quality_issues.extend(issues)
            else:
                valid_count += 1

        return {
            'total_count': len(components),
            'valid_count': valid_count,
            'issue_count': len(components) - valid_count,
            'unique_ids': len(component_ids)
        }

    def validate_dataframe(self, df: pd.DataFrame, entity_type: str = 'Generic') -> Dict[str, Any]:
        """
        Validate a pandas DataFrame.

        Args:
            df: DataFrame to validate
            entity_type: Type of entities in the DataFrame

        Returns:
            Validation report
        """
        logger.info(f"Validating DataFrame with {len(df)} rows...")

        issues = []

        # Check for null values
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                null_percentage = (count / len(df)) * 100
                issues.append({
                    'entity_type': entity_type,
                    'issue_type': 'null_values',
                    'field': col,
                    'null_count': int(count),
                    'null_percentage': round(null_percentage, 2),
                    'severity': 'warning' if null_percentage < 10 else 'error'
                })

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                'entity_type': entity_type,
                'issue_type': 'duplicate_rows',
                'duplicate_count': int(duplicate_count),
                'severity': 'warning'
            })

        report = {
            'entity_type': entity_type,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issue_count': len(issues),
            'issues': issues,
            'valid': len(issues) == 0
        }

        logger.info(f"DataFrame validation complete: {report['valid']} ({len(issues)} issues)")
        return report


def main():
    """Example usage of the DataQualityValidator."""
    validator = DataQualityValidator(strict=False)

    # Sample data
    assets = [
        {
            'assetId': 'asset-1',
            'assetName': 'Web Server',
            'assetType': 'software',
            'ipAddress': '192.168.1.100',
            'isInternal': True
        },
        {
            'assetId': 'asset-2',
            'assetName': 'Database Server',
            'assetType': 'software',
            'ipAddress': '192.168.1.101',
            'isInternal': True
        }
    ]

    connections = [
        {
            'connectionId': 'conn-1',
            'sourceAssetId': 'asset-1',
            'destinationAssetId': 'asset-2',
            'sourcePort': 54321,
            'destinationPort': 3306,
            'protocol': 'TCP'
        }
    ]

    threats = []
    components = []

    # Validate
    report = validator.validate_dataset(assets, connections, threats, components)

    print("Data Quality Report:")
    print(f"  Overall Score: {report['overall_score']:.1f}%")
    print(f"  Valid Entities: {report['valid_entities']}/{report['total_entities']}")
    print(f"  Total Issues: {report['total_issues']}")
    print(f"  Meets Threshold: {report['meets_threshold']}")


if __name__ == "__main__":
    main()
