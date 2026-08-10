"""
Flow Record Parser

Converts network flow records from CSE-CIC-IDS2018 into asset-centric graph entities
for the Asset Relationship Graph (ARG).

Flow Record → Graph Entities Mapping:
- Source IP → NetworkAsset (SoftwareAsset/HardwareAsset based on port/service)
- Destination IP → NetworkAsset
- Flow → NetworkConnection (relationship between assets)
- Attack label → ThreatEvent (linked to assets)
- Protocols/ports → SoftwareComponent (running services)
"""

import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import ipaddress
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowParser:
    """Parser for converting network flow records to ARG entities."""

    # Common service port mappings
    SERVICE_PORTS = {
        20: 'FTP-Data',
        21: 'FTP',
        22: 'SSH',
        23: 'Telnet',
        25: 'SMTP',
        53: 'DNS',
        80: 'HTTP',
        110: 'POP3',
        143: 'IMAP',
        443: 'HTTPS',
        445: 'SMB',
        3306: 'MySQL',
        3389: 'RDP',
        5432: 'PostgreSQL',
        5900: 'VNC',
        6379: 'Redis',
        8080: 'HTTP-Alt',
        8443: 'HTTPS-Alt',
        27017: 'MongoDB'
    }

    # Internal IP ranges (RFC1918)
    INTERNAL_RANGES = [
        ipaddress.IPv4Network('10.0.0.0/8'),
        ipaddress.IPv4Network('172.16.0.0/12'),
        ipaddress.IPv4Network('192.168.0.0/16')
    ]

    def __init__(self):
        """Initialize the flow parser."""
        self.assets: Dict[str, Dict] = {}
        self.connections: List[Dict] = []
        self.threat_events: List[Dict] = {}
        self.software_components: Dict[Tuple[str, int], Dict] = {}

    def parse_flows(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse flow records DataFrame into graph entities.

        Args:
            df: DataFrame with flow records

        Returns:
            Dictionary with parsed entities (assets, connections, threats, components)
        """
        logger.info(f"Parsing {len(df)} flow records...")

        # Reset state
        self.assets = {}
        self.connections = []
        self.threat_events = {}
        self.software_components = {}

        # Process each flow record
        for idx, row in df.iterrows():
            self._parse_flow_record(row)

        # Count entities
        stats = {
            'assets': len(self.assets),
            'connections': len(self.connections),
            'threat_events': len(self.threat_events),
            'software_components': len(self.software_components),
            'total_flows_processed': len(df)
        }

        logger.info(f"Parsed entities: {stats}")
        return {
            'assets': list(self.assets.values()),
            'connections': self.connections,
            'threat_events': list(self.threat_events.values()),
            'software_components': list(self.software_components.values()),
            'statistics': stats
        }

    def _parse_flow_record(self, row: pd.Series) -> None:
        """Parse a single flow record into graph entities."""
        # Extract IP addresses (handle different column naming conventions)
        src_ip = self._get_ip(row, ['src_ip', 'source_ip', 'source'])
        dest_ip = self._get_ip(row, ['dest_ip', 'destination_ip', 'destination'])
        src_port = self._get_port(row, ['src_port', 'source_port'])
        dest_port = self._get_port(row, ['dest_port', 'destination_port'])
        protocol = self._get_protocol(row)
        timestamp = self._get_timestamp(row)

        if not src_ip or not dest_ip:
            return

        # Determine flow direction and asset types
        src_is_internal = self._is_internal_ip(src_ip)
        dest_is_internal = self._is_internal_ip(dest_ip)

        # Create or update source asset
        src_asset = self._create_asset(src_ip, 'source', src_is_internal, row)

        # Create or update destination asset
        dest_asset = self._create_asset(dest_ip, 'destination', dest_is_internal, row)

        # Identify software components based on ports
        if dest_port:
            self._create_software_component(dest_ip, dest_port, protocol, row)

        # Create network connection
        connection = self._create_connection(
            src_ip, dest_ip, src_port, dest_port, protocol, row, timestamp
        )

        # Create threat event if this is an attack flow
        if self._is_attack_flow(row):
            threat = self._create_threat_event(src_ip, dest_ip, row, timestamp)
            if threat:
                self.threat_events[threat['threatId']] = threat

    def _get_ip(self, row: pd.Series, possible_cols: List[str]) -> Optional[str]:
        """Extract IP address from row using possible column names."""
        for col in possible_cols:
            if col in row and pd.notna(row[col]):
                ip_str = str(row[col]).strip()
                # Validate IP format
                try:
                    ipaddress.ip_address(ip_str)
                    return ip_str
                except ValueError:
                    continue
        return None

    def _get_port(self, row: pd.Series, possible_cols: List[str]) -> Optional[int]:
        """Extract port number from row."""
        for col in possible_cols:
            if col in row and pd.notna(row[col]):
                try:
                    port = int(row[col])
                    if 0 <= port <= 65535:
                        return port
                except (ValueError, TypeError):
                    continue
        return None

    def _get_protocol(self, row: pd.Series) -> str:
        """Extract protocol from row."""
        protocol_cols = ['protocol', 'proto']
        for col in protocol_cols:
            if col in row and pd.notna(row[col]):
                return str(row[col]).upper()
        return 'TCP'

    def _get_timestamp(self, row: pd.Series) -> Optional[datetime]:
        """Extract timestamp from row."""
        timestamp_cols = ['timestamp', 'time', 'flow_duration']
        for col in timestamp_cols:
            if col in row and pd.notna(row[col]):
                if isinstance(row[col], pd.Timestamp):
                    return row[col].to_pydatetime()
                elif isinstance(row[col], str):
                    try:
                        return pd.to_datetime(row[col]).to_pydatetime()
                    except Exception:
                        continue
        return None

    def _is_internal_ip(self, ip_str: str) -> bool:
        """Check if IP is in internal (RFC1918) range."""
        try:
            ip = ipaddress.ip_address(ip_str)
            for network in self.INTERNAL_RANGES:
                if ip in network:
                    return True
            return False
        except ValueError:
            return False

    def _create_asset(
        self,
        ip: str,
        role: str,
        is_internal: bool,
        row: pd.Series
    ) -> Dict:
        """Create or update an asset node."""
        asset_id = f"asset-{ip.replace('.', '-')}"

        if asset_id not in self.assets:
            # Determine asset type
            asset_type = 'network'
            if is_internal:
                asset_type = 'software'  # Assume internal IPs are compute assets

            self.assets[asset_id] = {
                'assetId': asset_id,
                'assetName': f"Asset {ip}",
                'assetType': asset_type,
                'ipAddress': ip,
                'isInternal': is_internal,
                'role': role,
                'provenanceSource': 'CSE-CIC-IDS2018',
                'flowCount': 0,
                'attackFlowCount': 0
            }

        # Update asset statistics
        self.assets[asset_id]['flowCount'] = self.assets[asset_id].get('flowCount', 0) + 1

        if self._is_attack_flow(row):
            self.assets[asset_id]['attackFlowCount'] = self.assets[asset_id].get('attackFlowCount', 0) + 1

        return self.assets[asset_id]

    def _create_software_component(
        self,
        ip: str,
        port: int,
        protocol: str,
        row: pd.Series
    ) -> Dict:
        """Create or update a software component (service)."""
        key = (ip, port)
        component_id = f"service-{ip.replace('.', '-')}-{port}"

        if key not in self.software_components:
            # Identify service
            service_name = self.SERVICE_PORTS.get(port, f'Unknown-{protocol}')

            self.software_components[key] = {
                'componentId': component_id,
                'assetId': f"asset-{ip.replace('.', '-')}",
                'serviceName': service_name,
                'port': port,
                'protocol': protocol,
                'componentType': 'network-service',
                'provenanceSource': 'CSE-CIC-IDS2018',
                'connectionCount': 0
            }

        self.software_components[key]['connectionCount'] += 1
        return self.software_components[key]

    def _create_connection(
        self,
        src_ip: str,
        dest_ip: str,
        src_port: Optional[int],
        dest_port: Optional[int],
        protocol: str,
        row: pd.Series,
        timestamp: Optional[datetime]
    ) -> Dict:
        """Create a network connection edge."""
        connection_id = f"conn-{hash((src_ip, dest_ip, src_port, dest_port, timestamp or '')) & 0x7FFFFFFF}"

        # Extract flow metrics if available
        flow_metrics = self._extract_flow_metrics(row)

        connection = {
            'connectionId': connection_id,
            'sourceAssetId': f"asset-{src_ip.replace('.', '-')}",
            'destinationAssetId': f"asset-{dest_ip.replace('.', '-')}",
            'sourcePort': src_port,
            'destinationPort': dest_port,
            'protocol': protocol,
            'timestamp': timestamp.isoformat() if timestamp else None,
            'isAttack': self._is_attack_flow(row),
            **flow_metrics,
            'provenanceSource': 'CSE-CIC-IDS2018'
        }

        self.connections.append(connection)
        return connection

    def _extract_flow_metrics(self, row: pd.Series) -> Dict:
        """Extract flow metrics from row."""
        metrics = {}

        # Common flow metric columns in CSE-CIC-IDS2018
        metric_mappings = {
            'flow_duration': 'flowDuration',
            'total_fwd_packets': 'totalFwdPackets',
            'total_bwd_packets': 'totalBwdPackets',
            'total_length_of_fwd_packet': 'totalFwdBytes',
            'total_length_of_bwd_packet': 'totalBwdBytes',
            'packet_length_mean': 'packetLengthMean',
            'packet_length_std': 'packetLengthStd',
            'packet_length_variance': 'packetLengthVariance',
            'flow_bytes_per_sec': 'flowBytesPerSec',
            'flow_packets_per_sec': 'flowPacketsPerSec'
        }

        for src_col, dest_col in metric_mappings.items():
            if src_col in row and pd.notna(row[src_col]):
                try:
                    metrics[dest_col] = float(row[src_col])
                except (ValueError, TypeError):
                    continue

        return metrics

    def _is_attack_flow(self, row: pd.Series) -> bool:
        """Check if flow is an attack flow."""
        # Check for attack indicators in various column names
        attack_indicators = ['label', 'label']

        for col in attack_indicators:
            if col in row and pd.notna(row[col]):
                label = str(row[col]).lower()
                return 'attack' in label or 'botnet' in label or 'ddos' in label

        return False

    def _create_threat_event(
        self,
        src_ip: str,
        dest_ip: str,
        row: pd.Series,
        timestamp: Optional[datetime]
    ) -> Optional[Dict]:
        """Create a threat event entity."""
        # Extract attack type
        attack_type = self._extract_attack_type(row)
        if not attack_type:
            return None

        threat_id = f"threat-{hash((src_ip, dest_ip, attack_type, str(timestamp))) & 0x7FFFFFFF}"

        threat = {
            'threatId': threat_id,
            'threatType': attack_type,
            'sourceAssetId': f"asset-{src_ip.replace('.', '-')}",
            'destinationAssetId': f"asset-{dest_ip.replace('.', '-')}",
            'timestamp': timestamp.isoformat() if timestamp else None,
            'confidence': 0.8,  # Default confidence from labeled dataset
            'provenanceSource': 'CSE-CIC-IDS2018'
        }

        return threat

    def _extract_attack_type(self, row: pd.Series) -> Optional[str]:
        """Extract attack type from row."""
        # Check various columns for attack type
        type_cols = ['label', 'attack_type', 'type', 'scenario']

        for col in type_cols:
            if col in row and pd.notna(row[col]):
                label = str(row[col])
                # Normalize attack type names
                if 'botnet' in label.lower():
                    return 'botnet'
                elif 'ddos' in label.lower():
                    return 'ddos'
                elif 'dos' in label.lower():
                    return 'dos'
                elif 'infiltration' in label.lower():
                    return 'infiltration'
                elif 'web' in label.lower() or 'xss' in label.lower() or 'sql' in label.lower():
                    return 'web_attack'
                elif 'benign' in label.lower():
                    return None  # Not an attack
                else:
                    return label.lower()

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get parsing statistics.

        Returns:
            Dictionary with parsing statistics
        """
        return {
            'total_assets': len(self.assets),
            'total_connections': len(self.connections),
            'total_threat_events': len(self.threat_events),
            'total_software_components': len(self.software_components),
            'internal_assets': sum(1 for a in self.assets.values() if a.get('isInternal')),
            'external_assets': sum(1 for a in self.assets.values() if not a.get('isInternal')),
            'assets_with_attacks': sum(1 for a in self.assets.values() if a.get('attackFlowCount', 0) > 0)
        }


def main():
    """Example usage of the FlowParser."""
    import pandas as pd

    # Create sample flow data
    sample_data = pd.DataFrame([
        {
            'timestamp': '2024-02-14 10:00:00',
            'src_ip': '192.168.1.100',
            'dest_ip': '10.0.0.50',
            'src_port': 54321,
            'dest_port': 443,
            'protocol': 'TCP',
            'flow_duration': 1.5,
            'total_fwd_packets': 10,
            'total_bwd_packets': 8,
            'label': 'Benign'
        },
        {
            'timestamp': '2024-02-14 10:00:01',
            'src_ip': '203.0.113.10',
            'dest_ip': '192.168.1.100',
            'src_port': 12345,
            'dest_port': 80,
            'protocol': 'TCP',
            'flow_duration': 2.0,
            'total_fwd_packets': 100,
            'total_bwd_packets': 50,
            'label': 'Botnet'
        }
    ])

    parser = FlowParser()
    entities = parser.parse_flows(sample_data)

    logger.info("Parsed entities:")
    logger.info(f"  Assets: {len(entities['assets'])}")
    logger.info(f"  Connections: {len(entities['connections'])}")
    logger.info(f"  Threat Events: {len(entities['threat_events'])}")
    logger.info(f"  Software Components: {len(entities['software_components'])}")


if __name__ == "__main__":
    main()
