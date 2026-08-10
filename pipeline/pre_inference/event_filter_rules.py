"""
Event Filtering Rules

Remove noise, low-priority events, and irrelevant traffic before ML inference.
These rules reduce computational load and improve signal-to-noise ratio for ML models.

Filtering Criteria:
- Duration: Filter very short flows (likely noise)
- Whitelist: Filter events from known trusted IPs
- Reputation: Filter events from highly trusted sources
- Port: Filter well-known safe ports (with additional checks)
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import ipaddress
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering an event."""
    should_process: bool  # True if event should proceed to ML models
    filter_reason: Optional[str]  # Reason if filtered, None if passed
    filter_category: Optional[str]  # Category of filter (e.g., 'duration', 'whitelist')


class EventFilterRules:
    """Event filtering rules for pre-inference stage."""

    # Default well-known safe ports (can be configured)
    DEFAULT_SAFE_PORTS = {22, 53, 80, 443, 123, 161, 162, 389, 636, 3306, 3389, 5432, 5672, 27017}

    # Default whitelist IPs (can be configured)
    DEFAULT_WHITELIST = [
        # Monitoring and management systems
        '127.0.0.1',  # Localhost
        '::1',  # IPv6 localhost
        # Add other trusted IPs as needed
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize filtering rules.

        Args:
            config: Configuration dictionary with filter parameters
                - min_flow_duration: Filter flows shorter than X seconds (default: 0.1)
                - whitelist_ips: List of whitelisted IPs (default: DEFAULT_WHITELIST)
                - max_reputation_score: Filter events with reputation > this (default: 0.9)
                - safe_ports: Set of safe ports (default: DEFAULT_SAFE_PORTS)
                - enable_duration_filter: Enable duration filtering (default: True)
                - enable_whitelist_filter: Enable whitelist filtering (default: True)
                - enable_reputation_filter: Enable reputation filtering (default: True)
                - enable_port_filter: Enable port filtering (default: False)
        """
        self.config = config or {}

        # Filter parameters with defaults
        self.min_flow_duration = self.config.get('min_flow_duration', 0.1)
        self.whitelist_ips = set(self.config.get('whitelist_ips', self.DEFAULT_WHITELIST))
        self.max_reputation_score = self.config.get('max_reputation_score', 0.9)
        self.safe_ports = set(self.config.get('safe_ports', self.DEFAULT_SAFE_PORTS))

        # Enable/disable filters
        self.enable_duration_filter = self.config.get('enable_duration_filter', True)
        self.enable_whitelist_filter = self.config.get('enable_whitelist_filter', True)
        self.enable_reputation_filter = self.config.get('enable_reputation_filter', True)
        self.enable_port_filter = self.config.get('enable_port_filter', False)

        # Statistics
        self.stats = {
            'total_events': 0,
            'filtered_events': 0,
            'passed_events': 0,
            'filtered_by_category': {}
        }

        logger.info(f"EventFilterRules initialized: min_duration={self.min_flow_duration}s, "
                   f"whitelist={len(self.whitelist_ips)} IPs, "
                   f"max_reputation={self.max_reputation_score}, "
                   f"safe_ports={len(self.safe_ports)}")

    def filter_by_duration(self, event: Dict[str, Any]) -> FilterResult:
        """
        Filter very short flows (likely noise).

        Args:
            event: Event dictionary with flow_duration field

        Returns:
            FilterResult with should_process and filter_reason
        """
        duration = event.get('flow_duration', event.get('duration', 0))

        if duration < self.min_flow_duration:
            return FilterResult(
                should_process=False,
                filter_reason=f"Flow duration too short: {duration:.3f}s (min: {self.min_flow_duration}s)",
                filter_category='duration'
            )

        return FilterResult(
            should_process=True,
            filter_reason=None,
            filter_category=None
        )

    def filter_by_whitelist(self, event: Dict[str, Any]) -> FilterResult:
        """
        Filter events from whitelisted IPs.

        Args:
            event: Event dictionary with src_ip and dst_ip fields

        Returns:
            FilterResult with should_process and filter_reason
        """
        src_ip = event.get('src_ip', '')
        dst_ip = event.get('dst_ip', '')

        if src_ip in self.whitelist_ips:
            return FilterResult(
                should_process=False,
                filter_reason=f"Source IP in whitelist: {src_ip}",
                filter_category='whitelist'
            )

        if dst_ip in self.whitelist_ips:
            return FilterResult(
                should_process=False,
                filter_reason=f"Destination IP in whitelist: {dst_ip}",
                filter_category='whitelist'
            )

        # Check for IP ranges (CIDR)
        for whitelisted in self.whitelist_ips:
            try:
                if '/' in whitelisted:  # CIDR notation
                    network = ipaddress.ip_network(whitelisted, strict=False)
                    if ipaddress.ip_address(src_ip) in network:
                        return FilterResult(
                            should_process=False,
                            filter_reason=f"Source IP in whitelist range: {src_ip} in {whitelisted}",
                            filter_category='whitelist'
                        )
                    if ipaddress.ip_address(dst_ip) in network:
                        return FilterResult(
                            should_process=False,
                            filter_reason=f"Destination IP in whitelist range: {dst_ip} in {whitelisted}",
                            filter_category='whitelist'
                        )
            except (ValueError, ipaddress.AddressValueError):
                continue

        return FilterResult(
            should_process=True,
            filter_reason=None,
            filter_category=None
        )

    def filter_by_reputation(self, event: Dict[str, Any]) -> FilterResult:
        """
        Filter events from highly trusted sources.

        Args:
            event: Event dictionary with reputation_score field

        Returns:
            FilterResult with should_process and filter_reason
        """
        reputation_score = event.get('reputation_score', event.get('trust_score', 0.0))

        if reputation_score > self.max_reputation_score:
            return FilterResult(
                should_process=False,
                filter_reason=f"High reputation score: {reputation_score:.3f} (max: {self.max_reputation_score})",
                filter_category='reputation'
            )

        return FilterResult(
            should_process=True,
            filter_reason=None,
            filter_category=None
        )

    def filter_by_port(self, event: Dict[str, Any]) -> FilterResult:
        """
        Filter well-known safe ports with additional checks.

        Note: Port filtering is disabled by default because legitimate attacks
        can occur on any port. Use with caution and customize for your environment.

        Args:
            event: Event dictionary with dst_port field

        Returns:
            FilterResult with should_process and filter_reason
        """
        dst_port = event.get('dst_port', event.get('port', 0))

        if dst_port in self.safe_ports:
            # Additional checks for safe ports
            # For example, check for unusual patterns on safe ports
            if self._check_safe_port_anomaly(event):
                # Anomaly detected on safe port, pass through
                return FilterResult(
                    should_process=True,
                    filter_reason=None,
                    filter_category=None
                )
            else:
                return FilterResult(
                    should_process=False,
                    filter_reason=f"Safe port without anomalies: {dst_port}",
                    filter_category='port'
                )

        return FilterResult(
            should_process=True,
            filter_reason=None,
            filter_category=None
        )

    def _check_safe_port_anomaly(self, event: Dict[str, Any]) -> bool:
        """
        Check for anomalies on safe ports.

        Args:
            event: Event dictionary

        Returns:
            True if anomaly detected, False otherwise
        """
        # Example checks for safe port anomalies:
        # 1. High volume of traffic on safe port
        # 2. Unusual protocol on safe port
        # 3. Suspicious payload on safe port

        # Check for high volume (e.g., >10MB on HTTP)
        dst_port = event.get('dst_port', 0)
        bytes_sent = event.get('bytes_sent', 0)

        if dst_port in {80, 443} and bytes_sent > 10 * 1024 * 1024:  # >10MB on HTTP/HTTPS
            return True  # Anomaly: large upload

        # Check for unusual protocol
        protocol = event.get('protocol', '').lower()
        if dst_port == 53 and protocol != 'udp':  # DNS should be UDP
            return True  # Anomaly: non-UDP on DNS port

        # Add more custom checks as needed

        return False

    def filter_by_internal_traffic(self, event: Dict[str, Any]) -> FilterResult:
        """
        Filter internal-to-internal traffic (can be configured).

        Args:
            event: Event dictionary with src_ip and dst_ip fields

        Returns:
            FilterResult with should_process and filter_reason
        """
        src_ip = event.get('src_ip', '')
        dst_ip = event.get('dst_ip', '')

        # Check if both IPs are private (RFC 1918)
        def is_private_ip(ip_str):
            try:
                ip = ipaddress.ip_address(ip_str)
                return ip.is_private
            except (ValueError, ipaddress.AddressValueError):
                return False

        if is_private_ip(src_ip) and is_private_ip(dst_ip):
            # Additional check for internal traffic anomalies
            if self._check_internal_traffic_anomaly(event):
                # Anomaly detected in internal traffic
                return FilterResult(
                    should_process=True,
                    filter_reason=None,
                    filter_category=None
                )
            else:
                return FilterResult(
                    should_process=False,
                    filter_reason="Internal-to-internal traffic without anomalies",
                    filter_category='internal_traffic'
                )

        return FilterResult(
            should_process=True,
            filter_reason=None,
            filter_category=None
        )

    def _check_internal_traffic_anomaly(self, event: Dict[str, Any]) -> bool:
        """
        Check for anomalies in internal traffic.

        Args:
            event: Event dictionary

        Returns:
            True if anomaly detected, False otherwise
        """
        # Example checks for internal traffic anomalies:
        # 1. Sudden spike in traffic volume
        # 2. Access to unusual internal ports
        # 3. Lateral movement patterns

        dst_port = event.get('dst_port', 0)

        # Check for access to sensitive internal ports
        sensitive_ports = {445, 3389, 5900}  # SMB, RDP, VNC
        if dst_port in sensitive_ports:
            return True  # Anomaly: access to sensitive internal port

        # Add more custom checks as needed

        return False

    def apply_all_filters(self, event: Dict[str, Any]) -> FilterResult:
        """
        Apply all enabled filtering rules to an event.

        Args:
            event: Event dictionary

        Returns:
            FilterResult with should_process and filter_reason
        """
        self.stats['total_events'] += 1

        # Define filter functions in order of priority
        filter_functions = []

        if self.enable_duration_filter:
            filter_functions.append(('duration', self.filter_by_duration))

        if self.enable_whitelist_filter:
            filter_functions.append(('whitelist', self.filter_by_whitelist))

        if self.enable_reputation_filter:
            filter_functions.append(('reputation', self.filter_by_reputation))

        if self.enable_port_filter:
            filter_functions.append(('port', self.filter_by_port))

        # Apply filters
        for category, filter_func in filter_functions:
            result = filter_func(event)

            if not result.should_process:
                self.stats['filtered_events'] += 1
                self.stats['filtered_by_category'][category] = \
                    self.stats['filtered_by_category'].get(category, 0) + 1

                logger.debug(f"Event filtered by {category}: {result.filter_reason}")
                return result

        # All filters passed
        self.stats['passed_events'] += 1
        return FilterResult(
            should_process=True,
            filter_reason=None,
            filter_category=None
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get filtering statistics.

        Returns:
            Dictionary with filtering statistics
        """
        total = self.stats['total_events']
        filtered = self.stats['filtered_events']
        passed = self.stats['passed_events']

        return {
            'total_events': total,
            'filtered_events': filtered,
            'passed_events': passed,
            'filter_rate': filtered / total if total > 0 else 0.0,
            'pass_rate': passed / total if total > 0 else 0.0,
            'filtered_by_category': self.stats['filtered_by_category']
        }

    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        self.stats = {
            'total_events': 0,
            'filtered_events': 0,
            'passed_events': 0,
            'filtered_by_category': {}
        }
        logger.info("Filtering statistics reset")


def main():
    """Example usage of EventFilterRules."""
    # Create filter rules with custom configuration
    config = {
        'min_flow_duration': 0.1,
        'whitelist_ips': ['127.0.0.1', '192.168.1.0/24'],
        'max_reputation_score': 0.9,
        'enable_duration_filter': True,
        'enable_whitelist_filter': True,
        'enable_reputation_filter': True,
        'enable_port_filter': False
    }

    filter_rules = EventFilterRules(config)

    # Test events
    test_events = [
        {
            'src_ip': '127.0.0.1',
            'dst_ip': '192.168.1.100',
            'flow_duration': 0.05,  # Too short
            'dst_port': 80,
            'reputation_score': 0.5
        },
        {
            'src_ip': '10.0.0.1',
            'dst_ip': '10.0.0.2',
            'flow_duration': 1.0,
            'dst_port': 443,
            'reputation_score': 0.95  # Too high
        },
        {
            'src_ip': '203.0.113.1',
            'dst_ip': '203.0.113.2',
            'flow_duration': 5.0,
            'dst_port': 22,
            'reputation_score': 0.3  # Should pass
        }
    ]

    for event in test_events:
        result = filter_rules.apply_all_filters(event)
        if result.should_process:
            print(f"Event passed filters: {event}")
        else:
            print(f"Event filtered: {result.filter_reason}")

    # Print statistics
    stats = filter_rules.get_statistics()
    print("\nFiltering Statistics:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Filtered: {stats['filtered_events']} ({stats['filter_rate']:.1%})")
    print(f"  Passed: {stats['passed_events']} ({stats['pass_rate']:.1%})")
    print(f"  By category: {stats['filtered_by_category']}")


if __name__ == "__main__":
    main()
