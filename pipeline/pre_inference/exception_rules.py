"""
Exception Handling Rules

Handle known patterns, edge cases, and false positives before ML inference.
These rules prevent known benign patterns from triggering unnecessary alerts.

Exception Types:
- Known benign patterns (whitelisted events)
- Maintenance windows (scheduled maintenance)
- Bulk transfers (legitimate large data transfers)
- Edge cases (special handling logic)
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass
from datetime import datetime, time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExceptionPattern:
    """Known benign pattern."""
    pattern: Dict[str, Any]
    reason: str
    pattern_id: str


@dataclass
class ExceptionResult:
    """Result of exception check."""
    is_exception: bool
    reason: Optional[str]
    exception_type: Optional[str]


class ExceptionRules:
    """Exception handling rules for pre-inference stage."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize exception rules.

        Args:
            config: Configuration dictionary with exception parameters
                - maintenance_windows: List of maintenance windows
                - bulk_transfer_thresholds: Bulk transfer thresholds
                - known_patterns: List of known benign patterns
                - enable_maintenance_check: Enable maintenance window check (default: True)
                - enable_bulk_transfer_check: Enable bulk transfer check (default: True)
                - enable_known_patterns_check: Enable known patterns check (default: True)
        """
        self.config = config or {}

        # Known benign patterns
        self.known_patterns: List[ExceptionPattern] = [
            # Default known patterns
            ExceptionPattern(
                pattern={'src_ip': '127.0.0.1', 'dst_port': 22},
                reason='Localhost SSH connection (administrative)',
                pattern_id='localhost-ssh'
            ),
            ExceptionPattern(
                pattern={'protocol': 'icmp', 'packet_count': {'<': 10}},
                reason='Low-volume ICMP traffic (routine connectivity checks)',
                pattern_id='low-icmp'
            ),
        ]

        # Add custom patterns from config
        if 'known_patterns' in self.config:
            for pattern_data in self.config['known_patterns']:
                self.add_known_pattern(
                    pattern=pattern_data.get('pattern', {}),
                    reason=pattern_data.get('reason', 'Custom pattern'),
                    pattern_id=pattern_data.get('pattern_id', f'custom-{len(self.known_patterns)}')
                )

        # Maintenance windows
        self.maintenance_windows = self.config.get('maintenance_windows', [])

        # Bulk transfer thresholds
        self.bulk_transfer_thresholds = self.config.get(
            'bulk_transfer_thresholds',
            {
                'min_bytes': 100 * 1024 * 1024,  # 100 MB
                'min_duration': 300,  # 5 minutes
                'min_packet_count': 10000
            }
        )

        # Enable/disable checks
        self.enable_maintenance_check = self.config.get('enable_maintenance_check', True)
        self.enable_bulk_transfer_check = self.config.get('enable_bulk_transfer_check', True)
        self.enable_known_patterns_check = self.config.get('enable_known_patterns_check', True)

        # Custom edge case handlers
        self.edge_case_handlers: Dict[str, Callable[[Dict[str, Any]], bool]] = {}

        # Statistics
        self.stats = {
            'total_checks': 0,
            'total_exceptions': 0,
            'exceptions_by_type': {}
        }

        logger.info(f"ExceptionRules initialized: {len(self.known_patterns)} known patterns, "
                   f"{len(self.maintenance_windows)} maintenance windows")

    def add_known_pattern(
        self,
        pattern: Dict[str, Any],
        reason: str,
        pattern_id: Optional[str] = None
    ) -> None:
        """
        Add a known benign pattern.

        Args:
            pattern: Dictionary of event attributes (supports operators)
            reason: Explanation of why this pattern is benign
            pattern_id: Unique identifier (auto-generated if not specified)
        """
        if pattern_id is None:
            pattern_id = f'pattern-{len(self.known_patterns)}'

        self.known_patterns.append(ExceptionPattern(
            pattern=pattern,
            reason=reason,
            pattern_id=pattern_id
        ))

        logger.info(f"Added known pattern: {pattern_id} - {reason}")

    def check_known_patterns(self, event: Dict[str, Any]) -> ExceptionResult:
        """
        Check if event matches any known benign pattern.

        Args:
            event: Event dictionary

        Returns:
            ExceptionResult with is_exception and reason
        """
        for known in self.known_patterns:
            if self._matches_pattern(event, known.pattern):
                return ExceptionResult(
                    is_exception=True,
                    reason=f"Known pattern: {known.reason} ({known.pattern_id})",
                    exception_type='known_pattern'
                )

        return ExceptionResult(
            is_exception=False,
            reason=None,
            exception_type=None
        )

    def _matches_pattern(self, event: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """
        Check if event matches pattern with support for operators.

        Pattern operators:
            {'key': value} - exact match
            {'key': {'>': value}} - greater than
            {'key': {'<': value}} - less than
            {'key': {'>=': value}} - greater than or equal
            {'key': {'<=': value}} - less than or equal
            {'key': {'!=': value}} - not equal
            {'key': {'in': [value1, value2]}} - in list
            {'key': {'regex': pattern}} - regex match

        Args:
            event: Event dictionary
            pattern: Pattern dictionary with operators

        Returns:
            True if event matches pattern
        """
        for key, value in pattern.items():
            if key not in event:
                return False

            event_value = event[key]

            # Check if value is an operator dictionary
            if isinstance(value, dict):
                # Operator matching
                if '>' in value:
                    if not (event_value > value['>']):
                        return False
                elif '<' in value:
                    if not (event_value < value['<']):
                        return False
                elif '>=' in value:
                    if not (event_value >= value['>=']):
                        return False
                elif '<=' in value:
                    if not (event_value <= value['<=']):
                        return False
                elif '!=' in value:
                    if event_value == value['!=']:
                        return False
                elif 'in' in value:
                    if event_value not in value['in']:
                        return False
                elif 'regex' in value:
                    if not re.match(value['regex'], str(event_value)):
                        return False
                else:
                    # Unknown operator
                    return False
            else:
                # Exact matching
                if event_value != value:
                    return False

        return True

    def check_maintenance_window(self, event: Dict[str, Any]) -> ExceptionResult:
        """
        Check if event occurs during scheduled maintenance.

        Args:
            event: Event dictionary with timestamp field

        Returns:
            ExceptionResult with is_exception and reason
        """
        if not self.maintenance_windows:
            return ExceptionResult(
                is_exception=False,
                reason=None,
                exception_type=None
            )

        # Get event timestamp
        event_time = event.get('timestamp')
        if event_time is None:
            return ExceptionResult(
                is_exception=False,
                reason=None,
                exception_type=None
            )

        # Parse timestamp (handles Unix timestamp and ISO format)
        if isinstance(event_time, (int, float)):
            dt = datetime.fromtimestamp(event_time)
        elif isinstance(event_time, str):
            try:
                dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
            except ValueError:
                return ExceptionResult(
                    is_exception=False,
                    reason=None,
                    exception_type=None
                )
        else:
            return ExceptionResult(
                is_exception=False,
                reason=None,
                exception_type=None
            )

        # Check if any maintenance window matches
        for window in self.maintenance_windows:
            if self._is_in_maintenance_window(dt, window):
                return ExceptionResult(
                    is_exception=True,
                    reason=f"Maintenance window exception: {window.get('name', 'unnamed')}",
                    exception_type='maintenance_window'
                )

        return ExceptionResult(
            is_exception=False,
            reason=None,
            exception_type=None
        )

    def _is_in_maintenance_window(
        self,
        dt: datetime,
        window: Dict[str, Any]
    ) -> bool:
        """
        Check if datetime is within maintenance window.

        Window format:
            {
                'name': 'Weekly backup',
                'days': ['Saturday', 'Sunday'],  # Optional, default: all days
                'start_time': '02:00',  # HH:MM format
                'end_time': '04:00',  # HH:MM format
                'timezone': 'UTC'  # Optional, default: UTC
            }

        Args:
            dt: Datetime to check
            window: Maintenance window definition

        Returns:
            True if dt is within maintenance window
        """
        # Check day of week
        if 'days' in window:
            day_name = dt.strftime('%A')
            if day_name not in window['days']:
                return False

        # Check time range
        start_time_str = window.get('start_time')
        end_time_str = window.get('end_time')

        if not start_time_str or not end_time_str:
            return False

        start_time = datetime.strptime(start_time_str, '%H:%M').time()
        end_time = datetime.strptime(end_time_str, '%H:%M').time()
        current_time = dt.time()

        # Check if current time is within window
        if start_time <= end_time:
            # Window does not cross midnight
            return start_time <= current_time <= end_time
        else:
            # Window crosses midnight (e.g., 22:00 - 02:00)
            return current_time >= start_time or current_time <= end_time

    def check_bulk_transfer(self, event: Dict[str, Any]) -> ExceptionResult:
        """
        Check if event is a legitimate bulk transfer.

        Args:
            event: Event dictionary with bytes, duration, packet_count fields

        Returns:
            ExceptionResult with is_exception and reason
        """
        bytes_total = (
            event.get('bytes_sent', 0) +
            event.get('bytes_received', 0)
        )
        duration = event.get('flow_duration', event.get('duration', 0))
        packet_count = event.get('packet_count', 0)

        # Check thresholds
        bytes_ok = bytes_total >= self.bulk_transfer_thresholds['min_bytes']
        duration_ok = duration >= self.bulk_transfer_thresholds['min_duration']
        packets_ok = packet_count >= self.bulk_transfer_thresholds['min_packet_count']

        if bytes_ok and duration_ok and packets_ok:
            # Check if it's a bulk transfer to/from a known safe location
            src_ip = event.get('src_ip', '')
            dst_ip = event.get('dst_ip', '')

            # Add custom logic for known safe bulk transfer locations
            # For example, backup servers, data warehouses, etc.

            return ExceptionResult(
                is_exception=True,
                reason=f"Bulk transfer exception: {bytes_total / (1024*1024):.2f}MB, "
                       f"{duration:.1f}s, {packet_count} packets",
                exception_type='bulk_transfer'
            )

        return ExceptionResult(
            is_exception=False,
            reason=None,
            exception_type=None
        )

    def register_edge_case_handler(
        self,
        name: str,
        handler: Callable[[Dict[str, Any]], bool]
    ) -> None:
        """
        Register a custom edge case handler.

        Args:
            name: Handler name
            handler: Function that takes an event and returns True if it's an edge case
        """
        self.edge_case_handlers[name] = handler
        logger.info(f"Registered edge case handler: {name}")

    def check_edge_cases(self, event: Dict[str, Any]) -> ExceptionResult:
        """
        Check if event matches any registered edge case handlers.

        Args:
            event: Event dictionary

        Returns:
            ExceptionResult with is_exception and reason
        """
        for name, handler in self.edge_case_handlers.items():
            try:
                if handler(event):
                    return ExceptionResult(
                        is_exception=True,
                        reason=f"Edge case exception: {name}",
                        exception_type='edge_case'
                    )
            except Exception as e:
                logger.warning(f"Edge case handler '{name}' failed: {e}")

        return ExceptionResult(
            is_exception=False,
            reason=None,
            exception_type=None
        )

    def check_all_exceptions(self, event: Dict[str, Any]) -> ExceptionResult:
        """
        Check if event is an exception using all enabled rules.

        Args:
            event: Event dictionary

        Returns:
            ExceptionResult with is_exception and reason
        """
        self.stats['total_checks'] += 1

        # Check known patterns
        if self.enable_known_patterns_check:
            result = self.check_known_patterns(event)
            if result.is_exception:
                self.stats['total_exceptions'] += 1
                self.stats['exceptions_by_type']['known_pattern'] = \
                    self.stats['exceptions_by_type'].get('known_pattern', 0) + 1
                return result

        # Check maintenance windows
        if self.enable_maintenance_check:
            result = self.check_maintenance_window(event)
            if result.is_exception:
                self.stats['total_exceptions'] += 1
                self.stats['exceptions_by_type']['maintenance_window'] = \
                    self.stats['exceptions_by_type'].get('maintenance_window', 0) + 1
                return result

        # Check bulk transfers
        if self.enable_bulk_transfer_check:
            result = self.check_bulk_transfer(event)
            if result.is_exception:
                self.stats['total_exceptions'] += 1
                self.stats['exceptions_by_type']['bulk_transfer'] = \
                    self.stats['exceptions_by_type'].get('bulk_transfer', 0) + 1
                return result

        # Check edge cases
        result = self.check_edge_cases(event)
        if result.is_exception:
            self.stats['total_exceptions'] += 1
            self.stats['exceptions_by_type']['edge_case'] = \
                self.stats['exceptions_by_type'].get('edge_case', 0) + 1
            return result

        # No exception
        return ExceptionResult(
            is_exception=False,
            reason=None,
            exception_type=None
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get exception statistics.

        Returns:
            Dictionary with exception statistics
        """
        exception_rate = 0.0
        if self.stats['total_checks'] > 0:
            exception_rate = self.stats['total_exceptions'] / self.stats['total_checks']

        return {
            'total_checks': self.stats['total_checks'],
            'total_exceptions': self.stats['total_exceptions'],
            'exception_rate': exception_rate,
            'exceptions_by_type': self.stats['exceptions_by_type'],
            'known_patterns_count': len(self.known_patterns),
            'maintenance_windows_count': len(self.maintenance_windows),
            'edge_case_handlers_count': len(self.edge_case_handlers)
        }

    def reset_statistics(self) -> None:
        """Reset exception statistics."""
        self.stats = {
            'total_checks': 0,
            'total_exceptions': 0,
            'exceptions_by_type': {}
        }
        logger.info("Exception statistics reset")


def main():
    """Example usage of ExceptionRules."""
    # Create exception rules with custom configuration
    config = {
        'known_patterns': [
            {
                'pattern': {'src_ip': '192.168.1.100', 'dst_port': 22},
                'reason': 'Admin workstation SSH connection',
                'pattern_id': 'admin-ssh'
            }
        ],
        'maintenance_windows': [
            {
                'name': 'Weekly backup',
                'days': ['Saturday'],
                'start_time': '02:00',
                'end_time': '04:00',
                'timezone': 'UTC'
            }
        ],
        'bulk_transfer_thresholds': {
            'min_bytes': 50 * 1024 * 1024,  # 50 MB
            'min_duration': 300,  # 5 minutes
            'min_packet_count': 5000
        }
    }

    exception_rules = ExceptionRules(config)

    # Test events
    test_events = [
        {
            'src_ip': '192.168.1.100',
            'dst_port': 22,
            'timestamp': datetime.now().timestamp(),
            'bytes_sent': 1000,
            'bytes_received': 500,
            'flow_duration': 1.0,
            'packet_count': 100
        },
        {
            'src_ip': '10.0.0.1',
            'dst_port': 80,
            'timestamp': datetime.now().timestamp(),
            'bytes_sent': 60 * 1024 * 1024,  # 60 MB
            'bytes_received': 10 * 1024 * 1024,
            'flow_duration': 400,  # > 5 minutes
            'packet_count': 6000
        },
        {
            'src_ip': '10.0.0.2',
            'dst_port': 443,
            'timestamp': datetime.now().timestamp(),
            'bytes_sent': 1000,
            'bytes_received': 500,
            'flow_duration': 1.0,
            'packet_count': 100
        }
    ]

    for event in test_events:
        result = exception_rules.check_all_exceptions(event)
        if result.is_exception:
            print(f"Event is an exception: {result.reason}")
        else:
            print(f"Event is not an exception")

    # Print statistics
    stats = exception_rules.get_statistics()
    print("\nException Statistics:")
    print(f"  Total checks: {stats['total_checks']}")
    print(f"  Total exceptions: {stats['total_exceptions']}")
    print(f"  Exception rate: {stats['exception_rate']:.2%}")
    print(f"  By type: {stats['exceptions_by_type']}")


if __name__ == "__main__":
    main()
