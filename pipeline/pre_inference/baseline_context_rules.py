"""
Baseline Context Rules

Establish normal behavior baselines for assets and compute deviations.
These rules track historical patterns and flag significant deviations from baselines.

Baseline Tracking:
- Exponential Moving Average (EMA) for trend tracking
- Standard deviation for volatility measurement
- Z-score deviation calculation
- Per-asset metric tracking

Supported Metrics:
- bytes_sent, bytes_received
- packet_count
- flow_duration
- Custom metrics (configurable)
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineData:
    """Baseline data for a single metric."""
    mean: float
    std: float
    count: int
    last_updated: float  # Unix timestamp


@dataclass
class BaselineResult:
    """Result of baseline comparison."""
    z_score: float
    is_anomalous: bool
    baseline_mean: float
    baseline_std: float
    metric_value: float


class BaselineContextRules:
    """Baseline context rules for pre-inference stage."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize baseline context rules.

        Args:
            config: Configuration dictionary with baseline parameters
                - alpha: EMA smoothing factor (default: 0.1)
                - min_samples: Minimum samples before using baseline (default: 10)
                - z_threshold: Z-score threshold for anomaly (default: 3.0)
                - max_age_seconds: Max age of baseline data in seconds (default: 86400)
                - tracked_metrics: List of metrics to track (default: standard metrics)
                - enable_baseline_update: Enable baseline updates (default: True)
        """
        self.config = config or {}

        # Baseline parameters with defaults
        self.alpha = self.config.get('alpha', 0.1)  # EMA smoothing factor
        self.min_samples = self.config.get('min_samples', 10)  # Min samples before using baseline
        self.z_threshold = self.config.get('z_threshold', 3.0)  # Z-score threshold
        self.max_age_seconds = self.config.get('max_age_seconds', 86400)  # 24 hours

        # Metrics to track
        self.tracked_metrics = self.config.get(
            'tracked_metrics',
            ['bytes_sent', 'bytes_received', 'packet_count', 'flow_duration']
        )

        # Enable/disable baseline updates
        self.enable_baseline_update = self.config.get('enable_baseline_update', True)

        # Baseline storage: {asset_id: {metric_name: BaselineData}}
        self.baselines: Dict[str, Dict[str, BaselineData]] = defaultdict(dict)

        # Statistics
        self.stats = {
            'total_updates': 0,
            'total_queries': 0,
            'total_anomalies': 0,
            'assets_tracked': 0,
            'metrics_tracked': 0
        }

        logger.info(f"BaselineContextRules initialized: alpha={self.alpha}, "
                   f"min_samples={self.min_samples}, z_threshold={self.z_threshold}")

    def update_baseline(
        self,
        asset_id: str,
        metric_name: str,
        value: float
    ) -> BaselineData:
        """
        Update baseline using exponential moving average (EMA).

        Formula:
            Baseline(t) = α * value + (1-α) * Baseline(t-1)

        Args:
            asset_id: Asset identifier
            metric_name: Metric name (e.g., 'bytes_sent')
            value: Observed metric value

        Returns:
            Updated BaselineData
        """
        if not self.enable_baseline_update:
            logger.debug(f"Baseline updates disabled, skipping update for {asset_id}.{metric_name}")
            # Return existing baseline or create placeholder
            if asset_id in self.baselines and metric_name in self.baselines[asset_id]:
                return self.baselines[asset_id][metric_name]
            else:
                return BaselineData(mean=0.0, std=0.0, count=0, last_updated=time.time())

        # Initialize asset if needed
        if asset_id not in self.baselines:
            self.baselines[asset_id] = {}
            self.stats['assets_tracked'] += 1

        # Initialize metric if needed
        if metric_name not in self.baselines[asset_id]:
            self.baselines[asset_id][metric_name] = BaselineData(
                mean=value,
                std=0.0,
                count=1,
                last_updated=time.time()
            )
            self.stats['metrics_tracked'] += 1
            self.stats['total_updates'] += 1
            return self.baselines[asset_id][metric_name]

        # Update using EMA
        baseline = self.baselines[asset_id][metric_name]
        old_mean = baseline.mean

        # EMA update
        new_mean = self.alpha * value + (1 - self.alpha) * old_mean

        # Update M2 (sum of squared deviations) using Welford's algorithm
        # M2 is stored in baseline.std; actual std = sqrt(M2 / (n-1))
        delta = value - old_mean
        new_count = baseline.count + 1
        delta2 = value - new_mean
        new_std = baseline.std + delta * delta2  # accumulate M2

        # Update baseline
        self.baselines[asset_id][metric_name] = BaselineData(
            mean=new_mean,
            std=new_std,
            count=new_count,
            last_updated=time.time()
        )

        self.stats['total_updates'] += 1

        logger.debug(f"Updated baseline for {asset_id}.{metric_name}: "
                    f"value={value}, mean={new_mean:.2f}, std={new_std:.2f}, count={new_count}")

        return self.baselines[asset_id][metric_name]

    def get_baseline_score(
        self,
        asset_id: str,
        metric_name: str,
        value: float
    ) -> float:
        """
        Compute baseline deviation score (z-score).

        Formula:
            z_score = (value - mean) / std

        Args:
            asset_id: Asset identifier
            metric_name: Metric name
            value: Observed metric value

        Returns:
            Z-score (number of standard deviations from baseline)
        """
        if asset_id not in self.baselines or metric_name not in self.baselines[asset_id]:
            return 0.0  # No baseline yet

        baseline = self.baselines[asset_id][metric_name]
        mean = baseline.mean
        # baseline.std stores M2 (Welford's sum of squared deviations)
        std = math.sqrt(baseline.std / (baseline.count - 1)) if baseline.count >= 2 else 0.0

        if std == 0:
            return 0.0  # No variation

        z_score = (value - mean) / std
        return z_score

    def check_baseline_anomaly(
        self,
        asset_id: str,
        metric_name: str,
        value: float,
        threshold: Optional[float] = None
    ) -> BaselineResult:
        """
        Check if value deviates significantly from baseline.

        Args:
            asset_id: Asset identifier
            metric_name: Metric name
            value: Observed metric value
            threshold: Z-score threshold (uses default if not specified)

        Returns:
            BaselineResult with z_score, is_anomalous, and baseline data
        """
        self.stats['total_queries'] += 1

        # Use provided threshold or default
        z_threshold = threshold if threshold is not None else self.z_threshold

        # Get baseline data
        if asset_id not in self.baselines or metric_name not in self.baselines[asset_id]:
            # No baseline yet
            return BaselineResult(
                z_score=0.0,
                is_anomalous=False,
                baseline_mean=0.0,
                baseline_std=0.0,
                metric_value=value
            )

        baseline = self.baselines[asset_id][metric_name]

        # Check if we have enough samples
        if baseline.count < self.min_samples:
            # Not enough samples for reliable baseline
            return BaselineResult(
                z_score=0.0,
                is_anomalous=False,
                baseline_mean=baseline.mean,
                baseline_std=baseline.std,
                metric_value=value
            )

        # Check if baseline is too old
        age_seconds = time.time() - baseline.last_updated
        if age_seconds > self.max_age_seconds:
            logger.warning(f"Baseline for {asset_id}.{metric_name} is too old ({age_seconds:.0f}s), "
                         f"consider refreshing")

        # Compute z-score
        z_score = self.get_baseline_score(asset_id, metric_name, value)

        # Check if anomalous
        is_anomalous = abs(z_score) > z_threshold

        if is_anomalous:
            self.stats['total_anomalies'] += 1
            logger.info(f"Baseline anomaly detected for {asset_id}.{metric_name}: "
                       f"value={value:.2f}, baseline={baseline.mean:.2f}±{baseline.std:.2f}, "
                       f"z_score={z_score:.2f}")

        return BaselineResult(
            z_score=z_score,
            is_anomalous=is_anomalous,
            baseline_mean=baseline.mean,
            baseline_std=baseline.std,
            metric_value=value
        )

    def update_baseline_from_event(
        self,
        asset_id: str,
        event: Dict[str, Any]
    ) -> Dict[str, BaselineResult]:
        """
        Update baselines for all tracked metrics from an event.

        Args:
            asset_id: Asset identifier
            event: Event dictionary with metric fields

        Returns:
            Dictionary of {metric_name: BaselineResult} for all tracked metrics
        """
        results = {}

        for metric_name in self.tracked_metrics:
            value = event.get(metric_name)
            if value is not None:
                try:
                    value_float = float(value)
                    # Update baseline
                    self.update_baseline(asset_id, metric_name, value_float)
                    # Check for anomaly
                    result = self.check_baseline_anomaly(asset_id, metric_name, value_float)
                    results[metric_name] = result
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert metric {metric_name}={value} to float: {e}")

        return results

    def get_baseline_context(
        self,
        asset_id: str
    ) -> Dict[str, Any]:
        """
        Get baseline context for an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            Dictionary with baseline context for all tracked metrics
        """
        context = {
            'asset_id': asset_id,
            'has_baselines': asset_id in self.baselines,
            'metrics': {}
        }

        if asset_id not in self.baselines:
            return context

        for metric_name in self.tracked_metrics:
            if metric_name in self.baselines[asset_id]:
                baseline = self.baselines[asset_id][metric_name]
                context['metrics'][metric_name] = {
                    'mean': baseline.mean,
                    'std': baseline.std,
                    'count': baseline.count,
                    'last_updated': datetime.fromtimestamp(baseline.last_updated).isoformat()
                }

        return context

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get baseline statistics.

        Returns:
            Dictionary with baseline statistics
        """
        anomaly_rate = 0.0
        if self.stats['total_queries'] > 0:
            anomaly_rate = self.stats['total_anomalies'] / self.stats['total_queries']

        return {
            'total_updates': self.stats['total_updates'],
            'total_queries': self.stats['total_queries'],
            'total_anomalies': self.stats['total_anomalies'],
            'anomaly_rate': anomaly_rate,
            'assets_tracked': self.stats['assets_tracked'],
            'metrics_tracked': self.stats['metrics_tracked'],
            'z_threshold': self.z_threshold,
            'min_samples': self.min_samples
        }

    def reset_statistics(self) -> None:
        """Reset baseline statistics (does not clear baseline data)."""
        self.stats = {
            'total_updates': 0,
            'total_queries': 0,
            'total_anomalies': 0,
            'assets_tracked': 0,
            'metrics_tracked': 0
        }
        logger.info("Baseline statistics reset")

    def clear_baseline(self, asset_id: Optional[str] = None) -> None:
        """
        Clear baseline data.

        Args:
            asset_id: Asset identifier (if None, clears all baselines)
        """
        if asset_id is None:
            self.baselines.clear()
            logger.info("Cleared all baseline data")
        elif asset_id in self.baselines:
            del self.baselines[asset_id]
            logger.info(f"Cleared baseline data for asset: {asset_id}")
        else:
            logger.warning(f"Asset not found in baselines: {asset_id}")

    def export_baselines(self) -> Dict[str, Any]:
        """
        Export all baseline data for persistence.

        Returns:
            Dictionary with all baseline data
        """
        export_data = {
            'config': {
                'alpha': self.alpha,
                'min_samples': self.min_samples,
                'z_threshold': self.z_threshold,
                'max_age_seconds': self.max_age_seconds,
                'tracked_metrics': self.tracked_metrics
            },
            'baselines': {},
            'statistics': self.stats,
            'exported_at': datetime.now().isoformat()
        }

        for asset_id, metrics in self.baselines.items():
            export_data['baselines'][asset_id] = {}
            for metric_name, baseline in metrics.items():
                export_data['baselines'][asset_id][metric_name] = {
                    'mean': baseline.mean,
                    'std': baseline.std,
                    'count': baseline.count,
                    'last_updated': baseline.last_updated
                }

        return export_data

    def import_baselines(self, data: Dict[str, Any]) -> None:
        """
        Import baseline data from export.

        Args:
            data: Dictionary with exported baseline data
        """
        # Restore configuration (optional)
        if 'config' in data:
            self.alpha = data['config'].get('alpha', self.alpha)
            self.min_samples = data['config'].get('min_samples', self.min_samples)
            self.z_threshold = data['config'].get('z_threshold', self.z_threshold)
            self.max_age_seconds = data['config'].get('max_age_seconds', self.max_age_seconds)
            self.tracked_metrics = data['config'].get('tracked_metrics', self.tracked_metrics)

        # Restore baselines
        if 'baselines' in data:
            for asset_id, metrics in data['baselines'].items():
                self.baselines[asset_id] = {}
                for metric_name, baseline_data in metrics.items():
                    self.baselines[asset_id][metric_name] = BaselineData(
                        mean=baseline_data['mean'],
                        std=baseline_data['std'],
                        count=baseline_data['count'],
                        last_updated=baseline_data['last_updated']
                    )

        # Restore statistics (optional)
        if 'statistics' in data:
            self.stats = data['statistics']

        logger.info(f"Imported baseline data for {len(self.baselines)} assets")


def main():
    """Example usage of BaselineContextRules."""
    # Create baseline rules with custom configuration
    config = {
        'alpha': 0.1,
        'min_samples': 5,
        'z_threshold': 2.0,
        'max_age_seconds': 3600,
        'tracked_metrics': ['bytes_sent', 'bytes_received']
    }

    baseline_rules = BaselineContextRules(config)

    # Simulate events for an asset
    asset_id = "server-001"

    # Normal traffic pattern
    normal_traffic = [
        {'bytes_sent': 1000, 'bytes_received': 500},
        {'bytes_sent': 1100, 'bytes_received': 550},
        {'bytes_sent': 950, 'bytes_received': 480},
        {'bytes_sent': 1050, 'bytes_received': 520},
        {'bytes_sent': 1000, 'bytes_received': 500},
    ]

    # Anomalous traffic
    anomalous_traffic = [
        {'bytes_sent': 5000, 'bytes_received': 2500},  # Spike
        {'bytes_sent': 100, 'bytes_received': 50},  # Drop
    ]

    print("Processing normal traffic...")
    for i, event in enumerate(normal_traffic, 1):
        results = baseline_rules.update_baseline_from_event(asset_id, event)
        print(f"  Event {i}: bytes_sent={event['bytes_sent']}, "
              f"bytes_sent_z={results['bytes_sent'].z_score:.2f}, "
              f"is_anomalous={results['bytes_sent'].is_anomalous}")

    print("\nProcessing anomalous traffic...")
    for i, event in enumerate(anomalous_traffic, 1):
        results = baseline_rules.update_baseline_from_event(asset_id, event)
        print(f"  Event {i}: bytes_sent={event['bytes_sent']}, "
              f"bytes_sent_z={results['bytes_sent'].z_score:.2f}, "
              f"is_anomalous={results['bytes_sent'].is_anomalous}")

    # Get baseline context
    context = baseline_rules.get_baseline_context(asset_id)
    print("\nBaseline Context:")
    print(f"  Asset: {context['asset_id']}")
    print(f"  Has baselines: {context['has_baselines']}")
    print(f"  Metrics tracked: {len(context['metrics'])}")
    for metric_name, metric_data in context['metrics'].items():
        print(f"    {metric_name}: mean={metric_data['mean']:.2f}, std={metric_data['std']:.2f}")

    # Get statistics
    stats = baseline_rules.get_statistics()
    print("\nStatistics:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Total anomalies: {stats['total_anomalies']}")
    print(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")


if __name__ == "__main__":
    main()
