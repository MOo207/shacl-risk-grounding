"""
Pre-Inference Pipeline

Orchestrates all pre-inference symbolic rules to prepare events for ML inference.
This pipeline validates, filters, establishes baseline context, and handles exceptions.

Pipeline Stages:
1. Ontology Validation - Ensure events comply with SHACL constraints
2. Event Filtering - Remove noise and low-priority events
3. Exception Handling - Handle known patterns and edge cases
4. Baseline Context - Establish and track normal behavior baselines

Output:
- Events ready for ML inference (passed all rules)
- Context information (baseline deviations, filter reasons, etc.)
- Statistics and monitoring data
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

from .ontology_validation_rules import OntologyValidationRules, ValidationResult
from .event_filter_rules import EventFilterRules, FilterResult
from .baseline_context_rules import BaselineContextRules
from .exception_rules import ExceptionRules, ExceptionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreInferenceResult:
    """Result of pre-inference processing."""
    should_infer: bool  # True if event should proceed to ML models
    context: Dict[str, Any]  # Context information for ML models
    filter_reason: Optional[str]  # Reason if filtered (e.g., validation failed, filtered)
    validation_result: Optional[ValidationResult]  # Ontology validation result
    filter_result: Optional[FilterResult]  # Event filter result
    exception_result: Optional[ExceptionResult]  # Exception check result
    baseline_results: Optional[Dict[str, Any]]  # Baseline comparison results
    processing_time_ms: float  # Time taken to process event


class PreInferencePipeline:
    """Pre-inference pipeline for symbolic rules orchestration."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pre-inference pipeline.

        Args:
            config: Configuration dictionary with pipeline parameters
                - ontology_validation_config: Config for OntologyValidationRules
                - event_filter_config: Config for EventFilterRules
                - baseline_context_config: Config for BaselineContextRules
                - exception_rules_config: Config for ExceptionRules
                - enable_ontology_validation: Enable ontology validation (default: True)
                - enable_event_filtering: Enable event filtering (default: True)
                - enable_exception_handling: Enable exception handling (default: True)
                - enable_baseline_context: Enable baseline context (default: True)
                - stop_on_violation: Stop pipeline on validation violation (default: False)
                - stop_on_filter: Stop pipeline on filter (default: True)
        """
        self.config = config or {}

        # Enable/disable stages
        self.enable_ontology_validation = self.config.get('enable_ontology_validation', True)
        self.enable_event_filtering = self.config.get('enable_event_filtering', True)
        self.enable_exception_handling = self.config.get('enable_exception_handling', True)
        self.enable_baseline_context = self.config.get('enable_baseline_context', True)

        # Stop conditions
        self.stop_on_violation = self.config.get('stop_on_violation', False)
        self.stop_on_filter = self.config.get('stop_on_filter', True)

        # Initialize rule engines
        self.ontology_rules = None
        self.filter_rules = None
        self.baseline_rules = None
        self.exception_rules = None

        if self.enable_ontology_validation:
            ontology_config = self.config.get('ontology_validation_config', {})
            self.ontology_rules = OntologyValidationRules(ontology_config)

        if self.enable_event_filtering:
            filter_config = self.config.get('event_filter_config', {})
            self.filter_rules = EventFilterRules(filter_config)

        if self.enable_baseline_context:
            baseline_config = self.config.get('baseline_context_config', {})
            self.baseline_rules = BaselineContextRules(baseline_config)

        if self.enable_exception_handling:
            exception_config = self.config.get('exception_rules_config', {})
            self.exception_rules = ExceptionRules(exception_config)

        # Statistics
        self.stats = {
            'total_events': 0,
            'events_passed': 0,
            'events_filtered': 0,
            'filtered_by_stage': {},
            'total_processing_time_ms': 0
        }

        logger.info("PreInferencePipeline initialized")

    def process_event(self, event: Dict[str, Any]) -> PreInferenceResult:
        """
        Process event through pre-inference pipeline.

        Args:
            event: Event dictionary to process

        Returns:
            PreInferenceResult with processing outcome and context
        """
        start_time = datetime.now()
        self.stats['total_events'] += 1

        context = {}
        validation_result = None
        filter_result = None
        exception_result = None
        baseline_results = None

        should_infer = True
        filter_reason = None

        # Stage 1: Ontology Validation
        if self.enable_ontology_validation and self.ontology_rules:
            validation_result = self.ontology_rules.validate_event(event)

            if not validation_result.is_valid:
                self.stats['events_filtered'] += 1
                self.stats['filtered_by_stage']['ontology_validation'] = \
                    self.stats['filtered_by_stage'].get('ontology_validation', 0) + 1

                context['validation_violations'] = validation_result.violations
                filter_reason = f"Ontology validation failed: {len(validation_result.violations)} violations"

                if self.stop_on_violation:
                    should_infer = False
                    logger.warning(f"Event filtered by ontology validation: {filter_reason}")
                    return self._create_result(
                        should_infer, context, filter_reason,
                        validation_result, filter_result, exception_result,
                        baseline_results, start_time
                    )

        # Stage 2: Event Filtering
        if self.enable_event_filtering and self.filter_rules:
            filter_result = self.filter_rules.apply_all_filters(event)

            if not filter_result.should_process:
                self.stats['events_filtered'] += 1
                self.stats['filtered_by_stage']['event_filter'] = \
                    self.stats['filtered_by_stage'].get('event_filter', 0) + 1

                context['filter_category'] = filter_result.filter_category
                filter_reason = filter_result.filter_reason

                if self.stop_on_filter:
                    should_infer = False
                    logger.debug(f"Event filtered: {filter_reason}")
                    return self._create_result(
                        should_infer, context, filter_reason,
                        validation_result, filter_result, exception_result,
                        baseline_results, start_time
                    )

        # Stage 3: Exception Handling
        if self.enable_exception_handling and self.exception_rules:
            exception_result = self.exception_rules.check_all_exceptions(event)

            if exception_result.is_exception:
                self.stats['events_filtered'] += 1
                self.stats['filtered_by_stage']['exception_handling'] = \
                    self.stats['filtered_by_stage'].get('exception_handling', 0) + 1

                context['exception_type'] = exception_result.exception_type
                filter_reason = exception_result.reason

                should_infer = False
                logger.info(f"Event is an exception: {filter_reason}")
                return self._create_result(
                    should_infer, context, filter_reason,
                    validation_result, filter_result, exception_result,
                    baseline_results, start_time
                )

        # Stage 4: Baseline Context (always run for passed events)
        if self.enable_baseline_context and self.baseline_rules and should_infer:
            asset_id = event.get('asset_id', event.get('src_ip', 'unknown'))
            baseline_results = self.baseline_rules.update_baseline_from_event(asset_id, event)

            # Add baseline deviations to context
            context['baseline_deviations'] = {}
            for metric_name, result in baseline_results.items():
                context['baseline_deviations'][f'{metric_name}_z_score'] = result.z_score
                context['baseline_deviations'][f'{metric_name}_is_anomalous'] = result.is_anomalous
                context['baseline_deviations'][f'{metric_name}_baseline_mean'] = result.baseline_mean
                context['baseline_deviations'][f'{metric_name}_baseline_std'] = result.baseline_std

        # Event passed all pre-inference stages
        if should_infer:
            self.stats['events_passed'] += 1
            logger.debug(f"Event passed pre-inference pipeline")

        return self._create_result(
            should_infer, context, filter_reason,
            validation_result, filter_result, exception_result,
            baseline_results, start_time
        )

    def _create_result(
        self,
        should_infer: bool,
        context: Dict[str, Any],
        filter_reason: Optional[str],
        validation_result: Optional[ValidationResult],
        filter_result: Optional[FilterResult],
        exception_result: Optional[ExceptionResult],
        baseline_results: Optional[Dict[str, Any]],
        start_time: datetime
    ) -> PreInferenceResult:
        """
        Create PreInferenceResult.

        Args:
            should_infer: Whether event should proceed to ML models
            context: Context information
            filter_reason: Reason if filtered
            validation_result: Validation result
            filter_result: Filter result
            exception_result: Exception result
            baseline_results: Baseline results
            start_time: Start timestamp

        Returns:
            PreInferenceResult
        """
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        self.stats['total_processing_time_ms'] += processing_time_ms

        return PreInferenceResult(
            should_infer=should_infer,
            context=context,
            filter_reason=filter_reason,
            validation_result=validation_result,
            filter_result=filter_result,
            exception_result=exception_result,
            baseline_results=baseline_results,
            processing_time_ms=processing_time_ms
        )

    def process_batch(
        self,
        events: List[Dict[str, Any]]
    ) -> List[PreInferenceResult]:
        """
        Process a batch of events through pre-inference pipeline.

        Args:
            events: List of event dictionaries

        Returns:
            List of PreInferenceResult objects
        """
        results = []

        for event in events:
            result = self.process_event(event)
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        total = self.stats['total_events']
        filtered = self.stats['events_filtered']
        passed = self.stats['events_passed']

        avg_processing_time = 0.0
        if total > 0:
            avg_processing_time = self.stats['total_processing_time_ms'] / total

        return {
            'total_events': total,
            'events_passed': passed,
            'events_filtered': filtered,
            'pass_rate': passed / total if total > 0 else 0.0,
            'filter_rate': filtered / total if total > 0 else 0.0,
            'filtered_by_stage': self.stats['filtered_by_stage'],
            'total_processing_time_ms': self.stats['total_processing_time_ms'],
            'avg_processing_time_ms': avg_processing_time,
            'stages_enabled': {
                'ontology_validation': self.enable_ontology_validation,
                'event_filtering': self.enable_event_filtering,
                'exception_handling': self.enable_exception_handling,
                'baseline_context': self.enable_baseline_context
            }
        }

    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            'total_events': 0,
            'events_passed': 0,
            'events_filtered': 0,
            'filtered_by_stage': {},
            'total_processing_time_ms': 0
        }

        # Reset component statistics
        if self.ontology_rules:
            self.ontology_rules.reset_statistics()
        if self.filter_rules:
            self.filter_rules.reset_statistics()
        if self.baseline_rules:
            self.baseline_rules.reset_statistics()
        if self.exception_rules:
            self.exception_rules.reset_statistics()

        logger.info("Pre-inference pipeline statistics reset")

    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all components.

        Returns:
            Dictionary with component statistics
        """
        stats = {
            'pipeline': self.get_statistics()
        }

        if self.ontology_rules:
            stats['ontology_validation'] = self.ontology_rules.get_statistics()

        if self.filter_rules:
            stats['event_filter'] = self.filter_rules.get_statistics()

        if self.baseline_rules:
            stats['baseline_context'] = self.baseline_rules.get_statistics()

        if self.exception_rules:
            stats['exception_handling'] = self.exception_rules.get_statistics()

        return stats

    def export_baselines(self) -> Optional[Dict[str, Any]]:
        """
        Export baseline data from baseline rules.

        Returns:
            Dictionary with baseline data or None if baseline rules disabled
        """
        if self.baseline_rules:
            return self.baseline_rules.export_baselines()
        return None

    def import_baselines(self, data: Dict[str, Any]) -> None:
        """
        Import baseline data to baseline rules.

        Args:
            data: Dictionary with baseline data
        """
        if self.baseline_rules:
            self.baseline_rules.import_baselines(data)
            logger.info("Baseline data imported")

    def close(self) -> None:
        """Close connections and cleanup."""
        if self.ontology_rules:
            self.ontology_rules.close()

        logger.info("Pre-inference pipeline closed")


def main():
    """Example usage of PreInferencePipeline."""
    # Create pipeline configuration
    config = {
        'enable_ontology_validation': True,
        'enable_event_filtering': True,
        'enable_exception_handling': True,
        'enable_baseline_context': True,
        'stop_on_violation': False,
        'stop_on_filter': True,
        'ontology_validation_config': {
            'enable_shacl_validation': False,  # Disable SHACL for demo
            'enable_custom_constraints': True,
            'fail_on_violation': False
        },
        'event_filter_config': {
            'min_flow_duration': 0.1,
            'whitelist_ips': ['127.0.0.1'],
            'enable_duration_filter': True,
            'enable_whitelist_filter': True,
            'enable_reputation_filter': False
        },
        'exception_rules_config': {
            'enable_known_patterns_check': True,
            'enable_maintenance_check': False,
            'enable_bulk_transfer_check': False
        },
        'baseline_context_config': {
            'alpha': 0.1,
            'min_samples': 3,
            'z_threshold': 2.0,
            'tracked_metrics': ['bytes_sent', 'bytes_received']
        }
    }

    pipeline = PreInferencePipeline(config)

    # Test events
    test_events = [
        {
            'asset_id': 'server-001',
            'src_ip': '10.0.0.1',
            'dst_ip': '10.0.0.2',
            'src_port': 12345,
            'dst_port': 443,
            'protocol': 'tcp',
            'flow_duration': 0.5,
            'bytes_sent': 1000,
            'bytes_received': 500,
            'packet_count': 100
        },
        {
            'asset_id': 'server-002',
            'src_ip': '127.0.0.1',  # Should be filtered (whitelist)
            'dst_ip': '192.168.1.100',
            'src_port': 22,
            'dst_port': 80,
            'protocol': 'tcp',
            'flow_duration': 1.0,
            'bytes_sent': 2000,
            'bytes_received': 1000,
            'packet_count': 200
        },
        {
            'asset_id': 'server-001',
            'src_ip': '10.0.0.3',
            'dst_ip': '10.0.0.4',
            'src_port': 54321,
            'dst_port': 22,
            'protocol': 'tcp',
            'flow_duration': 2.0,
            'bytes_sent': 5000,  # Anomalous spike
            'bytes_received': 2500,
            'packet_count': 300
        }
    ]

    print("Processing events through pre-inference pipeline...")
    for i, event in enumerate(test_events, 1):
        result = pipeline.process_event(event)
        print(f"\nEvent {i}:")
        print(f"  Should infer: {result.should_infer}")
        print(f"  Filter reason: {result.filter_reason}")
        print(f"  Processing time: {result.processing_time_ms:.2f}ms")
        if result.context.get('baseline_deviations'):
            print(f"  Baseline deviations: {result.context['baseline_deviations']}")

    # Print statistics
    stats = pipeline.get_statistics()
    print("\nPipeline Statistics:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Events passed: {stats['events_passed']}")
    print(f"  Events filtered: {stats['events_filtered']}")
    print(f"  Pass rate: {stats['pass_rate']:.2%}")
    print(f"  Filter rate: {stats['filter_rate']:.2%}")
    print(f"  Filtered by stage: {stats['filtered_by_stage']}")
    print(f"  Avg processing time: {stats['avg_processing_time_ms']:.2f}ms")

    # Print component statistics
    component_stats = pipeline.get_component_statistics()
    print("\nComponent Statistics:")
    for component, comp_stats in component_stats.items():
        print(f"  {component}:")
        for key, value in comp_stats.items():
            print(f"    {key}: {value}")

    # Close pipeline
    pipeline.close()


if __name__ == "__main__":
    main()
