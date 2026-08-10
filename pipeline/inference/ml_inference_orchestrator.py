"""
ML Inference Orchestration for INTERSYMBOLIC-GRC

Coordinates the tri-stage inference pipeline:
1. Pre-Inference: Symbolic rules (validation, filtering, baseline context)
2. In-Inference: ML models (anomaly detection, graph scoring, probabilistic risk)
3. Output: Risk signals with confidence, explanations, and context

This module provides the main orchestration class that integrates
pre-inference symbolic rules, feature extraction, sub-symbolic ML models,
and risk signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

from pipeline.pre_inference import PreInferencePipeline
from pipeline.features import FeatureExtractionPipeline

# ML model backends (optional — deleted in codebase cleanup; graceful fallback)
try:
    from models import (
        AnomalyDetectionModel,
        GraphBehavioralScoring,
        ProbabilisticRiskPipeline,
        EnsembleAnomalyDetector,
    )
    from models.anomaly_detection import create_isolation_forest, create_autoencoder
    from models.graph_behavioral_scoring import create_graph_scorer
    from models.probabilistic_risk_indicators import create_bayesian_updater, create_risk_aggregator
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    AnomalyDetectionModel = None
    GraphBehavioralScoring = None
    ProbabilisticRiskPipeline = None
    EnsembleAnomalyDetector = None
    create_isolation_forest = None
    create_autoencoder = None
    create_graph_scorer = None
    create_bayesian_updater = None
    create_risk_aggregator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceStage(Enum):
    """Inference pipeline stages"""
    PRE_INFERENCE = "pre_inference"
    FEATURE_EXTRACTION = "feature_extraction"
    ANOMALY_DETECTION = "anomaly_detection"
    GRAPH_SCORING = "graph_scoring"
    PROBABILISTIC_RISK = "probabilistic_risk"
    POST_INFERENCE = "post_inference"


@dataclass
class InferenceInput:
    """Input for inference pipeline"""
    event: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


@dataclass
class InferenceResult:
    """Result of inference pipeline"""
    event_id: str
    timestamp: datetime
    stage: str
    risk_signals: List['RiskSignal'] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None
    overall_risk_score: float = 0.0
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'stage': self.stage,
            'risk_signals': [s.to_dict() for s in self.risk_signals],
            'explanations': self.explanations,
            'metadata': self.metadata,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score,
            'overall_risk_score': self.overall_risk_score,
            'confidence': self.confidence
        }


@dataclass
class RiskSignal:
    """Risk signal from inference pipeline"""
    type: str  # 'anomaly', 'behavioral', 'risk', 'violation'
    source: str  # 'anomaly_detection', 'graph_scoring', 'probabilistic_risk', 'pre_inference'
    score: float  # 0-1
    description: str
    entity_type: str  # 'asset', 'connection', 'threat', 'component'
    entity_id: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'source': self.source,
            'score': self.score,
            'description': self.description,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'confidence': self.confidence,
            'tags': self.tags,
            'metadata': self.metadata
        }


class MLInferenceOrchestrator:
    """
    Orchestrates the tri-stage ML inference pipeline for INTERSYMBOLIC-GRC

    Pipeline Stages:
    1. Pre-Inference: Symbolic rules validate and filter events
    2. Feature Extraction: Extract features for ML models
    3. Anomaly Detection: Detect anomalies in event data
    4. Graph Scoring: Calculate behavioral scores from ARG
    5. Probabilistic Risk: Combine signals into risk scores
    6. Post-Inference: Map to GRC entities
    """

    def __init__(
        self,
        neo4j_driver=None,
        use_enhanced_models: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize ML Inference Orchestrator

        Args:
            neo4j_driver: Neo4j driver for graph-based scoring
            use_enhanced_models: Whether to use ensemble and advanced models
            confidence_threshold: Minimum confidence to flag as anomaly
        """
        self.neo4j_driver = neo4j_driver
        self.confidence_threshold = confidence_threshold
        self.use_enhanced_models = use_enhanced_models

        # Initialize pre-inference pipeline
        self.pre_inference = PreInferencePipeline()

        # Initialize feature extraction pipeline
        self.feature_extractor = FeatureExtractionPipeline()

        # Initialize ML models
        self.anomaly_detector: Optional[EnsembleAnomalyDetector] = None
        self.graph_scorer: Optional[GraphBehavioralScoring] = None
        self.risk_pipeline: Optional[ProbabilisticRiskPipeline] = None

        # Cache for models
        self.models_loaded = False

        # Statistics
        self.stats = {
            'total_events': 0,
            'events_passed_pre_inference': 0,
            'events_processed': 0,
            'anomalies_detected': 0,
            'risk_signals_generated': 0
        }

    def load_models(self) -> None:
        """
        Load ML models (lazy loading)

        Creates:
        - EnsembleAnomalyDetector (Isolation Forest + Autoencoder)
        - GraphBehavioralScoring
        - ProbabilisticRiskPipeline
        """
        if self.models_loaded:
            logger.info("Models already loaded, skipping...")
            return

        if not _MODELS_AVAILABLE:
            logger.warning("ML model backends not available (models/ directory removed). "
                           "Orchestrator runs in rule-only mode.")
            self.models_loaded = True
            return

        logger.info("Loading ML models...")

        try:
            # Initialize ensemble anomaly detector
            if self.use_enhanced_models:
                logger.info("Creating EnsembleAnomalyDetector...")
                self.anomaly_detector = EnsembleAnomalyDetector(
                    contamination=0.1,
                    n_estimators=100,
                    random_state=42
                )
                logger.info("EnsembleAnomalyDetector created")
            else:
                logger.info("Using simple Isolation Forest...")
                self.anomaly_detector = create_isolation_forest(
                    contamination=0.1,
                    random_state=42
                )

            # Initialize graph behavioral scorer
            if self.neo4j_driver is not None:
                logger.info("Initializing GraphBehavioralScoring...")
                self.graph_scorer = create_graph_scorer(neo4j_driver=self.neo4j_driver)
                logger.info("GraphBehavioralScoring initialized")
            else:
                logger.warning("Neo4j driver not configured, graph scoring disabled")
                self.graph_scorer = None

            # Initialize probabilistic risk pipeline
            logger.info("Initializing ProbabilisticRiskPipeline...")
            self.risk_pipeline = create_risk_pipeline()
            logger.info("ProbabilisticRiskPipeline initialized")

            self.models_loaded = True
            logger.info("All ML models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            raise

    def run(
        self,
        inference_input: InferenceInput
    ) -> InferenceResult:
        """
        Run complete inference pipeline on a single event

        Args:
            inference_input: Input event and context

        Returns:
            InferenceResult with risk signals and explanations
        """
        self.stats['total_events'] += 1

        logger.debug(f"Running inference on event {inference_input.event_id}")

        # Stage 1: Pre-Inference
        stage_result = self._run_pre_inference(inference_input)
        if not stage_result['passed']:
            logger.debug(f"Event {inference_input.event_id} failed pre-inference: {stage_result['reason']}")
            return InferenceResult(
                event_id=inference_input.event_id,
                timestamp=inference_input.timestamp or datetime.utcnow(),
                stage=InferenceStage.PRE_INFERENCE.value,
                explanations=[stage_result['reason']],
                metadata={'pre_inference': stage_result}
            )

        self.stats['events_passed_pre_inference'] += 1

        # Stage 2: Feature Extraction
        features = self._extract_features(stage_result['context'])

        # Stage 3: Anomaly Detection
        anomaly_result = self._run_anomaly_detection(features)
        anomaly_signal = anomaly_result['signal']

        # Stage 4: Graph Scoring (if enabled)
        graph_signals = []
        if self.graph_scorer is not None and self.models_loaded:
            graph_signals = self._run_graph_scoring(inference_input)

        # Stage 5: Probabilistic Risk
        risk_signals = [anomaly_signal] + graph_signals
        risk_result = self._run_probabilistic_risk(risk_signals, inference_input)

        # Stage 6: Post-Inference (GRC mapping)
        post_inference = self._run_post_inference(
            inference_result=risk_result,
            inference_input=inference_input
        )

        # Combine results
        result = InferenceResult(
            event_id=inference_input.event_id,
            timestamp=inference_input.timestamp or datetime.utcnow(),
            stage=InferenceStage.POST_INFERENCE.value,
            risk_signals=risk_result['signals'],
            explanations=post_inference['explanations'],
            metadata={
                **post_inference['metadata'],
                'anomaly_result': anomaly_result,
                'graph_signals': graph_signals,
                'pre_inference': stage_result
            },
            is_anomaly=anomaly_result['is_anomaly'],
            anomaly_score=anomaly_result.get('score'),
            overall_risk_score=risk_result['overall_risk'],
            confidence=post_inference.get('confidence', 0.5)
        )

        # Update statistics
        self.stats['events_processed'] += 1
        if result.is_anomaly:
            self.stats['anomalies_detected'] += 1
        self.stats['risk_signals_generated'] += len(result.risk_signals)

        return result

    def _run_pre_inference(
        self,
        inference_input: InferenceInput
    ) -> Dict[str, Any]:
        """
        Stage 1: Run pre-inference symbolic rules

        Returns:
            Dict with passed flag, context, and reason
        """
        stage_result = {
            'passed': False,
            'context': {},
            'reason': ''
        }

        try:
            # Run pre-inference pipeline
            context = self.pre_inference.run(
                event=inference_input.event,
                context=inference_input.context or {}
            )

            # Check if event passes validation
            if context['validation_result']['passed']:
                stage_result['passed'] = True
                stage_result['context'] = context
                stage_result['reason'] = 'Event passed all pre-inference validation rules'
            else:
                stage_result['passed'] = False
                stage_result['reason'] = f'Event failed validation: {context["validation_result"]["reasons"][:3]}'
                stage_result['context'] = context

        except Exception as e:
            logger.error(f"Pre-inference failed: {e}", exc_info=True)
            stage_result['passed'] = False
            stage_result['reason'] = f'Pre-inference error: {str(e)}'

        return stage_result

    def _extract_features(
        self,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Stage 2: Extract features for ML models

        Args:
            context: Context from pre-inference stage

        Returns:
            Feature DataFrame
        """
        try:
            # Extract features using feature extraction pipeline
            features_df = self.feature_extractor.run(context=context)

            if features_df is None or features_df.empty:
                logger.warning("No features extracted")
                return pd.DataFrame()

            logger.debug(f"Extracted {len(features_df)} features")
            return features_df

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _run_anomaly_detection(
        self,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Stage 3: Run anomaly detection

        Args:
            features: Feature DataFrame

        Returns:
            Dict with signal and is_anomaly flags
        """
        if features.empty:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'signal': RiskSignal(
                    type='anomaly',
                    source='anomaly_detection',
                    score=0.0,
                    description='No features extracted',
                    entity_type='general',
                    entity_id='unknown',
                    confidence=0.0
                )
            }

        try:
            # Load models if not already loaded
            if not self.models_loaded:
                self.load_models()

            # Get anomaly score
            if isinstance(self.anomaly_detector, EnsembleAnomalyDetector):
                anomaly_score, is_anomaly = self.anomaly_detector.predict(features)
            else:
                anomaly_score = self.anomaly_detector.predict(features)
                is_anomaly = anomaly_score < 0

            return {
                'is_anomaly': is_anomaly,
                'score': float(anomaly_score) if not np.isnan(anomaly_score) else 0.0,
                'signal': RiskSignal(
                    type='anomaly',
                    source='anomaly_detection',
                    score=float(anomaly_score) if not np.isnan(anomaly_score) else 0.0,
                    description=f'Anomaly detected with score {anomaly_score:.3f}'
                        if is_anomaly else 'No anomaly detected',
                    entity_type='threat_event',
                    entity_id=f'anomaly_{self.stats["events_processed"]}',
                    confidence=0.8
                )
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return {
                'is_anomaly': False,
                'score': 0.0,
                'signal': RiskSignal(
                    type='anomaly',
                    source='anomaly_detection',
                    score=0.0,
                    description=f'Anomaly detection error: {str(e)}',
                    entity_type='threat_event',
                    entity_id=f'anomaly_{self.stats["events_processed"]}',
                    confidence=0.0
                )
            }

    def _run_graph_scoring(
        self,
        inference_input: InferenceInput
    ) -> List[RiskSignal]:
        """
        Stage 4: Run graph-based behavioral scoring

        Args:
            inference_input: Input event

        Returns:
            List of risk signals from graph scoring
        """
        if self.graph_scorer is None or not self.models_loaded:
            return []

        try:
            # Compute graph scores
            scores = self.graph_scorer.compute_scores(
                asset_id=inference_input.event.get('assetId'),
                connection_id=inference_input.event.get('connectionId')
            )

            signals = []
            for score_type, score in scores.items():
                if score['score'] > 0.5:  # Threshold
                    signals.append(RiskSignal(
                        type='behavioral',
                        source='graph_scoring',
                        score=score['score'],
                        description=f'{score_type}: {score["description"]}',
                        entity_type=score['entity_type'],
                        entity_id=score['entity_id'],
                        confidence=0.75
                    ))

            return signals

        except Exception as e:
            logger.error(f"Graph scoring failed: {e}", exc_info=True)
            return []

    def _run_probabilistic_risk(
        self,
        risk_signals: List[RiskSignal],
        inference_input: InferenceInput
    ) -> Dict[str, Any]:
        """
        Stage 5: Combine signals into probabilistic risk scores

        Args:
            risk_signals: List of risk signals
            inference_input: Input event

        Returns:
            Dict with overall risk score and signals
        """
        try:
            if self.risk_pipeline is None or not self.models_loaded:
                # Return simple aggregation
                if risk_signals:
                    max_score = max(s.score for s in risk_signals)
                    avg_score = np.mean([s.score for s in risk_signals])
                    overall_risk = max(max_score, avg_score)
                else:
                    overall_risk = 0.0

                return {
                    'signals': risk_signals,
                    'overall_risk': overall_risk
                }

            # Use probabilistic risk pipeline
            risk_result = self.risk_pipeline.run(
                signals=risk_signals,
                context=inference_input.context or {}
            )

            return risk_result

        except Exception as e:
            logger.error(f"Probabilistic risk calculation failed: {e}", exc_info=True)
            return {
                'signals': risk_signals,
                'overall_risk': 0.0
            }

    def _run_post_inference(
        self,
        inference_result: InferenceResult,
        inference_input: InferenceInput
    ) -> Dict[str, Any]:
        """
        Stage 6: Post-inference GRC mapping

        Args:
            inference_result: Result from earlier stages
            inference_input: Original input

        Returns:
            Dict with explanations and metadata
        """
        explanations = []
        metadata = {}

        try:
            # Map risk signals to GRC entities
            risk_signals = inference_result.risk_signals

            if risk_signals:
                explanations.append(
                    f'Generated {len(risk_signals)} risk signals: '
                    f'{", ".join(set(s.type for s in risk_signals))}'
                )

                # Extract entity information
                entities = set(s.entity_id for s in risk_signals)
                explanations.append(f'Associated entities: {", ".join(entities)}')

            # Add confidence
            confidence = inference_result.confidence
            if confidence >= 0.7:
                confidence_level = 'high'
            elif confidence >= 0.5:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'

            explanations.append(f'Overall confidence: {confidence_level} ({confidence:.2f})')

            metadata = {
                'entity_ids': list(set(s.entity_id for s in risk_signals)),
                'signal_types': list(set(s.type for s in risk_signals)),
                'risk_level': 'critical' if inference_result.overall_risk_score > 0.8
                             else 'high' if inference_result.overall_risk_score > 0.6
                             else 'medium' if inference_result.overall_risk_score > 0.4
                             else 'low'
            }

        except Exception as e:
            logger.error(f"Post-inference failed: {e}", exc_info=True)

        return {
            'explanations': explanations,
            'metadata': metadata
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats

    def print_statistics(self) -> None:
        """Print statistics to console"""
        print("\n" + "=" * 60)
        print("ML INFERENCE ORCHESTRATOR STATISTICS")
        print("=" * 60)
        print(f"Total Events: {self.stats['total_events']}")
        print(f"Events Passed Pre-Inference: {self.stats['events_passed_pre_inference']}")
        print(f"Events Processed: {self.stats['events_processed']}")
        print(f"Anomalies Detected: {self.stats['anomalies_detected']}")
        print(f"Risk Signals Generated: {self.stats['risk_signals_generated']}")
        print("=" * 60)


def create_risk_pipeline():
    """Create probabilistic risk pipeline. Returns None if model backends unavailable."""
    if not _MODELS_AVAILABLE:
        logger.warning("models/ backends unavailable — create_risk_pipeline() returns None")
        return None
    from models.probabilistic_risk_indicators import (
        BayesianRiskUpdater,
        RiskAggregator,
    )
    bayesian_updater = BayesianRiskUpdater()
    risk_aggregator = RiskAggregator()
    return ProbabilisticRiskPipeline(
        risk_updaters=[bayesian_updater],
        risk_aggregator=risk_aggregator,
        confidence_threshold=0.6
    )
