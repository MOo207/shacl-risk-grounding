"""
ML Inference Orchestration

This module contains the orchestration layer that integrates pre-inference symbolic rules,
feature extraction, sub-symbolic ML models, and risk signal generation.

The inference orchestration follows the INTERSYMBOLIC-GRC tri-stage pipeline:
1. Pre-Inference: Symbolic rules (validation, filtering, baseline context, exceptions)
2. In-Inference: ML models (anomaly detection, graph scoring, probabilistic risk)
3. Output: Risk signals with confidence, explanations, and context
"""

from .ml_inference_orchestrator import MLInferenceOrchestrator, InferenceResult, RiskSignal

__all__ = [
    'MLInferenceOrchestrator',
    'InferenceResult',
    'RiskSignal',
]

__version__ = '0.1.0'
