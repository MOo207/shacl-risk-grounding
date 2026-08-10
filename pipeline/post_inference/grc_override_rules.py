"""
Post-Inference Override Rules for INTERSYMBOLIC-GRC

Overrides ML predictions when:
1. ML confidence is LOW (< threshold)
2. Symbolic evidence is STRONG (multiple conditions met)

These rules are calibrated on validation set to minimize accuracy loss
while providing compliance enforcement.

Usage:
    from pipeline.post_inference.grc_override_rules import OverrideRules
    override = OverrideRules(confidence_threshold=0.7)
    final_label = override.apply(row, ml_prediction, ml_confidence)
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OverrideRule:
    """Override rule configuration"""
    rule_id: str
    description: str
    target_class: str
    enabled: bool
    threshold: Optional[float]  # None = disabled
    condition_fn: callable  # Function that takes row and returns bool
    min_confidence_threshold: float = 0.7  # Only override if ML confidence below this


class OverrideRules:
    """
    Post-inference override rules engine

    Overrides ML predictions ONLY when:
    1. ML confidence < threshold (uncertain prediction)
    2. Symbolic evidence is strong (rule conditions met)

    This prevents catastrophic accuracy loss by only overriding
    uncertain predictions with strong symbolic evidence.
    """

    # Default configuration (will be calibrated)
    DEFAULT_RULES = {
        "rule_syn_ddos": {
            "threshold": 200.0,
            "enabled": False,  # Disabled by default (needs calibration)
            "description": "SYN Flag Cnt > threshold → DDoS",
            "target_class": "DDoS"
        },
        "rule_flow_dos": {
            "threshold": 500.0,
            "enabled": False,  # Disabled by default (needs calibration)
            "description": "Flow Pkts/s > threshold → DoS",
            "target_class": "DoS"
        },
        "rule_bruteforce": {
            "threshold_fwd": 3.0,
            "threshold_syn": 1.0,
            "enabled": False,  # Disabled by default (needs calibration)
            "description": "Tot Fwd Pkts <= threshold_fwd AND SYN = threshold_syn → BruteForce",
            "target_class": "BruteForce"
        },
        "rule_infiltration": {
            "threshold_len": 200000.0,
            "threshold_dur": 600.0,
            "enabled": False,  # Disabled by default (needs calibration)
            "description": "TotLen Fwd Pkts > threshold_len AND Duration > threshold_dur → Infiltration",
            "target_class": "Infiltration"
        }
    }

    def __init__(self, confidence_threshold: float = 0.7, rules_config: Optional[Dict] = None):
        """
        Initialize override rules engine

        Args:
            confidence_threshold: ML confidence below which overrides are allowed
            rules_config: Optional custom rule configuration (calibrated values)
        """
        self.confidence_threshold = confidence_threshold
        self.rules_config = rules_config or self.DEFAULT_RULES.copy()
        self.override_count = 0
        self.total_predictions = 0

        logger.info(f"Initialized OverrideRules with confidence_threshold={confidence_threshold}")
        enabled_rules = sum(1 for r in self.rules_config.values() if r.get("enabled", False))
        logger.info(f"  Enabled rules: {enabled_rules}/{len(self.rules_config)}")

    def load_calibration(self, calibration_path: str):
        """
        Load calibrated thresholds from file

        Args:
            calibration_path: Path to calibration JSON file
        """
        import json
        from pathlib import Path

        cal_path = Path(calibration_path)
        if not cal_path.exists():
            logger.warning(f"Calibration file not found: {calibration_path}")
            return

        with open(cal_path, 'r') as f:
            calibration = json.load(f)

        # Apply calibrated values
        for rule_id, cal_data in calibration.items():
            if rule_id in self.rules_config:
                self.rules_config[rule_id]["threshold"] = cal_data.get("best_threshold")
                self.rules_config[rule_id]["enabled"] = cal_data.get("enabled", False)

        logger.info(f"Loaded calibration from {calibration_path}")

    def apply(
        self,
        row: Dict[str, Any],
        ml_prediction: str,
        ml_confidence: float
    ) -> str:
        """
        Apply override rules to ML prediction

        Args:
            row: Feature row as dict
            ml_prediction: ML model prediction
            ml_confidence: ML model confidence (0.0-1.0)

        Returns:
            Final prediction (either ML or overridden)
        """
        self.total_predictions += 1

        # If ML confidence is high, trust the model
        if ml_confidence >= self.confidence_threshold:
            logger.debug(f"ML confidence high ({ml_confidence:.3f} >= {self.confidence_threshold}), trusting prediction")
            return ml_prediction

        # Check each enabled override rule
        for rule_id, rule_config in self.rules_config.items():
            if not rule_config.get("enabled", False):
                continue

            if self._check_rule(row, rule_config):
                override_label = rule_config["target_class"]
                self.override_count += 1
                logger.debug(
                    f"Override triggered: {rule_id} "
                    f"ML: {ml_prediction} ({ml_confidence:.3f}) → {override_label}"
                )
                return override_label

        # No override triggered, return ML prediction
        return ml_prediction

    def _check_rule(self, row: Dict[str, Any], rule_config: Dict) -> bool:
        """
        Check if a single override rule triggers

        Args:
            row: Feature row
            rule_config: Rule configuration

        Returns:
            True if rule triggers
        """
        rule_id = rule_config.get("rule_id", "unknown")

        try:
            # Extract numeric features safely
            syn = float(row.get("SYN Flag Cnt", 0))
            pps = float(row.get("Flow Pkts/s", 0))
            tfwd = float(row.get("Tot Fwd Pkts", 0))
            tlen = float(row.get("TotLen Fwd Pkts", 0))
            dur = float(row.get("Flow Duration", 0))
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to extract features for rule {rule_id}: {e}")
            return False

        # Check specific rules
        if rule_id == "rule_syn_ddos":
            threshold = rule_config.get("threshold", 200.0)
            if threshold is not None and syn > threshold:
                return True

        elif rule_id == "rule_flow_dos":
            threshold = rule_config.get("threshold", 500.0)
            if threshold is not None and pps > threshold:
                return True

        elif rule_id == "rule_bruteforce":
            threshold_fwd = rule_config.get("threshold_fwd", 3.0)
            threshold_syn = rule_config.get("threshold_syn", 1.0)
            if (threshold_fwd is not None and threshold_syn is not None and
                tfwd <= threshold_fwd and syn == threshold_syn):
                return True

        elif rule_id == "rule_infiltration":
            threshold_len = rule_config.get("threshold_len", 200000.0)
            threshold_dur = rule_config.get("threshold_dur", 600.0)
            if (threshold_len is not None and threshold_dur is not None and
                tlen > threshold_len and dur > threshold_dur):
                return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get override statistics

        Returns:
            Dict with statistics
        """
        override_rate = (self.override_count / self.total_predictions * 100
                       if self.total_predictions > 0 else 0.0)

        return {
            "total_predictions": self.total_predictions,
            "override_count": self.override_count,
            "override_rate": override_rate,
            "confidence_threshold": self.confidence_threshold,
            "enabled_rules": sum(1 for r in self.rules_config.values() if r.get("enabled", False))
        }

    def print_statistics(self):
        """Print statistics to console"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("OVERRIDE RULES STATISTICS")
        print("=" * 60)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Overrides Applied: {stats['override_count']}")
        print(f"Override Rate: {stats['override_rate']:.2f}%")
        print(f"Confidence Threshold: {stats['confidence_threshold']}")
        print(f"Enabled Rules: {stats['enabled_rules']}/{len(self.rules_config)}")
        print("=" * 60)


def create_override_rules(
    confidence_threshold: float = 0.7,
    calibration_path: Optional[str] = None
) -> OverrideRules:
    """
    Create override rules engine

    Args:
        confidence_threshold: ML confidence below which overrides are allowed
        calibration_path: Optional path to calibrated thresholds file

    Returns:
        OverrideRules instance
    """
    override = OverrideRules(confidence_threshold=confidence_threshold)

    if calibration_path:
        override.load_calibration(calibration_path)

    return override
