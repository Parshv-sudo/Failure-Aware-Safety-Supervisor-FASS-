"""
Rule-based risk assessor for the ADAS Supervision Framework.

Implements ``BaseRiskModel`` with a configurable weighted-sum formula.
All inputs are normalised to [0, 1] before weighting, and the final
score is clamped to [0, 1].

Formula::

    risk = w1·(1 − confidence)
         + w2·(1 / max(TTC, ε)) / (1 / ε)
         + w3·(speed / speed_max)
         + w4·road_complexity

Where ε prevents division-by-zero and (1/ε) scales the TTC term
to [0, 1].
"""

import logging
from typing import List

from perception.object_detector import Detection
from supervision.risk_model_interface import BaseRiskModel, RiskInput, RiskOutput
from utils.math_utils import clamp

logger = logging.getLogger(__name__)


class RuleBasedRiskModel(BaseRiskModel):
    """Weighted-sum risk model with per-term normalisation.

    Args:
        weights: Dict with keys ``w1_confidence``, ``w2_ttc``,
            ``w3_speed``, ``w4_complexity``.
        epsilon: TTC floor (seconds).
        speed_max: Maximum expected ego speed (m/s) for normalisation.
    """

    def __init__(
        self,
        weights: dict = None,
        epsilon: float = 0.1,
        speed_max: float = 40.0,
    ):
        w = weights or {}
        self.w1 = w.get("w1_confidence", 0.30)
        self.w2 = w.get("w2_ttc", 0.35)
        self.w3 = w.get("w3_speed", 0.15)
        self.w4 = w.get("w4_complexity", 0.20)
        self.epsilon = max(epsilon, 1e-6)
        self.speed_max = max(speed_max, 1e-6)

    # ------------------------------------------------------------------
    # BaseRiskModel interface
    # ------------------------------------------------------------------

    def compute_risk(self, inputs: RiskInput) -> RiskOutput:
        """Compute weighted risk from normalised inputs.

        Args:
            inputs: ``RiskInput`` with all fields already in [0, 1].

        Returns:
            ``RiskOutput`` with clamped ``risk`` and component breakdown.
        """
        confidence_term = self.w1 * (1.0 - inputs.confidence)
        ttc_term = self.w2 * inputs.ttc
        speed_term = self.w3 * inputs.speed_factor
        complexity_term = self.w4 * inputs.road_complexity

        raw = confidence_term + ttc_term + speed_term + complexity_term
        risk = clamp(raw, 0.0, 1.0)

        return RiskOutput(
            risk=risk,
            components={
                "confidence_term": round(confidence_term, 6),
                "ttc_term": round(ttc_term, 6),
                "speed_term": round(speed_term, 6),
                "complexity_term": round(complexity_term, 6),
            },
        )


class RiskAssessor:
    """High-level risk assessor that normalises raw inputs and delegates
    scoring to a ``BaseRiskModel``.

    Args:
        model: Risk model implementation (defaults to rule-based).
        epsilon: TTC floor for normalisation.
        speed_max: Speed ceiling for normalisation (m/s).
    """

    def __init__(
        self,
        model: BaseRiskModel = None,
        epsilon: float = 0.1,
        speed_max: float = 40.0,
    ):
        self.epsilon = max(epsilon, 1e-6)
        self.speed_max = max(speed_max, 1e-6)
        self.model = model or RuleBasedRiskModel(
            epsilon=self.epsilon, speed_max=self.speed_max
        )

    def assess(
        self,
        detections: List[Detection],
        ttcs: List[float],
        ego_speed: float,
        road_complexity: float,
        ego_vehicle = None,
    ) -> RiskOutput:
        """Run the full risk pipeline: normalise → model → output.

        Args:
            detections: Current-tick detections with confidence.
            ttcs: Per-detection TTC values (seconds).
            ego_speed: Ego scalar speed (m/s).
            road_complexity: ODD road complexity ∈ [0, 1].
            ego_vehicle: Optional CARLA vehicle object for full kinematic extraction.

        Returns:
            ``RiskOutput`` with score and component breakdown.
        """
        # Aggregate perception: use worst (lowest) confidence
        if detections:
            worst_conf = min(d.confidence for d in detections)
        else:
            worst_conf = 1.0  # no detections → fully confident scene

        # TTC normalisation: 1/max(TTC,ε) / (1/ε) → [0,1]
        min_ttc = min(ttcs) if ttcs else float("inf")
        ttc_norm = (1.0 / max(min_ttc, self.epsilon)) / (1.0 / self.epsilon)
        ttc_norm = clamp(ttc_norm, 0.0, 1.0)

        # Speed normalisation
        speed_norm = clamp(ego_speed / self.speed_max, 0.0, 1.0)

        # Road complexity already in [0,1]
        rc = clamp(road_complexity, 0.0, 1.0)

        inputs = RiskInput(
            confidence=worst_conf,
            ttc=ttc_norm,
            speed_factor=speed_norm,
            road_complexity=rc,
        )
        
        # Pass raw detections and speed if the model supports it
        import inspect
        sig = inspect.signature(self.model.compute_risk)
        kwargs = {}
        if "raw_detections" in sig.parameters:
            kwargs["raw_detections"] = detections
        if "ego_speed" in sig.parameters:
            kwargs["ego_speed"] = ego_speed
        if "ego_vehicle" in sig.parameters:
            kwargs["ego_vehicle"] = ego_vehicle
        
        return self.model.compute_risk(inputs, **kwargs)
