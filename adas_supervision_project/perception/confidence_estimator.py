"""
Confidence estimator for the ADAS Supervision Framework.

Refines raw detection confidence scores by applying temporal decay,
weather degradation, and occlusion penalties.  Maintains per-object
confidence memory for smooth temporal tracking.
"""

import logging
from typing import Dict, List

from perception.object_detector import Detection
from utils.math_utils import clamp

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Estimates and tracks detection confidence over time.

    Confidence decays each tick an object is *not re-detected*, and
    is refreshed (with possible degradation) when it *is* detected.

    Args:
        decay_rate: Per-tick multiplicative decay (e.g. 0.02 → 2 %%).
        weather_factor: Multiplicative penalty ∈ (0, 1] for weather.
    """

    def __init__(self, decay_rate: float = 0.02, weather_factor: float = 1.0):
        self.decay_rate = decay_rate
        self.weather_factor = clamp(weather_factor, 0.1, 1.0)
        # actor_id → last tracked confidence
        self._memory: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Refine confidence for a batch of detections.

        Also decays confidence for previously tracked objects that did
        **not** appear in this cycle.

        Args:
            detections: Current-tick detections (modified in-place).

        Returns:
            The same list with updated ``confidence`` values.
        """
        seen_ids = set()

        for det in detections:
            seen_ids.add(det.actor_id)
            raw = det.confidence * self.weather_factor

            # Blend with memory (exponential moving average)
            prev = self._memory.get(det.actor_id)
            if prev is not None:
                refined = 0.7 * raw + 0.3 * prev
            else:
                refined = raw

            det.confidence = clamp(refined, 0.0, 1.0)
            self._memory[det.actor_id] = det.confidence

        # Decay unseen objects
        for aid in list(self._memory):
            if aid not in seen_ids:
                self._memory[aid] = max(
                    0.0, self._memory[aid] - self.decay_rate
                )
                if self._memory[aid] <= 0.0:
                    del self._memory[aid]

        return detections

    def set_weather_factor(self, factor: float):
        """Dynamically update the weather degradation factor.

        Args:
            factor: Value in (0, 1].  Lower = worse visibility.
        """
        self.weather_factor = clamp(factor, 0.1, 1.0)

    def get_tracked_count(self) -> int:
        """Number of objects currently in the confidence memory."""
        return len(self._memory)
