"""
Time-To-Collision (TTC) calculator for the ADAS Supervision Framework.

Computes TTC for each detection based on distance and closing
velocity.  Returns ``float('inf')`` for non-approaching objects and
applies an epsilon clamp for numerical stability in downstream risk
calculations.
"""

import logging
from typing import List

from perception.object_detector import Detection

logger = logging.getLogger(__name__)


class TTCCalculator:
    """Computes Time-To-Collision from detections.

    Args:
        epsilon: Minimum TTC floor (seconds) to prevent division-by-
            zero explosions in downstream ``1 / TTC`` risk terms.
            Default ``0.1``.
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = max(epsilon, 1e-6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, detections: List[Detection]) -> List[float]:
        """Compute TTC for each detection.

        Args:
            detections: Current-tick detection list.

        Returns:
            List of TTC values (seconds), aligned with *detections*.
            ``float('inf')`` means the object is not approaching.
        """
        ttcs: List[float] = []
        for det in detections:
            ttc = self._single_ttc(det.distance, det.relative_velocity)
            ttcs.append(ttc)
        return ttcs

    def compute_min(self, detections: List[Detection]) -> float:
        """Return the minimum TTC across all detections.

        Returns:
            Minimum TTC (seconds), or ``float('inf')`` if no
            approaching objects exist.
        """
        ttcs = self.compute(detections)
        return min(ttcs) if ttcs else float("inf")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _single_ttc(self, distance: float, closing_speed: float) -> float:
        """TTC for a single object.

        Args:
            distance: Range in metres.
            closing_speed: Positive means the object is approaching.

        Returns:
            TTC in seconds (ε-clamped), or ``inf`` if diverging.
        """
        if closing_speed <= 0.0:
            return float("inf")
        ttc = distance / closing_speed
        return max(ttc, self.epsilon)
