"""
Driver model for the ADAS Supervision Framework.

Simulates human driver behaviour including:
    - **Attention level** ∈ [0, 1] with configurable decay and
      recovery on alerts.
    - **Overtrust mode** — attention decays over time as the driver
      becomes complacent, increasing response latency.
    - **Hazard-coupled response** — response time depends on
      attention, road complexity, speed, and alert urgency.

The model allows studying early vs. late warning effectiveness
and false-positive tolerance.
"""

import logging
import math
import random
from typing import Optional

from utils.math_utils import clamp

logger = logging.getLogger(__name__)


class DriverModel:
    """Simulated driver with configurable attention dynamics.

    Args:
        initial_attention: Starting attention level ∈ [0, 1].
        decay_rate: Per-tick additive attention decay.
        recovery_on_alert: Additive attention boost when an alert fires.
        attention_epsilon: Floor for attention to prevent division
            explosion in response-time calculation.
        random_seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        initial_attention: float = 0.95,
        decay_rate: float = 0.005,
        recovery_on_alert: float = 0.3,
        attention_epsilon: float = 0.05,
        random_seed: int = 42,
    ):
        self.attention = initial_attention
        self._decay_rate = decay_rate
        self._recovery = recovery_on_alert
        self._epsilon = max(attention_epsilon, 1e-3)
        self._rng = random.Random(random_seed)
        self._initial = initial_attention

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, alert_active: bool = False):
        """Advance the driver model by one tick.

        If an alert is active, attention recovers; otherwise it
        decays (overtrust simulation).

        Args:
            alert_active: Whether any takeover alert is active this tick.
        """
        if alert_active:
            self.attention = clamp(
                self.attention + self._recovery, 0.0, 1.0
            )
        else:
            self.attention = clamp(
                self.attention - self._decay_rate, 0.0, 1.0
            )

    def sample_response_time(
        self,
        base_mean: float,
        base_std: float,
        distribution: str = "lognormal",
        road_complexity: float = 0.0,
        ego_speed: float = 0.0,
        speed_max: float = 40.0,
        complexity_alpha: float = 0.3,
        speed_beta: float = 0.2,
        alert_urgency: float = 1.0,
    ) -> float:
        """Sample a hazard-coupled driver response time.

        Formula::

            response = base_sample
                     × (1 + α · complexity)
                     × (1 + β · speed_norm)
                     / alert_urgency
                     / max(attention, ε)

        Args:
            base_mean: Mean of the base distribution (seconds).
            base_std: Std deviation of the base distribution.
            distribution: ``"gaussian"`` or ``"lognormal"``.
            road_complexity: ∈ [0, 1].
            ego_speed: m/s.
            speed_max: Speed ceiling for normalisation.
            complexity_alpha: Complexity coupling coefficient.
            speed_beta: Speed coupling coefficient.
            alert_urgency: Urgency factor (higher → shorter response).

        Returns:
            Response time in seconds (≥ 0.1).
        """
        # Base sample
        if distribution == "lognormal" and base_mean > 0:
            sigma2 = math.log(1 + (base_std / base_mean) ** 2)
            mu = math.log(base_mean) - sigma2 / 2
            base = self._rng.lognormvariate(mu, math.sqrt(sigma2))
        else:
            base = max(0.1, self._rng.gauss(base_mean, base_std))

        speed_norm = clamp(ego_speed / max(speed_max, 1e-6), 0.0, 1.0)
        safe_attention = max(self.attention, self._epsilon)

        response = (
            base
            * (1.0 + complexity_alpha * road_complexity)
            * (1.0 + speed_beta * speed_norm)
            / max(alert_urgency, 0.1)
            / safe_attention
        )
        return max(response, 0.1)

    def reset(self):
        """Reset attention to initial value."""
        self.attention = self._initial

    @property
    def is_attentive(self) -> bool:
        """True if attention > 50 %%."""
        return self.attention > 0.5
