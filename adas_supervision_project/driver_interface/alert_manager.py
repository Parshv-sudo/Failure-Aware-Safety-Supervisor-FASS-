"""
Alert manager for the ADAS Supervision Framework.

Maps risk levels to alert tiers and outputs human-readable
console alerts during simulation.  Designed for future extension
to audio/visual HMI channels.
"""

import logging
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Severity tiers for driver-facing alerts."""

    NONE = auto()
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


# ANSI colour codes for console output (no dependency required)
_COLOURS = {
    AlertLevel.NONE: "\033[0m",       # reset
    AlertLevel.INFO: "\033[94m",      # blue
    AlertLevel.WARNING: "\033[93m",   # yellow
    AlertLevel.CRITICAL: "\033[91m",  # red
    AlertLevel.EMERGENCY: "\033[41m", # red bg
}
_RESET = "\033[0m"


class AlertManager:
    """Manages driver-facing alerts.

    Provides a thin abstraction over alert level determination and
    output so that the same alerts can later be routed to a real
    HMI, audio system, or GUI overlay.

    Args:
        thresholds: Dict with keys ``stage_1_warning`` through
            ``stage_4_emergency_braking`` (float risk thresholds).
    """

    def __init__(self, thresholds: dict = None):
        t = thresholds or {}
        self._t1 = t.get("stage_1_warning", 0.4)
        self._t2 = t.get("stage_2_speed_reduction", 0.6)
        self._t3 = t.get("stage_3_takeover_request", 0.75)
        self._t4 = t.get("stage_4_emergency_braking", 0.9)
        self._hysteresis = 0.05  # De-escalation requires risk to drop this much below threshold
        self._current_level = AlertLevel.NONE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, risk: float) -> AlertLevel:
        """Determine alert level from current risk score.

        Uses hysteresis to prevent rapid oscillation when risk hovers
        near a threshold boundary.  Once at a level, risk must drop
        ``_hysteresis`` (default 0.05) below the threshold to de-escalate.

        Args:
            risk: Normalised risk ∈ [0, 1].

        Returns:
            ``AlertLevel`` corresponding to the risk score.
        """
        hyst = self._hysteresis

        # Determine the raw level from thresholds
        if risk >= self._t4:
            raw_level = AlertLevel.EMERGENCY
        elif risk >= self._t3:
            raw_level = AlertLevel.CRITICAL
        elif risk >= self._t2:
            raw_level = AlertLevel.WARNING
        elif risk >= self._t1:
            raw_level = AlertLevel.INFO
        else:
            raw_level = AlertLevel.NONE

        # Apply hysteresis: only de-escalate if risk drops sufficiently
        if raw_level.value < self._current_level.value:
            # Check if risk is low enough to actually de-escalate
            # (must be below the current level's threshold minus hysteresis)
            current_threshold = self._level_threshold(self._current_level)
            if risk > current_threshold - hyst:
                # Stay at current level — risk hasn't dropped enough
                return self._current_level

        level = raw_level
        if level != self._current_level:
            self._on_level_change(self._current_level, level, risk)
            self._current_level = level

        return level

    def _level_threshold(self, level: AlertLevel) -> float:
        """Return the escalation threshold for a given alert level."""
        return {
            AlertLevel.INFO: self._t1,
            AlertLevel.WARNING: self._t2,
            AlertLevel.CRITICAL: self._t3,
            AlertLevel.EMERGENCY: self._t4,
        }.get(level, 0.0)

    @property
    def current_level(self) -> AlertLevel:
        """Most recently evaluated alert level."""
        return self._current_level

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_level_change(
        self,
        prev: AlertLevel,
        new: AlertLevel,
        risk: float,
    ):
        """Emit a console alert when the level changes."""
        colour = _COLOURS.get(new, "")
        logger.warning(
            "%s⚠ ALERT %s → %s  (risk=%.3f)%s",
            colour,
            prev.name,
            new.name,
            risk,
            _RESET,
        )
