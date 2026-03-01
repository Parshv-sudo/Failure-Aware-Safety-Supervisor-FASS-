#!/usr/bin/env python
"""
FASS Risk Engine
==================
Fuses ML risk predictions with deterministic physical safety rules.

Architecture:
    ML Risk Score (0-1) ─┐
                         ├─→ Fused Risk Score → VotingLogic
    Deterministic Rules ─┘

Deterministic rules include:
    - TTC-based risk (time-to-collision threshold)
    - Distance-based risk (proximity threshold)
    - Sensor failure risk (degradation threshold)

CRITICAL SAFETY RULE:
    Deterministic thresholds ALWAYS override ML when they indicate
    higher risk.  The fused risk is max(ML_risk, deterministic_risk).
    This ensures the ML model can INCREASE sensitivity but NEVER
    decrease it below physics-based minimums.

ISO 26262 Note:
    This module implements the "diversity" principle — redundant
    risk assessment paths that are independent of each other.
    The deterministic path has no dependency on ML model weights.
"""

import time
from typing import Optional

from ..training.config import FASSConfig, DEFAULT_CONFIG


class RiskEngine:
    """Fuses ML predictions with deterministic safety rules.

    Safety Rationale:
        ML models can fail silently (adversarial inputs, distribution shift,
        sensor corruption).  This engine provides a physics-based safety floor
        that is completely independent of learned weights.  The fusion rule
        `max(ML_risk, deterministic_risk)` guarantees that the ML model can
        only INCREASE sensitivity, never decrease it below physics-based
        minimums.  This is the core "diversity" principle of ISO 26262.

    Parameters
    ----------
    config : FASSConfig
        Safety thresholds and configuration.
    """

    def __init__(self, config: FASSConfig = None):
        self.config = config or DEFAULT_CONFIG

        # State
        self._ml_risk = 0.0
        self._ml_uncertainty = 0.0
        self._deterministic_risk = 0.0
        self._fused_risk = 0.0
        self._override_active = False
        self._last_update_time = 0.0

        # History for trend detection
        self._risk_history = []
        self._max_history = 50

    def update(
        self,
        ml_risk: float,
        ml_uncertainty: float,
        min_distance: float = 999.0,
        min_ttc: float = 999.0,
        sensor_failures: int = 0,
        ego_speed: float = 0.0,
    ):
        """Update the risk engine with new ML prediction and physical measurements.

        Safety Rationale:
            This method implements the fusion logic that is critical for
            ASIL-D compliance.  The deterministic risk is computed from
            direct physical measurements (TTC, distance, sensor count)
            without any dependency on ML weights.  The fusion always takes
            the MAXIMUM of both paths to ensure the ML model cannot mask
            a genuine threat.

            If ML uncertainty exceeds 0.3 (high epistemic uncertainty),
            a conservative boost is applied to the fused risk.  This
            implements the principle: "when uncertain, err on the side
            of caution."

        Parameters
        ----------
        ml_risk : float
            ML-predicted risk score [0, 1].
        ml_uncertainty : float
            ML-predicted total uncertainty.
        min_distance : float
            Minimum distance to any detected object (meters).
        min_ttc : float
            Minimum time-to-collision (seconds).
        sensor_failures : int
            Number of failed sensors.
        ego_speed : float
            Ego vehicle speed (m/s).
        """
        self._ml_risk = ml_risk
        self._ml_uncertainty = ml_uncertainty
        self._last_update_time = time.time()

        # ── Deterministic risk assessment ──
        # These rules are INDEPENDENT of the ML model.

        # TTC risk
        ttc_risk = 0.0
        if min_ttc < self.config.ttc_emergency_s:
            ttc_risk = 1.0  # CRITICAL
        elif min_ttc < self.config.ttc_warning_s:
            ttc_risk = 0.5  # HIGH (reduced for dense traffic)
        elif min_ttc < 5.0:
            ttc_risk = 0.15  # MODERATE (reduced to avoid over-triggering in traffic)

        # Distance risk
        dist_risk = 0.0
        if min_distance < self.config.min_distance_stop_m:
            dist_risk = 1.0  # CRITICAL
        elif min_distance < self.config.min_distance_warn_m:
            dist_risk = 0.5  # HIGH (reduced for dense traffic)
        elif min_distance < 8.0:
            dist_risk = max(0.0, 1.0 - min_distance / 8.0) * 0.3

        # Sensor degradation risk
        sensor_risk = 0.0
        if sensor_failures >= self.config.sensor_failure_threshold:
            sensor_risk = 0.9  # CRITICAL — safe stop
        elif sensor_failures >= 1:
            sensor_risk = 0.3  # MODERATE — reduced confidence

        # Speed-adjusted risk
        speed_factor = min(1.0, ego_speed / 30.0)  # Higher speed = higher risk
        speed_adjusted_dist_risk = dist_risk * (0.5 + 0.5 * speed_factor)

        # Combined deterministic risk
        self._deterministic_risk = max(ttc_risk, speed_adjusted_dist_risk, sensor_risk)

        # ── Fusion: ML + Deterministic ──
        # SAFETY RULE: Take the MAXIMUM.  Deterministic overrides ML.
        self._fused_risk = max(ml_risk, self._deterministic_risk)
        self._override_active = (self._deterministic_risk > ml_risk)

        # If uncertainty is very high, boost risk conservatively
        if ml_uncertainty > 0.3:
            uncertainty_boost = min(0.2, ml_uncertainty * 0.3)
            self._fused_risk = min(1.0, self._fused_risk + uncertainty_boost)

        # Track history
        self._risk_history.append(self._fused_risk)
        if len(self._risk_history) > self._max_history:
            self._risk_history.pop(0)

    @property
    def fused_risk(self) -> float:
        return self._fused_risk

    @property
    def is_overriding(self) -> bool:
        """True if deterministic rules are currently overriding ML.

        Safety Rationale:
            When this is True, it means the physics-based assessment found
            higher risk than the ML model.  This is a safety-critical
            indicator that should be logged and monitored — persistent
            overriding may indicate ML model degradation.
        """
        return self._override_active

    @property
    def risk_trend(self) -> str:
        """Detect if risk is rising, falling, or stable."""
        if len(self._risk_history) < 5:
            return 'UNKNOWN'
        recent = self._risk_history[-5:]
        delta = recent[-1] - recent[0]
        if delta > 0.1:
            return 'RISING'
        elif delta < -0.1:
            return 'FALLING'
        return 'STABLE'

    def get_state(self) -> dict:
        """Return full risk engine state for logging.

        Safety Rationale:
            Every field in this state dict is included in safety logs
            for post-incident reconstruction.  The decomposition of risk
            into ml_risk vs deterministic_risk enables auditors to determine
            whether the ML model or physics rules drove each decision.
        """
        return {
            'ml_risk': round(self._ml_risk, 5),
            'ml_uncertainty': round(self._ml_uncertainty, 5),
            'deterministic_risk': round(self._deterministic_risk, 5),
            'fused_risk': round(self._fused_risk, 5),
            'override_active': self._override_active,
            'risk_trend': self.risk_trend,
            'timestamp': self._last_update_time,
        }
