#!/usr/bin/env python
"""
Deterministic Safety Overrides for FASS
=========================================
Hard-coded safety thresholds that CANNOT be overridden by ML predictions.

These represent physics-based and engineering limits that are independent
of any learned model.  They form the "safety floor" of the FASS system.

Override Rules:
    1. TTC < 0.5s      → EMERGENCY BRAKE (imminent collision)
    2. Distance < 1.5m  → EMERGENCY STOP (object extremely close)
    3. Sensor failures ≥ 2 → SAFE STOP (insufficient perception)
    4. Ego speed > 200 km/h → SPEED LIMIT (physical implausibility check)

CRITICAL SAFETY PRINCIPLE:
    These overrides exist because ML models can fail silently.  A neural
    network might output risk=0.1 while an object is <2m away.  The
    deterministic path catches this regardless of ML output.

ISO 26262 Note:
    These thresholds are derived from vehicle dynamics analysis and
    ASIL-D requirements.  They must be validated through physical testing
    and must NOT be modified without a formal safety review.
"""

from ..training.config import FASSConfig, DEFAULT_CONFIG


class DeterministicOverrides:
    """Physics-based safety overrides independent of ML.

    Parameters
    ----------
    config : FASSConfig
        Safety thresholds.
    """

    def __init__(self, config: FASSConfig = None):
        self.config = config or DEFAULT_CONFIG
        self._override_count = 0

    def check(
        self,
        min_distance: float = 999.0,
        min_ttc: float = 999.0,
        sensor_failures: int = 0,
        ego_speed: float = 0.0,
    ) -> dict:
        """Check all deterministic safety conditions.

        Parameters
        ----------
        min_distance : float
            Closest object distance (meters).
        min_ttc : float
            Minimum time-to-collision (seconds).
        sensor_failures : int
            Number of failed sensors.
        ego_speed : float
            Ego vehicle speed (m/s).

        Returns
        -------
        dict with:
            override : bool — True if any condition triggers
            reason   : str  — human-readable explanation
            severity : str  — 'CRITICAL', 'HIGH', 'MODERATE'
            actions  : list — recommended actions
        """
        reasons = []
        actions = []
        severity = 'NONE'

        # Rule 1: TTC emergency
        if min_ttc < self.config.ttc_emergency_s:
            reasons.append(f'TTC={min_ttc:.2f}s < {self.config.ttc_emergency_s}s')
            actions.append('EMERGENCY_BRAKE')
            severity = 'CRITICAL'

        # Rule 2: Distance emergency
        if min_distance < self.config.min_distance_stop_m:
            reasons.append(f'Distance={min_distance:.2f}m < {self.config.min_distance_stop_m}m')
            actions.append('EMERGENCY_STOP')
            severity = 'CRITICAL'

        # Rule 3: Sensor failure
        if sensor_failures >= self.config.sensor_failure_threshold:
            reasons.append(f'Sensor failures={sensor_failures} >= {self.config.sensor_failure_threshold}')
            actions.append('SAFE_STOP')
            if severity != 'CRITICAL':
                severity = 'HIGH'

        # Rule 4: Speed implausibility (> 200 km/h ≈ 55.6 m/s)
        max_plausible_speed = 55.6  # m/s
        if ego_speed > max_plausible_speed:
            reasons.append(f'Speed={ego_speed:.1f}m/s > {max_plausible_speed}m/s (implausible)')
            actions.append('SPEED_LIMIT')
            if severity == 'NONE':
                severity = 'MODERATE'

        # Rule 5: TTC warning (non-emergency)
        if min_ttc < self.config.ttc_warning_s and 'EMERGENCY_BRAKE' not in actions:
            reasons.append(f'TTC={min_ttc:.2f}s < {self.config.ttc_warning_s}s (warning)')
            actions.append('GENTLE_BRAKE')
            if severity == 'NONE':
                severity = 'MODERATE'

        # Rule 6: Distance warning
        if (min_distance < self.config.min_distance_warn_m and
                'EMERGENCY_STOP' not in actions):
            reasons.append(f'Distance={min_distance:.2f}m < {self.config.min_distance_warn_m}m (warning)')
            actions.append('SPEED_LIMIT')
            if severity == 'NONE':
                severity = 'MODERATE'

        override = len(reasons) > 0
        if override:
            self._override_count += 1

        return {
            'override': override,
            'reason': '; '.join(reasons) if reasons else '',
            'severity': severity,
            'actions': actions,
            'override_count': self._override_count,
        }

    def self_test(self) -> bool:
        """Built-in verification of override logic.

        Runs a suite of known inputs and asserts expected outputs.
        Returns True if all tests pass.

        ISO 26262: This constitutes a unit-level safety test
        that should be run at system startup.
        """
        tests_passed = 0
        tests_total = 0

        # Test 1: TTC emergency
        tests_total += 1
        r = self.check(min_ttc=0.3)
        assert r['override'] is True, "TTC emergency should trigger"
        assert 'EMERGENCY_BRAKE' in r['actions'], "Should recommend EMERGENCY_BRAKE"
        tests_passed += 1

        # Test 2: Distance emergency
        tests_total += 1
        r = self.check(min_distance=1.0)
        assert r['override'] is True, "Distance emergency should trigger"
        assert 'EMERGENCY_STOP' in r['actions'], "Should recommend EMERGENCY_STOP"
        tests_passed += 1

        # Test 3: Sensor failure
        tests_total += 1
        r = self.check(sensor_failures=3)
        assert r['override'] is True, "Sensor failure should trigger"
        assert 'SAFE_STOP' in r['actions'], "Should recommend SAFE_STOP"
        tests_passed += 1

        # Test 4: Normal conditions — NO override
        tests_total += 1
        r = self.check(min_distance=50.0, min_ttc=10.0, sensor_failures=0, ego_speed=15.0)
        assert r['override'] is False, "Normal conditions should NOT trigger"
        tests_passed += 1

        # Test 5: Speed implausibility
        tests_total += 1
        r = self.check(ego_speed=70.0)  # > 55.6 m/s
        assert r['override'] is True, "Speed implausibility should trigger"
        tests_passed += 1

        # Test 6: Combined critical
        tests_total += 1
        r = self.check(min_ttc=0.3, min_distance=1.0, sensor_failures=2)
        assert r['override'] is True, "Combined critical should trigger"
        assert r['severity'] == 'CRITICAL', "Severity should be CRITICAL"
        tests_passed += 1

        print(f"[DeterministicOverrides] Self-test: {tests_passed}/{tests_total} passed")
        return tests_passed == tests_total
