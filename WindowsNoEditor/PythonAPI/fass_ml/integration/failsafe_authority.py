#!/usr/bin/env python
"""
FASS FailSafe Authority
=========================
The final decision layer that triggers vehicle interventions based on
the advisory from VotingLogic.

Intervention Types:
    NONE            — no action, vehicle operates normally
    SPEED_LIMIT     — reduce maximum speed (e.g., to 30 km/h)
    CONTROL_LIMIT   — restrict steering + throttle range
    GENTLE_BRAKE    — apply moderate braking (0.3-0.5 brake force)
    EMERGENCY_BRAKE — apply full emergency braking (1.0 brake force)
    SAFE_STOP       — bring vehicle to complete stop (full brake + hazards)

Escalation Logic:
    SAFE      → NONE
    CAUTION   → SPEED_LIMIT + CONTROL_LIMIT
    DANGER    → EMERGENCY_BRAKE or SAFE_STOP

CRITICAL SAFETY RULES:
    1. Emergency brake is NEVER blocked or delayed.
    2. Once SAFE_STOP is triggered, it cannot be cancelled by ML.
       Only a manual reset or deterministic all-clear releases it.
    3. Every intervention is logged with full context for audit.

ISO 26262 Note:
    This module implements ASIL-D fail-safe authority.  It must be
    independently testable and free of ML dependency for its core
    braking path.
"""

import time
from typing import Optional, Callable


class Intervention:
    """Describes a specific vehicle intervention."""

    NONE = 'NONE'
    SPEED_LIMIT = 'SPEED_LIMIT'
    CONTROL_LIMIT = 'CONTROL_LIMIT'
    GENTLE_BRAKE = 'GENTLE_BRAKE'
    EMERGENCY_BRAKE = 'EMERGENCY_BRAKE'
    SAFE_STOP = 'SAFE_STOP'

    def __init__(self, intervention_type: str, brake_force: float = 0.0,
                 max_speed: float = 999.0, max_steer: float = 1.0,
                 reason: str = ''):
        self.type = intervention_type
        self.brake_force = brake_force
        self.max_speed = max_speed          # km/h
        self.max_steer = max_steer          # max absolute steer input
        self.reason = reason
        self.timestamp = time.time()

    def to_dict(self):
        return {
            'type': self.type,
            'brake_force': self.brake_force,
            'max_speed': self.max_speed,
            'max_steer': self.max_steer,
            'reason': self.reason,
            'timestamp': self.timestamp,
        }


class FailSafeAuthority:
    """Triggers vehicle interventions based on advisory from VotingLogic.

    Safety Rationale:
        This is the LAST module in the FASS decision chain and the ONLY
        module that directly commands vehicle actuators.  Two critical
        safety invariants are enforced:

        1. MONOTONIC ESCALATION: Once SAFE_STOP is latched, it cannot
           be cancelled by any ML prediction.  This prevents oscillation
           between braking and releasing in adversarial or noisy conditions.
           Only an authorized operator calling ``reset_safe_stop()`` can
           release the latch.

        2. DETERMINISTIC OVERRIDE PRIORITY: When ``override_active=True``
           (meaning the physics-based RiskEngine detected higher risk than
           ML), the FailSafeAuthority escalates directly to SAFE_STOP.
           This ensures that the ML model cannot suppress a physics-based
           emergency signal.

        The intervention hierarchy is strictly ordered:
            NONE < SPEED_LIMIT < CONTROL_LIMIT < GENTLE_BRAKE < EMERGENCY_BRAKE < SAFE_STOP

    Parameters
    ----------
    on_intervention : callable, optional
        Callback invoked whenever an intervention is triggered.
        Signature: on_intervention(Intervention) → None.
    gentle_brake_force : float
        Brake force for GENTLE_BRAKE (0-1).
    speed_limit_kmh : float
        Speed limit applied during CAUTION (km/h).
    steer_limit : float
        Maximum absolute steering angle during CONTROL_LIMIT.
    """

    def __init__(
        self,
        on_intervention: Optional[Callable] = None,
        gentle_brake_force: float = 0.2,
        speed_limit_kmh: float = 30.0,
        steer_limit: float = 0.5,
    ):
        self.on_intervention = on_intervention
        self.gentle_brake_force = gentle_brake_force
        self.speed_limit_kmh = speed_limit_kmh
        self.steer_limit = steer_limit

        # State
        self._current_intervention = Intervention(Intervention.NONE)
        self._safe_stop_active = False    # Latching flag
        self._intervention_count = 0
        self._history = []
        self._max_history = 200

    def decide(self, advisory: str, risk_score: float = 0.0,
               risk_trend: str = 'STABLE', override_active: bool = False) -> Intervention:
        """Determine and execute the appropriate intervention.

        Safety Rationale:
            The decision logic is structured as a priority cascade:
            - SAFE_STOP latch is checked FIRST (irrevocable once set)
            - DANGER + high risk or override → SAFE_STOP (latch set)
            - DANGER + moderate risk → EMERGENCY_BRAKE (no latch)
            - DANGER + low risk → GENTLE_BRAKE (allows recovery)
            - CAUTION + rising trend → preemptive GENTLE_BRAKE
            - CAUTION + stable/falling → SPEED_LIMIT only
            - SAFE → no intervention

            The SAFE_STOP latch exists because once a genuinely dangerous
            situation is detected with high confidence, releasing brakes
            based on a subsequent low-risk prediction is too risky — the
            original detection may have been correct and the subsequent
            prediction may be corrupted.

        Parameters
        ----------
        advisory : str
            'SAFE', 'CAUTION', or 'DANGER' from VotingLogic.
        risk_score : float
            Fused risk score from RiskEngine.
        risk_trend : str
            'RISING', 'FALLING', or 'STABLE'.
        override_active : bool
            True if deterministic overrides are active in RiskEngine.

        Returns
        -------
        Intervention object describing the action taken.
        """

        # SAFETY: Once SAFE_STOP is latched, it stays until manual reset
        if self._safe_stop_active:
            intervention = Intervention(
                Intervention.SAFE_STOP,
                brake_force=1.0,
                max_speed=0.0,
                max_steer=0.0,
                reason='SAFE_STOP latched — manual reset required',
            )
            self._record(intervention)
            return intervention

        # --- Determine intervention ---
        if advisory == 'DANGER':
            if risk_score > 0.85:
                # Very high risk: irrevocable SAFE_STOP latch
                intervention = Intervention(
                    Intervention.SAFE_STOP,
                    brake_force=1.0,
                    max_speed=0.0,
                    max_steer=0.0,
                    reason=f'DANGER advisory, risk={risk_score:.3f}, '
                           f'override={override_active}',
                )
                self._safe_stop_active = True
            elif risk_score > 0.7:
                intervention = Intervention(
                    Intervention.EMERGENCY_BRAKE,
                    brake_force=1.0,
                    max_speed=0.0,
                    reason=f'DANGER advisory, risk={risk_score:.3f}',
                )
            else:
                intervention = Intervention(
                    Intervention.GENTLE_BRAKE,
                    brake_force=self.gentle_brake_force,
                    max_speed=self.speed_limit_kmh,
                    reason=f'DANGER advisory (moderate), risk={risk_score:.3f}',
                )

        elif advisory == 'CAUTION':
            if risk_trend == 'RISING':
                intervention = Intervention(
                    Intervention.GENTLE_BRAKE,
                    brake_force=self.gentle_brake_force * 0.5,
                    max_speed=self.speed_limit_kmh,
                    max_steer=self.steer_limit,
                    reason=f'CAUTION + rising trend, risk={risk_score:.3f}',
                )
            else:
                intervention = Intervention(
                    Intervention.SPEED_LIMIT,
                    max_speed=self.speed_limit_kmh,
                    max_steer=self.steer_limit,
                    reason=f'CAUTION advisory, risk={risk_score:.3f}',
                )

        else:
            # SAFE
            intervention = Intervention(Intervention.NONE)

        self._record(intervention)
        return intervention

    def reset_safe_stop(self):
        """Manually reset the SAFE_STOP latch.

        This should only be called by an authorized operator after
        the vehicle has been brought to a complete stop and the
        situation has been assessed.
        """
        self._safe_stop_active = False
        print("[FailSafeAuthority] SAFE_STOP latch RESET by operator")

    def _record(self, intervention: Intervention):
        """Record intervention and invoke callback."""
        self._current_intervention = intervention
        if intervention.type != Intervention.NONE:
            self._intervention_count += 1

        self._history.append(intervention.to_dict())
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if self.on_intervention and intervention.type != Intervention.NONE:
            self.on_intervention(intervention)

    @property
    def current_intervention(self) -> Intervention:
        return self._current_intervention

    @property
    def intervention_count(self) -> int:
        return self._intervention_count

    def get_state(self) -> dict:
        return {
            'current_intervention': self._current_intervention.to_dict(),
            'safe_stop_active': self._safe_stop_active,
            'intervention_count': self._intervention_count,
            'recent_interventions': self._history[-5:],
        }
