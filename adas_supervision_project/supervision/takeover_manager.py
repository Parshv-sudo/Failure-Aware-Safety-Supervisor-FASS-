"""
Takeover manager for the ADAS Supervision Framework.

Implements the four-stage takeover escalation driven by the
``SimulationStateMachine``:

    Stage 1 → Warning (visual alert)
    Stage 2 → Gradual speed reduction
    Stage 3 → Takeover request
    Stage 4 → Emergency braking fallback

Driver response time is sampled from configurable distributions
(Gaussian / lognormal), coupled to hazard context (alert level,
road complexity, speed).
"""

import logging
import math
import random
from enum import IntEnum
from typing import Optional

from core.simulation_state import SimulationState, SimulationStateMachine
from driver_interface.alert_manager import AlertLevel, AlertManager
from driver_interface.control_transition import ControlTransition
from utils.math_utils import clamp

logger = logging.getLogger(__name__)


class TakeoverStage(IntEnum):
    """Takeover escalation stages."""
    NONE = 0
    WARNING = 1
    SPEED_REDUCTION = 2
    TAKEOVER_REQUEST = 3
    EMERGENCY_BRAKING = 4


class TakeoverManager:
    """Manages the four-stage takeover escalation.

    Args:
        state_machine: The simulation state machine.
        alert_manager: Alert manager for driver-facing alerts.
        control_transition: Control transition handler.
        thresholds: Dict with ``stage_1_warning`` through
            ``stage_4_emergency_braking`` keys (float risk thresholds).
        response_config: Driver response time configuration.
        hazard_config: Hazard coupling coefficients.
        random_seed: RNG seed for response-time sampling.
    """

    def __init__(
        self,
        state_machine: SimulationStateMachine,
        alert_manager: AlertManager,
        control_transition: ControlTransition,
        thresholds: dict = None,
        response_config: dict = None,
        hazard_config: dict = None,
        random_seed: int = 42,
    ):
        self.sm = state_machine
        self.alert_mgr = alert_manager
        self.ctrl_trans = control_transition

        t = thresholds or {}
        self._t1 = t.get("stage_1_warning", 0.4)
        self._t2 = t.get("stage_2_speed_reduction", 0.6)
        self._t3 = t.get("stage_3_takeover_request", 0.75)
        self._t4 = t.get("stage_4_emergency_braking", 0.9)

        self._rng = random.Random(random_seed)
        self._response_cfg = response_config or {}
        self._hazard_cfg = hazard_config or {}

        self._stage = TakeoverStage.NONE
        self._takeover_start_tick: Optional[int] = None
        self._simulated_response_time: Optional[float] = None
        self._response_elapsed: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stage(self) -> TakeoverStage:
        """Current takeover stage."""
        return self._stage

    @property
    def response_time(self) -> Optional[float]:
        """Last sampled driver response time (seconds)."""
        return self._simulated_response_time

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def update(
        self,
        risk: float,
        ego_speed: float,
        road_complexity: float,
        vehicle,          # carla.Vehicle
        delta_t: float,
        tick: int,
    ) -> TakeoverStage:
        """Evaluate risk and progress takeover staging.

        Must be called once per tick while the simulation is in
        ``RUNNING`` or ``TAKEOVER_ACTIVE`` state.

        Args:
            risk: Normalised risk score ∈ [0, 1].
            ego_speed: m/s.
            road_complexity: ∈ [0, 1].
            vehicle: Ego ``carla.Vehicle``.
            delta_t: Fixed time-step.
            tick: Current tick count.

        Returns:
            Active ``TakeoverStage`` after evaluation.
        """
        alert_level = self.alert_mgr.evaluate(risk)

        # ---- Escalation logic ------------------------------------------
        new_stage = self._risk_to_stage(risk)

        if new_stage > self._stage:
            self._escalate(new_stage, risk, ego_speed, road_complexity, tick)
        elif new_stage < self._stage and new_stage == TakeoverStage.NONE:
            self._de_escalate()

        # ---- Stage actions ---------------------------------------------
        if self._stage == TakeoverStage.SPEED_REDUCTION:
            self.ctrl_trans.gradual_slow_down(vehicle, delta_t)

        elif self._stage == TakeoverStage.TAKEOVER_REQUEST:
            self.ctrl_trans.gradual_slow_down(vehicle, delta_t)
            self._response_elapsed += delta_t
            if (
                self._simulated_response_time is not None
                and self._response_elapsed >= self._simulated_response_time
            ):
                logger.info(
                    "Driver responded after %.2f s (simulated).",
                    self._response_elapsed,
                )
                self._de_escalate()

        elif self._stage == TakeoverStage.EMERGENCY_BRAKING:
            self.ctrl_trans.emergency_brake(vehicle)
            if ego_speed < 0.3:
                logger.info("Vehicle stopped — minimum risk condition.")
                self.sm.transition(SimulationState.TERMINATED)

        return self._stage

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _risk_to_stage(self, risk: float) -> TakeoverStage:
        """Map risk score to a takeover stage."""
        if risk >= self._t4:
            return TakeoverStage.EMERGENCY_BRAKING
        if risk >= self._t3:
            return TakeoverStage.TAKEOVER_REQUEST
        if risk >= self._t2:
            return TakeoverStage.SPEED_REDUCTION
        if risk >= self._t1:
            return TakeoverStage.WARNING
        return TakeoverStage.NONE

    def _escalate(
        self,
        new_stage: TakeoverStage,
        risk: float,
        ego_speed: float,
        road_complexity: float,
        tick: int,
    ):
        """Move to a higher takeover stage."""
        prev = self._stage
        self._stage = new_stage
        self._takeover_start_tick = tick
        self._response_elapsed = 0.0

        # Sample driver response time (hazard-coupled)
        self._simulated_response_time = self._sample_response(
            new_stage, ego_speed, road_complexity
        )

        logger.warning(
            "⚡ Takeover escalation: Stage %d → %d  (risk=%.3f, "
            "response_time=%.2f s)",
            prev, new_stage, risk,
            self._simulated_response_time or 0.0,
        )

        # State machine transitions
        if new_stage >= TakeoverStage.WARNING:
            if self.sm.state == SimulationState.RUNNING:
                self.sm.transition(SimulationState.TAKEOVER_ACTIVE)
        if new_stage == TakeoverStage.EMERGENCY_BRAKING:
            self.sm.transition(SimulationState.FALLBACK_BRAKING)

    def _de_escalate(self):
        """Return to normal running."""
        logger.info("Takeover de-escalated → NONE")
        self._stage = TakeoverStage.NONE
        self._simulated_response_time = None
        self._response_elapsed = 0.0
        if self.sm.state in (
            SimulationState.TAKEOVER_ACTIVE,
            SimulationState.FALLBACK_BRAKING,
        ):
            self.sm.transition(SimulationState.RUNNING)

    def _sample_response(
        self,
        stage: TakeoverStage,
        ego_speed: float,
        road_complexity: float,
    ) -> float:
        """Sample driver response time from configured distribution.

        ``response = base × (1 + α·complexity) × (1 + β·speed_norm)
                     / alert_urgency``

        Args:
            stage: Active takeover stage (determines base distribution).
            ego_speed: m/s.
            road_complexity: [0, 1].

        Returns:
            Response time in seconds (≥ 0.1).
        """
        dist_type = self._response_cfg.get("distribution", "lognormal")
        stage_key = f"stage_{int(stage)}"
        params = self._response_cfg.get("stage_params", {}).get(
            stage_key, {"mean": 2.0, "std": 0.5}
        )
        mean = params.get("mean", 2.0)
        std = params.get("std", 0.5)

        # Base sample
        if dist_type == "lognormal":
            # lognormal parameterisation
            sigma2 = math.log(1 + (std / mean) ** 2)
            mu = math.log(mean) - sigma2 / 2
            base = self._rng.lognormvariate(mu, math.sqrt(sigma2))
        else:
            base = max(0.1, self._rng.gauss(mean, std))

        # Hazard coupling
        hc = self._hazard_cfg
        alpha = hc.get("complexity_alpha", 0.3)
        beta = hc.get("speed_beta", 0.2)
        urgency_map = hc.get("alert_urgency_factors", {})
        urgency = urgency_map.get(stage_key, 1.0)

        speed_norm = clamp(ego_speed / 40.0, 0.0, 1.0)

        response = (
            base
            * (1.0 + alpha * road_complexity)
            * (1.0 + beta * speed_norm)
            / max(urgency, 0.1)
        )
        return max(response, 0.1)
