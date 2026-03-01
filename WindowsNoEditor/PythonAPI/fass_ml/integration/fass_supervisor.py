#!/usr/bin/env python
"""
FASS Supervisor — Top-Level Orchestrator
==========================================
Ties together all FASS modules in a single tick() call:

    Sensors → FeatureExtractor → InferenceEngine → RiskEngine → VotingLogic → FailSafeAuthority

Usage in a CARLA tick loop:
    supervisor = FASSSupervisor(checkpoint_path='./fass_checkpoints/best_model.pt')

    while True:
        world.tick()
        result = supervisor.tick(sensor_data, ego_state, weather, sensor_health)
        # result contains: risk, uncertainty, advisory, intervention, full_state

ISO 26262 Note:
    This orchestrator enforces a deterministic execution order.
    No module can skip or bypass the safety chain.
"""

import time
from typing import Optional

from .inference import FASSInferenceEngine
from .risk_engine import RiskEngine
from .voting_logic import VotingLogic, EvaluatorVote
from .failsafe_authority import FailSafeAuthority
from ..safety.safety_logger import SafetyLogger
from ..safety.deterministic_overrides import DeterministicOverrides
from ..training.config import FASSConfig, DEFAULT_CONFIG


class FASSSupervisor:
    """Top-level FASS orchestrator.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to trained model checkpoint.
    config : FASSConfig, optional
        Configuration.
    log_dir : str, optional
        Directory for safety logs.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        config: FASSConfig = None,
        log_dir: str = None,
    ):
        self.config = config or DEFAULT_CONFIG

        # Initialize all modules
        self.inference = FASSInferenceEngine(
            checkpoint_path=checkpoint_path,
            config=self.config,
        )
        self.risk_engine = RiskEngine(config=self.config)
        self.voting_logic = VotingLogic()
        self.failsafe = FailSafeAuthority(
            on_intervention=self._on_intervention,
        )
        self.overrides = DeterministicOverrides(config=self.config)
        self.logger = SafetyLogger(
            log_dir=log_dir or self.config.log_dir,
        )

        self._tick_count = 0
        self._total_latency_ms = 0.0
        print(f"[FASS Supervisor] Initialized with {self.inference.model.summary()}")

    def tick(
        self,
        sensor_data: dict,
        ego_state: dict,
        weather: dict = None,
        sensor_health: dict = None,
        scenario_id: str = None,
    ) -> dict:
        """Execute one full FASS supervision cycle.

        Parameters
        ----------
        sensor_data : dict
            Must contain 'detected_objects' list.
        ego_state : dict
            Ego kinematics (speed, acceleration, steer, etc.).
        weather : dict, optional
        sensor_health : dict, optional
        scenario_id : str, optional
            Scenario tag for log traceability.

        Returns
        -------
        dict with full supervision result.
        """
        t0 = time.perf_counter()
        self._tick_count += 1

        # ── Step 1: Check deterministic overrides FIRST ──
        # These take priority over everything else.
        detected_objects = sensor_data.get('detected_objects', [])
        min_dist = min((o['distance'] for o in detected_objects), default=999.0)
        ego_speed = ego_state.get('speed', 0.0)

        # Compute min TTC
        min_ttc = 999.0
        for obj in detected_objects:
            relative_speed = ego_speed - obj.get('speed', 0.0) * 0.5
            if relative_speed > 0.1:
                ttc = obj['distance'] / relative_speed
                min_ttc = min(min_ttc, ttc)

        sensor_failures = 0
        if sensor_health:
            sensor_failures = sum(1 for v in sensor_health.values() if not v)

        override_result = self.overrides.check(
            min_distance=min_dist,
            min_ttc=min_ttc,
            sensor_failures=sensor_failures,
            ego_speed=ego_speed,
        )

        # ── Step 2: ML Inference ──
        risk, uncertainty, ml_advisory = self.inference.predict(
            sensor_data, ego_state, weather, sensor_health
        )

        # ── Step 3: Risk Engine Fusion ──
        self.risk_engine.update(
            ml_risk=risk,
            ml_uncertainty=uncertainty,
            min_distance=min_dist,
            min_ttc=min_ttc,
            sensor_failures=sensor_failures,
            ego_speed=ego_speed,
        )

        # ── Step 4: Multi-Evaluator Voting ──
        self.voting_logic.clear_votes()

        # ML evaluator vote
        ml_confidence = max(0.0, 1.0 - uncertainty)
        self.voting_logic.submit_vote(EvaluatorVote(
            name='ML_RISK',
            advisory=ml_advisory,
            confidence=ml_confidence,
            risk_score=risk,
        ))

        # TTC physics evaluator
        if min_ttc < self.config.ttc_emergency_s:
            ttc_advisory = 'DANGER'
            ttc_conf = 0.95
        elif min_ttc < self.config.ttc_warning_s:
            ttc_advisory = 'CAUTION'
            ttc_conf = 0.8
        else:
            ttc_advisory = 'SAFE'
            ttc_conf = 0.5
        self.voting_logic.submit_vote(EvaluatorVote(
            name='TTC_PHYSICS',
            advisory=ttc_advisory,
            confidence=ttc_conf,
            risk_score=max(0, 1.0 - min_ttc / 3.0),
        ))

        # Distance evaluator
        if min_dist < self.config.min_distance_stop_m:
            dist_advisory = 'DANGER'
            dist_conf = 0.95
        elif min_dist < self.config.min_distance_warn_m:
            dist_advisory = 'CAUTION'
            dist_conf = 0.8
        else:
            dist_advisory = 'SAFE'
            dist_conf = 0.5
        self.voting_logic.submit_vote(EvaluatorVote(
            name='DISTANCE',
            advisory=dist_advisory,
            confidence=dist_conf,
            risk_score=max(0, 1.0 - min_dist / 15.0),
        ))

        # Execute vote
        final_advisory = self.voting_logic.vote()

        # Override with deterministic if needed
        if override_result['override']:
            final_advisory = 'DANGER'

        # ── Step 5: FailSafe Authority ──
        intervention = self.failsafe.decide(
            advisory=final_advisory,
            risk_score=self.risk_engine.fused_risk,
            risk_trend=self.risk_engine.risk_trend,
            override_active=self.risk_engine.is_overriding,
        )

        # ── Step 6: Logging ──
        tick_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += tick_latency_ms

        detailed = self.inference.predict_detailed(
            sensor_data, ego_state, weather, sensor_health
        )

        log_entry = {
            'tick': self._tick_count,
            'scenario_id': scenario_id,
            'ml_risk': risk,
            'ml_uncertainty': uncertainty,
            'ml_advisory': ml_advisory,
            'epistemic_unc': detailed.get('epistemic_unc', 0),
            'aleatoric_unc': detailed.get('aleatoric_unc', 0),
            'fused_risk': self.risk_engine.fused_risk,
            'deterministic_override': override_result['override'],
            'override_reason': override_result.get('reason', ''),
            'final_advisory': final_advisory,
            'intervention': intervention.to_dict(),
            'risk_engine_state': self.risk_engine.get_state(),
            'voting_state': self.voting_logic.get_state(),
            'min_distance': min_dist,
            'min_ttc': min_ttc,
            'ego_speed': ego_speed,
            'sensor_failures': sensor_failures,
            'tick_latency_ms': round(tick_latency_ms, 2),
        }

        # Log to safety logger
        if final_advisory == 'DANGER' or intervention.type != 'NONE':
            self.logger.log_critical_event(log_entry)
        else:
            self.logger.log_prediction(log_entry)

        return {
            'risk': risk,
            'uncertainty': uncertainty,
            'advisory': final_advisory,
            'intervention': intervention,
            'fused_risk': self.risk_engine.fused_risk,
            'override_active': override_result['override'],
            'tick_latency_ms': tick_latency_ms,
        }

    def _on_intervention(self, intervention):
        """Callback when an intervention is triggered."""
        print(f"[FASS] ⚠ INTERVENTION: {intervention.type} | "
              f"brake={intervention.brake_force:.1f} | "
              f"reason={intervention.reason}")

    def get_stats(self) -> dict:
        """Get supervisor statistics."""
        avg_latency = (self._total_latency_ms / self._tick_count
                       if self._tick_count > 0 else 0)
        return {
            'total_ticks': self._tick_count,
            'avg_tick_latency_ms': round(avg_latency, 2),
            'total_interventions': self.failsafe.intervention_count,
            'safe_stop_active': self.failsafe._safe_stop_active,
        }
