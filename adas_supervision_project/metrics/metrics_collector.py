"""
Episode-level metrics collector for the ADAS Supervision Framework.

Accumulates per-tick data and computes KPIs on ``finalize()``:

    - False takeover count
    - Missed hazard count (N-consecutive-tick based)
    - Minimum TTC observed
    - Time in TAKEOVER_ACTIVE
    - Driver reaction latency distribution (mean, std, percentiles)
    - Maximum risk score
    - Collision count
    - Episode outcome classification
    - Scenario metadata

Outputs ``logs/<sim_id>_metrics.json``.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects per-tick telemetry and produces episode-level KPIs.

    Args:
        simulation_id: Unique run identifier.
        output_dir: Directory for the metrics JSON file.
        scenario_name: Active scenario label.
        seed_values: Dict of seeds used (``random_seed``, ``traffic_seed``).
        map_type: Map type string (``"default"`` or ``"real_world"``).
        hazard_ttc_threshold: TTC (seconds) below which a hazard exists.
        consecutive_tick_requirement: N ticks a hazard must persist
            to qualify as a *missed* hazard.
        safety_ttc_threshold: If ``min_TTC`` during a takeover episode
            exceeds this, the takeover is classified as *false*.
    """

    def __init__(
        self,
        simulation_id: str,
        output_dir: str,
        scenario_name: str = "",
        seed_values: dict = None,
        map_type: str = "default",
        hazard_ttc_threshold: float = 2.0,
        consecutive_tick_requirement: int = 5,
        safety_ttc_threshold: float = 4.0,
    ):
        self._sim_id = simulation_id
        self._out_dir = output_dir
        self._scenario = scenario_name
        self._seeds = seed_values or {}
        self._map_type = map_type
        self._hazard_thresh = hazard_ttc_threshold
        self._consec_req = consecutive_tick_requirement
        self._safety_thresh = safety_ttc_threshold

        # Accumulators
        self._min_ttc = float("inf")
        self._max_risk = 0.0
        self._collision_count = 0
        self._reaction_latencies: List[float] = []

        # Takeover tracking
        self._takeover_active_ticks = 0
        self._in_takeover = False
        self._takeover_min_ttc = float("inf")
        self._false_takeover_count = 0

        # Missed hazard tracking
        self._hazard_streak = 0
        self._missed_hazard_count = 0

        # State timing
        self._delta_t: float = 0.05  # will be set on first tick

    # ------------------------------------------------------------------
    # Per-tick ingestion
    # ------------------------------------------------------------------

    def record_tick(
        self,
        min_ttc: float,
        risk: float,
        takeover_active: bool,
        collision_this_tick: bool,
        response_time: Optional[float],
        delta_t: float = 0.05,
    ):
        """Feed one tick of data.

        Args:
            min_ttc: Minimum TTC this tick (seconds or inf).
            risk: Current risk score.
            takeover_active: Whether takeover stage > 0.
            collision_this_tick: Whether a new collision occurred.
            response_time: Sampled driver response time (if any).
            delta_t: Simulation time-step.
        """
        self._delta_t = delta_t

        # Global extremes
        if min_ttc < self._min_ttc:
            self._min_ttc = min_ttc
        if risk > self._max_risk:
            self._max_risk = risk
        if collision_this_tick:
            self._collision_count += 1

        # Reaction latency
        if response_time is not None and response_time > 0:
            self._reaction_latencies.append(response_time)

        # Takeover duration + false-takeover detection
        if takeover_active:
            self._takeover_active_ticks += 1
            if not self._in_takeover:
                # Start of new takeover episode
                self._in_takeover = True
                self._takeover_min_ttc = min_ttc
            else:
                if min_ttc < self._takeover_min_ttc:
                    self._takeover_min_ttc = min_ttc
        else:
            if self._in_takeover:
                # End of takeover episode — classify
                if self._takeover_min_ttc > self._safety_thresh:
                    self._false_takeover_count += 1
                self._in_takeover = False
                self._takeover_min_ttc = float("inf")

        # Missed hazard (N consecutive ticks under threshold, no takeover)
        if min_ttc < self._hazard_thresh and not takeover_active:
            self._hazard_streak += 1
            if self._hazard_streak == self._consec_req:
                self._missed_hazard_count += 1
        else:
            self._hazard_streak = 0

    # ------------------------------------------------------------------
    # Finalise & output
    # ------------------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """Compute KPIs and write summary JSON.

        Returns:
            The KPI dictionary.
        """
        # Close any open takeover episode
        if self._in_takeover and self._takeover_min_ttc > self._safety_thresh:
            self._false_takeover_count += 1

        # Reaction latency statistics
        latency_stats = self._latency_stats()

        # Episode outcome
        outcome = self._classify_outcome()

        summary: Dict[str, Any] = {
            "simulation_id": self._sim_id,
            "scenario_name": self._scenario,
            "seed_values": self._seeds,
            "map_type": self._map_type,
            "episode_outcome": outcome,
            "false_takeover_count": self._false_takeover_count,
            "missed_hazard_count": self._missed_hazard_count,
            "min_ttc_observed": (
                round(self._min_ttc, 4)
                if self._min_ttc != float("inf")
                else None
            ),
            "takeover_active_duration_s": round(
                self._takeover_active_ticks * self._delta_t, 3
            ),
            "reaction_latency_distribution": latency_stats,
            "max_risk_score": round(self._max_risk, 6),
            "collision_count": self._collision_count,
        }

        # Write to file
        os.makedirs(self._out_dir, exist_ok=True)
        path = os.path.join(self._out_dir, f"{self._sim_id}_metrics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Metrics written to %s", path)

        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify_outcome(self) -> str:
        """Determine episode outcome label."""
        if self._collision_count > 0:
            return "collision"
        if self._missed_hazard_count > 0:
            return "missed_hazard"
        if self._takeover_active_ticks > 0:
            return "manual_override"
        return "safe"

    def _latency_stats(self) -> Dict[str, Any]:
        """Compute reaction-latency statistics."""
        if not self._reaction_latencies:
            return {}
        vals = sorted(self._reaction_latencies)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        std = math.sqrt(var)

        def percentile(p):
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < n else f
            d = k - f
            return round(vals[f] + d * (vals[c] - vals[f]), 4)

        return {
            "count": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(vals[0], 4),
            "max": round(vals[-1], 4),
            "p50": percentile(50),
            "p95": percentile(95),
        }
