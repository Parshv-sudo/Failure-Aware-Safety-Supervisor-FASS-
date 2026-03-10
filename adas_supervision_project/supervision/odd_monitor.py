"""
Operational Design Domain (ODD) monitor for the ADAS Supervision Framework.

Computes a composite *road complexity* metric ∈ [0, 1] from four
sub-factors:

1. **Curvature magnitude** — |dθ/ds| from sequential waypoints.
2. **Intersection proximity** — distance to nearest junction.
3. **Lane-width variance** — deviation from nominal lane width.
4. **Speed-limit context** — ratio of ego speed to road speed limit.
"""

import logging
import math
from typing import Optional

import carla

from utils.math_utils import clamp, compute_curvature, compute_distance_2d

logger = logging.getLogger(__name__)


class ODDMonitor:
    """Monitors the Operational Design Domain and outputs road complexity.

    Args:
        world: Active ``carla.World``.
        config: ``road_complexity`` section of config.
    """

    def __init__(self, world: "carla.World", config: dict = None):
        self.world = world
        self._map: carla.Map = world.get_map()
        cfg = config or {}

        self.w_curv = cfg.get("curvature_weight", 0.35)
        self.w_junc = cfg.get("intersection_weight", 0.30)
        self.w_lane = cfg.get("lane_width_weight", 0.15)
        self.w_speed = cfg.get("speed_limit_weight", 0.20)

        self.nominal_lane_width = cfg.get("nominal_lane_width", 3.5)
        self.curvature_max = cfg.get("curvature_max", 0.05)
        self.intersection_range = cfg.get("intersection_range", 50.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_complexity(
        self,
        ego_location: "carla.Location",
        ego_speed: float,
    ) -> float:
        """Compute road complexity at the ego vehicle's current position.

        Args:
            ego_location: Ego world location.
            ego_speed: Ego scalar speed m/s.

        Returns:
            Complexity score ∈ [0, 1].
        """
        wp = self._map.get_waypoint(
            ego_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return 0.5  # uncertain — mid-range

        curv = self._curvature_factor(wp)
        junc = self._intersection_factor(wp)
        lane = self._lane_width_factor(wp)
        spd = self._speed_limit_factor(wp, ego_speed)

        complexity = (
            self.w_curv * curv
            + self.w_junc * junc
            + self.w_lane * lane
            + self.w_speed * spd
        )
        return clamp(complexity, 0.0, 1.0)

    def get_road_type(self, ego_location: "carla.Location") -> str:
        """Return a simplified road-type label for the ego position.

        Returns:
            One of ``"highway"``, ``"urban"``, ``"intersection"``,
            ``"unknown"``.
        """
        wp = self._map.get_waypoint(
            ego_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return "unknown"
        if wp.is_junction:
            return "intersection"
        # Heuristic: speed limit ≥ 80 → highway
        speed_limit = self._get_speed_limit(wp)
        if speed_limit and speed_limit >= 80:
            return "highway"
        return "urban"

    # ------------------------------------------------------------------
    # Sub-factor computations
    # ------------------------------------------------------------------

    def _curvature_factor(self, wp: "carla.Waypoint") -> float:
        """κ = |Δθ / Δs| from two waypoints 2 m apart, normalised."""
        next_wps = wp.next(2.0)
        if not next_wps:
            return 0.0
        nwp = next_wps[0]
        h_a = wp.transform.rotation.yaw
        h_b = nwp.transform.rotation.yaw
        kappa = compute_curvature(h_a, h_b, 2.0)
        return clamp(kappa / self.curvature_max, 0.0, 1.0)

    def _intersection_factor(self, wp: "carla.Waypoint") -> float:
        """Proximity to nearest junction — closer → higher complexity."""
        if wp.is_junction:
            return 1.0
        # Look ahead for a junction
        scan_wp = wp
        dist_travelled = 0.0
        step = 2.0
        while dist_travelled < self.intersection_range:
            next_wps = scan_wp.next(step)
            if not next_wps:
                break
            scan_wp = next_wps[0]
            dist_travelled += step
            if scan_wp.is_junction:
                return clamp(
                    1.0 - dist_travelled / self.intersection_range, 0.0, 1.0
                )
        return 0.0  # no junction in range

    def _lane_width_factor(self, wp: "carla.Waypoint") -> float:
        """Deviation of current lane width from nominal."""
        lw = wp.lane_width
        deviation = abs(lw - self.nominal_lane_width) / self.nominal_lane_width
        return clamp(deviation, 0.0, 1.0)

    def _speed_limit_factor(self, wp: "carla.Waypoint", ego_speed: float) -> float:
        """Ratio of ego speed to road speed limit.

        Over-speed → complexity, under-speed → lower complexity.
        """
        limit = self._get_speed_limit(wp)
        if limit is None or limit <= 0:
            return 0.5  # unknown limit
        limit_ms = limit / 3.6  # km/h → m/s
        ratio = ego_speed / max(limit_ms, 1e-6)
        return clamp(ratio, 0.0, 1.0)

    @staticmethod
    def _get_speed_limit(wp: "carla.Waypoint") -> Optional[float]:
        """Attempt to read speed limit (km/h) for a waypoint.

        CARLA exposes speed limits via ``Landmark`` or via the
        ``TrafficManager``.  We fall back to a heuristic when neither
        is available.
        """
        # CARLA 0.9.15: no direct wp.speed_limit — use heuristic
        # based on road type. This can be improved with landmark queries.
        road_id = wp.road_id
        if wp.is_junction:
            return 30.0
        lane_count = 0
        try:
            # Count lanes to infer speed limit
            left = wp.get_left_lane()
            right = wp.get_right_lane()
            lane_count = 1 + (1 if left else 0) + (1 if right else 0)
        except Exception:
            pass
        if lane_count >= 3:
            return 90.0
        if lane_count == 2:
            return 50.0
        return 50.0
