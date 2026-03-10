"""
Highway cut-in scenario for the ADAS Supervision Framework.

A deterministic scenario where the ego vehicle drives on a highway
and an NPC vehicle performs a lane-change cut-in from an adjacent
lane, forcing the perception/supervision pipeline to react.

Spawns and NPC trajectories are fully seeded for replay
reproducibility.
"""

import logging
import random
import time

import carla

logger = logging.getLogger(__name__)


class HighwayCutInScenario:
    """Deterministic highway cut-in scenario.

    Args:
        client: ``carla.Client``.
        world: Active ``carla.World``.
        ego_vehicle: The ego ``carla.Vehicle``.
        config: Scenario section of config.
        random_seed: Seed for NPC spawning.
    """

    def __init__(
        self,
        client: "carla.Client",
        world: "carla.World",
        ego_vehicle: "carla.Vehicle",
        config: dict = None,
        random_seed: int = 42,
    ):
        self.client = client
        self.world = world
        self.ego = ego_vehicle
        self._cfg = config or {}
        self._rng = random.Random(random_seed)
        self._npc_vehicles: list = []
        self._npc_cutin: "carla.Vehicle" = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self):
        """Spawn traffic and the cut-in NPC ahead of the ego vehicle.

        The NPC is placed in an adjacent lane slightly ahead of ego
        and scripted to change lanes after a short delay.
        """
        bp_library = self.world.get_blueprint_library()
        vehicle_bps = list(bp_library.filter("vehicle.*"))

        ego_transform = self.ego.get_transform()
        ego_wp = self.world.get_map().get_waypoint(
            ego_transform.location, lane_type=carla.LaneType.Driving
        )

        # ---- Cut-in NPC ------------------------------------------------
        # Place in adjacent lane, 25 m ahead
        adjacent_wp = ego_wp.get_left_lane() or ego_wp.get_right_lane()
        if adjacent_wp is None:
            logger.warning("No adjacent lane found — using same lane ahead.")
            adjacent_wp = ego_wp

        ahead_wps = adjacent_wp.next(25.0)
        if ahead_wps and ahead_wps[0]:
            try:
                npc_spawn = ahead_wps[0].transform
                npc_spawn.location.z += 0.5  # lift to avoid ground collision
                bp = self._rng.choice(vehicle_bps)
                if bp.has_attribute("color"):
                    bp.set_attribute("color", "255,0,0")
                self._npc_cutin = self.world.try_spawn_actor(bp, npc_spawn)
                if self._npc_cutin:
                    self._npc_vehicles.append(self._npc_cutin)
                    logger.info("Cut-in NPC spawned at %s", npc_spawn.location)
            except Exception as e:
                logger.warning(f"Failed to spawn cut-in NPC: {e}")

        # ---- Background traffic ----------------------------------------
        n_traffic = self._cfg.get("traffic_vehicles", 10)
        spawn_points = self.world.get_map().get_spawn_points()
        self._rng.shuffle(spawn_points)
        tm = self.client.get_trafficmanager()
        for sp in spawn_points[:n_traffic]:
            bp = self._rng.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                self._npc_vehicles.append(npc)

        logger.info(
            "Scenario setup complete — %d NPCs total.", len(self._npc_vehicles)
        )

    def tick(self, elapsed_seconds: float):
        """Called every simulation tick to script NPC behaviour.

        After ~2 s the cut-in NPC switches to the ego lane.

        Args:
            elapsed_seconds: Wall-clock seconds since scenario start.
        """
        if self._npc_cutin is None or not self._npc_cutin.is_alive:
            return

        if 2.0 < elapsed_seconds < 2.5:
            # Force the NPC into ego's lane by applying steer
            ctrl = carla.VehicleControl(
                throttle=0.6, steer=-0.35, brake=0.0
            )
            self._npc_cutin.apply_control(ctrl)
        elif 2.5 <= elapsed_seconds < 3.5:
            # Straighten out the vehicle after the lane change
            ctrl = carla.VehicleControl(
                throttle=0.5, steer=0.1, brake=0.0
            )
            self._npc_cutin.apply_control(ctrl)
        elif elapsed_seconds >= 3.5 and not getattr(self, '_cutin_done', False):
            # Hand the vehicle back to the Traffic Manager so it drives smoothly
            tm = self.client.get_trafficmanager()
            self._npc_cutin.set_autopilot(True, tm.get_port())
            
            # Optionally configure it to drive normally without immediate lane changes
            tm.auto_lane_change(self._npc_cutin, False)
            tm.vehicle_percentage_speed_difference(self._npc_cutin, -10.0) # drive a bit faster
            
            self._cutin_done = True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Destroy all NPC actors spawned by this scenario."""
        for npc in self._npc_vehicles:
            if npc is not None and npc.is_alive:
                npc.destroy()
        self._npc_vehicles.clear()
        self._npc_cutin = None
        logger.info("Scenario NPCs destroyed.")
