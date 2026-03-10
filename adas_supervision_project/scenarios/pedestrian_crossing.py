"""
Pedestrian crossing scenario for the ADAS Supervision Framework.

A deterministic scenario where a pedestrian crosses the road ahead
of the ego vehicle, testing perception detection, TTC computation,
and takeover response.

Spawns are seeded and weather is deterministic for replay.
"""

import logging
import random

import carla

logger = logging.getLogger(__name__)


class PedestrianCrossingScenario:
    """Deterministic pedestrian crossing scenario.

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
        self._pedestrian = None
        self._ped_controller = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self):
        """Spawn a pedestrian ahead of the ego vehicle and background traffic."""
        bp_library = self.world.get_blueprint_library()

        ego_transform = self.ego.get_transform()
        ego_wp = self.world.get_map().get_waypoint(
            ego_transform.location, lane_type=carla.LaneType.Driving
        )

        # ---- Pedestrian -------------------------------------------------
        walker_bps = list(bp_library.filter("walker.pedestrian.*"))
        if walker_bps:
            walker_bp = self._rng.choice(walker_bps)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            ahead_wps = ego_wp.next(30.0)
            if ahead_wps:
                spawn_loc = ahead_wps[0].transform.location
                # Offset to the side of the road
                spawn_loc.y -= 4.0
                spawn_loc.z += 0.5
                spawn_transform = carla.Transform(
                    spawn_loc,
                    carla.Rotation(yaw=90),  # face across road
                )
                self._pedestrian = self.world.try_spawn_actor(
                    walker_bp, spawn_transform
                )
                if self._pedestrian:
                    logger.info("Pedestrian spawned at %s", spawn_loc)

                    # Attach AI controller
                    ctrl_bp = bp_library.find("controller.ai.walker")
                    self._ped_controller = self.world.spawn_actor(
                        ctrl_bp, carla.Transform(), attach_to=self._pedestrian
                    )

        # ---- Background traffic ----------------------------------------
        vehicle_bps = list(bp_library.filter("vehicle.*"))
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
            "Pedestrian crossing scenario ready — %d traffic NPCs.",
            len(self._npc_vehicles),
        )

    def tick(self, elapsed_seconds: float):
        """Script pedestrian movement.

        At ~3 s the pedestrian begins crossing.

        Args:
            elapsed_seconds: Seconds since scenario start.
        """
        if self._ped_controller is None:
            return

        if 3.0 <= elapsed_seconds < 3.2:
            # Start crossing
            ego_loc = self.ego.get_location()
            target = carla.Location(
                x=ego_loc.x + 30,
                y=ego_loc.y + 8,  # cross to other side
                z=ego_loc.z,
            )
            self._ped_controller.start()
            self._ped_controller.go_to_location(target)
            self._ped_controller.set_max_speed(1.4)  # walking speed

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Destroy all NPC actors."""
        if self._ped_controller and self._ped_controller.is_alive:
            self._ped_controller.stop()
            self._ped_controller.destroy()
        if self._pedestrian and self._pedestrian.is_alive:
            self._pedestrian.destroy()
        for npc in self._npc_vehicles:
            if npc and npc.is_alive:
                npc.destroy()
        self._npc_vehicles.clear()
        logger.info("Pedestrian crossing scenario cleaned up.")
