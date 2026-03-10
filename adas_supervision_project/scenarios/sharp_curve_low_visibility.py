"""
Sharp curve with low visibility scenario for the ADAS Supervision Framework.

A deterministic scenario where the ego vehicle approaches a sharp
curve under foggy conditions (reduced visibility), testing the
road-complexity computation, confidence degradation due to weather,
and takeover response.

Weather is set to heavy fog for deterministic low visibility.
"""

import logging
import random

import carla

logger = logging.getLogger(__name__)


class SharpCurveLowVisibilityScenario:
    """Deterministic sharp-curve + fog scenario.

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
        self._original_weather = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self):
        """Set foggy weather and spawn sparse traffic."""
        bp_library = self.world.get_blueprint_library()

        # Save and set fog weather
        self._original_weather = self.world.get_weather()
        fog = carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=10.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=15.0,
            fog_density=80.0,
            fog_distance=10.0,
            fog_falloff=2.0,
            wetness=30.0,
        )
        self.world.set_weather(fog)
        logger.info("Weather set to HEAVY FOG (visibility ≈ 10 m)")

        # Sparse traffic
        vehicle_bps = list(bp_library.filter("vehicle.*"))
        n_traffic = max(1, self._cfg.get("traffic_vehicles", 10) // 3)
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
            "Sharp curve scenario ready — %d NPCs, fog active.",
            len(self._npc_vehicles),
        )

    def tick(self, elapsed_seconds: float):
        """Per-tick hook.

        The fog already provides the challenge — no scripted NPC
        behaviour needed.  The road geometry (curves) and reduced
        perception confidence do the work.

        Args:
            elapsed_seconds: Seconds since scenario start.
        """
        pass  # Weather + road geometry drive the challenge

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Restore weather and destroy NPCs."""
        if self._original_weather is not None:
            self.world.set_weather(self._original_weather)
            logger.info("Weather restored.")
        for npc in self._npc_vehicles:
            if npc and npc.is_alive:
                npc.destroy()
        self._npc_vehicles.clear()
        logger.info("Sharp curve scenario cleaned up.")
