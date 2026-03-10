"""
Vehicle manager for the ADAS Supervision Framework.

Handles ego vehicle spawning (with deterministic spawn-point selection),
vehicle state queries, and control application.
"""

import logging
import random
from typing import Tuple

import carla

logger = logging.getLogger(__name__)


class VehicleManager:
    """Manages the ego vehicle lifecycle.

    Args:
        world: Active ``carla.World`` handle.
        blueprint_filter: Blueprint id string (e.g. ``"vehicle.tesla.model3"``).
        spawn_index: Index into the map's spawn-point list.
        random_seed: Seed for deterministic blueprint selection fallback.
    """

    def __init__(
        self,
        world: "carla.World",
        blueprint_filter: str = "vehicle.tesla.model3",
        spawn_index: int = 0,
        random_seed: int = 42,
    ):
        self.world = world
        self.vehicle: carla.Vehicle = None
        self._rng = random.Random(random_seed)
        
        # EMA filter state for smooth autopilot
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0

        bp_library = world.get_blueprint_library()
        bp_list = bp_library.filter(blueprint_filter)
        if not bp_list:
            logger.warning(
                "Blueprint '%s' not found — falling back to any vehicle.",
                blueprint_filter,
            )
            bp_list = bp_library.filter("vehicle.*")

        self._blueprint = self._rng.choice(list(bp_list))
        if self._blueprint.has_attribute("color"):
            self._blueprint.set_attribute("color", "0,0,0")  # black

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in current map.")
        self._spawn_index = min(spawn_index, len(spawn_points) - 1)
        self._spawn_points = spawn_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn(self) -> "carla.Vehicle":
        """Spawn the ego vehicle and return it.

        Returns:
            The spawned ``carla.Vehicle`` actor.

        Raises:
            RuntimeError: If spawning fails at the selected point.
        """
        transform = self._spawn_points[self._spawn_index]
        # Lift spawn point slightly to prevent Town04/Town05 ground-clipping
        transform.location.z += 1.0
        
        self.vehicle = self.world.try_spawn_actor(self._blueprint, transform)
        if self.vehicle is None:
            # Try a nearby spawn point
            for sp in self._spawn_points:
                sp.location.z += 1.0
                self.vehicle = self.world.try_spawn_actor(self._blueprint, sp)
                if self.vehicle is not None:
                    break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at any point.")
        logger.info(
            "Ego vehicle spawned: %s at %s",
            self.vehicle.type_id,
            self.vehicle.get_transform().location,
        )
        return self.vehicle

    def get_transform(self) -> "carla.Transform":
        """Current ego transform (location + rotation)."""
        return self.vehicle.get_transform()

    def get_location(self) -> "carla.Location":
        """Current ego location."""
        return self.vehicle.get_location()

    def get_velocity(self) -> "carla.Vector3D":
        """Current ego velocity vector (m/s)."""
        return self.vehicle.get_velocity()

    def get_velocity_tuple(self) -> Tuple[float, float, float]:
        """Velocity as a plain tuple ``(vx, vy, vz)``."""
        v = self.vehicle.get_velocity()
        return (v.x, v.y, v.z)

    def get_speed(self) -> float:
        """Scalar speed in m/s."""
        v = self.vehicle.get_velocity()
        return (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5

    def get_control(self) -> "carla.VehicleControl":
        """Current vehicle control state."""
        return self.vehicle.get_control()

    def apply_control(self, control: "carla.VehicleControl"):
        """Apply a ``carla.VehicleControl`` to the ego vehicle."""
        self.vehicle.apply_control(control)

    def smooth_control(self, control: "carla.VehicleControl", alpha_steer: float = 0.1, alpha_pedal: float = 0.2) -> "carla.VehicleControl":
        """Apply an EMA low-pass filter to raw control inputs to damp robotic movements.
        
        Args:
            control: The raw target control from Autopilot.
            alpha_steer: Smoothing factor for steering (lower = smoother/slower).
            alpha_pedal: Smoothing factor for throttle/brake.
            
        Returns:
            A new smoothed ``carla.VehicleControl``.
        """
        self._last_steer += alpha_steer * (control.steer - self._last_steer)
        self._last_throttle += alpha_pedal * (control.throttle - self._last_throttle)
        self._last_brake += alpha_pedal * (control.brake - self._last_brake)
        
        # Snap to 0 if extremely close
        if abs(self._last_steer) < 0.01: self._last_steer = 0.0
        if abs(self._last_throttle) < 0.01: self._last_throttle = 0.0
        if abs(self._last_brake) < 0.01: self._last_brake = 0.0

        return carla.VehicleControl(
            throttle=self._last_throttle,
            steer=self._last_steer,
            brake=self._last_brake,
            hand_brake=control.hand_brake,
            reverse=control.reverse,
            manual_gear_shift=control.manual_gear_shift,
            gear=control.gear
        )

    def reset_smoothing(self, current_control: "carla.VehicleControl"):
        """Syncs the EMA filter to a specific control state (e.g., when ADAS hands back control)."""
        self._last_steer = current_control.steer
        self._last_throttle = current_control.throttle
        self._last_brake = current_control.brake

    def set_autopilot(self, enabled: bool = True):
        """Enable or disable CARLA's built-in autopilot."""
        self.vehicle.set_autopilot(enabled)
        logger.info("Autopilot %s", "ON" if enabled else "OFF")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Remove the ego vehicle from the simulation."""
        if self.vehicle is not None:
            self.vehicle.destroy()
            logger.info("Ego vehicle destroyed.")
            self.vehicle = None
