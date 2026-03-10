"""
Sensor manager for the ADAS Supervision Framework.

Attaches RGB camera, collision sensor, lane-invasion sensor, and IMU
to the ego vehicle.  Sensor data is buffered via callbacks and made
available through thread-safe accessors.
"""

import logging
import threading
from typing import Any, Dict, Optional

import carla

logger = logging.getLogger(__name__)


class SensorManager:
    """Manages the ego vehicle's sensor suite.

    Args:
        world: Active ``carla.World`` handle.
        vehicle: The ego ``carla.Vehicle`` actor to attach sensors to.
        config: Sensor-related config dictionary (``cfg.get("sensors")``).
    """

    def __init__(
        self,
        world: "carla.World",
        vehicle: "carla.Vehicle",
        config: Dict[str, Any],
    ):
        self.world = world
        self.vehicle = vehicle
        self._config = config or {}
        self._sensors: Dict[str, carla.Actor] = {}
        self._lock = threading.Lock()

        # Buffered latest data
        self._latest: Dict[str, Any] = {
            "rgb": None,
            "collision": None,
            "lane_invasion": None,
            "imu": None,
        }
        self._collision_history: list = []

        self._attach_all()

    # ------------------------------------------------------------------
    # Sensor attachment
    # ------------------------------------------------------------------

    def _attach_all(self):
        """Attach every configured sensor to the ego vehicle."""
        bp_lib = self.world.get_blueprint_library()

        # -- RGB camera --------------------------------------------------
        cam_cfg = self._config.get("rgb_camera", {})
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(cam_cfg.get("image_size_x", 800)))
        cam_bp.set_attribute("image_size_y", str(cam_cfg.get("image_size_y", 600)))
        cam_bp.set_attribute("fov", str(cam_cfg.get("fov", 90)))
        pos = cam_cfg.get("position", {"x": 1.5, "y": 0.0, "z": 2.4})
        cam_transform = carla.Transform(
            carla.Location(x=pos["x"], y=pos["y"], z=pos["z"])
        )
        cam = self.world.try_spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        if cam:
            cam.listen(self._on_rgb)
            self._sensors["rgb"] = cam
            logger.info("Attached RGB camera.")
        else:
            logger.warning("Failed to attach RGB camera. Collision overlap?")

        # -- Collision sensor --------------------------------------------
        if self._config.get("collision", {}).get("enabled", True):
            col_bp = bp_lib.find("sensor.other.collision")
            col = self.world.try_spawn_actor(
                col_bp, carla.Transform(), attach_to=self.vehicle
            )
            if col:
                col.listen(self._on_collision)
                self._sensors["collision"] = col
                logger.info("Attached collision sensor.")
            else:
                logger.warning("Failed to attach collision sensor.")

        # -- Lane invasion sensor ----------------------------------------
        if self._config.get("lane_invasion", {}).get("enabled", True):
            li_bp = bp_lib.find("sensor.other.lane_invasion")
            li = self.world.try_spawn_actor(
                li_bp, carla.Transform(), attach_to=self.vehicle
            )
            if li:
                li.listen(self._on_lane_invasion)
                self._sensors["lane_invasion"] = li
                logger.info("Attached lane-invasion sensor.")
            else:
                logger.warning("Failed to attach lane-invasion sensor.")

        # -- IMU ---------------------------------------------------------
        if self._config.get("imu", {}).get("enabled", True):
            imu_bp = bp_lib.find("sensor.other.imu")
            imu = self.world.try_spawn_actor(
                imu_bp, carla.Transform(), attach_to=self.vehicle
            )
            if imu:
                imu.listen(self._on_imu)
                self._sensors["imu"] = imu
                logger.info("Attached IMU sensor.")
            else:
                logger.warning("Failed to attach IMU sensor.")

    # ------------------------------------------------------------------
    # Callbacks (called from CARLA sensor threads)
    # ------------------------------------------------------------------

    def _on_rgb(self, image):
        with self._lock:
            self._latest["rgb"] = image

    def _on_collision(self, event):
        with self._lock:
            self._latest["collision"] = event
            self._collision_history.append(event)

    def _on_lane_invasion(self, event):
        with self._lock:
            self._latest["lane_invasion"] = event

    def _on_imu(self, data):
        with self._lock:
            self._latest["imu"] = data

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_latest(self, sensor_name: str) -> Optional[Any]:
        """Return the most recent data for *sensor_name*.

        Args:
            sensor_name: One of ``"rgb"``, ``"collision"``,
                ``"lane_invasion"``, ``"imu"``.
        """
        with self._lock:
            return self._latest.get(sensor_name)

    def get_collision_history(self) -> list:
        """Return the full collision event history."""
        with self._lock:
            return list(self._collision_history)

    def has_collided(self) -> bool:
        """True if at least one collision has been recorded."""
        with self._lock:
            return len(self._collision_history) > 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Stop listening and destroy all attached sensors."""
        for name, sensor in self._sensors.items():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
                logger.info("Destroyed sensor: %s", name)
        self._sensors.clear()
