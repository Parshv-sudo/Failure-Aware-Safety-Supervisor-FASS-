"""
CARLA client wrapper for the ADAS Supervision Framework.

Establishes a connection to a running CARLA server, configures
synchronous mode, seeds the traffic manager, and provides world-tick
management and graceful cleanup.
"""

import glob
import os
import sys
import time
import logging

# ---------------------------------------------------------------------------
# Dynamically add the CARLA egg / whl to sys.path so ``import carla`` works
# regardless of where this script is launched from.
# ---------------------------------------------------------------------------
_CARLA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_EGG_PATTERN = os.path.join(
    _CARLA_ROOT,
    "WindowsNoEditor",
    "PythonAPI",
    "carla",
    "dist",
    "carla-*%d.%d-%s.egg"
    % (
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64",
    ),
)

try:
    sys.path.append(glob.glob(_EGG_PATTERN)[0])
except IndexError:
    # Egg not found — carla may already be installed as a pip package.
    pass

import carla  # noqa: E402  (must come after path manipulation)

logger = logging.getLogger(__name__)


class CarlaClient:
    """Manages the lifecycle of a CARLA client connection.

    Responsibilities:
        * Connect to server at *host*:*port* with automatic retry.
        * Enable synchronous mode with a fixed delta.
        * Seed the traffic manager for deterministic spawning.
        * Provide ``tick()`` and ``cleanup()`` helpers.

    Args:
        host: CARLA server hostname.
        port: CARLA server port.
        timeout: Client timeout in seconds.
        sync: Whether to enable synchronous mode.
        fixed_delta: Simulation fixed time-step (seconds).
        traffic_seed: Seed for the traffic manager RNG.
        max_retries: Number of connection attempts before giving up.
        retry_interval: Seconds between retry attempts.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 30.0,
        sync: bool = True,
        fixed_delta: float = 0.05,
        traffic_seed: int = 42,
        max_retries: int = 10,
        retry_interval: float = 3.0,
    ):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        # --- Retry loop: wait for CARLA server to be ready ---------------
        self.world = None
        for attempt in range(1, max_retries + 1):
            try:
                self.world = self.client.get_world()
                break
            except RuntimeError as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Could not connect to CARLA at {host}:{port} "
                        f"after {max_retries} attempts. "
                        f"Make sure CarlaUE4.exe is running."
                    ) from exc
                logger.warning(
                    "Connection attempt %d/%d failed (%s) — retrying "
                    "in %.0f s…",
                    attempt, max_retries, exc, retry_interval,
                )
                time.sleep(retry_interval)

        logger.info(
            "Connected to CARLA server at %s:%d  (map: %s)",
            host,
            port,
            self.world.get_map().name,
        )

        self._original_settings = self.world.get_settings()
        self.sync = sync
        self.fixed_delta = fixed_delta

        if sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_delta
            self.world.apply_settings(settings)
            logger.info(
                "Synchronous mode ON  (Δt = %.3f s)", fixed_delta
            )

        # Traffic manager seed
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(sync)
        self.traffic_manager.set_random_device_seed(traffic_seed)
        logger.info("Traffic manager seed set to %d", traffic_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self):
        """Advance the simulation by one step (synchronous mode)."""
        if self.sync:
            self.world.tick()

    def get_world(self):
        """Return the current CARLA world handle."""
        return self.world

    def get_client(self):
        """Return the raw ``carla.Client`` object."""
        return self.client

    def get_blueprint_library(self):
        """Shortcut to ``world.get_blueprint_library()``."""
        return self.world.get_blueprint_library()

    def set_weather(self, preset_name: str):
        """Apply a named weather preset.

        Args:
            preset_name: One of the ``carla.WeatherParameters`` presets,
                e.g. ``"ClearNoon"``, ``"WetCloudySunset"``.
        """
        preset = getattr(carla.WeatherParameters, preset_name, None)
        if preset is None:
            logger.warning("Unknown weather preset '%s' — skipping.", preset_name)
            return
        self.world.set_weather(preset)
        logger.info("Weather set to %s", preset_name)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Restore original world settings (async mode, etc.)."""
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            logger.info("Restored original CARLA settings.")
