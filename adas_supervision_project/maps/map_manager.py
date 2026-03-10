"""
Map manager for the ADAS Supervision Framework.

Orchestrates map loading — either from a real-world OSM file
(converted to OpenDRIVE on-the-fly) or by loading a built-in CARLA town.
"""

import logging
import os

import carla

from maps.osm_loader import OSMLoader
from maps.xodr_converter import XODRConverter

logger = logging.getLogger(__name__)


class MapManager:
    """Decides which map to load and applies it to the CARLA world.

    Args:
        client: ``carla.Client`` instance.
        map_config: The ``map`` section of the YAML config.
    """

    def __init__(self, client: "carla.Client", map_config: dict, on_world_loaded=None):
        self.client = client
        self._cfg = map_config or {}
        self._on_world_loaded = on_world_loaded

    def load(self) -> "carla.World":
        """Load the appropriate map and return the (possibly new) world.

        Returns:
            The active ``carla.World`` after map loading.
        """
        map_type = self._cfg.get("type", "default")

        if map_type == "real_world":
            world = self._load_real_world()
        else:
            world = self._load_default_town()

        if self._on_world_loaded:
            self._on_world_loaded(world)
            
        return world

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_real_world(self) -> "carla.World":
        """Load an OSM file, convert to XODR, and apply to CARLA."""
        source = self._cfg.get("source", "")
        # Resolve relative paths against the project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        osm_path = os.path.join(project_root, source) if not os.path.isabs(source) else source

        if not os.path.isfile(osm_path):
            logger.warning(
                "OSM file '%s' not found — falling back to default town.", osm_path
            )
            return self._load_default_town()

        try:
            loader = OSMLoader(osm_path)
            osm_data = loader.load()

            osm_settings = self._cfg.get("osm_settings", {})
            converter = XODRConverter(osm_settings)
            xodr_data = converter.convert(osm_data)

            logger.info("Loading OpenDRIVE world from converted OSM data…")
            return self.client.generate_opendrive_world(xodr_data)
        except Exception as exc:
            logger.error("Real-world map load failed: %s — falling back.", exc)
            return self._load_default_town()

    def _load_default_town(self) -> "carla.World":
        """Load a built-in CARLA town."""
        town = self._cfg.get("name", self._cfg.get("fallback_town", "Town03"))
        logger.info("Loading CARLA town: %s", town)
        return self.client.load_world(town)
