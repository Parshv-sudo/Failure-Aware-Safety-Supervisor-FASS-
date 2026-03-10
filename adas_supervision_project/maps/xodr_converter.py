"""
OSM → OpenDRIVE converter for the ADAS Supervision Framework.

Wraps ``carla.Osm2Odr.convert()`` with configurable
``carla.Osm2OdrSettings``.
"""

import logging
import os

import carla

logger = logging.getLogger(__name__)


class XODRConverter:
    """Converts raw OSM XML data to OpenDRIVE (.xodr) format.

    Args:
        settings_dict: Dictionary of conversion settings from config
            (``lane_width``, ``traffic_lights``, etc.).
    """

    # Default OSM way types to include in the conversion
    DEFAULT_WAY_TYPES = [
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "unclassified", "residential",
    ]

    def __init__(self, settings_dict: dict = None):
        self._settings_dict = settings_dict or {}

    def convert(self, osm_data: str) -> str:
        """Convert OSM XML string to OpenDRIVE XML string.

        Args:
            osm_data: Raw ``.osm`` file content.

        Returns:
            OpenDRIVE XML string.

        Raises:
            RuntimeError: If CARLA conversion fails.
        """
        settings = carla.Osm2OdrSettings()
        settings.set_osm_way_types(self.DEFAULT_WAY_TYPES)
        settings.default_lane_width = self._settings_dict.get("lane_width", 6.0)
        settings.generate_traffic_lights = self._settings_dict.get(
            "traffic_lights", True
        )
        settings.all_junctions_with_traffic_lights = self._settings_dict.get(
            "all_junctions_lights", False
        )
        settings.center_map = self._settings_dict.get("center_map", True)

        xodr_data = carla.Osm2Odr.convert(osm_data, settings)
        if not xodr_data:
            raise RuntimeError("CARLA Osm2Odr conversion returned empty data.")

        logger.info("OSM → XODR conversion successful (%d bytes).", len(xodr_data))
        return xodr_data

    def convert_and_save(
        self, osm_data: str, output_path: str
    ) -> str:
        """Convert and write the resulting XODR to disk.

        Args:
            osm_data: Raw OSM XML string.
            output_path: File path to write the ``.xodr`` file.

        Returns:
            The OpenDRIVE XML string (same as written to disk).
        """
        xodr = self.convert(osm_data)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(xodr)
        logger.info("XODR written to %s", output_path)
        return xodr
