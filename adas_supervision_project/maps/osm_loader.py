"""
OSM file loader for the ADAS Supervision Framework.

Reads and validates OpenStreetMap (.osm) XML files from disk.
"""

import logging
import os
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class OSMLoader:
    """Loads and validates an OpenStreetMap XML file.

    Args:
        osm_path: Path to the ``.osm`` file.
    """

    def __init__(self, osm_path: str):
        self.osm_path = os.path.abspath(osm_path)
        self._raw_data: str = ""

    def load(self) -> str:
        """Read the OSM file and return its raw XML string.

        Returns:
            The full XML content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid OSM XML.
        """
        if not os.path.isfile(self.osm_path):
            raise FileNotFoundError(f"OSM file not found: {self.osm_path}")

        with open(self.osm_path, "r", encoding="utf-8") as fh:
            self._raw_data = fh.read()

        self._validate()
        logger.info("Loaded OSM file: %s (%d bytes)", self.osm_path, len(self._raw_data))
        return self._raw_data

    def _validate(self):
        """Minimal XML structure check — root element must be ``<osm>``."""
        try:
            root = ET.fromstring(self._raw_data)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML in OSM file: {exc}") from exc
        if root.tag != "osm":
            raise ValueError(
                f"Expected root element <osm>, got <{root.tag}>"
            )
