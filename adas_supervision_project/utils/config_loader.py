"""
Configuration loader for the ADAS Supervision Framework.

Loads simulation_config.yaml, validates required keys, and computes a
SHA-256 hash of the raw config file for forensic traceability.
"""

import hashlib
import os
import yaml


class Config:
    """Singleton-style configuration container.

    Attributes:
        data: Parsed YAML dictionary.
        config_hash: SHA-256 hex digest of the raw config file.
        config_path: Absolute path to the loaded config file.
    """

    _instance = None

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "simulation_config.yaml",
            )
        self.config_path = os.path.abspath(config_path)
        self._load(self.config_path)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get(self, *keys, default=None):
        """Retrieve a nested config value using dot-style key sequence.

        Example::

            cfg.get("risk", "weights", "w1_confidence")
        """
        node = self.data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self, path: str):
        """Read YAML and compute file hash."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        self.config_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self.data = yaml.safe_load(raw)
        self._validate()

    def _validate(self):
        """Minimal structural validation."""
        required_sections = ["carla", "map", "logging"]
        for section in required_sections:
            if section not in self.data:
                raise ValueError(
                    f"Missing required config section: '{section}'"
                )

    @classmethod
    def instance(cls, config_path: str = None) -> "Config":
        """Return (or create) the singleton Config instance."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton — useful for testing."""
        cls._instance = None

    def __repr__(self):
        return f"Config(path={self.config_path!r}, hash={self.config_hash[:12]}…)"
