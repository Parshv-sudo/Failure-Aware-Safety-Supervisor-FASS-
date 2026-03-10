"""
Empty scenario for the ADAS Supervision Framework.

Spawns no NPCs, allowing for baseline ambient risk testing of 
the ego vehicle's ML models or Rule-Based framework.
"""

import logging
import carla

logger = logging.getLogger(__name__)


class EmptyScenario:
    """Deterministic empty scenario with no traffic.

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
        pass

    def setup(self):
        """Do nothing. Ego vehicle drives on empty road."""
        logger.info("Empty Scenario setup complete — 0 NPCs total.")

    def tick(self, elapsed_seconds: float):
        """Do nothing."""
        pass

    def cleanup(self):
        """Do nothing."""
        logger.info("Empty Scenario NPCs destroyed.")
