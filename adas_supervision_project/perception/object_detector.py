"""
Simulated object detector for the ADAS Supervision Framework.

Queries the CARLA world for nearby actors and produces detection
records that include object class, distance, relative velocity,
and an initial confidence estimate.  Noise, occlusion, and
misclassification injection are applied to simulate real sensor
imperfections.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import carla

from utils.math_utils import compute_distance, speed_from_velocity

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected object record.

    Attributes:
        actor_id: CARLA actor id (for correlation).
        object_class: Detected class label.
        true_class: Ground-truth class label (may differ if misclassified).
        distance: Distance from ego in metres.
        relative_velocity: Closing speed in m/s (positive = approaching).
        location: World (x, y, z) of the detected object.
        confidence: Initial confidence score ∈ [0, 1].
        occluded: Whether the object is synthetically occluded.
    """
    actor_id: int = 0
    object_class: str = "unknown"
    true_class: str = "unknown"
    distance: float = 0.0
    relative_velocity: float = 0.0
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    confidence: float = 1.0
    occluded: bool = False


class ObjectDetector:
    """Simulated perception front-end.

    Uses ground-truth CARLA world state augmented with noise,
    occlusion, and misclassification to model real-world perception
    limitations.

    Args:
        world: Active ``carla.World``.
        ego_vehicle: The ego ``carla.Vehicle``.
        config: ``perception`` section of the YAML config.
        random_seed: RNG seed for reproducibility.
    """

    _CLASS_MAP = {
        "vehicle": "vehicle",
        "walker": "pedestrian",
        "traffic": "traffic_sign",
    }
    _ALL_CLASSES = list(_CLASS_MAP.values())

    def __init__(
        self,
        world: "carla.World",
        ego_vehicle: "carla.Vehicle",
        config: dict = None,
        random_seed: int = 42,
    ):
        self.world = world
        self.ego = ego_vehicle
        cfg = config or {}
        self.detection_range = cfg.get("detection_range", 50.0)
        self.noise_std = cfg.get("noise_std", 0.5)
        self.occlusion_prob = cfg.get("occlusion_probability", 0.05)
        self.misclass_prob = cfg.get("misclassification_probability", 0.01)
        self._rng = random.Random(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> List[Detection]:
        """Run one detection cycle and return a list of detections.

        Returns:
            List of ``Detection`` records for nearby actors.
        """
        ego_loc = self.ego.get_location()
        ego_vel = self.ego.get_velocity()
        ego_pos = (ego_loc.x, ego_loc.y, ego_loc.z)
        ego_v = (ego_vel.x, ego_vel.y, ego_vel.z)

        detections: List[Detection] = []

        for actor in self.world.get_actors():
            # Skip self and sensors
            if actor.id == self.ego.id:
                continue
            if not (
                actor.type_id.startswith("vehicle")
                or actor.type_id.startswith("walker")
            ):
                continue

            loc = actor.get_location()
            pos = (loc.x, loc.y, loc.z)
            dist = compute_distance(ego_pos, pos)

            if dist > self.detection_range:
                continue

            # Relative velocity (closing speed)
            vel = actor.get_velocity()
            other_v = (vel.x, vel.y, vel.z)
            # Project relative velocity onto ego→other axis
            dx = pos[0] - ego_pos[0]
            dy = pos[1] - ego_pos[1]
            dz = pos[2] - ego_pos[2]
            d = max(dist, 1e-6)
            unit = (dx / d, dy / d, dz / d)
            rel_vx = ego_v[0] - other_v[0]
            rel_vy = ego_v[1] - other_v[1]
            rel_vz = ego_v[2] - other_v[2]
            closing_speed = rel_vx * unit[0] + rel_vy * unit[1] + rel_vz * unit[2]

            # Determine class
            true_cls = self._classify(actor.type_id)
            det_cls = true_cls

            # Misclassification injection
            if self._rng.random() < self.misclass_prob:
                det_cls = self._rng.choice(self._ALL_CLASSES)

            # Noise on distance
            noisy_dist = max(0.0, dist + self._rng.gauss(0, self.noise_std))

            # Occlusion
            occluded = self._rng.random() < self.occlusion_prob

            # Base confidence (decreases with distance)
            base_conf = max(0.0, 1.0 - (dist / self.detection_range) ** 2)
            if occluded:
                base_conf *= 0.3

            detections.append(
                Detection(
                    actor_id=actor.id,
                    object_class=det_cls,
                    true_class=true_cls,
                    distance=noisy_dist,
                    relative_velocity=closing_speed,
                    location=pos,
                    confidence=base_conf,
                    occluded=occluded,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(type_id: str) -> str:
        """Map a CARLA ``type_id`` to a simplified class label."""
        for prefix, label in ObjectDetector._CLASS_MAP.items():
            if type_id.startswith(prefix):
                return label
        return "unknown"
