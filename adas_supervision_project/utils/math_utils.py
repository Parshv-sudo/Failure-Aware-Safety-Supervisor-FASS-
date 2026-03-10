"""
Mathematical utility functions for the ADAS Supervision Framework.

Provides vector math, angle normalisation, curvature computation,
and general-purpose numerical helpers used across modules.
"""

import math
from typing import Tuple


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* to the closed interval [min_val, max_val].

    Args:
        value: The number to clamp.
        min_val: Lower bound.
        max_val: Upper bound.

    Returns:
        Clamped value.
    """
    return max(min_val, min(value, max_val))


def compute_distance(
    pos_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3-D points.

    Args:
        pos_a: (x, y, z) of point A.
        pos_b: (x, y, z) of point B.

    Returns:
        Scalar distance in the same units as input.
    """
    return math.sqrt(
        (pos_a[0] - pos_b[0]) ** 2
        + (pos_a[1] - pos_b[1]) ** 2
        + (pos_a[2] - pos_b[2]) ** 2
    )


def compute_distance_2d(
    pos_a: Tuple[float, float],
    pos_b: Tuple[float, float],
) -> float:
    """Euclidean distance in the XY plane.

    Args:
        pos_a: (x, y) of point A.
        pos_b: (x, y) of point B.

    Returns:
        Scalar distance.
    """
    return math.sqrt(
        (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
    )


def compute_relative_velocity(
    vel_ego: Tuple[float, float, float],
    vel_other: Tuple[float, float, float],
) -> float:
    """Closing speed along the ego-to-other vector.

    A positive value means the other object is approaching.

    Args:
        vel_ego: (vx, vy, vz) of ego vehicle.
        vel_other: (vx, vy, vz) of the other actor.

    Returns:
        Signed relative speed (positive = closing).
    """
    return math.sqrt(
        (vel_ego[0] - vel_other[0]) ** 2
        + (vel_ego[1] - vel_other[1]) ** 2
        + (vel_ego[2] - vel_other[2]) ** 2
    )


def normalize_angle(angle_deg: float) -> float:
    """Normalise an angle to the range (-180, 180] degrees.

    Args:
        angle_deg: Angle in degrees.

    Returns:
        Normalised angle in degrees.
    """
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg <= -180.0:
        angle_deg += 360.0
    return angle_deg


def compute_curvature(
    heading_a_deg: float,
    heading_b_deg: float,
    arc_length: float,
) -> float:
    """Approximate curvature κ = |Δθ / Δs|  from two sequential waypoints.

    Args:
        heading_a_deg: Heading at waypoint A (degrees).
        heading_b_deg: Heading at waypoint B (degrees).
        arc_length: Distance between the two waypoints (metres).

    Returns:
        Curvature in 1/m.  Returns 0.0 if arc_length ≤ 0.
    """
    if arc_length <= 0.0:
        return 0.0
    delta_heading = abs(normalize_angle(heading_b_deg - heading_a_deg))
    return math.radians(delta_heading) / arc_length


def speed_from_velocity(vel: Tuple[float, float, float]) -> float:
    """Scalar speed (m/s) from a 3-D velocity vector.

    Args:
        vel: (vx, vy, vz).

    Returns:
        |v| in m/s.
    """
    return math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
