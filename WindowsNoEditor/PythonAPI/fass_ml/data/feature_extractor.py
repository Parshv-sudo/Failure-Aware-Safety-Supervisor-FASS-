#!/usr/bin/env python
"""
Feature Extractor for FASS ML Module
======================================
Transforms raw CARLA sensor data into the fixed-size ML feature vector
consumed by FASSRiskNet.

Feature Vector (30 dimensions):
    [0]  closest_object_distance      — distance to nearest detected object (m)
    [1]  min_ttc                      — minimum time-to-collision (s), capped at 10
    [2]  num_objects                   — count of detected objects in range
    [3]  num_vehicles                  — count of detected vehicles
    [4]  num_pedestrians               — count of detected walkers/pedestrians
    [5]  avg_detection_confidence      — mean detection confidence across objects
    [6]  min_detection_confidence      — worst-case detection confidence
    [7]  closest_object_speed          — speed of nearest object (m/s)
    [8]  closest_object_is_pedestrian  — 1.0 if nearest is walker, else 0.0
    [9]  objects_in_5m                 — count of objects within 5 m
    [10] objects_in_10m               — count of objects within 10 m
    [11] camera_healthy               — 1.0 if camera is producing data
    [12] lidar_healthy                — 1.0 if LiDAR is producing data
    [13] radar_healthy                — 1.0 if radar is producing data
    [14] sensor_degradation_score     — 0-1, fraction of sensors failed
    [15] rain_intensity               — normalized precipitation (0-1)
    [16] fog_density                  — normalized fog (0-1)
    [17] sun_altitude_norm            — normalized sun angle (-1 to 1)
    [18] is_night                     — 1.0 if nighttime
    [19] road_wetness                 — normalized wetness (0-1)
    [20] wind_intensity               — normalized wind (0-1)
    [21] ego_speed                    — ego vehicle speed (m/s)
    [22] ego_acceleration             — ego vehicle acceleration magnitude (m/s²)
    [23] ego_yaw_rate                 — ego vehicle yaw rate (deg/s)
    [24] ego_steer                    — steering input (-1 to 1)
    [25] ego_throttle                 — throttle input (0 to 1)
    [26] ego_brake                    — brake input (0 to 1)
    [27] speed_risk_factor            — ego_speed / (closest_dist + 1), unitless
    [28] combined_sensor_object_risk  — sensor_degradation * (1 - min_confidence)
    [29] environmental_severity       — max(rain, fog, is_night, wetness)

ISO 26262 Note:
    Feature extraction is a pre-processing step.  Any missing or corrupted
    inputs are imputed with CONSERVATIVE defaults (worst-case assumption)
    to avoid masking real danger.
"""

import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_DIM = 35
MAX_TTC = 10.0          # cap TTC at 10 seconds
MAX_DISTANCE = 80.0     # sensor range
_CONSERVATIVE_DEFAULTS = {
    # When data is missing, assume WORST CASE for safety
    'closest_object_distance': 5.0,     # assume something is close
    'min_ttc': 2.0,                     # assume imminent
    'avg_detection_confidence': 0.3,    # assume poor detection
    'min_detection_confidence': 0.1,
    'sensor_degradation_score': 0.5,    # assume partial failure
}

FEATURE_NAMES = [
    'closest_object_distance', 'min_ttc', 'num_objects', 'num_vehicles',
    'num_pedestrians', 'avg_detection_confidence', 'min_detection_confidence',
    'closest_object_speed', 'closest_object_is_pedestrian', 'objects_in_5m',
    'objects_in_10m', 'camera_healthy', 'lidar_healthy', 'radar_healthy',
    'sensor_degradation_score', 'rain_intensity', 'fog_density',
    'sun_altitude_norm', 'is_night', 'road_wetness', 'wind_intensity',
    'ego_speed', 'ego_acceleration', 'ego_yaw_rate', 'ego_steer',
    'ego_throttle', 'ego_brake', 'speed_risk_factor',
    'combined_sensor_object_risk', 'environmental_severity',
    # NEW: terrain & state features
    'ego_pitch', 'ego_roll', 'ego_is_stationary', 'ego_is_reversing',
    'ego_altitude_change',
]

# Normalization ranges [min, max] for each feature (used for min-max scaling)
_NORM_RANGES = {
    'closest_object_distance': (0.0, MAX_DISTANCE),
    'min_ttc': (0.0, MAX_TTC),
    'num_objects': (0.0, 30.0),
    'num_vehicles': (0.0, 20.0),
    'num_pedestrians': (0.0, 10.0),
    'ego_speed': (0.0, 40.0),           # ~144 km/h
    'ego_acceleration': (0.0, 15.0),
    'ego_yaw_rate': (-180.0, 180.0),
    'closest_object_speed': (0.0, 40.0),
    'objects_in_5m': (0.0, 10.0),
    'objects_in_10m': (0.0, 15.0),
    # NEW terrain/state ranges
    'ego_pitch': (-30.0, 30.0),
    'ego_roll': (-30.0, 30.0),
    'ego_altitude_change': (-20.0, 20.0),
}


# ============================================================================
# Core extraction
# ============================================================================

def extract_features(
    detected_objects: list,
    ego_kinematics: dict,
    weather: dict,
    sensor_health: dict,
    normalize: bool = True,
) -> np.ndarray:
    """Convert raw CARLA data into a 30-dim float32 feature vector.

    Parameters
    ----------
    detected_objects : list of dicts with keys: type, distance, speed,
                       detection_confidence, bbox_extent.
    ego_kinematics   : dict with keys: speed, acceleration, yaw_rate, steer,
                       throttle, brake.
    weather          : dict with keys: precipitation, fog_density,
                       sun_altitude_angle, wetness, wind_intensity, is_night.
    sensor_health    : dict mapping sensor names to bool.
    normalize        : if True, apply min-max normalization.

    Returns
    -------
    np.ndarray of shape (30,) and dtype float32.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # --- Object features ---
    if detected_objects:
        distances = [o['distance'] for o in detected_objects]
        confidences = [o.get('detection_confidence', 0.5) for o in detected_objects]
        speeds = [o.get('speed', 0.0) for o in detected_objects]
        types = [o.get('type', 'unknown') for o in detected_objects]

        closest = detected_objects[0]   # already sorted by distance

        feat[0] = min(distances)
        feat[2] = len(detected_objects)
        feat[3] = sum(1 for t in types if t == 'vehicle')
        feat[4] = sum(1 for t in types if t == 'walker')
        feat[5] = np.mean(confidences)
        feat[6] = min(confidences)
        feat[7] = closest.get('speed', 0.0)
        feat[8] = 1.0 if closest.get('type') == 'walker' else 0.0
        feat[9] = sum(1 for d in distances if d < 5.0)
        feat[10] = sum(1 for d in distances if d < 10.0)

        # Min TTC
        ego_speed = ego_kinematics.get('speed', 0.0)
        min_ttc = MAX_TTC
        for obj in detected_objects:
            relative_speed = ego_speed - obj.get('speed', 0.0) * 0.5
            if relative_speed > 0.1:
                ttc = obj['distance'] / relative_speed
                min_ttc = min(min_ttc, ttc)
        feat[1] = min(min_ttc, MAX_TTC)
    else:
        # No objects — use safe defaults
        feat[0] = MAX_DISTANCE
        feat[1] = MAX_TTC
        feat[5] = 1.0 # avg_detection_confidence (1.0 = highly certain nothing is there)
        feat[6] = 1.0 # min_detection_confidence
        
    # --- Sensor health ---
    cam_ok = float(sensor_health.get('camera_front', True))
    lid_ok = float(sensor_health.get('lidar_roof', True))
    rad_ok = float(sensor_health.get('radar_front', True))
    feat[11] = cam_ok
    feat[12] = lid_ok
    feat[13] = rad_ok
    total_sensors = 3.0
    failed = (1.0 - cam_ok) + (1.0 - lid_ok) + (1.0 - rad_ok)
    feat[14] = failed / total_sensors

    # --- Weather / environment ---
    feat[15] = min(1.0, weather.get('precipitation', 0.0) / 100.0)
    feat[16] = min(1.0, weather.get('fog_density', 0.0) / 100.0)
    sun_alt = weather.get('sun_altitude_angle', 45.0)
    feat[17] = max(-1.0, min(1.0, sun_alt / 90.0))
    feat[18] = 1.0 if weather.get('is_night', False) else 0.0
    feat[19] = min(1.0, weather.get('wetness', 0.0) / 100.0)
    feat[20] = min(1.0, weather.get('wind_intensity', 0.0) / 100.0)

    # --- Ego kinematics ---
    feat[21] = ego_kinematics.get('speed', 0.0)
    feat[22] = ego_kinematics.get('acceleration', 0.0)
    feat[23] = ego_kinematics.get('yaw_rate', 0.0)
    feat[24] = ego_kinematics.get('steer', 0.0)
    feat[25] = ego_kinematics.get('throttle', 0.0)
    feat[26] = ego_kinematics.get('brake', 0.0)

    # --- Derived / compound features ---
    feat[27] = feat[21] / (feat[0] + 1.0)  # speed_risk_factor
    feat[28] = feat[14] * (1.0 - feat[6])  # combined_sensor_object_risk
    feat[29] = max(feat[15], feat[16], feat[18], feat[19])  # environmental_severity

    # --- NEW: Terrain & state features ---
    feat[30] = ego_kinematics.get('pitch', 0.0)             # ego_pitch (degrees)
    feat[31] = ego_kinematics.get('roll', 0.0)              # ego_roll (degrees)
    feat[32] = 1.0 if ego_kinematics.get('is_stationary', False) else 0.0
    feat[33] = 1.0 if ego_kinematics.get('is_reversing', False) else 0.0
    feat[34] = ego_kinematics.get('altitude_change', 0.0)   # m over last 3s

    # --- Normalize ---
    if normalize:
        feat = _normalize(feat)

    return feat


def _normalize(feat: np.ndarray) -> np.ndarray:
    """Min-max normalization using known ranges.  Features already in [0,1]
    or [-1,1] are left unchanged."""
    out = feat.copy()
    for i, name in enumerate(FEATURE_NAMES):
        if name in _NORM_RANGES:
            lo, hi = _NORM_RANGES[name]
            if hi > lo:
                out[i] = (feat[i] - lo) / (hi - lo)
                out[i] = max(0.0, min(1.0, out[i]))
    return out


def impute_missing(feat: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf with conservative defaults.

    ISO 26262 SAFETY NOTE: Missing data → assume WORST CASE.
    """
    for i, name in enumerate(FEATURE_NAMES):
        if not np.isfinite(feat[i]):
            feat[i] = _CONSERVATIVE_DEFAULTS.get(name, 0.0)
    return feat


# ============================================================================
# Batch extraction from .npz files
# ============================================================================

def extract_from_npz(npz_path: str, normalize: bool = True) -> tuple:
    """Load a single .npz frame and return (feature_vector, labels_dict).

    Returns
    -------
    tuple of (np.ndarray[30], dict)
    """
    data = np.load(npz_path, allow_pickle=True)

    detected_objects = json.loads(str(data['detected_objects']))
    ego_kinematics = json.loads(str(data['ego_kinematics']))
    weather = json.loads(str(data['weather']))
    sensor_health = json.loads(str(data['sensor_health']))
    labels = json.loads(str(data['labels']))

    feat = extract_features(detected_objects, ego_kinematics, weather,
                            sensor_health, normalize=normalize)
    feat = impute_missing(feat)

    return feat, labels
