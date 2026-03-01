#!/usr/bin/env python
"""
CARLA Data Collector for FASS ML Module
========================================
Connects to CARLA server in synchronous mode, spawns an ego vehicle with
a full sensor suite, and collects per-tick snapshots for ML training.

Sensors attached:
    - RGB Camera (front)
    - LiDAR (roof-mounted)
    - Radar (front)
    - Collision detector
    - GNSS
    - IMU

Each frame is saved as a compressed .npz archive containing:
    - detected_objects  : list of dicts (type, bbox, velocity, distance)
    - ego_kinematics    : dict (speed, accel, yaw_rate, steer, location, rotation)
    - weather           : dict (rain, fog, sun, wetness, wind)
    - sensor_health     : dict (camera_ok, lidar_ok, radar_ok)
    - collision_event   : dict or None (other_actor, impulse)
    - metadata          : dict (frame_id, timestamp, map_name, scenario_id)

ISO 26262 Note:
    This module is a DATA SOURCE only.  It does not make control decisions.
    Sensor health flags are critical for downstream degradation awareness.

Usage:
    python carla_data_collector.py --host localhost --port 2000 --frames 500 --out ./fass_data
"""

import glob
import os
import sys
import time
import math
import argparse
import json
import numpy as np
from queue import Queue, Empty
from collections import defaultdict

# ---------------------------------------------------------------------------
# CARLA egg path resolution (standard CARLA pattern)
# ---------------------------------------------------------------------------
try:
    sys.path.append(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

try:
    import carla
except ImportError:
    raise ImportError(
        "Cannot import carla.  Make sure the CARLA .egg is in PythonAPI/carla/dist/ "
        "or the carla package is installed."
    )


# ============================================================================
# Sensor callback helpers
# ============================================================================

class SensorCallbackManager:
    """Thread-safe sensor data accumulator.

    Each sensor pushes data into a shared queue on its callback.  The main
    loop drains the queue once per world tick, ensuring all sensor data is
    aligned to the same simulation frame.
    """

    def __init__(self):
        self._queue = Queue()
        self._latest = {}  # sensor_name -> latest data
        self._health = defaultdict(lambda: True)

    def make_callback(self, sensor_name: str):
        """Return a closure suitable for ``sensor.listen(...)``."""
        def _cb(data):
            self._queue.put((sensor_name, data))
        return _cb

    def collect(self, expected_count: int, timeout: float = 2.0):
        """Block until *expected_count* sensor payloads arrive or timeout."""
        received = 0
        while received < expected_count:
            try:
                name, data = self._queue.get(timeout=timeout)
                self._latest[name] = data
                self._health[name] = True
                received += 1
            except Empty:
                # Mark missing sensors as unhealthy
                for name in list(self._health):
                    if name not in self._latest:
                        self._health[name] = False
                break

    def get(self, name: str):
        return self._latest.get(name)

    def get_health(self) -> dict:
        return dict(self._health)

    def reset_frame(self):
        self._latest.clear()


# ============================================================================
# Object detection helpers (from ground-truth actors)
# ============================================================================

def _extract_detected_objects(world, ego_vehicle, max_distance: float = 80.0):
    """Extract detected objects from CARLA world state (ground truth).

    In a real ADS, this would come from perception.  Here we use CARLA's
    ground truth for dataset label generation and pair it with sensor data.
    """
    ego_loc = ego_vehicle.get_location()
    objects_out = []

    for actor in world.get_actors():
        # Skip ego and sensors
        if actor.id == ego_vehicle.id:
            continue

        # Only vehicles, walkers, and static obstacles
        type_id = actor.type_id
        if not any(type_id.startswith(t) for t in ('vehicle.', 'walker.', 'static.')):
            continue

        actor_loc = actor.get_location()
        distance = ego_loc.distance(actor_loc)
        if distance > max_distance:
            continue

        vel = actor.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        bbox = actor.bounding_box
        obj = {
            'actor_id': actor.id,
            'type': type_id.split('.')[0],           # 'vehicle' | 'walker' | 'static'
            'type_id': type_id,
            'distance': round(distance, 3),
            'speed': round(speed, 3),
            'location': [actor_loc.x, actor_loc.y, actor_loc.z],
            'velocity': [vel.x, vel.y, vel.z],
            'bbox_extent': [bbox.extent.x, bbox.extent.y, bbox.extent.z],
            # Simulated detection confidence — degrades with distance and weather
            'detection_confidence': max(0.0, min(1.0, 1.0 - (distance / max_distance) * 0.5)),
        }
        objects_out.append(obj)

    # Sort by distance (nearest first)
    objects_out.sort(key=lambda o: o['distance'])
    return objects_out


def _extract_ego_kinematics(ego_vehicle):
    """Extract ego vehicle kinematics."""
    vel = ego_vehicle.get_velocity()
    accel = ego_vehicle.get_acceleration()
    ang_vel = ego_vehicle.get_angular_velocity()
    transform = ego_vehicle.get_transform()
    control = ego_vehicle.get_control()

    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    accel_mag = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

    return {
        'speed': round(speed, 4),
        'acceleration': round(accel_mag, 4),
        'velocity': [round(vel.x, 4), round(vel.y, 4), round(vel.z, 4)],
        'yaw_rate': round(ang_vel.z, 4),
        'location': [transform.location.x, transform.location.y, transform.location.z],
        'rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
        'steer': round(control.steer, 4),
        'throttle': round(control.throttle, 4),
        'brake': round(control.brake, 4),
    }


def _extract_weather(world):
    """Extract weather / environment state."""
    w = world.get_weather()
    return {
        'cloudiness': w.cloudiness,
        'precipitation': w.precipitation,
        'precipitation_deposits': w.precipitation_deposits,  # road wetness
        'wind_intensity': w.wind_intensity,
        'sun_altitude_angle': w.sun_altitude_angle,
        'fog_density': w.fog_density,
        'fog_distance': w.fog_distance,
        'fog_falloff': w.fog_falloff,
        'wetness': w.wetness,
        # Derived flags
        'is_night': w.sun_altitude_angle < -5.0,
        'is_rainy': w.precipitation > 30.0,
        'is_foggy': w.fog_density > 20.0,
    }


# ============================================================================
# Collision callback
# ============================================================================

class CollisionTracker:
    """Tracks collision events on the ego vehicle."""

    def __init__(self):
        self._events = []

    def callback(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._events.append({
            'frame': event.frame,
            'other_actor': event.other_actor.type_id if event.other_actor else 'unknown',
            'impulse': [impulse.x, impulse.y, impulse.z],
            'intensity': round(intensity, 3),
        })

    def pop_events(self):
        events = list(self._events)
        self._events.clear()
        return events


# ============================================================================
# Main data collection loop
# ============================================================================

def collect_data(
    host: str = 'localhost',
    port: int = 2000,
    num_frames: int = 500,
    output_dir: str = './fass_data',
    scenario_id: str = 'default',
    delta_seconds: float = 0.05,
):
    """Run synchronous data collection in CARLA.

    Parameters
    ----------
    host, port : CARLA server connection.
    num_frames : Number of simulation ticks to record.
    output_dir : Directory for .npz output files.
    scenario_id : Tag for traceability in logs.
    delta_seconds : Fixed simulation timestep (20 Hz default).
    """
    os.makedirs(output_dir, exist_ok=True)

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    # --- Save / restore settings ---
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta_seconds
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = carla_map.get_spawn_points()

    actors_to_destroy = []
    sensor_manager = SensorCallbackManager()
    collision_tracker = CollisionTracker()

    try:
        # --- Spawn ego vehicle ---
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        ego_vehicle.set_autopilot(True)
        actors_to_destroy.append(ego_vehicle)

        # --- Attach sensors ---
        sensor_specs = [
            ('camera_front', 'sensor.camera.rgb', carla.Transform(
                carla.Location(x=1.5, z=2.4))),
            ('lidar_roof', 'sensor.lidar.ray_cast', carla.Transform(
                carla.Location(z=2.5))),
            ('radar_front', 'sensor.other.radar', carla.Transform(
                carla.Location(x=2.0, z=1.0))),
            ('gnss', 'sensor.other.gnss', carla.Transform()),
            ('imu', 'sensor.other.imu', carla.Transform()),
        ]

        sensor_count = 0
        for name, bp_name, transform in sensor_specs:
            bp = blueprint_library.find(bp_name)
            # Configure LiDAR for reasonable density
            if 'lidar' in bp_name:
                bp.set_attribute('points_per_second', '100000')
                bp.set_attribute('range', '80.0')
            sensor = world.spawn_actor(bp, transform, attach_to=ego_vehicle)
            sensor.listen(sensor_manager.make_callback(name))
            actors_to_destroy.append(sensor)
            sensor_count += 1

        # Collision sensor
        col_bp = blueprint_library.find('sensor.other.collision')
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego_vehicle)
        col_sensor.listen(collision_tracker.callback)
        actors_to_destroy.append(col_sensor)

        print(f"[FASS DataCollector] Ego vehicle spawned.  Collecting {num_frames} frames...")
        print(f"[FASS DataCollector] Scenario ID: {scenario_id}")
        print(f"[FASS DataCollector] Output dir : {output_dir}")

        # --- Main collection loop ---
        for frame_idx in range(num_frames):
            world.tick()
            sensor_manager.collect(expected_count=sensor_count, timeout=2.0)

            # Extract all data
            detected_objects = _extract_detected_objects(world, ego_vehicle)
            ego_kinematics = _extract_ego_kinematics(ego_vehicle)
            weather = _extract_weather(world)
            sensor_health = sensor_manager.get_health()
            collision_events = collision_tracker.pop_events()

            snapshot = world.get_snapshot()
            metadata = {
                'frame_id': snapshot.frame,
                'timestamp': snapshot.timestamp.elapsed_seconds,
                'map_name': carla_map.name,
                'scenario_id': scenario_id,
                'frame_idx': frame_idx,
            }

            # --- Compute ground-truth risk label ---
            # Risk = f(min_distance, min_TTC, collision)
            min_dist = min((o['distance'] for o in detected_objects), default=999.0)
            ego_speed = ego_kinematics['speed']
            min_ttc = 999.0
            for obj in detected_objects:
                # Simplified TTC: closing speed along line of sight
                relative_speed = ego_speed - obj['speed'] * 0.5  # approximate
                if relative_speed > 0.1:
                    ttc = obj['distance'] / relative_speed
                    min_ttc = min(min_ttc, ttc)

            # Ground-truth risk label (0-1)
            distance_risk = max(0.0, 1.0 - min_dist / 30.0)
            ttc_risk = max(0.0, 1.0 - min_ttc / 5.0)
            collision_risk = 1.0 if collision_events else 0.0
            risk_label = min(1.0, max(distance_risk, ttc_risk, collision_risk))

            # Severity label
            severity = 0.0
            if collision_events:
                severity = max(e['intensity'] for e in collision_events)
                severity = min(1.0, severity / 500.0)  # normalize

            labels = {
                'risk': round(risk_label, 4),
                'severity': round(severity, 4),
                'min_distance': round(min_dist, 3),
                'min_ttc': round(min_ttc, 3),
                'collision': bool(collision_events),
            }

            # --- Save frame ---
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.npz")
            np.savez_compressed(
                frame_path,
                detected_objects=json.dumps(detected_objects),
                ego_kinematics=json.dumps(ego_kinematics),
                weather=json.dumps(weather),
                sensor_health=json.dumps(sensor_health),
                collision_events=json.dumps(collision_events),
                metadata=json.dumps(metadata),
                labels=json.dumps(labels),
            )

            sensor_manager.reset_frame()

            if (frame_idx + 1) % 50 == 0:
                print(f"  [{frame_idx + 1}/{num_frames}] risk={risk_label:.3f} "
                      f"dist={min_dist:.1f}m ttc={min_ttc:.1f}s "
                      f"collision={bool(collision_events)}")

        print(f"[FASS DataCollector] Done.  {num_frames} frames saved to {output_dir}")

    finally:
        # Cleanup
        for actor in reversed(actors_to_destroy):
            if actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FASS CARLA Data Collector — gather sensor data for ML training")
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--frames', type=int, default=500, help='Number of frames to collect')
    parser.add_argument('--out', default='./fass_data', help='Output directory')
    parser.add_argument('--scenario-id', default='default', help='Scenario tag for traceability')
    parser.add_argument('--delta', type=float, default=0.05, help='Simulation timestep (seconds)')
    args = parser.parse_args()

    collect_data(
        host=args.host,
        port=args.port,
        num_frames=args.frames,
        output_dir=args.out,
        scenario_id=args.scenario_id,
        delta_seconds=args.delta,
    )


if __name__ == '__main__':
    main()
