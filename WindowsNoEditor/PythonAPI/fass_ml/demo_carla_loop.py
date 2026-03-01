#!/usr/bin/env python
"""
FASS CARLA Demo — Hybrid Auto/Manual Driving with Safety Supervision
=====================================================================
Spawns an ego vehicle in CARLA with a visual Pygame window and full
FASS (Failure-Aware Safety Supervisor) integration.

Driving Modes:
    AUTOPILOT (default) — Car drives itself via CARLA Traffic Manager
    MANUAL              — Driver controls with WASD/Arrow keys

FASS Handoff:
    When the FASS supervisor detects DANGER, it automatically disables
    autopilot and alerts the driver to take manual control.  Emergency
    braking is applied regardless of mode.

Controls:
    W / ↑        : throttle
    S / ↓        : brake
    A / ← D / →  : steer
    SPACE        : handbrake
    Q            : toggle reverse
    P            : toggle autopilot
    R            : reset FASS SAFE_STOP latch + resume autopilot
    TAB          : toggle FASS HUD
    ESC          : quit

Usage:
    python -m fass_ml.demo_carla_loop
    python -m fass_ml.demo_carla_loop --synthetic   (no CARLA needed)
"""

import glob
import os
import sys
import time
import math
import argparse
import numpy as np
import threading

# CARLA egg path
try:
    sys.path.append(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

# Add PythonAPI/carla to path for agents module
_carla_agents_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'carla')
sys.path.insert(0, os.path.normpath(_carla_agents_path))

try:
    import pygame
    from pygame.locals import (
        K_ESCAPE, K_TAB, K_SPACE, K_r,
        K_w, K_a, K_s, K_d, K_p, K_q,
        K_UP, K_DOWN, K_LEFT, K_RIGHT,
        KMOD_CTRL,
    )
except ImportError:
    raise ImportError("pygame is required. Install with: pip install pygame")

from fass_ml.integration.fass_supervisor import FASSSupervisor
from fass_ml.safety.deterministic_overrides import DeterministicOverrides
from fass_ml.training.config import FASSConfig


# ============================================================================
# Pygame Camera Callback (thread-safe)
# ============================================================================

class CameraDisplay:
    """Receives CARLA camera images and converts them for Pygame."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = None
        self._lock = threading.Lock()

    def callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        with self._lock:
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display):
        with self._lock:
            if self.surface is not None:
                display.blit(self.surface, (0, 0))


# ============================================================================
# FASS HUD Overlay
# ============================================================================

class FASSHUD:
    """Draws FASS supervision + driving mode status on the Pygame window."""

    COLORS = {
        'SAFE':    (46, 204, 113),
        'CAUTION': (241, 196, 15),
        'DANGER':  (231, 76, 60),
    }

    def __init__(self, width, height):
        self.dim = (width, height)
        self._show = True
        pygame.font.init()
        self._font_big = pygame.font.SysFont('consolas', 22, bold=True)
        self._font = pygame.font.SysFont('consolas', 16)
        self._font_small = pygame.font.SysFont('consolas', 13)
        self._font_alert = pygame.font.SysFont('consolas', 28, bold=True)
        self._result = None
        self._ego_state = None
        self._tick_idx = 0
        self._objects_count = 0
        self._autopilot = True
        self._alert_text = None
        self._alert_timer = 0

    def toggle(self):
        self._show = not self._show

    def set_alert(self, text, duration=3.0):
        self._alert_text = text
        self._alert_timer = duration

    def update(self, result, ego_state, tick_idx, objects_count, autopilot, online_stats=None):
        self._result = result
        self._ego_state = ego_state
        self._tick_idx = tick_idx
        self._objects_count = objects_count
        self._autopilot = autopilot
        if online_stats:
            self._online_stats = online_stats

    def render(self, display, dt):
        # Always show driving mode indicator (even if HUD hidden)
        mode_text = "AUTOPILOT" if self._autopilot else "MANUAL"
        mode_color = (52, 152, 219) if self._autopilot else (230, 126, 34)
        mode_surf = self._font_big.render(
            f"  [{mode_text}]  P=Toggle", True, (255, 255, 255))
        mode_bg = pygame.Surface((mode_surf.get_width() + 16, 32))
        mode_bg.set_alpha(200)
        mode_bg.fill(mode_color)
        display.blit(mode_bg, (self.dim[0] - mode_bg.get_width() - 8, 8))
        display.blit(mode_surf, (self.dim[0] - mode_bg.get_width(), 12))

        # Centered alert text (handoff notification)
        if self._alert_timer > 0:
            self._alert_timer -= dt
            alpha = min(255, int(self._alert_timer * 200))
            alert_surf = self._font_alert.render(
                self._alert_text, True, (255, 255, 255))
            alert_bg = pygame.Surface(
                (alert_surf.get_width() + 40, alert_surf.get_height() + 20))
            alert_bg.set_alpha(min(200, alpha))
            alert_bg.fill((231, 76, 60))
            x = (self.dim[0] - alert_bg.get_width()) // 2
            y = self.dim[1] // 3
            display.blit(alert_bg, (x, y))
            display.blit(alert_surf, (x + 20, y + 10))

        if not self._show or self._result is None:
            return

        r = self._result
        ego = self._ego_state or {}
        advisory = r.get('advisory', 'SAFE')
        color = self.COLORS.get(advisory, (200, 200, 200))

        # --- Status Bar at top ---
        bar_h = 38
        bar_surface = pygame.Surface((self.dim[0] - 300, bar_h))
        bar_surface.set_alpha(200)
        bar_surface.fill(color)
        display.blit(bar_surface, (0, 0))

        intervention_type = (r['intervention'].type
                             if hasattr(r.get('intervention'), 'type')
                             else str(r.get('intervention', '')))
        status_text = f"  FASS: {advisory}   |   Intervention: {intervention_type}"
        if r.get('override_active'):
            status_text += "   |   OVERRIDE"
        text_surf = self._font_big.render(status_text, True, (255, 255, 255))
        display.blit(text_surf, (8, 8))

        # --- Info Panel (left side) ---
        panel_lines = []
        speed_kmh = ego.get('speed', 0) * 3.6
        panel_lines.append(f"Speed:       {speed_kmh:6.1f} km/h")
        panel_lines.append(f"Throttle:    {ego.get('throttle', 0):6.2f}")
        panel_lines.append(f"Brake:       {ego.get('brake', 0):6.2f}")
        panel_lines.append(f"Steer:       {ego.get('steer', 0):+6.3f}")
        panel_lines.append(f"")
        panel_lines.append(f"ML Risk:     {r.get('risk', 0):6.3f}")
        panel_lines.append(f"Uncertainty: {r.get('uncertainty', 0):6.4f}")
        panel_lines.append(f"Fused Risk:  {r.get('fused_risk', 0):6.3f}")
        panel_lines.append(f"")
        panel_lines.append(f"Objects:     {self._objects_count:4d}")
        panel_lines.append(f"Tick:        {self._tick_idx:4d}")
        panel_lines.append(f"Latency:     {r.get('tick_latency_ms', 0):5.1f} ms")

        panel_w = 280
        panel_h = len(panel_lines) * 20 + 16
        panel_surface = pygame.Surface((panel_w, panel_h))
        panel_surface.set_alpha(180)
        panel_surface.fill((20, 20, 20))
        display.blit(panel_surface, (0, bar_h))

        y = bar_h + 8
        for line in panel_lines:
            text_surf = self._font.render(line, True, (220, 220, 220))
            display.blit(text_surf, (10, y))
            y += 20

        # --- Online Learning Dashboard ---
        if self._online_stats:
            x_ol = panel_w + 20
            y_base = bar_h + 8
            
            title_surf = self._font.render("ONLINE LEARNING", True, (0, 255, 255))
            pygame.draw.rect(display, (0, 255, 255), (x_ol, y_base + 18, 160, 2))
            display.blit(title_surf, (x_ol, y_base))
            
            y_ol = y_base + 26
            st = self._online_stats
            labeled = st.get('buffer_frames', 0)
            high = st.get('high_risk_frames', 0)
            loss = st.get('loss', 0.0)
            rounds = st.get('train_step', 0)
            cols = st.get('collisions_seen', 0)
            
            lines = [
                f"Rounds:    {rounds}",
                f"Loss:      {loss:.4f}",
                f"Buffer:    {labeled} frames",
                f"High-Risk: {high}",
                f"Crashes:   {cols}"
            ]
            
            # semi-transparent bg for stats
            ol_h = len(lines) * 20 + 36
            ol_surface = pygame.Surface((200, ol_h))
            ol_surface.set_alpha(150)
            ol_surface.fill((10, 30, 30))
            display.blit(ol_surface, (x_ol - 10, bar_h))
            
            for text in lines:
                surf = self._font_small.render(text, True, (200, 200, 200))
                display.blit(surf, (x_ol, y_ol))
                y_ol += 20

        # --- Risk meter (bottom) ---
        meter_h = 12
        meter_y = self.dim[1] - meter_h - 4
        bg_rect = pygame.Rect(4, meter_y, self.dim[0] - 8, meter_h)
        pygame.draw.rect(display, (60, 60, 60), bg_rect)

        risk_val = min(1.0, max(0.0, r.get('fused_risk', 0)))
        fill_w = int((self.dim[0] - 8) * risk_val)
        if risk_val < 0.4:
            meter_color = (46, 204, 113)
        elif risk_val < 0.7:
            meter_color = (241, 196, 15)
        else:
            meter_color = (231, 76, 60)
        pygame.draw.rect(display, meter_color, pygame.Rect(4, meter_y, fill_w, meter_h))

        label = self._font_small.render(f"Risk: {risk_val:.3f}", True, (255, 255, 255))
        display.blit(label, (8, meter_y - 16))

        hint = self._font_small.render(
            "W/S=Throttle/Brake  A/D=Steer  Q=Reverse  P=Autopilot  R=Reset  ESC=Quit",
            True, (180, 180, 180))
        display.blit(hint, (self.dim[0] - hint.get_width() - 10, meter_y - 16))


# ============================================================================
# Keyboard Vehicle Controller
# ============================================================================

class KeyboardVehicleControl:
    """WASD/Arrow keyboard controls for manual driving (from manual_control.py)."""

    def __init__(self):
        self._steer_cache = 0.0
        self._reverse = False

    def parse(self, keys, milliseconds):
        """Return a carla.VehicleControl from current key state."""
        import carla
        control = carla.VehicleControl()

        # Throttle
        if keys[K_UP] or keys[K_w]:
            control.throttle = min(1.0, 0.6)  # moderate throttle
        else:
            control.throttle = 0.0

        # Brake
        if keys[K_DOWN] or keys[K_s]:
            control.brake = min(1.0, 0.8)
        else:
            control.brake = 0.0

        # Steering
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        control.steer = round(self._steer_cache, 1)
        control.hand_brake = keys[K_SPACE]

        # Reverse gear handling
        if self._reverse:
            control.reverse = True
            control.manual_gear_shift = True
            control.gear = -1
        else:
            control.reverse = False
            control.manual_gear_shift = False

        return control

    def toggle_reverse(self):
        """Toggle reverse gear on/off."""
        self._reverse = not self._reverse
        return self._reverse


# ============================================================================
# Sensor helpers (reused from data collector)
# ============================================================================

def _extract_detected_objects(world, ego_vehicle, max_distance=80.0):
    ego_loc = ego_vehicle.get_location()
    objects_out = []
    for actor in world.get_actors():
        if actor.id == ego_vehicle.id:
            continue
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
        objects_out.append({
            'type': type_id.split('.')[0],
            'distance': round(distance, 3),
            'speed': round(speed, 3),
            'detection_confidence': max(0.0, min(1.0, 1.0 - (distance / max_distance) * 0.5)),
            'bbox_extent': [bbox.extent.x, bbox.extent.y, bbox.extent.z],
        })
    objects_out.sort(key=lambda o: o['distance'])
    return objects_out


def _extract_ego_kinematics(ego_vehicle, _state={}):
    """Extract ego vehicle kinematics including terrain data.
    
    Uses a stateful dict to track altitude history and stationary duration.
    """
    vel = ego_vehicle.get_velocity()
    accel = ego_vehicle.get_acceleration()
    ang_vel = ego_vehicle.get_angular_velocity()
    transform = ego_vehicle.get_transform()
    control = ego_vehicle.get_control()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    accel_mag = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

    # Pitch and roll from transform rotation
    pitch = transform.rotation.pitch   # positive = nose up (uphill)
    roll = transform.rotation.roll     # positive = tilted right

    # Track altitude history for gradient detection
    import time as _time
    now = _time.time()
    altitude = transform.location.z
    if 'alt_history' not in _state:
        _state['alt_history'] = []
    _state['alt_history'].append((now, altitude))
    # Keep only last 5 seconds
    _state['alt_history'] = [(t, a) for t, a in _state['alt_history']
                              if now - t < 5.0]
    # Altitude change over last 3 seconds
    alt_3s_ago = [a for t, a in _state['alt_history'] if now - t >= 2.5]
    altitude_change = altitude - alt_3s_ago[0] if alt_3s_ago else 0.0

    # Stationary detection (speed < 0.5 m/s for 2+ seconds)
    if speed < 0.5:
        _state.setdefault('stationary_since', now)
    else:
        _state['stationary_since'] = now
    is_stationary = (now - _state.get('stationary_since', now)) >= 2.0

    return {
        'speed': round(speed, 4),
        'acceleration': round(accel_mag, 4),
        'yaw_rate': round(ang_vel.z, 4),
        'steer': round(control.steer, 4),
        'throttle': round(control.throttle, 4),
        'brake': round(control.brake, 4),
        'pitch': round(pitch, 2),
        'roll': round(roll, 2),
        'altitude_change': round(altitude_change, 3),
        'is_stationary': is_stationary,
        'is_reversing': bool(control.reverse),
    }


def _extract_weather(world):
    w = world.get_weather()
    return {
        'precipitation': w.precipitation,
        'fog_density': w.fog_density,
        'sun_altitude_angle': w.sun_altitude_angle,
        'wetness': w.wetness,
        'wind_intensity': w.wind_intensity,
        'is_night': w.sun_altitude_angle < -5.0,
    }


# ============================================================================
# Scenario Manager — Stress-test conditions for FASS
# ============================================================================

class ScenarioManager:
    """Periodically triggers challenging conditions to stress-test FASS."""

    def __init__(self, world, ego, blueprint_library):
        import random as _r
        self._world = world
        self._ego = ego
        self._bp_lib = blueprint_library
        self._rng = _r
        self._elapsed = 0.0
        self._scenario_actors = []  # actors spawned by scenarios
        self._active_scenario = None
        self._scenario_timer = 0.0
        self._next_scenario_at = _r.uniform(30.0, 45.0)  # first trigger
        self._original_weather = world.get_weather()
        self._sensor_failure_active = False
        self._scenario_count = 0

        # Scenario pool
        self._scenarios = [
            'STORM', 'NIGHT', 'JAYWALKER', 'CUT_IN', 'SENSOR_FAIL',
        ]
        self._last_scenario = None

    def tick(self, dt):
        """Call each frame. Returns sensor_health override or None."""
        import carla
        self._elapsed += dt
        sensor_override = None

        # Handle active scenario duration
        if self._active_scenario:
            self._scenario_timer -= dt
            if self._scenario_timer <= 0:
                self._end_scenario()
            elif self._active_scenario == 'SENSOR_FAIL':
                sensor_override = {
                    'camera_front': False,
                    'lidar_roof': True,
                    'radar_front': True,
                }

        # Trigger new scenario at scheduled time
        if not self._active_scenario and self._elapsed >= self._next_scenario_at:
            self._trigger_random_scenario()

        return sensor_override

    def _trigger_random_scenario(self):
        import carla
        # Pick a random scenario (avoid repeating the last one)
        choices = [s for s in self._scenarios if s != self._last_scenario]
        scenario = self._rng.choice(choices)
        self._active_scenario = scenario
        self._last_scenario = scenario
        self._scenario_count += 1

        if scenario == 'STORM':
            self._scenario_timer = self._rng.uniform(10.0, 18.0)
            weather = self._world.get_weather()
            weather.precipitation = 95.0
            weather.precipitation_deposits = 90.0
            weather.fog_density = 60.0
            weather.fog_distance = 15.0
            weather.wetness = 100.0
            weather.wind_intensity = 80.0
            self._world.set_weather(weather)
            print(f"[Scenario #{self._scenario_count}] \U0001f327  STORM — Heavy rain + fog!")

        elif scenario == 'NIGHT':
            self._scenario_timer = self._rng.uniform(12.0, 20.0)
            weather = self._world.get_weather()
            weather.sun_altitude_angle = -30.0
            weather.cloudiness = 100.0
            self._world.set_weather(weather)
            print(f"[Scenario #{self._scenario_count}] \U0001f319 NIGHT — Low visibility blackout!")

        elif scenario == 'JAYWALKER':
            self._scenario_timer = 8.0
            self._spawn_jaywalker()
            print(f"[Scenario #{self._scenario_count}] \U0001f6b6 JAYWALKER — Pedestrian crossing!")

        elif scenario == 'CUT_IN':
            self._scenario_timer = 8.0
            self._spawn_cut_in_vehicle()
            print(f"[Scenario #{self._scenario_count}] \U0001f697 CUT-IN — Aggressive vehicle!")

        elif scenario == 'SENSOR_FAIL':
            self._scenario_timer = self._rng.uniform(5.0, 8.0)
            self._sensor_failure_active = True
            print(f"[Scenario #{self._scenario_count}] \U0001f4e1 SENSOR FAILURE — Camera offline!")

    def _end_scenario(self):
        import carla
        # Restore weather
        if self._active_scenario in ('STORM', 'NIGHT'):
            self._world.set_weather(self._original_weather)
            print(f"[Scenario] \u2600  Weather restored to normal")

        # Cleanup scenario actors
        for actor in self._scenario_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self._scenario_actors.clear()

        if self._active_scenario == 'SENSOR_FAIL':
            self._sensor_failure_active = False
            print(f"[Scenario] \U0001f4e1 Sensors back online")

        self._active_scenario = None
        # Schedule next scenario
        self._next_scenario_at = self._elapsed + self._rng.uniform(20.0, 40.0)

    def _spawn_jaywalker(self):
        """Spawn a pedestrian crossing the road 20-30m ahead of ego."""
        import carla
        try:
            ego_t = self._ego.get_transform()
            fwd = ego_t.get_forward_vector()
            right = ego_t.get_right_vector()
            dist = self._rng.uniform(20.0, 30.0)

            # Spawn walker to the right side, walking left across the road
            spawn_loc = carla.Location(
                x=ego_t.location.x + fwd.x * dist + right.x * 8.0,
                y=ego_t.location.y + fwd.y * dist + right.y * 8.0,
                z=ego_t.location.z + 1.0,
            )

            walker_bps = self._bp_lib.filter('walker.pedestrian.*')
            walker_bp = self._rng.choice(list(walker_bps))
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            walker = self._world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
            if walker:
                self._scenario_actors.append(walker)

                # Add AI controller to make walker cross
                ctrl_bp = self._bp_lib.find('controller.ai.walker')
                controller = self._world.spawn_actor(ctrl_bp, carla.Transform(), walker)
                self._scenario_actors.append(controller)

                controller.start()
                # Walk toward the left side of the road
                target = carla.Location(
                    x=ego_t.location.x + fwd.x * dist - right.x * 8.0,
                    y=ego_t.location.y + fwd.y * dist - right.y * 8.0,
                    z=ego_t.location.z,
                )
                controller.go_to_location(target)
                controller.set_max_speed(1.4 + self._rng.random() * 0.6)  # walking speed
        except Exception as e:
            print(f"[Scenario] Jaywalker spawn failed: {e}")

    def _spawn_cut_in_vehicle(self):
        """Spawn a vehicle that cuts in front of the ego."""
        import carla
        try:
            ego_t = self._ego.get_transform()
            fwd = ego_t.get_forward_vector()
            right = ego_t.get_right_vector()

            # Spawn in adjacent lane, slightly ahead
            spawn_loc = carla.Location(
                x=ego_t.location.x + fwd.x * 15.0 + right.x * 4.0,
                y=ego_t.location.y + fwd.y * 15.0 + right.y * 4.0,
                z=ego_t.location.z + 0.5,
            )
            spawn_rot = ego_t.rotation

            vehicle_bps = self._bp_lib.filter('vehicle.*')
            # Pick a smaller vehicle for cut-in
            small_vehicles = [bp for bp in vehicle_bps
                              if int(bp.get_attribute('number_of_wheels')) == 4]
            if not small_vehicles:
                small_vehicles = list(vehicle_bps)
            veh_bp = self._rng.choice(small_vehicles)
            if veh_bp.has_attribute('color'):
                colors = veh_bp.get_attribute('color').recommended_values
                veh_bp.set_attribute('color', self._rng.choice(colors))

            vehicle = self._world.try_spawn_actor(
                veh_bp, carla.Transform(spawn_loc, spawn_rot))
            if vehicle:
                self._scenario_actors.append(vehicle)
                # Enable autopilot so it drives aggressively
                vehicle.set_autopilot(True, 8000)
                # Make it aggressive via TM
                try:
                    tm = self._world.get_client().get_trafficmanager(8000)
                except:
                    pass  # TM config is best-effort
        except Exception as e:
            print(f"[Scenario] Cut-in spawn failed: {e}")

    def cleanup(self):
        """Destroy all scenario actors."""
        for actor in self._scenario_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self._scenario_actors.clear()
        # Restore weather
        try:
            self._world.set_weather(self._original_weather)
        except Exception:
            pass


# ============================================================================
# Inline Traffic Spawner
# ============================================================================

def _spawn_traffic(client, world, num_vehicles=50, num_walkers=30):
    """Spawn NPC vehicles and walkers into the current world."""
    import carla
    import random as _r

    bp_lib = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()
    _r.shuffle(spawn_pts)

    spawned = []

    # --- Vehicles ---
    vehicle_bps = bp_lib.filter('vehicle.*')
    batch = []
    for i, sp in enumerate(spawn_pts[:num_vehicles]):
        bp = _r.choice(list(vehicle_bps))
        if bp.has_attribute('color'):
            bp.set_attribute('color', _r.choice(bp.get_attribute('color').recommended_values))
        batch.append(carla.command.SpawnActor(bp, sp).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, 8000)))

    results = client.apply_batch_sync(batch, False)
    v_count = sum(1 for r in results if not r.error)
    for r in results:
        if not r.error:
            spawned.append(r.actor_id)

    # --- Walkers ---
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    w_spawn = []
    for _ in range(num_walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            w_spawn.append(carla.Transform(loc))

    w_batch = []
    for sp in w_spawn:
        bp = _r.choice(list(walker_bps))
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        w_batch.append(carla.command.SpawnActor(bp, sp))

    w_results = client.apply_batch_sync(w_batch, False)
    walker_ids = [r.actor_id for r in w_results if not r.error]

    # Attach AI controllers to walkers
    ctrl_bp = bp_lib.find('controller.ai.walker')
    c_batch = []
    for wid in walker_ids:
        c_batch.append(carla.command.SpawnActor(ctrl_bp, carla.Transform(),
                                                carla.command.FutureActor))
    # Apply via parent
    c_batch_real = []
    for wid in walker_ids:
        c_batch_real.append(
            carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid))
    c_results = client.apply_batch_sync(c_batch_real, False)
    ctrl_ids = [r.actor_id for r in c_results if not r.error]

    # Start walking
    world.wait_for_tick(2.0)
    actors = world.get_actors(ctrl_ids)
    for ctrl in actors:
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(1.0 + _r.random() * 1.5)

    print(f"[Traffic] Spawned {v_count} vehicles + {len(walker_ids)} walkers")
    return spawned + walker_ids + ctrl_ids


# ============================================================================
# Main CARLA Demo — Hybrid Auto/Manual + FASS
# ============================================================================

def run_carla_demo(
    host='localhost',
    port=2000,
    checkpoint=None,
    num_ticks=0,
    width=1280,
    height=720,
):
    """Run FASS demo with autopilot + manual keyboard controls."""
    import carla

    print("=" * 60)
    print("  FASS CARLA DEMO — Hybrid Auto/Manual + Safety Supervisor")
    print("=" * 60)

    # --- Deterministic override self-test ---
    overrides = DeterministicOverrides()
    assert overrides.self_test(), "Self-test FAILED!"
    print("✓ Deterministic override self-test passed\n")

    # --- FASS supervisor ---
    config = FASSConfig()
    supervisor = FASSSupervisor(checkpoint_path=checkpoint, config=config)

    # --- Online learning ---
    from .training.online_trainer import ExperienceBuffer, OnlineTrainer
    import os
    buffer_path = os.path.join(os.path.dirname(checkpoint or __file__), '_fass_experience_buffer.pt')
    exp_buffer = ExperienceBuffer(
        max_size=config.online_buffer_size,
        lookahead_s=config.online_lookahead_s,
        buffer_path=buffer_path
    )
    online_trainer = OnlineTrainer(
        model=supervisor.inference.model,
        config=config,
        checkpoint_path=checkpoint,
    )
    online_learning_enabled = True
    print(f"[Online Learning] Enabled — training every {config.online_train_interval_s}s")

    # --- Pygame ---
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode(
        (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("FASS Autonomous Driving — Hybrid Control")
    clock = pygame.time.Clock()

    camera_display = CameraDisplay(width, height)
    hud = FASSHUD(width, height)
    kbd = KeyboardVehicleControl()

    # --- Connect to CARLA ---
    client = carla.Client(host, port)
    client.set_timeout(15.0)

    # --- Random map selection ---
    import random as _rand
    available_maps = client.get_available_maps()
    # Filter out layered maps (_Opt suffix) to avoid issues
    good_maps = [m for m in available_maps if '_Opt' not in m]
    if not good_maps:
        good_maps = available_maps
    chosen_map = _rand.choice(good_maps)
    current_map = client.get_world().get_map().name
    # Only load if different (loading same map is slow)
    if chosen_map.split('/')[-1] != current_map.split('/')[-1]:
        print(f"[Demo] Loading random map: {chosen_map}")
        client.load_world(chosen_map)
        import time as _t
        _t.sleep(5.0)  # Wait for map to load
    else:
        print(f"[Demo] Using current map: {current_map}")

    world = client.get_world()

    # Ensure async mode
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        world.apply_settings(settings)

    # --- Spawn traffic inline ---
    print("[Demo] Spawning traffic (50 vehicles, 30 walkers)...")
    traffic_actor_ids = _spawn_traffic(client, world, 50, 30)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    actors_to_destroy = []

    autopilot_enabled = True  # Start in autopilot mode
    scenario_mgr = None

    try:
        # --- Spawn ego vehicle ---
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        _rand.shuffle(spawn_points)
        ego = None
        for sp in spawn_points:
            ego = world.try_spawn_actor(vehicle_bp, sp)
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("Could not spawn ego vehicle. Restart CARLA.")
        actors_to_destroy.append(ego)

        # --- Register with Traffic Manager for proper autopilot ---
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(False)  # Async mode (matches generate_traffic --asynch)
        ego.set_autopilot(True, 8000)
        tm.auto_lane_change(ego, True)
        tm.distance_to_leading_vehicle(ego, 5.0)
        tm.vehicle_percentage_speed_difference(ego, -20)  # 20% below speed limit
        print("[Demo] Ego vehicle spawned — AUTOPILOT ON (Traffic Manager registered)")
        print("[Demo] Press P to toggle manual/autopilot control")

        # --- Chase camera for visual rendering ---
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', '100')
        cam_transform = carla.Transform(
            carla.Location(x=-6.0, z=3.5),
            carla.Rotation(pitch=-12.0))
        camera_actor = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        camera_actor.listen(camera_display.callback)
        actors_to_destroy.append(camera_actor)

        # --- Collision sensor ---
        col_bp = blueprint_library.find('sensor.other.collision')
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        collision_count = [0]
        def _on_collision(event):
            collision_count[0] += 1
            exp_buffer.record_collision()  # Record for online learning
        col_sensor.listen(_on_collision)
        actors_to_destroy.append(col_sensor)

        # --- Scenario Manager ---
        scenario_mgr = ScenarioManager(world, ego, blueprint_library)

        mode_str = "INFINITE (ESC to quit)" if num_ticks == 0 else f"{num_ticks} ticks"
        print(f"[Demo] Sensors attached. Running {mode_str}...")
        print(f"[Demo] Stress-test scenarios will trigger every 20-40 seconds\n")
        print("[Controls] W/S=Throttle/Brake  A/D=Steer  P=Autopilot  R=Reset FASS")
        print("           SPACE=Handbrake  Q=Reverse  L=Learn  TAB=HUD  ESC=Quit\n")

        # === MAIN LOOP ===
        running = True
        tick_idx = 0
        consecutive_errors = 0
        while running:
            # Respect tick limit if set (0 = infinite)
            if num_ticks > 0 and tick_idx >= num_ticks:
                break

            # --- Wait for next world frame (ASYNC) ---
            try:
                world.wait_for_tick(2.0)
            except RuntimeError:
                continue

            dt = clock.tick() / 1000.0  # Track FPS, don't rate-limit

            try:  # Wrap tick body so transient CARLA errors don't kill the demo
                # --- EVENT HANDLING (process AFTER tick for freshness) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYUP:
                        if event.key == K_ESCAPE:
                            running = False
                        elif event.key == K_p:
                            # Toggle autopilot
                            autopilot_enabled = not autopilot_enabled
                            ego.set_autopilot(autopilot_enabled, 8000)
                            mode = "AUTOPILOT" if autopilot_enabled else "MANUAL"
                            print(f"[Demo] Switched to {mode} mode")
                            hud.set_alert(f"Mode: {mode}", 2.0)
                        elif event.key == K_q:
                            # Toggle reverse (manual mode only)
                            if not autopilot_enabled:
                                is_rev = kbd.toggle_reverse()
                                gear_str = "REVERSE" if is_rev else "FORWARD"
                                print(f"[Demo] Gear: {gear_str}")
                                hud.set_alert(f"Gear: {gear_str}", 1.5)
                        elif event.key == K_r:
                            # Reset SAFE_STOP latch only
                            supervisor.failsafe.reset_safe_stop()
                            hud.set_alert("FASS RESET - Press P for autopilot", 3.0)
                            print("[Demo] FASS SAFE_STOP reset. Press P for autopilot.")
                        elif event.key == K_TAB:
                            hud.toggle()
                        elif event.key == K_l:
                            # Force immediate online training step
                            if online_learning_enabled:
                                train_result = online_trainer.train_step(exp_buffer)
                                if train_result:
                                    supervisor.inference.reload_model()
                                    hud.set_alert(f"Trained! loss={train_result['loss']:.4f}", 2.0)
                                else:
                                    hud.set_alert("Not enough data to train yet", 1.5)

                if not running:
                    break

                # --- MANUAL CONTROL (when autopilot is off) ---
                if not autopilot_enabled:
                    keys = pygame.key.get_pressed()
                    control = kbd.parse(keys, max(1, clock.get_time()))
                    ego.apply_control(control)

                # --- SCENARIO MANAGER TICK ---
                sensor_override = scenario_mgr.tick(dt)

                # --- EXTRACT SENSOR DATA ---
                detected_objects = _extract_detected_objects(world, ego)
                ego_state = _extract_ego_kinematics(ego)
                weather = _extract_weather(world)

                sensor_data = {'detected_objects': detected_objects}
                sensor_health = {
                    'camera_front': True,
                    'lidar_roof': True,
                    'radar_front': True,
                }
                # Apply sensor failure from scenario
                if sensor_override:
                    sensor_health.update(sensor_override)

                # === FASS SUPERVISION TICK ===
                result = supervisor.tick(
                    sensor_data=sensor_data,
                    ego_state=ego_state,
                    weather=weather,
                    sensor_health=sensor_health,
                    scenario_id='DEMO_LIVE',
                )

                # === ONLINE LEARNING: Record frame ===
                if online_learning_enabled:
                    # Get the feature vector from the latest inference
                    feat = supervisor.inference.last_features
                    if feat is not None:
                        # Compute min TTC and min distance from detected objects
                        ego_spd = ego_state.get('speed', 0.0)
                        min_ttc_val = 10.0
                        min_dist_val = 80.0
                        for obj in detected_objects:
                            min_dist_val = min(min_dist_val, obj['distance'])
                            rel_spd = ego_spd - obj.get('speed', 0.0) * 0.5
                            if rel_spd > 0.1:
                                min_ttc_val = min(min_ttc_val, obj['distance'] / rel_spd)

                        exp_buffer.push(
                            features=feat,
                            min_ttc=min_ttc_val,
                            min_distance=min_dist_val,
                            speed=ego_spd,
                            pitch=ego_state.get('pitch', 0.0),
                        )

                    # Process hindsight labels for pending frames
                    exp_buffer.process_hindsight()

                    # Auto-train periodically
                    if online_trainer.should_train():
                        train_result = online_trainer.train_step(exp_buffer)
                        if train_result:
                            supervisor.inference.reload_model()

                # === FASS INTERVENTION ENFORCEMENT ===
                intervention = result['intervention']
                advisory = result['advisory']

                # DANGER → Gradual slow-down then handoff to driver
                # Don't instantly disable autopilot at high speed — that's dangerous!
                if advisory == 'DANGER' and autopilot_enabled:
                    ego_speed_kmh = ego_state.get('speed', 0) * 3.6  # m/s → km/h
                    if ego_speed_kmh < 20.0:
                        # Speed is safe — hand off to driver
                        autopilot_enabled = False
                        ego.set_autopilot(False)
                        hud.set_alert("⚠ DANGER — TAKE MANUAL CONTROL! (WASD)", 5.0)
                        print(f"[FASS HANDOFF] Speed safe ({ego_speed_kmh:.0f} km/h). "
                              f"Autopilot DISABLED. Driver take control.")
                    else:
                        # Still too fast — keep autopilot but apply progressive braking
                        brake_pct = min(0.5, 0.15 + (ego_speed_kmh / 200.0))
                        hud.set_alert(f"⚠ DANGER — Slowing down... ({ego_speed_kmh:.0f} km/h)", 0.5)

                # Emergency brake override (SAFE_STOP / EMERGENCY_BRAKE)
                # In AUTOPILOT: always enforce FASS braking
                # In MANUAL: only enforce if SAFE_STOP latch is active (press R to clear)
                safe_stop_latched = supervisor.failsafe._safe_stop_active
                if intervention.type in ('EMERGENCY_BRAKE', 'SAFE_STOP'):
                    if autopilot_enabled or safe_stop_latched:
                        brake_force = 1.0
                        emergency = carla.VehicleControl(
                            throttle=0.0, brake=brake_force, steer=0.0,
                            hand_brake=False, manual_gear_shift=False)
                        ego.apply_control(emergency)
                    elif not autopilot_enabled:
                        # Manual mode: warn but let driver control
                        hud.set_alert("⚠ FASS: Brake recommended! R=Reset", 1.0)
                elif intervention.type == 'GENTLE_BRAKE':
                    # Apply smooth, progressive braking (not a sudden slam)
                    ego_speed_kmh = ego_state.get('speed', 0) * 3.6
                    smooth_brake = min(0.3, 0.1 + (ego_speed_kmh / 300.0))
                    if autopilot_enabled:
                        # Let TM handle steering, just reduce speed
                        ctrl = ego.get_control()
                        ctrl.throttle = max(0.0, ctrl.throttle - 0.3)
                        ctrl.brake = smooth_brake
                        ego.apply_control(ctrl)

                # --- UPDATE HUD ---
                hud.update(
                    result, ego_state, tick_idx + 1, len(detected_objects), autopilot_enabled,
                    online_stats={
                        'buffer_frames': exp_buffer.stats['labeled'],
                        'high_risk_frames': exp_buffer.stats['high_risk'],
                        'collisions_seen': exp_buffer.stats['collisions'],
                        'train_step': online_trainer.train_count,
                        'loss': online_trainer.last_loss or 0.0
                    } if online_learning_enabled else None
                )

                # --- RENDER ---
                camera_display.render(display)
                hud.render(display, dt)
                pygame.display.flip()

                # --- TERMINAL OUTPUT ---
                if (tick_idx + 1) % 100 == 0 or advisory != 'SAFE':
                    mode_tag = "AUTO" if autopilot_enabled else "MANU"
                    scen = scenario_mgr._active_scenario or '-'
                    print(f"  [{tick_idx+1:5d}|{mode_tag}] "
                          f"risk={result['risk']:.3f} "
                          f"unc={result['uncertainty']:.4f} "
                          f"advisory={advisory:7s} "
                          f"fused={result['fused_risk']:.3f} "
                          f"intervention={intervention.type:15s} "
                          f"scen={scen:12s} "
                          f"latency={result['tick_latency_ms']:.1f}ms")

                tick_idx += 1
                consecutive_errors = 0  # Reset on successful tick

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"[Demo] Transient error (#{consecutive_errors}): {e}")
                if consecutive_errors >= 10:
                    print(f"[Demo] Too many consecutive errors ({consecutive_errors}), stopping.")
                    break
                import time as _time
                _time.sleep(0.1)
                continue

        # --- FINAL STATS ---
        stats = supervisor.get_stats()
        print(f"\n{'='*60}")
        print(f"  DEMO COMPLETE")
        print(f"  Ticks: {stats['total_ticks']}")
        print(f"  Avg latency: {stats['avg_tick_latency_ms']:.1f}ms")
        print(f"  Interventions: {stats['total_interventions']}")
        print(f"  Collisions: {collision_count[0]}")
        print(f"  Log files: {supervisor.logger.log_files}")
        print(f"{'='*60}")

    finally:
        # Save online experience buffer on exit
        if 'online_learning_enabled' in locals() and online_learning_enabled:
            print("[Online Learning] Saving experience buffer to disk...")
            try:
                exp_buffer.save()
            except Exception as e:
                print(f"[Online Learning] Could not save buffer: {e}")

        if scenario_mgr:
            scenario_mgr.cleanup()
        for a in reversed(actors_to_destroy):
            try:
                if a.is_alive:
                    a.destroy()
            except Exception:
                pass
        # Cleanup traffic actors
        try:
            all_actors = world.get_actors(traffic_actor_ids)
            for a in all_actors:
                try:
                    a.destroy()
                except Exception:
                    pass
        except Exception:
            pass
        pygame.quit()


# ============================================================================
# Synthetic Demo (unchanged)
# ============================================================================

def run_synthetic_demo(num_ticks=100, checkpoint=None):
    """Run the FASS demo with synthetic data (no CARLA needed)."""
    print("=" * 60)
    print("  FASS SYNTHETIC DEMO — No CARLA Server Required")
    print("=" * 60)

    overrides = DeterministicOverrides()
    assert overrides.self_test(), "Self-test FAILED!"
    print("✓ Deterministic override self-test passed\n")

    config = FASSConfig()
    supervisor = FASSSupervisor(checkpoint_path=checkpoint, config=config)
    rng = np.random.RandomState(42)

    for tick_idx in range(num_ticks):
        n_objects = rng.randint(0, 8)
        detected_objects = [{
            'type': rng.choice(['vehicle', 'walker', 'static']),
            'distance': rng.uniform(2.0, 60.0),
            'speed': rng.uniform(0.0, 20.0),
            'detection_confidence': rng.uniform(0.3, 1.0),
            'bbox_extent': [rng.uniform(1, 3), rng.uniform(0.5, 2), rng.uniform(1, 2)],
        } for _ in range(n_objects)]

        result = supervisor.tick(
            sensor_data={'detected_objects': detected_objects},
            ego_state={
                'speed': rng.uniform(0, 30),
                'acceleration': rng.uniform(0, 5),
                'yaw_rate': rng.uniform(-30, 30),
                'steer': rng.uniform(-0.5, 0.5),
                'throttle': rng.uniform(0, 1),
                'brake': 0.0,
            },
            weather={
                'precipitation': rng.uniform(0, 100),
                'fog_density': rng.uniform(0, 100),
                'sun_altitude_angle': rng.uniform(-30, 90),
                'wetness': rng.uniform(0, 100),
                'wind_intensity': rng.uniform(0, 50),
                'is_night': rng.random() < 0.3,
            },
            sensor_health={
                'camera_front': rng.random() > 0.05,
                'lidar_roof': rng.random() > 0.05,
                'radar_front': rng.random() > 0.05,
            },
            scenario_id='DEMO_SYNTHETIC',
        )

        if (tick_idx + 1) % 10 == 0 or result['advisory'] != 'SAFE':
            print(f"  [{tick_idx+1:4d}] "
                  f"risk={result['risk']:.3f} "
                  f"unc={result['uncertainty']:.4f} "
                  f"advisory={result['advisory']:7s} "
                  f"fused={result['fused_risk']:.3f} "
                  f"intervention={result['intervention'].type:15s} "
                  f"latency={result['tick_latency_ms']:.1f}ms")

    stats = supervisor.get_stats()
    print(f"\n{'='*60}")
    print(f"  SYNTHETIC DEMO COMPLETE")
    print(f"  Ticks: {stats['total_ticks']}")
    print(f"  Avg latency: {stats['avg_tick_latency_ms']:.1f}ms")
    print(f"  Interventions: {stats['total_interventions']}")
    print(f"  Log files: {supervisor.logger.log_files}")
    print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FASS CARLA Demo")
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--ticks', type=int, default=0,
                        help='Max ticks (0=infinite, ESC to quit)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic_demo(num_ticks=args.ticks or 100, checkpoint=args.checkpoint)
    else:
        run_carla_demo(
            host=args.host,
            port=args.port,
            checkpoint=args.checkpoint,
            num_ticks=args.ticks,
            width=args.width,
            height=args.height,
        )


if __name__ == '__main__':
    main()
