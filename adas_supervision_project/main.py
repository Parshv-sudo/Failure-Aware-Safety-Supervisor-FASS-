#!/usr/bin/env python
"""
ADAS Level 2 Supervision & Takeover Management Framework
=========================================================

State-machine-driven simulation loop:

    INITIALIZING → RUNNING → TAKEOVER_ACTIVE → FALLBACK_BRAKING → TERMINATED

Per-tick pipeline::

    tick → perceive → assess risk → takeover staging → log

Run with::

    python main.py [--config path/to/config.yaml]

Requires a running CARLA server (``CarlaUE4.exe``).
"""

import argparse
import logging as stdlib_logging
import os
import sys
import time
import uuid

import glob
# ---------------------------------------------------------------------------
# Dynamically add the CARLA egg / whl to sys.path so ``import carla`` works
# regardless of where this script is launched from.
# ---------------------------------------------------------------------------
_CARLA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
_EGG_PATTERN = os.path.join(
    _CARLA_ROOT,
    "WindowsNoEditor",
    "PythonAPI",
    "carla",
    "dist",
    "carla-*%d.%d-%s.egg"
    % (
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64",
    ),
)
try:
    sys.path.append(glob.glob(_EGG_PATTERN)[0])
except IndexError:
    pass

import carla

# ---------------------------------------------------------------------------
# Resolve project root and ensure it is on ``sys.path``
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import Config
from core.simulation_state import SimulationState, SimulationStateMachine
from carla_interface.carla_client import CarlaClient
from carla_interface.vehicle_manager import VehicleManager
from carla_interface.sensor_manager import SensorManager
from maps.map_manager import MapManager
from perception.object_detector import ObjectDetector
from perception.confidence_estimator import ConfidenceEstimator
from perception.ttc_calculator import TTCCalculator
from supervision.risk_assessor import RiskAssessor, RuleBasedRiskModel
from supervision.fass_integrated_risk_model import FASSIntegratedRiskModel
from supervision.odd_monitor import ODDMonitor
from supervision.takeover_manager import TakeoverManager, TakeoverStage
from driver_interface.alert_manager import AlertManager
from driver_interface.control_transition import ControlTransition
from driver_interface.driver_model import DriverModel
from flight_recorder.event_logger import EventLogger
from flight_recorder.blackbox_recorder import BlackboxRecorder
from metrics.metrics_collector import MetricsCollector
from scenarios.highway_cut_in import HighwayCutInScenario
from scenarios.empty_scenario import EmptyScenario
from visualizer import PygameVisualizer

logger = stdlib_logging.getLogger("adas")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ADAS Level 2 Supervision Framework"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "simulation_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Override the CARLA map/town to load",
    )
    return parser.parse_args()


def setup_logging(level_name: str = "INFO"):
    """Configure the root logger."""
    level = getattr(stdlib_logging, level_name.upper(), stdlib_logging.INFO)
    stdlib_logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    """State-machine-driven main loop."""
    args = parse_args()
    cfg = Config.instance(args.config)

    setup_logging(cfg.get("logging", "console_log_level", default="INFO"))
    logger.info("Config loaded  (hash=%s…)", cfg.config_hash[:12])

    simulation_id = uuid.uuid4().hex[:16]
    logger.info("Simulation ID: %s", simulation_id)

    # ==================================================================
    # State machine
    # ==================================================================
    sm = SimulationStateMachine(SimulationState.INITIALIZING)

    # ==================================================================
    # INITIALIZING
    # ==================================================================
    carla_cfg = cfg.get("carla", default={})
    seeds_cfg = cfg.get("seeds", default={})

    client = CarlaClient(
        host=carla_cfg.get("host", "localhost"),
        port=carla_cfg.get("port", 2000),
        timeout=carla_cfg.get("timeout", 10.0),
        sync=carla_cfg.get("synchronous_mode", True),
        fixed_delta=carla_cfg.get("fixed_delta_seconds", 0.05),
        traffic_seed=seeds_cfg.get("traffic_seed", 42),
    )

    # Map
    map_cfg = cfg.get("map", default={})
    if args.map:
        if args.map.endswith('.osm') or args.map.endswith('.xodr'):
            map_cfg["type"] = "real_world"
            map_cfg["source"] = args.map
        else:
            map_cfg["name"] = args.map
        
    def _on_world_loaded(w):
        if carla_cfg.get("synchronous_mode", True):
            settings = w.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = carla_cfg.get("fixed_delta_seconds", 0.05)
            w.apply_settings(settings)
            
    map_mgr = MapManager(client.get_client(), map_cfg, on_world_loaded=_on_world_loaded)
    world = map_mgr.load()
    client.world = world

    # Weather
    weather = cfg.get("scenario", "weather", default="ClearNoon")
    client.set_weather(weather)

    # Ego vehicle
    veh_cfg = cfg.get("vehicle", default={})
    veh_mgr = VehicleManager(
        world,
        blueprint_filter=veh_cfg.get("blueprint_filter", "vehicle.tesla.model3"),
        spawn_index=veh_cfg.get("spawn_index", 0),
        random_seed=seeds_cfg.get("random_seed", 42),
    )
    ego = veh_mgr.spawn()
    veh_mgr.set_autopilot(True)

    # Smooth out Autopilot behavior (Native CARLA TM)
    tm = client.traffic_manager
    tm.auto_lane_change(ego, False)
    tm.distance_to_leading_vehicle(ego, 5.0)           # Increase following distance to prevent hard brakes
    tm.vehicle_percentage_speed_difference(ego, 15.0)  # Drive 15% below speed limit for smoother cornering
    tm.ignore_lights_percentage(ego, 0.0)
    tm.keep_right_rule_percentage(ego, 0.0)

    # Sensors
    sensor_mgr = SensorManager(world, ego, cfg.get("sensors", default={}))

    # Perception
    detector = ObjectDetector(
        world, ego,
        config=cfg.get("perception", default={}),
        random_seed=seeds_cfg.get("random_seed", 42),
    )
    conf_est = ConfidenceEstimator(
        decay_rate=cfg.get("perception", "confidence_decay_rate", default=0.02),
    )
    risk_cfg = cfg.get("risk", default={})
    ttc_calc = TTCCalculator(epsilon=risk_cfg.get("epsilon", 0.1))

    # Risk
    sup_cfg = cfg.get("supervision", default={})
    if sup_cfg.get("risk_model") == "fass_ml":
        logger.info("Using FASS ML Integrated Risk Model")
        checkpoint_path = os.path.join(PROJECT_ROOT, sup_cfg.get("fass_ml", {}).get("checkpoint_path", ""))
        model = FASSIntegratedRiskModel(checkpoint_path=checkpoint_path)
    else:
        logger.info("Using Rule-Based Risk Model")
        model = RuleBasedRiskModel(
            weights=risk_cfg.get("weights", {}),
            epsilon=risk_cfg.get("epsilon", 0.1),
            speed_max=risk_cfg.get("speed_max", 40.0),
        )
    risk_assessor = RiskAssessor(
        model=model,
        epsilon=risk_cfg.get("epsilon", 0.1),
        speed_max=risk_cfg.get("speed_max", 40.0),
    )

    # ODD
    odd_monitor = ODDMonitor(world, cfg.get("road_complexity", default={}))

    # Alert & control
    takeover_cfg = cfg.get("takeover", default={})
    alert_mgr = AlertManager(takeover_cfg.get("thresholds", {}))
    ctrl_trans = ControlTransition(
        speed_reduction_rate=takeover_cfg.get("speed_reduction_rate", 2.0)
    )

    # Driver model (overtrust / attention dynamics)
    driver_cfg = cfg.get("overtrust", default={})
    driver_model = DriverModel(
        initial_attention=driver_cfg.get("initial_attention", 0.95),
        decay_rate=driver_cfg.get("decay_rate_per_tick", 0.005),
        recovery_on_alert=driver_cfg.get("recovery_on_alert", 0.3),
        attention_epsilon=driver_cfg.get("attention_epsilon", 0.05),
        random_seed=seeds_cfg.get("random_seed", 42),
    )

    # Takeover manager
    takeover_mgr = TakeoverManager(
        state_machine=sm,
        alert_manager=alert_mgr,
        control_transition=ctrl_trans,
        thresholds=takeover_cfg.get("thresholds", {}),
        response_config=cfg.get("driver_response", default={}),
        hazard_config=cfg.get("driver_response", "hazard_coupling", default={}),
        random_seed=seeds_cfg.get("random_seed", 42),
    )

    # Logging
    log_cfg = cfg.get("logging", default={})
    log_dir = os.path.join(PROJECT_ROOT, log_cfg.get("output_dir", "logs"))

    event_logger = EventLogger(log_dir, simulation_id)
    blackbox = BlackboxRecorder(
        output_dir=log_dir,
        simulation_id=simulation_id,
        config_hash=cfg.config_hash,
        config_snapshot=cfg.data,
        scenario_name=cfg.get("scenario", "active", default="highway_cut_in"),
        log_every_n_ticks=log_cfg.get("log_every_n_ticks", 1),
    )

    # Metrics collector
    metrics_cfg = cfg.get("metrics", default={})
    metrics_collector = MetricsCollector(
        simulation_id=simulation_id,
        output_dir=log_dir,
        scenario_name=cfg.get("scenario", "active", default="highway_cut_in"),
        seed_values=seeds_cfg,
        map_type=cfg.get("map", "type", default="default"),
        hazard_ttc_threshold=metrics_cfg.get("hazard_ttc_threshold", 2.0),
        consecutive_tick_requirement=metrics_cfg.get("consecutive_tick_requirement", 5),
        safety_ttc_threshold=metrics_cfg.get("safety_ttc_threshold", 4.0),
    )

    # Register state-change logging
    def _log_state_change(prev, new):
        event_logger.log("state_change", {
            "from": prev.name, "to": new.name
        })

    for s in SimulationState:
        sm.on_enter(s, _log_state_change)

    # Scenario
    scenario_cfg = cfg.get("scenario", default={})
    scenario_type = scenario_cfg.get("type", "deterministic")
    
    if scenario_type == "empty":
        scenario = EmptyScenario(
            client=client.get_client(),
            world=world,
            ego_vehicle=ego,
            config=scenario_cfg,
            random_seed=seeds_cfg.get("random_seed", 42),
        )

    else:
        scenario = HighwayCutInScenario(
            client=client.get_client(),
            world=world,
            ego_vehicle=ego,
            config=scenario_cfg,
            random_seed=seeds_cfg.get("random_seed", 42),
        )
        
    # Tick the CARLA world once to let the physics engine move the newly spawned Ego
    # vehicle to its actual map coordinates, instead of the default Location(0,0,0)
    world.tick()
    scenario.setup()

    # Visualizer
    enable_vis = cfg.get("visualization", "enabled", default=True)
    vis = None
    if enable_vis:
        logger.info("Starting PyGame Visualizer")
        vis = PygameVisualizer(width=1280, height=720)
        vis.attach_camera(world, ego)

    duration = scenario_cfg.get("duration_seconds", 60)
    delta_t = carla_cfg.get("fixed_delta_seconds", 0.05)

    # ==================================================================
    # Transition → RUNNING
    # ==================================================================
    sm.transition(SimulationState.RUNNING)

    logger.info("=" * 60)
    if duration > 0:
        logger.info("  SIMULATION RUNNING  —  %d s  @  %.0f Hz", duration, 1 / delta_t)
    else:
        logger.info("  SIMULATION RUNNING  —  INFINITE  @  %.0f Hz (Press ESC to stop)", 1 / delta_t)
    logger.info("=" * 60)

    tick_count = 0
    grace_ticks = 40  # 2s startup grace to avoid false lane invasion on spawn
    start_time = time.time()
    try:
        while True:
            # 1 — determine simulated time
            elapsed = tick_count * delta_t
            
            if duration > 0 and elapsed >= duration:
                logger.info("Reached maximum simulation duration (%.1fs). terminating.", duration)
                sm.transition(SimulationState.TERMINATED)
                break

            # Inform CARLA C++ Server to process the tick
            client.tick()
            tick_count += 1
            
            # 2 — scenario scripted behaviour
            scenario.tick(elapsed)

            # 3 — perception
            detections = detector.detect()
            detections = conf_est.update(detections)
            ttcs = ttc_calc.compute(detections)
            min_ttc = min(ttcs) if ttcs else float("inf")

            # 4 — ODD
            ego_loc = veh_mgr.get_location()
            ego_speed = veh_mgr.get_speed()
            road_complexity = odd_monitor.compute_complexity(ego_loc, ego_speed)

            # 5 — risk assessment
            risk_out = risk_assessor.assess(
                detections=detections, 
                ttcs=ttcs, 
                ego_speed=ego_speed, 
                road_complexity=road_complexity,
                ego_vehicle=veh_mgr.vehicle
            )

            # 5b — deterministic off-road / collision safety fallbacks
            current_cols = len(sensor_mgr.get_collision_history())
            _prev_cols = getattr(sensor_mgr, '_tracked_col_count', 0)
            collision_this_tick = current_cols > _prev_cols and tick_count > 5
            if collision_this_tick:
                sensor_mgr._tracked_col_count = current_cols
                risk_out.risk = 1.0
                risk_out.components["deterministic_collision"] = 1.0
            else:
                # Skip lane invasion checks during startup grace period
                if grace_ticks > 0:
                    grace_ticks -= 1
                else:
                    li_event = sensor_mgr.get_latest("lane_invasion")
                    if li_event and li_event.frame > getattr(sensor_mgr, '_last_li_frame', -1):
                        sensor_mgr._last_li_frame = li_event.frame
                        # Sustain lane invasion penalty for 30 ticks (1.5s) to allow TakeoverManager to escalate
                        sensor_mgr._li_ticks_remaining = 30 
                        
                    if getattr(sensor_mgr, '_li_ticks_remaining', 0) > 0:
                        sensor_mgr._li_ticks_remaining -= 1
                        risk_out.risk = max(risk_out.risk, 0.65)
                        risk_out.components["deterministic_lane_invasion"] = 0.65

            # 6 — driver model (attention dynamics)
            alert_active = takeover_mgr.stage > 0
            driver_model.tick(alert_active=alert_active)

            # 7 — takeover staging (state-machine-driven)
            takeover_stage = takeover_mgr.update(
                risk=risk_out.risk,
                ego_speed=ego_speed,
                road_complexity=road_complexity,
                vehicle=ego,
                delta_t=delta_t,
                tick=tick_count,
            )

            # 7.5 — Smooth Native Autopilot (EMA Filter)
            # If ADAS is not actively braking, Autopilot is driving.
            # We intercept Autopilot's raw erratic control, smooth it, and forcefully apply it.
            if takeover_stage < TakeoverStage.SPEED_REDUCTION.value:
                raw_control = veh_mgr.get_control()
                smoothed_control = veh_mgr.smooth_control(raw_control, alpha_steer=0.15, alpha_pedal=0.10)
                veh_mgr.apply_control(smoothed_control)
            else:
                # ADAS is currently controlling the car. Sync the filter so it doesn't 
                # jerk when Autopilot re-assumes control later.
                veh_mgr.reset_smoothing(veh_mgr.get_control())

            # 8 — metrics ingestion
            metrics_collector.record_tick(
                min_ttc=min_ttc,
                risk=risk_out.risk,
                takeover_active=alert_active,
                collision_this_tick=collision_this_tick,
                response_time=takeover_mgr.response_time,
                delta_t=delta_t,
            )

            # 9 — GUI update
            if vis:
                vis.update_telemetry(
                    risk=risk_out.risk,
                    takeover_stage=takeover_stage,
                    speed=ego_speed,
                    ttc=min_ttc,
                    confidence=1.0 - risk_out.components.get("confidence_term", 0),
                    detections=len(detections),
                    run_state=sm.state.name,
                    control=veh_mgr.get_control()
                )
                keep_running = vis.tick()
                if not keep_running:
                    logger.info("User closed visualizer.")
                    sm.transition(SimulationState.TERMINATED)
                    break

            # 9 — console output (every 20 ticks ≈ 1 s)
            if tick_count % 20 == 0:
                logger.info(
                    "t=%.1fs | state=%s | stage=%d | attn=%.2f | speed=%.1f | "
                    "risk=%.3f | TTC=%.2f | complexity=%.2f | det=%d",
                    elapsed, sm.state.name, takeover_stage,
                    driver_model.attention, ego_speed,
                    risk_out.risk, min_ttc,
                    road_complexity, len(detections),
                )

            # 10 — collision event logging
            if collision_this_tick:
                last_col = sensor_mgr.get_latest("collision")
                if last_col:
                    event_logger.log("collision", {
                        "other_actor": str(last_col.other_actor.type_id)
                        if last_col.other_actor else "static",
                        "tick": tick_count,
                    })

            # 11 — blackbox
            ctrl = veh_mgr.get_control()
            pos = veh_mgr.get_location()
            blackbox.record_tick({
                "tick": tick_count,
                "elapsed_s": round(elapsed, 4),
                "sim_state": sm.state.name,
                "takeover_stage": int(takeover_stage),
                "position": {"x": round(pos.x, 3), "y": round(pos.y, 3), "z": round(pos.z, 3)},
                "speed": round(ego_speed, 4),
                "steering": round(ctrl.steer, 4),
                "throttle": round(ctrl.throttle, 4),
                "brake": round(ctrl.brake, 4),
                "detections": len(detections),
                "min_ttc": round(min_ttc, 4) if min_ttc != float("inf") else None,
                "risk": round(risk_out.risk, 6),
                "confidence_term": risk_out.components.get("confidence_term", 0),
                "ttc_term": risk_out.components.get("ttc_term", 0),
                "speed_term": risk_out.components.get("speed_term", 0),
                "complexity_term": risk_out.components.get("complexity_term", 0),
                "road_complexity": round(road_complexity, 4),
                "driver_response_time": takeover_mgr.response_time,
                "driver_attention": round(driver_model.attention, 4),
            })

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sm.transition(SimulationState.TERMINATED)

    finally:
        wall_time = time.time() - start_time
        logger.info("Simulation ended — %d ticks in %.1f s", tick_count, wall_time)
        logger.info("State history: %s", " → ".join(s.name for s in sm.history))

        # Finalize metrics
        metrics_summary = metrics_collector.finalize()
        logger.info(
            "Episode outcome: %s | collisions=%d | missed_hazards=%d",
            metrics_summary.get("episode_outcome"),
            metrics_summary.get("collision_count", 0),
            metrics_summary.get("missed_hazard_count", 0),
        )

        blackbox.close()
        event_logger.close()
        scenario.cleanup()
        if vis:
            vis.cleanup()
        sensor_mgr.destroy()
        veh_mgr.destroy()
        client.cleanup()

        logger.info("All resources released. Goodbye.")


if __name__ == "__main__":
    main()
