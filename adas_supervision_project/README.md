# ADAS Level 2 Supervision & Takeover Management Framework

> An academic-grade, modular Python framework that runs on top of **CARLA 0.9.15** to simulate, supervise, and analyze Level 2 ADAS driving with real-world map support, risk assessment, staged takeover management, and forensic-grade logging.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│  │   CARLA    │  │Perception│  │ Supervision│  │ Logging  │ │
│  │ Interface  │  │ Pipeline │  │  Pipeline  │  │ System   │ │
│  ├───────────┤  ├──────────┤  ├────────────┤  ├──────────┤ │
│  │CarlaClient│→ │ObjectDet │→ │RiskAssessor│→ │Blackbox  │ │
│  │VehicleMgr │  │ConfidEst │  │ODD Monitor │  │EventLog  │ │
│  │SensorMgr  │  │TTCCalc   │  │TakeoverMgr │  │ReplayMgr │ │
│  └───────────┘  └──────────┘  └────────────┘  └──────────┘ │
│  ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│  │   Maps    │  │ Scenarios│  │  Driver IF  │  │ Metrics  │ │
│  │ OSM→XODR │  │CutIn/Ped │  │AlertMgr    │  │Collector │ │
│  │MapManager │  │Curve/... │  │DriverModel │  │KPI/Stats │ │
│  └───────────┘  └──────────┘  └────────────┘  └──────────┘ │
└──────────────────────────────────────────────────────────────┘
                         ↕ TCP
                  ┌──────────────┐
                  │ CARLA Server │
                  │(CarlaUE4.exe)│
                  └──────────────┘
```

## Quick Start

### Prerequisites

- **CARLA 0.9.15** (WindowsNoEditor build) — already present at project root
- **Python 3.7+**
- **PyYAML** — `pip install pyyaml`

### Running

1. **Start CARLA server**:
   ```powershell
   .\WindowsNoEditor\CarlaUE4.exe
   ```

2. **Run the framework** (from the CARLA root):
   ```powershell
   cd adas_supervision_project
   python main.py
   ```

3. **With a custom config**:
   ```powershell
   python main.py --config path/to/my_config.yaml
   ```

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `carla_interface/` | CARLA connection, vehicle spawn, sensor attachment |
| `maps/` | OSM loading → OpenDRIVE conversion → world generation |
| `perception/` | Simulated object detection, confidence estimation, TTC |
| `supervision/` | Risk assessment (extensible interface), ODD monitoring, takeover |
| `driver_interface/` | Alert management, control transitions, driver model |
| `scenarios/` | Deterministic, replayable driving scenarios |
| `logging/` | Forensic blackbox, event logs, replay validation |
| `metrics/` | Episode-level KPIs and statistical summaries |
| `core/` | Simulation state machine |
| `utils/` | Config loader, math utilities |

---

## Real-World Map Loading

### How It Works

1. Place a `.osm` file in `maps/` (or reference any path in config).
2. Set config:
   ```yaml
   map:
     type: real_world
     source: "maps/sample_region.osm"
   ```
3. The pipeline: **OSM XML → `carla.Osm2Odr.convert()` → OpenDRIVE → `client.generate_opendrive_world()`**
4. If the OSM file is missing or conversion fails, the system falls back to `fallback_town` (e.g. `Town03`).

### Config Options

```yaml
map:
  osm_settings:
    lane_width: 6.0
    traffic_lights: true
    all_junctions_lights: false
    center_map: true
```

---

## Risk Assessment

### Formula (Normalized)

```
risk = w1·(1 − confidence)                        # [0, 1]
     + w2·(1/max(TTC, ε)) / (1/ε)                 # [0, 1]
     + w3·(speed / speed_max)                      # [0, 1]
     + w4·road_complexity                          # [0, 1]

Output: clamp(risk, 0, 1)
```

All 4 component terms are logged individually in the blackbox for post-hoc dominance analysis.

### Road Complexity (4 sub-factors)

| Factor | Method |
|--------|--------|
| Curvature | `\|dθ/ds\|` from waypoints |
| Intersection proximity | Distance to nearest junction |
| Lane width variance | Deviation from nominal |
| Speed limit context | Ego speed / road limit |

---

## Takeover Logic (Phase 2+)

Four staged escalation driven by a simulation state machine:

| Stage | Trigger | Action |
|-------|---------|--------|
| 1 — Warning | `risk > 0.4` | Visual/audio alert |
| 2 — Speed Reduction | `risk > 0.6` | Gradual deceleration |
| 3 — Takeover Request | `risk > 0.75` | Explicit handover prompt |
| 4 — Emergency Braking | `risk > 0.9` | Full stop fallback |

Driver response time is sampled from configurable distributions (Gaussian/lognormal), coupled to hazard context.

---

## Log Format

### Blackbox (`.jsonl`)

**Line 1 — Header:**
```json
{
  "record_type": "header",
  "simulation_id": "abc123",
  "config_hash": "sha256...",
  "carla_version": "0.9.15",
  "scenario": "highway_cut_in",
  "config_snapshot": { ... }
}
```

**Subsequent lines — Tick records:**
```json
{
  "record_type": "tick",
  "tick": 42,
  "elapsed_s": 2.1,
  "position": {"x": 100.5, "y": -200.3, "z": 0.1},
  "speed": 12.5,
  "risk": 0.45,
  "confidence_term": 0.09,
  "ttc_term": 0.21,
  "speed_term": 0.05,
  "complexity_term": 0.10,
  "min_ttc": 3.2,
  "detections": 5
}
```

### Events (`.jsonl`)

```json
{
  "timestamp": "2026-03-01T...",
  "event_type": "collision",
  "data": {"other_actor": "vehicle.audi.a2", "tick": 120}
}
```

---

## Phased Development

| Phase | Contents | Status |
|-------|----------|--------|
| **1** | CARLA connection, vehicle, sensors, maps, perception, simple risk, single scenario, basic logging | ✅ |
| **2** | State machine, 4-stage takeover, alert manager, control transitions | Planned |
| **3** | Driver model, metrics collector, replay validation, additional scenarios | Planned |
| **4** | Overtrust simulation, full metrics analysis, batch experiment support | Planned |

---

## Design Principles

- **Zero CARLA modifications** — all code lives inside `adas_supervision_project/`
- **Deterministic reproducibility** — seeds for CARLA RNG, traffic manager, perception noise
- **Extensible risk model** — abstract `BaseRiskModel` interface (rule-based now, ML-ready)
- **Forensic-grade logging** — every run tagged with UUID, config hash, full snapshot
- **Configurable everything** — weights, thresholds, distributions, log frequency in one YAML

---

## License

Academic project — MIT License.
