"""
FASS ML Module — Failure-Aware Safety Supervisor
=================================================
Machine-learning-based risk prediction with uncertainty estimation for
autonomous driving supervision inside CARLA 0.9.15.

ISO 26262 Alignment:
    This module is an ADVISORY layer.  Deterministic fail-safe thresholds
    (see safety/deterministic_overrides.py) ALWAYS override ML predictions
    when safety-critical conditions are detected.

Package layout:
    data/           – CARLA sensor hooks, feature extraction, dataset, scenario gen
    models/         – Bayesian / MC-Dropout risk prediction network + losses
    training/       – Train, evaluate, config
    integration/    – Inference engine, RiskEngine, VotingLogic, FailSafeAuthority
    safety/         – Logging, deterministic overrides
"""

__version__ = "0.1.0"
