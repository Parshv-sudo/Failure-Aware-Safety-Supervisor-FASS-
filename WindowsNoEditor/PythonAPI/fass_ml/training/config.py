#!/usr/bin/env python
"""
FASS Training Configuration
==============================
Single source of truth for all hyperparameters, paths, thresholds,
and seeds.  Every tunable value lives here for reproducibility.

ISO 26262 Note:
    Configuration traceability is required for safety qualification.
    Any change to these values must be documented and re-validated.
"""

import os
import random
import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class FASSConfig:
    """All FASS ML configuration in one place.

    Sections:
        Model       — network architecture
        Training    — optimization loop
        Safety      — deterministic override thresholds
        Evaluation  — MC-sampling, calibration
        Paths       — input/output directories
        Seeds       — reproducibility
    """

    # ── Model ─────────────────────────────────────────────────────────
    input_dim: int = 35
    hidden_dims: tuple = (128, 64, 32)
    dropout_p: float = 0.15

    # ── Training ──────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10
    grad_clip_norm: float = 1.0

    # Online learning
    online_lr: float = 1e-4               # lower LR for fine-tuning
    online_train_interval_s: float = 60.0 # seconds between training steps
    online_buffer_size: int = 2000        # experience replay buffer size
    online_batch_size: int = 32           # mini-batch for online updates
    online_grad_steps: int = 5            # gradient steps per training round
    online_lookahead_s: float = 3.0       # hindsight labeling window

    # Loss function
    safety_weight: float = 3.0           # false-negative penalty multiplier
    high_risk_threshold: float = 0.5     # label threshold for extra weighting
    high_risk_multiplier: float = 5.0    # extra weight for high-risk samples

    # ── Safety Thresholds (deterministic overrides) ───────────────────
    # These CANNOT be overridden by ML predictions.
    ttc_emergency_s: float = 0.5         # TTC < 0.5s → emergency brake
    ttc_warning_s: float = 1.5           # TTC < 1.5s → caution
    min_distance_stop_m: float = 1.5     # distance < 1.5m → emergency stop
    min_distance_warn_m: float = 3.0     # distance < 3m → caution
    sensor_failure_threshold: int = 2    # ≥2 sensors failed → safe stop
    max_inference_latency_ms: float = 50.0  # warn if exceeded

    # ── Evaluation ────────────────────────────────────────────────────
    mc_samples: int = 30                 # MC-Dropout forward passes
    ece_n_bins: int = 10                 # Expected Calibration Error bins
    latency_benchmark_iters: int = 1000  # iterations for latency test
    risk_threshold: float = 0.6          # advisory threshold

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: str = './fass_data'
    checkpoint_dir: str = './fass_checkpoints'
    log_dir: str = './fass_logs'
    eval_output: str = './fass_eval_report.json'

    # ── Seeds ─────────────────────────────────────────────────────────
    global_seed: int = 42
    scenario_seeds: list = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    def set_deterministic(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config for logging / reproducibility."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d


# Default configuration instance
DEFAULT_CONFIG = FASSConfig()
