#!/usr/bin/env python
"""
FASS Inference Engine
=======================
Low-latency inference module callable by the FASS RiskEngine.

Provides:
    predict(sensor_data, ego_state) → (risk, uncertainty, advisory)

Design Priorities:
    - Latency: <50ms including MC-Dropout uncertainty estimation.
    - Thread-safety: stateless per call, model weights are read-only.
    - Fallback: if inference fails, returns WORST-CASE estimates.

ISO 26262 Note:
    This module is the ML ↔ FASS bridge.  Its outputs are ADVISORY ONLY.
    The RiskEngine applies deterministic overrides before any intervention.
    All predictions are logged with timestamps for traceability.
"""

import os
import time
import numpy as np
import torch
from typing import Tuple, Optional

from ..models.risk_model import FASSRiskNet
from ..data.feature_extractor import extract_features, impute_missing, FEATURE_DIM
from ..training.config import FASSConfig, DEFAULT_CONFIG


class FASSInferenceEngine:
    """Production inference engine for FASS risk prediction.

    Parameters
    ----------
    checkpoint_path : str
        Path to saved model checkpoint (.pt file).
    config : FASSConfig
        Configuration (for MC samples, thresholds, etc.).
    device : str
        'cpu' or 'cuda'.

    Example
    -------
    >>> engine = FASSInferenceEngine('./fass_checkpoints/best_model.pt')
    >>> risk, uncertainty, advisory = engine.predict(sensor_data, ego_state)
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        config: FASSConfig = None,
        device: str = 'cpu',
    ):
        self.config = config or DEFAULT_CONFIG
        self.device = device
        self._latency_warnings = 0
        self._checkpoint_path = checkpoint_path
        self.last_features = None  # Cached for online learning

        # Load model
        self.model = FASSRiskNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout_p=self.config.dropout_p,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            # Use strict=False to allow loading old checkpoints with
            # different input dimensions (new weights init randomly)
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"[FASS Inference] Model loaded from {checkpoint_path}")
        else:
            print("[FASS Inference] WARNING: No checkpoint — using uninitialized model")

        self.model.to(device)
        self.model.eval()

    def reload_model(self, checkpoint_path: str = None):
        """Hot-reload model weights from a checkpoint file.

        Used by OnlineTrainer after saving an updated checkpoint.
        """
        path = checkpoint_path or self._checkpoint_path
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            train_count = ckpt.get('online_train_count', '?')
            print(f"[FASS Inference] Model hot-reloaded (online step #{train_count})")
            return True
        return False

    def predict(
        self,
        sensor_data: dict,
        ego_state: dict,
        weather: dict = None,
        sensor_health: dict = None,
        mc_samples: int = None,
    ) -> Tuple[float, float, str]:
        """Run risk prediction with uncertainty.

        Parameters
        ----------
        sensor_data : dict
            Detected objects list under key 'detected_objects'.
        ego_state : dict
            Ego vehicle kinematics (speed, acceleration, yaw_rate, steer, etc.).
        weather : dict, optional
            Weather/environment state.
        sensor_health : dict, optional
            Sensor health flags.
        mc_samples : int, optional
            Override number of MC-Dropout samples.

        Returns
        -------
        risk : float
            Risk score [0, 1].
        uncertainty : float
            Total uncertainty (epistemic + aleatoric).
        advisory : str
            'SAFE', 'CAUTION', or 'DANGER'.

        SAFETY: If inference fails for ANY reason, returns (1.0, 1.0, 'DANGER')
        to ensure fail-safe behavior.
        """
        t0 = time.perf_counter()

        try:
            # Extract features
            detected_objects = sensor_data.get('detected_objects', [])
            feat = extract_features(
                detected_objects=detected_objects,
                ego_kinematics=ego_state,
                weather=weather or {},
                sensor_health=sensor_health or {},
                normalize=True,
            )
            feat = impute_missing(feat)
            self.last_features = feat  # Cache for online learning

            # Convert to tensor
            x = torch.from_numpy(feat).unsqueeze(0).to(self.device)

            # MC-Dropout prediction
            n = mc_samples or self.config.mc_samples
            result = self.model.predict_with_uncertainty(
                x, n_samples=n, risk_threshold=self.config.risk_threshold
            )

            risk = result['risk_mean']
            uncertainty = result['total_unc']
            advisory = result['advisory']

        except Exception as e:
            # SAFETY: Any failure → assume DANGER
            print(f"[FASS Inference] ERROR during prediction: {e}")
            print("[FASS Inference] FAILSAFE: Returning worst-case risk=1.0")
            risk = 1.0
            uncertainty = 1.0
            advisory = 'DANGER'

        # Latency check
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if latency_ms > self.config.max_inference_latency_ms:
            self._latency_warnings += 1
            if self._latency_warnings <= 10:
                print(f"[FASS Inference] ⚠ Latency {latency_ms:.1f}ms > "
                      f"{self.config.max_inference_latency_ms}ms threshold")

        return risk, uncertainty, advisory

    def predict_detailed(
        self,
        sensor_data: dict,
        ego_state: dict,
        weather: dict = None,
        sensor_health: dict = None,
    ) -> dict:
        """Full prediction with decomposed uncertainty and metadata.

        Returns a dict with all prediction details for logging.
        """
        t0 = time.perf_counter()

        try:
            detected_objects = sensor_data.get('detected_objects', [])
            feat = extract_features(
                detected_objects=detected_objects,
                ego_kinematics=ego_state,
                weather=weather or {},
                sensor_health=sensor_health or {},
                normalize=True,
            )
            feat = impute_missing(feat)
            x = torch.from_numpy(feat).unsqueeze(0).to(self.device)

            result = self.model.predict_with_uncertainty(
                x, n_samples=self.config.mc_samples,
                risk_threshold=self.config.risk_threshold,
            )

            result['inference_ok'] = True
            result['features'] = feat.tolist()

        except Exception as e:
            result = {
                'risk_mean': 1.0,
                'risk_std': 0.0,
                'epistemic_unc': 1.0,
                'aleatoric_unc': 1.0,
                'total_unc': 2.0,
                'advisory': 'DANGER',
                'inference_ok': False,
                'error': str(e),
            }

        result['latency_ms'] = round((time.perf_counter() - t0) * 1000.0, 2)
        result['timestamp'] = time.time()

        return result
