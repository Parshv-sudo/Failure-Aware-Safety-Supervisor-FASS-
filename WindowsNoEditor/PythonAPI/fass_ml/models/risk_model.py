#!/usr/bin/env python
"""
FASSRiskNet — Uncertainty-Aware Risk Prediction Model
======================================================
A compact MLP with MC-Dropout for probabilistic risk estimation.

Architecture:
    Input (30) → Dense(128) → ReLU → Dropout(p) →
                 Dense(64)  → ReLU → Dropout(p) →
                 Dense(32)  → ReLU → Dropout(p) →
                 Output: risk_score (sigmoid, 0-1)
                         log_variance (real, aleatoric uncertainty)

Uncertainty Estimation:
    Epistemic (model uncertainty):
        Multiple forward passes with dropout enabled at inference time
        (Monte Carlo Dropout).  Variance across passes = epistemic uncertainty.

    Aleatoric (data noise):
        The network directly predicts log-variance as a second output head.
        σ² = exp(log_var) captures irreducible observation noise.

    Total uncertainty = epistemic + aleatoric

Design Choices:
    - Small network (< 15K params) for <50ms CPU inference.
    - Sigmoid output ensures risk ∈ [0, 1].
    - log_variance output avoids negative variance via exp().
    - MC-Dropout does NOT require a separate Bayesian framework.

ISO 26262 Note:
    ML predictions are ADVISORY.  They feed into RiskEngine, which
    applies deterministic safety thresholds as hard overrides.
    The model must NEVER be the sole decision-maker for braking
    or steering interventions.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..data.feature_extractor import FEATURE_DIM


class FASSRiskNet(nn.Module):
    """Monte Carlo Dropout risk prediction network.

    Parameters
    ----------
    input_dim : int
        Feature vector dimension (default: 30).
    hidden_dims : tuple
        Hidden layer sizes.
    dropout_p : float
        Dropout probability (used at both train AND inference time for MC-Dropout).
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple = (128, 64, 32),
        dropout_p: float = 0.15,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
            ])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)

        # Risk score head — sigmoid ensures [0, 1]
        self.risk_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid(),
        )

        # Aleatoric uncertainty head — outputs log(σ²)
        self.log_var_head = nn.Linear(prev_dim, 1)

        self._dropout_p = dropout_p
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim).

        Returns
        -------
        risk : Tensor of shape (batch, 1), values in [0, 1].
        log_var : Tensor of shape (batch, 1), log-variance (aleatoric).
        """
        features = self.backbone(x)
        risk = self.risk_head(features)
        log_var = self.log_var_head(features)
        return risk, log_var

    def enable_mc_dropout(self):
        """Enable dropout layers at inference time for MC-Dropout."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
        risk_threshold: float = 0.5,
    ) -> dict:
        """Run MC-Dropout inference and decompose uncertainty.

        Parameters
        ----------
        x : Tensor of shape (1, input_dim) or (batch, input_dim).
        n_samples : Number of stochastic forward passes.
        risk_threshold : Threshold for collision advisory.

        Returns
        -------
        dict with keys:
            risk_mean       : float — mean predicted risk (0-1)
            risk_std        : float — std across MC samples
            epistemic_unc   : float — model uncertainty (variance of means)
            aleatoric_unc   : float — data noise (mean of predicted variances)
            total_unc       : float — epistemic + aleatoric
            advisory        : str   — 'SAFE' | 'CAUTION' | 'DANGER'
            latency_ms      : float — wall-clock inference time
            mc_samples      : int
        """
        t0 = time.perf_counter()

        self.enable_mc_dropout()

        risk_samples = []
        var_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                risk, log_var = self.forward(x)
                risk_samples.append(risk.cpu().numpy())
                var_samples.append(np.exp(log_var.cpu().numpy()))

        risk_samples = np.concatenate(risk_samples, axis=0)  # (n_samples * batch, 1)
        var_samples = np.concatenate(var_samples, axis=0)

        # Decompose uncertainty
        risk_mean = float(np.mean(risk_samples))
        risk_std = float(np.std(risk_samples))
        epistemic_unc = float(np.var(risk_samples))          # variance of means
        aleatoric_unc = float(np.mean(var_samples))          # mean of variances
        total_unc = epistemic_unc + aleatoric_unc

        # Collision advisory
        if risk_mean > 0.7 or (risk_mean > risk_threshold and total_unc < 0.1):
            advisory = 'DANGER'
        elif risk_mean > risk_threshold:
            advisory = 'CAUTION'
        else:
            advisory = 'SAFE'

        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.eval()  # Restore eval mode

        return {
            'risk_mean': round(risk_mean, 5),
            'risk_std': round(risk_std, 5),
            'epistemic_unc': round(epistemic_unc, 6),
            'aleatoric_unc': round(aleatoric_unc, 6),
            'total_unc': round(total_unc, 6),
            'advisory': advisory,
            'latency_ms': round(latency_ms, 2),
            'mc_samples': n_samples,
        }

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        return (
            f"FASSRiskNet(input={self._input_dim}, "
            f"params={self.param_count:,}, "
            f"dropout={self._dropout_p})"
        )
