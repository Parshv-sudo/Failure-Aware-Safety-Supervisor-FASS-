#!/usr/bin/env python
"""
Loss Functions for FASS Risk Prediction
=========================================
Custom loss functions that:
    1. Model aleatoric uncertainty (heteroscedastic Gaussian NLL).
    2. Apply asymmetric penalties to reduce false negatives.
    3. Weight high-risk samples more heavily for safety.

Risk-Weighted Loss:
    L = (1/(2σ²)) * (y - ŷ)² + (1/2) * log(σ²) + safety_penalty

    Where:
        σ² = exp(log_var)     — predicted aleatoric variance
        y  = ground-truth risk label
        ŷ  = predicted risk (sigmoid output)

    Safety Penalty:
        When the model UNDER-predicts risk (ŷ < y), an asymmetric
        penalty is added:  λ * max(0, y - ŷ)²
        This directly reduces false-negative rate.

ISO 26262 Note:
    In safety-critical systems, false negatives (failing to detect danger)
    are far more dangerous than false positives (unnecessary warnings).
    The asymmetric loss reflects this ASIL requirement.
"""

import torch
import torch.nn as nn


class RiskWeightedLoss(nn.Module):
    """Heteroscedastic Gaussian NLL with asymmetric safety penalty.

    Parameters
    ----------
    safety_weight : float
        Multiplier for false-negative penalty.  Higher = more conservative.
    high_risk_threshold : float
        Risk labels above this threshold get additional weight.
    high_risk_multiplier : float
        Extra weight for high-risk samples.
    """

    def __init__(
        self,
        safety_weight: float = 3.0,
        high_risk_threshold: float = 0.5,
        high_risk_multiplier: float = 5.0,
    ):
        super().__init__()
        self.safety_weight = safety_weight
        self.high_risk_threshold = high_risk_threshold
        self.high_risk_multiplier = high_risk_multiplier

    def forward(
        self,
        pred_risk: torch.Tensor,
        log_var: torch.Tensor,
        target_risk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the risk-weighted loss.

        Parameters
        ----------
        pred_risk : (batch, 1) — predicted risk score [0, 1].
        log_var   : (batch, 1) — predicted log-variance (aleatoric).
        target_risk : (batch,) or (batch, 1) — ground-truth risk label.

        Returns
        -------
        Scalar loss tensor.
        """
        target = target_risk.view_as(pred_risk)

        # --- Heteroscedastic Gaussian NLL ---
        precision = torch.exp(-log_var)  # 1/σ²
        nll = 0.5 * precision * (target - pred_risk) ** 2 + 0.5 * log_var

        # --- Asymmetric false-negative penalty ---
        # Penalise when pred < target (under-prediction of risk)
        under_pred = torch.clamp(target - pred_risk, min=0.0)
        fn_penalty = self.safety_weight * under_pred ** 2

        # --- High-risk sample weighting ---
        sample_weights = torch.ones_like(target)
        high_risk_mask = target > self.high_risk_threshold
        sample_weights[high_risk_mask] = self.high_risk_multiplier

        # --- Baseline anchor penalty ---
        # If the target is VERY safe (< 0.1) but we predict high, penalize false positives.
        # This prevents the asymmetric safety penalty from dragging the empty baseline up to 0.5.
        safe_mask = target < 0.1
        over_pred = torch.clamp(pred_risk - target, min=0.0)
        fp_penalty = 2.0 * over_pred ** 2
        fp_penalty = fp_penalty * safe_mask.float() # only apply to actually safe samples

        # --- Combined loss ---
        per_sample_loss = sample_weights * (nll + fn_penalty) + fp_penalty

        return per_sample_loss.mean()


class BinaryCollisionLoss(nn.Module):
    """Optional auxiliary loss: binary cross-entropy on collision prediction.

    Can be added as a multi-task head for explicit collision classification.
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        collision_labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.bce(pred_logits, collision_labels.view_as(pred_logits))
