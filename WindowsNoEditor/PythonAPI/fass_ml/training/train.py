#!/usr/bin/env python
"""
FASS Training Script
======================
Training loop with validation, early stopping, model checkpointing,
and per-epoch metric logging.

Usage:
    python -m fass_ml.training.train --data-dir ./fass_data --epochs 50

    Or with synthetic data (no CARLA needed):
    python -m fass_ml.training.train --synthetic --epochs 20

ISO 26262 Note:
    Training reproducibility is ensured via deterministic seeding.
    All training metrics are logged for audit trail.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import FASSConfig, DEFAULT_CONFIG
from ..models.risk_model import FASSRiskNet
from ..models.losses import RiskWeightedLoss


def _create_synthetic_data(n_samples: int = 2000, seed: int = 42):
    """Generate synthetic training data for testing without CARLA.

    Creates realistic-looking feature vectors and risk labels based
    on simple heuristics (distance, TTC, weather).
    """
    rng = np.random.RandomState(seed)

    features = rng.randn(n_samples, 35).astype(np.float32) * 0.3 + 0.5
    features = np.clip(features, 0, 1)

    # Generate correlated risk labels
    # Risk correlates with: close objects (feat 0 low), low TTC (feat 1 low),
    # many objects (feat 2 high), poor weather (feat 29 high)
    risk = (
        (1.0 - features[:, 0]) * 0.3 +     # close objects
        (1.0 - features[:, 1]) * 0.3 +     # low TTC
        features[:, 14] * 0.1 +             # sensor degradation
        features[:, 29] * 0.1 +             # environmental severity
        features[:, 21] * 0.2               # high speed
    )
    risk = np.clip(risk + rng.randn(n_samples) * 0.05, 0, 1).astype(np.float32)

    return features, risk


def train(config: FASSConfig = None, synthetic: bool = False):
    """Run the full training pipeline.

    Parameters
    ----------
    config : FASSConfig
        Configuration.  Uses DEFAULT_CONFIG if None.
    synthetic : bool
        If True, use synthetic data instead of CARLA-collected data.
    """
    if config is None:
        config = DEFAULT_CONFIG

    config.set_deterministic()
    config.ensure_dirs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[FASS Train] Device: {device}")

    # --- Data ---
    if synthetic:
        print("[FASS Train] Using synthetic data")
        X_all, y_all = _create_synthetic_data(n_samples=3000, seed=config.global_seed)

        # Split
        n = len(X_all)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_val, y_val = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
        X_test, y_test = X_all[n_train+n_val:], y_all[n_train+n_val:]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=config.batch_size)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            batch_size=config.batch_size)
    else:
        from ..data.dataset import create_dataloaders
        loaders = create_dataloaders(config.data_dir, batch_size=config.batch_size,
                                     seed=config.global_seed)
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']

    # --- Model ---
    model = FASSRiskNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout_p=config.dropout_p,
    ).to(device)

    print(f"[FASS Train] {model.summary()}")

    # --- Optimizer & Loss ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = RiskWeightedLoss(
        safety_weight=config.safety_weight,
        high_risk_threshold=config.high_risk_threshold,
        high_risk_multiplier=config.high_risk_multiplier,
    )

    # --- Training loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(config.max_epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred_risk, log_var = model(batch_x)
            loss = criterion(pred_risk, log_var, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred_risk, log_var = model(batch_x)
                loss = criterion(pred_risk, log_var, batch_y)
                val_losses.append(loss.item())
                val_preds.append(pred_risk.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        scheduler.step(avg_val_loss)

        # Compute metrics
        if val_preds:
            val_preds_np = np.concatenate(val_preds).flatten()
            val_targets_np = np.concatenate(val_targets).flatten()
            val_mae = np.mean(np.abs(val_preds_np - val_targets_np))

            # Collision accuracy (binary at threshold)
            pred_danger = val_preds_np > config.risk_threshold
            true_danger = val_targets_np > config.risk_threshold
            tp = np.sum(pred_danger & true_danger)
            fp = np.sum(pred_danger & ~true_danger)
            fn = np.sum(~pred_danger & true_danger)
            tn = np.sum(~pred_danger & ~true_danger)

            fn_rate = fn / max(fn + tp, 1)
            accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        else:
            val_mae = 0.0
            fn_rate = 0.0
            accuracy = 0.0

        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': round(avg_train_loss, 6),
            'val_loss': round(avg_val_loss, 6),
            'val_mae': round(float(val_mae), 6),
            'fn_rate': round(float(fn_rate), 4),
            'accuracy': round(float(accuracy), 4),
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(epoch_info)

        print(f"  Epoch {epoch+1:3d}/{config.max_epochs} | "
              f"train_loss={avg_train_loss:.5f} | "
              f"val_loss={avg_val_loss:.5f} | "
              f"val_mae={val_mae:.4f} | "
              f"FN_rate={fn_rate:.3f} | "
              f"acc={accuracy:.3f}")

        # ── Early stopping & checkpointing ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            ckpt_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config.to_dict(),
            }, ckpt_path)
            print(f"    [*] Saved best model (val_loss={avg_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"    ✗ Early stopping at epoch {epoch+1}")
                break

    # ── Save training history ──
    history_path = os.path.join(config.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[FASS Train] Done. Best val_loss={best_val_loss:.5f}")
    print(f"[FASS Train] Checkpoint: {config.checkpoint_dir}/best_model.pt")
    print(f"[FASS Train] History:    {history_path}")

    return model, history


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FASS ML Training")
    parser.add_argument('--data-dir', default='./fass_data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (no CARLA needed)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = FASSConfig(
        data_dir=args.data_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        global_seed=args.seed,
    )

    train(config=config, synthetic=args.synthetic)


if __name__ == '__main__':
    main()
