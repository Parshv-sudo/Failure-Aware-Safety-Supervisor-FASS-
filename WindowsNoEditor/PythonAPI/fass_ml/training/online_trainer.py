#!/usr/bin/env python
"""
Online Learning for FASS
==========================
Fine-tunes the FASSRiskNet model during live CARLA demo using
hindsight experience replay.

How it works:
    1. ExperienceBuffer collects (features, sensor_readings) each tick
    2. After a lookahead delay, ground truth labels are computed from
       what *actually* happened (collision, near-miss, safe driving)
    3. OnlineTrainer periodically runs mini-batch gradient updates
    4. Updated model is hot-reloaded into the inference engine

ISO 26262 Note:
    Online-trained models are NOT safety-qualified until validated
    offline.  The deterministic overrides remain the safety floor.
"""

import os
import time
import collections
import numpy as np
import torch
import torch.nn as nn

from ..models.losses import RiskWeightedLoss
from ..training.config import FASSConfig, DEFAULT_CONFIG


# ============================================================================
# Experience Buffer with Hindsight Labeling
# ============================================================================

Frame = collections.namedtuple('Frame', [
    'timestamp',        # wall-clock time
    'features',         # np.ndarray (35,)
    'min_ttc',          # float — raw TTC at this frame
    'min_distance',     # float — raw closest distance
    'speed',            # float — ego speed m/s
    'pitch',            # float — terrain pitch degrees
    'collision',        # bool — collision happened at or after this frame
    'label',            # float or None — ground truth risk (set by hindsight)
])


class ExperienceBuffer:
    """Circular buffer that collects live frames and labels them via hindsight.

    Parameters
    ----------
    max_size : int
        Maximum frames to keep.
    lookahead_s : float
        Seconds to wait before computing hindsight label.
    """

    def __init__(self, max_size: int = 2000, lookahead_s: float = 3.0, buffer_path: str = None):
        self.max_size = max_size
        self.lookahead_s = lookahead_s
        self.buffer_path = buffer_path
        self._buffer = collections.deque(maxlen=max_size)
        
        # Try to load existing buffer data
        if self.buffer_path and os.path.exists(self.buffer_path):
            try:
                data = torch.load(self.buffer_path)
                for f_data in data.get('frames', []):
                    # We only load fully labeled frames
                    frame = Frame(
                        timestamp=f_data['timestamp'],
                        features=f_data['features'],
                        min_ttc=f_data['min_ttc'],
                        min_distance=f_data['min_distance'],
                        speed=f_data['speed'],
                        pitch=f_data['pitch'],
                        collision=f_data['collision'],
                        label=f_data['label']
                    )
                    self._buffer.append(frame)
                print(f"[ExperienceBuffer] Loaded {len(self._buffer)} past experiences from {self.buffer_path}")
            except Exception as e:
                print(f"[ExperienceBuffer] WARNING: Could not load saved buffer: {e}")

        self._pending = collections.deque()   # frames awaiting labeling
        self._collision_times = []            # timestamps of collisions
        self._labeled_count = 0

    def record_collision(self, timestamp: float = None):
        """Record that a collision happened (called from CARLA callback)."""
        t = timestamp or time.time()
        self._collision_times.append(t)
        # Keep only last 60 seconds of collision records
        cutoff = t - 60.0
        self._collision_times = [c for c in self._collision_times if c > cutoff]

    def push(self, features: np.ndarray, min_ttc: float, min_distance: float,
             speed: float, pitch: float = 0.0):
        """Add a new frame to the buffer (label computed later via hindsight)."""
        frame = Frame(
            timestamp=time.time(),
            features=features.copy(),
            min_ttc=min_ttc,
            min_distance=min_distance,
            speed=speed,
            pitch=pitch,
            collision=False,
            label=None,
        )
        self._pending.append(frame)

    def process_hindsight(self):
        """Label pending frames whose lookahead window has elapsed."""
        now = time.time()
        newly_labeled = 0

        while self._pending and (now - self._pending[0].timestamp) >= self.lookahead_s:
            frame = self._pending.popleft()
            label = self._compute_label(frame, now)
            # Create labeled frame (namedtuple is immutable, so replace)
            labeled = frame._replace(label=label)
            self._buffer.append(labeled)
            self._labeled_count += 1
            newly_labeled += 1

        return newly_labeled

    def _compute_label(self, frame: Frame, now: float) -> float:
        """Compute ground truth risk using hindsight.

        Checks what happened in the lookahead window after this frame:
        - Collision → 1.0
        - Very low TTC → 0.85
        - Very close distance → 0.7
        - Moderate TTC → 0.5
        - Moderate distance → 0.3
        - Normal → 0.05

        Plus modifiers for terrain and conditions.
        """
        t0 = frame.timestamp
        t1 = t0 + self.lookahead_s

        # Check if a collision occurred in the lookahead window
        collision_in_window = any(t0 <= ct <= t1 for ct in self._collision_times)

        if collision_in_window:
            return 1.0

        # Use the sensor readings at this frame for labeling
        # (future frames in pending could refine this, but current readings
        #  are a good proxy since they're what led to the outcome)
        if frame.min_ttc < 0.5:
            base_risk = 0.85
        elif frame.min_distance < 1.5:
            base_risk = 0.7
        elif frame.min_ttc < 1.5:
            base_risk = 0.5
        elif frame.min_distance < 5.0:
            base_risk = 0.3
        else:
            base_risk = 0.05

        # Terrain modifiers
        if frame.pitch < -5.0 and frame.speed > 8.3:  # downhill + >30km/h
            base_risk = min(1.0, base_risk + 0.1)
        if frame.speed < 0.5:  # stationary / parking
            base_risk = max(0.0, base_risk - 0.1)

        return round(min(1.0, max(0.0, base_risk)), 3)

    def sample_batch(self, batch_size: int = 32):
        """Sample a balanced mini-batch for training.

        Returns (features_tensor, labels_tensor) or None if not enough data.
        """
        labeled = [f for f in self._buffer if f.label is not None]
        if len(labeled) < batch_size:
            return None

        # Balanced sampling: 50% high-risk, 50% normal
        high_risk = [f for f in labeled if f.label > 0.3]
        low_risk = [f for f in labeled if f.label <= 0.3]

        if not high_risk or not low_risk:
            # Fall back to random sampling
            indices = np.random.choice(len(labeled), batch_size, replace=True)
            selected = [labeled[i] for i in indices]
        else:
            n_high = min(batch_size // 2, len(high_risk))
            n_low = batch_size - n_high
            selected = (
                list(np.random.choice(high_risk, n_high, replace=True)) +
                list(np.random.choice(low_risk, n_low, replace=True))
            )

        features = np.array([f.features for f in selected], dtype=np.float32)
        labels = np.array([f.label for f in selected], dtype=np.float32)

        return (
            torch.from_numpy(features),
            torch.from_numpy(labels),
        )

    def save(self):
        """Save labeled frames to disk for persistence across runs."""
        if not self.buffer_path:
            return

        labeled = [f for f in self._buffer if f.label is not None]
        if not labeled:
            return

        # Convert to dicts for serialization
        frames_dict = []
        for f in labeled:
            frames_dict.append({
                'timestamp': f.timestamp,
                'features': f.features,
                'min_ttc': f.min_ttc,
                'min_distance': f.min_distance,
                'speed': f.speed,
                'pitch': f.pitch,
                'collision': f.collision,
                'label': f.label
            })

        data = {'frames': frames_dict}
        try:
            torch.save(data, self.buffer_path)
            # print(f"[ExperienceBuffer] Saved {len(frames_dict)} frames to {self.buffer_path}")
        except Exception as e:
            print(f"[ExperienceBuffer] ERROR saving buffer: {e}")

    @property
    def stats(self):
        labeled = [f for f in self._buffer if f.label is not None]
        high = sum(1 for f in labeled if f.label > 0.3)
        return {
            'total_frames': len(self._buffer),
            'pending': len(self._pending),
            'labeled': len(labeled),
            'high_risk': high,
            'low_risk': len(labeled) - high,
            'collisions': len(self._collision_times),
        }


# ============================================================================
# Online Trainer
# ============================================================================

class OnlineTrainer:
    """Periodically fine-tunes FASSRiskNet on live experience data.

    Parameters
    ----------
    model : FASSRiskNet
        The model to train (from inference engine).
    config : FASSConfig
        Configuration.
    checkpoint_path : str
        Path to save updated checkpoints.
    """

    def __init__(self, model, config: FASSConfig = None,
                 checkpoint_path: str = None):
        self.config = config or DEFAULT_CONFIG
        self.model = model
        self.checkpoint_path = checkpoint_path

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.online_lr,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = RiskWeightedLoss(
            safety_weight=self.config.safety_weight,
            high_risk_threshold=self.config.high_risk_threshold,
            high_risk_multiplier=self.config.high_risk_multiplier,
        )

        self._last_train_time = time.time()
        self._train_count = 0
        self._total_loss_history = []

    def should_train(self) -> bool:
        """Check if enough time has passed for a training step."""
        return (time.time() - self._last_train_time) >= self.config.online_train_interval_s

    def train_step(self, buffer: ExperienceBuffer) -> dict:
        """Run a mini-batch training step.

        Returns dict with training stats, or None if not enough data.
        """
        # First, label any pending frames
        buffer.process_hindsight()

        batch = buffer.sample_batch(self.config.online_batch_size)
        if batch is None:
            return None

        features, labels = batch
        self.model.train()

        step_losses = []
        for _ in range(self.config.online_grad_steps):
            self.optimizer.zero_grad()
            pred_risk, log_var = self.model(features)
            loss = self.criterion(pred_risk, log_var, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            step_losses.append(loss.item())

        self.model.eval()
        self._train_count += 1
        self._last_train_time = time.time()

        avg_loss = np.mean(step_losses)
        self._total_loss_history.append(avg_loss)

        # Save checkpoint
        if self.checkpoint_path:
            torch.save({
                'epoch': -1,  # online
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': avg_loss,
                'config': self.config.to_dict(),
                'online_train_count': self._train_count,
            }, self.checkpoint_path)

        stats = buffer.stats
        result = {
            'train_step': self._train_count,
            'loss': round(avg_loss, 5),
            'grad_steps': self.config.online_grad_steps,
            'buffer_frames': stats['labeled'],
            'high_risk_frames': stats['high_risk'],
            'collisions_seen': stats['collisions'],
        }

        # Log improvement
        if len(self._total_loss_history) >= 2:
            result['loss_delta'] = round(
                self._total_loss_history[-1] - self._total_loss_history[-2], 5)

        print(f"\n[Online Learning] Step #{self._train_count}: "
              f"loss={avg_loss:.4f} | "
              f"frames={stats['labeled']} "
              f"(high_risk={stats['high_risk']}) | "
              f"collisions={stats['collisions']}")

        return result

    @property
    def train_count(self):
        return self._train_count

    @property
    def last_loss(self):
        return self._total_loss_history[-1] if self._total_loss_history else None
