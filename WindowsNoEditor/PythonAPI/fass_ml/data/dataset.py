#!/usr/bin/env python
"""
PyTorch Dataset for FASS ML Module
====================================
Loads .npz frames produced by carla_data_collector.py, extracts features
via feature_extractor, and provides (feature_vector, risk_label) pairs
for training.

Supports:
    - Automatic train / val / test split
    - Risk-weighted sampling (oversample high-risk frames)
    - On-the-fly feature augmentation (optional noise injection)

ISO 26262 Note:
    Dataset integrity is critical.  Corrupted frames are LOGGED and SKIPPED
    rather than silently dropped.
"""

import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Local import
from .feature_extractor import extract_from_npz, FEATURE_DIM


class FASSDataset(Dataset):
    """PyTorch dataset for FASS risk prediction training.

    Parameters
    ----------
    data_dir : str
        Directory containing .npz frame files.
    split : str
        One of 'train', 'val', 'test'.  Splits are deterministic given
        the same seed.
    split_ratios : tuple
        (train, val, test) ratios summing to 1.0.
    seed : int
        Random seed for reproducible splitting.
    augment : bool
        If True, apply small noise to features during training.
    normalize : bool
        If True, apply feature normalization.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        augment: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        assert split in ('train', 'val', 'test')
        assert abs(sum(split_ratios) - 1.0) < 1e-6

        self.data_dir = data_dir
        self.split = split
        self.augment = augment and (split == 'train')
        self.normalize = normalize

        # Discover all frame files
        all_files = sorted(glob.glob(os.path.join(data_dir, 'frame_*.npz')))
        if not all_files:
            raise FileNotFoundError(f"No frame files found in {data_dir}")

        # Deterministic split
        rng = random.Random(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)

        n = len(all_files)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        if split == 'train':
            selected = indices[:n_train]
        elif split == 'val':
            selected = indices[n_train:n_train + n_val]
        else:
            selected = indices[n_train + n_val:]

        self.files = [all_files[i] for i in selected]

        # Pre-load labels for sampling weights
        self._risk_labels = []
        valid_files = []
        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)
                labels = json.loads(str(data['labels']))
                self._risk_labels.append(labels['risk'])
                valid_files.append(f)
            except Exception as e:
                print(f"[FASS Dataset] WARNING: Corrupted frame skipped: {f} ({e})")

        self.files = valid_files
        self._risk_labels = np.array(self._risk_labels, dtype=np.float32)

        print(f"[FASS Dataset] {split}: {len(self.files)} frames  "
              f"risk_mean={self._risk_labels.mean():.3f}  "
              f"risk_max={self._risk_labels.max():.3f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat, labels = extract_from_npz(self.files[idx], normalize=self.normalize)

        if self.augment:
            # Small Gaussian noise for training augmentation
            noise = np.random.normal(0, 0.01, size=feat.shape).astype(np.float32)
            feat = feat + noise

        features = torch.from_numpy(feat)
        risk = torch.tensor(labels['risk'], dtype=torch.float32)

        return features, risk

    def get_sample_weights(self, high_risk_multiplier: float = 5.0):
        """Compute per-sample weights for risk-weighted sampling.

        High-risk frames (>0.5) are upsampled by ``high_risk_multiplier``.
        This reduces false-negative rate during training.
        """
        weights = np.ones_like(self._risk_labels)
        weights[self._risk_labels > 0.5] = high_risk_multiplier
        weights[self._risk_labels > 0.8] = high_risk_multiplier * 2
        return torch.from_numpy(weights).double()


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    risk_weighted_sampling: bool = True,
) -> dict:
    """Create train/val/test DataLoaders.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a DataLoader.
    """
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = FASSDataset(data_dir, split=split, seed=seed,
                         augment=(split == 'train'))

        sampler = None
        shuffle = (split == 'train')
        if split == 'train' and risk_weighted_sampling and len(ds) > 0:
            weights = ds.get_sample_weights()
            sampler = WeightedRandomSampler(weights, num_samples=len(ds),
                                            replacement=True)
            shuffle = False  # sampler handles ordering

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=(split == 'train'),
        )

    return loaders
