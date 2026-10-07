"""
Shared utilities: reproducibility, EMA tracker, metric averaging, plotting helpers.
"""
import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Tracks a running mean of a scalar quantity."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class EMATracker:
    """
    Per-sample exponential-moving-average tracker for temporal difficulty memory.
    d_i^(t) = alpha * d_i^(t-1) + (1 - alpha) * d_tilde_i^(t)
    Indexed by a stable per-sample id (dataset index), so the memory persists
    across epochs regardless of shuffling / mini-batch composition.
    """
    def __init__(self, n_samples: int, alpha: float = 0.9, init_val: float = 0.5):
        self.alpha = alpha
        self.values = torch.full((n_samples,), init_val, dtype=torch.float32)
        self.initialized = torch.zeros(n_samples, dtype=torch.bool)

    def update(self, idx: torch.Tensor, new_val: torch.Tensor):
        idx = idx.cpu()
        new_val = new_val.detach().cpu()
        first_seen = ~self.initialized[idx]
        # First observation: seed directly, no averaging distortion.
        self.values[idx[first_seen]] = new_val[first_seen]
        seen_before = self.initialized[idx]
        if seen_before.any():
            prev = self.values[idx[seen_before]]
            self.values[idx[seen_before]] = self.alpha * prev + (1 - self.alpha) * new_val[seen_before]
        self.initialized[idx] = True

    def get(self, idx: torch.Tensor):
        return self.values[idx.cpu()]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def normalize_batch(x, mean, std, device):
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return (x - mean) / std
