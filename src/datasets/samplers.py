from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels):
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights))
