from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def task_balance_labels(df, task_type):
    """Derive the per-row class label used for balancing, given a task type.

    For binary tasks: the binary target (0/1).
    For multiclass tasks: class4_label (0-3).
    For multitask/hierarchical tasks: the combined auth*2 + enc label (0-3),
        so all four quadrants (real/fake x encoded/not_encoded) are counted.

    Returns a list of integer labels, one per row of ``df``.
    """
    if task_type == "binary":
        return df["target_value"].astype(int).tolist()
    if task_type == "multiclass":
        return df["class4_label"].astype(int).tolist()
    if task_type in {"multitask", "hierarchical"}:
        return (df["auth_label"].astype(int) * 2 + df["enc_label"].astype(int)).tolist()
    raise ValueError(f"Unsupported task_type={task_type}")


def balanced_subset_indices(labels, seed=42, max_per_class=None):
    """Return positional indices of an equal-count subset of ``labels``.

    Every class is downsampled to the smallest class count (optionally capped
    further at ``max_per_class``), giving a deterministic, leakage-neutral
    balanced subset. The returned indices are shuffled.

    Args:
        labels: list/array of integer class labels, one per row.
        seed: RNG seed for deterministic selection.
        max_per_class: optional hard cap on rows kept per class.

    Returns:
        list[int] of positional indices into ``labels``.
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(labels, return_counts=True)
    n_per = int(counts.min())
    if max_per_class is not None:
        n_per = min(n_per, int(max_per_class))

    keep = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        keep.extend(idx[:n_per].tolist())
    rng.shuffle(keep)
    return [int(i) for i in keep]


def make_balanced_sampler(labels):
    """Create a WeightedRandomSampler that balances across all classes.

    Each sample is weighted by 1/count(its_class), so that every class
    is sampled with equal probability per epoch.

    Args:
        labels: list/array of integer class labels, one per sample.

    Returns:
        WeightedRandomSampler with replacement, length = len(labels).
    """
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights))


def make_task_balanced_sampler(dataset):
    """Create a balanced sampler appropriate for the dataset's task type.

    For binary tasks: balances on the binary target (0/1).
    For multiclass tasks: balances on class4_label (0-3).
    For multitask/hierarchical tasks: balances on the combined
        auth*2 + enc label (0-3), so all four quadrants are equally represented.

    Args:
        dataset: an AudioManifestDataset instance (must have .df and .task_cfg).

    Returns:
        WeightedRandomSampler.
    """
    labels = task_balance_labels(dataset.df, dataset.task_cfg["type"])
    return make_balanced_sampler(labels)
