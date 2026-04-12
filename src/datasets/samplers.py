from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


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
    task_type = dataset.task_cfg["type"]
    df = dataset.df

    if task_type == "binary":
        labels = df["target_value"].astype(int).tolist()
    elif task_type == "multiclass":
        labels = df["class4_label"].astype(int).tolist()
    elif task_type in {"multitask", "hierarchical"}:
        # Balance on the 4-class combination so that all quadrants
        # (real/fake × encoded/not_encoded) are equally represented
        labels = (df["auth_label"].astype(int) * 2 + df["enc_label"].astype(int)).tolist()
    else:
        raise ValueError(f"Unsupported task_type={task_type}")

    return make_balanced_sampler(labels)
