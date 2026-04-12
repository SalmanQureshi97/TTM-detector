from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)


def filter_manifest(df: pd.DataFrame, datasets, split):
    out = df[df["source_dataset"].isin(datasets)].copy()
    if split is not None:
        out = out[out["split"] == split].copy()
    return out


def assign_group_splits(
    df: pd.DataFrame,
    group_col: str = "track_id",
    stratify_col: str = "class4_label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits with group-aware stratification.

    Splits are performed at the track level (group_col) to prevent leakage.
    Stratification is done per source_dataset × stratify_col combination to
    maintain class proportions across splits within each dataset.

    Args:
        df: Master manifest DataFrame.
        group_col: Column to group by (all rows with same value stay together).
        stratify_col: Column to stratify on (e.g., class4_label).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        DataFrame with 'split' column assigned.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    df = df.copy()
    df["split"] = ""

    # Build a track-level table: one row per unique track_id, with its
    # dominant class4_label and source_dataset for stratification.
    track_table = (
        df.groupby(group_col)
        .agg(
            strat_label=(stratify_col, "first"),
            source_dataset=("source_dataset", "first"),
        )
        .reset_index()
    )
    # Stratify on the combination of dataset and label
    track_table["strat_key"] = (
        track_table["source_dataset"] + "_" + track_table["strat_label"].astype(str)
    )

    # Split tracks into train / temp, then temp into val / test
    # We stratify within each strat_key group to preserve class proportions
    tracks = track_table[group_col].values
    strat_keys = track_table["strat_key"].values

    train_tracks, temp_tracks, temp_strat = _stratified_group_split(
        tracks, strat_keys, train_size=train_ratio, seed=seed,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    val_tracks, test_tracks, _ = _stratified_group_split(
        temp_tracks, temp_strat, train_size=val_share, seed=seed,
    )

    train_set = set(train_tracks)
    val_set = set(val_tracks)
    test_set = set(test_tracks)

    df.loc[df[group_col].isin(train_set), "split"] = "train"
    df.loc[df[group_col].isin(val_set), "split"] = "val"
    df.loc[df[group_col].isin(test_set), "split"] = "test"

    # Log split summary
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        n_tracks = subset[group_col].nunique()
        n_rows = len(subset)
        log.info("Split %-5s: %6d tracks, %7d rows", split_name, n_tracks, n_rows)

    return df


def _stratified_group_split(groups, strat_labels, train_size, seed):
    """Split groups into two sets, stratified by strat_labels.

    Falls back to unstratified GroupShuffleSplit for strata that are too
    small (fewer than 2 unique groups).

    Returns:
        (train_groups, test_groups, test_strat_labels)
    """
    groups = np.asarray(groups)
    strat_labels = np.asarray(strat_labels)

    train_mask = np.zeros(len(groups), dtype=bool)

    unique_strata = np.unique(strat_labels)
    rng = np.random.RandomState(seed)

    for stratum in unique_strata:
        mask = strat_labels == stratum
        stratum_groups = groups[mask]
        unique_in_stratum = np.unique(stratum_groups)

        if len(unique_in_stratum) < 2:
            # Too few groups to split — put in train
            train_mask[mask] = True
            continue

        n_train = max(1, int(round(len(unique_in_stratum) * train_size)))
        n_train = min(n_train, len(unique_in_stratum) - 1)  # ensure at least 1 in test

        shuffled = rng.permutation(unique_in_stratum)
        train_groups_stratum = set(shuffled[:n_train])

        for i in np.where(mask)[0]:
            if groups[i] in train_groups_stratum:
                train_mask[i] = True

    test_mask = ~train_mask
    return groups[train_mask], groups[test_mask], strat_labels[test_mask]
