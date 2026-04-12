from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def filter_manifest(df: pd.DataFrame, datasets, split):
    out = df[df["source_dataset"].isin(datasets)].copy()
    if split is not None:
        out = out[out["split"] == split].copy()
    return out


def assign_group_splits(
    df: pd.DataFrame,
    group_col: str = "track_id",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    df = df.copy()
    groups = df[group_col].astype(str).to_numpy()
    indices = df.index.to_numpy()

    first_split = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(first_split.split(indices, groups=groups))
    df["split"] = ""
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df[group_col].astype(str).to_numpy()
    temp_indices = temp_df.index.to_numpy()
    val_share = val_ratio / (val_ratio + test_ratio)

    second_split = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=seed)
    val_sub_idx, test_sub_idx = next(second_split.split(temp_indices, groups=temp_groups))

    df.loc[temp_indices[val_sub_idx], "split"] = "val"
    df.loc[temp_indices[test_sub_idx], "split"] = "test"
    return df
