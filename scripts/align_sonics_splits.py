"""Override the SONICS rows in the master manifest to match SONICS' official splits.

Why: the SpecTTTra ALPHA-120s checkpoint was pretrained on SONICS using a specific
train/val/test partition. If we evaluate the model on any SONICS row it has already
seen during pretraining, that's leakage. This script aligns the ``split`` column for
every SONICS row in our manifest to the SONICS-official split (looked up by the
basename of the audio file). FMA and FakeMusicCaps rows are left untouched -- the
SpecTTTra backbone never saw them, so our group-aware splits are fine.

Encoded variants (``sonics_real_encoded``, ``sonics_fake_encoded``) inherit the
SONICS-official split through filename matching -- the stem of an encoded file is
identical to its source's stem.

Usage:
    python scripts/align_sonics_splits.py \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_with_splits.csv \\
        --sonics-train /home/jovyan/Thesis/Code/data/sonics_splits/train.csv \\
        --sonics-valid /home/jovyan/Thesis/Code/data/sonics_splits/valid.csv \\
        --sonics-test  /home/jovyan/Thesis/Code/data/sonics_splits/test.csv \\
        --output /home/jovyan/Thesis/Code/data/manifests/master_manifest_sonics_aligned.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="Path to master_manifest_with_splits.csv")
    p.add_argument("--sonics-train", required=True, help="SONICS official train.csv")
    p.add_argument("--sonics-valid", required=True, help="SONICS official valid.csv")
    p.add_argument("--sonics-test",  required=True, help="SONICS official test.csv")
    p.add_argument("--output", required=True, help="Output manifest CSV path")
    p.add_argument("--filename-col", default="filename",
                   help="Column in the SONICS CSVs holding the audio filename "
                        "(default: 'filename')")
    return p.parse_args()


def _build_official_map(paths_by_split, filename_col):
    """basename-without-extension -> 'train' | 'val' | 'test'."""
    mapping = {}
    for path, split in paths_by_split:
        df = pd.read_csv(path, low_memory=False)
        if filename_col not in df.columns:
            raise ValueError(
                f"{path}: missing column '{filename_col}'. "
                f"Found columns: {list(df.columns)}"
            )
        for fn in df[filename_col].dropna():
            stem = Path(str(fn)).stem
            mapping[stem] = split
    return mapping


def _resolve_track_conflicts(df):
    """Force every track_id to a single split (majority vote).

    Encoded variants of a track must end up in the same split as the
    track's source. Since encoded files inherit the source's stem and
    therefore the same SONICS-official split, conflicts here would be
    rare -- but the resolver guarantees the invariant either way.
    """
    spans = df.groupby("track_id")["split"].nunique()
    conflicted = int((spans > 1).sum())
    if conflicted == 0:
        return df, 0

    def majority(splits):
        c = Counter(splits.tolist())
        return c.most_common(1)[0][0]

    track_split = df.groupby("track_id")["split"].apply(majority)
    df = df.copy()
    df["split"] = df["track_id"].map(track_split)
    return df, conflicted


def main():
    args = parse_args()

    official_map = _build_official_map(
        [(args.sonics_train, "train"),
         (args.sonics_valid, "val"),
         (args.sonics_test,  "test")],
        filename_col=args.filename_col,
    )
    print(f"SONICS official map: {len(official_map)} unique stems "
          f"(train+val+test).")

    m = pd.read_csv(args.manifest, low_memory=False)
    m["track_id"] = m["track_id"].astype(str)
    m["_stem"] = m["filepath"].astype(str).map(lambda p: Path(p).stem)

    sonics_mask = m["source_dataset"] == "SONICS"
    n_sonics = int(sonics_mask.sum())
    print(f"SONICS rows in manifest: {n_sonics:,}")

    matched = m.loc[sonics_mask, "_stem"].isin(official_map)
    n_matched = int(matched.sum())
    n_unmatched = n_sonics - n_matched
    print(f"  matched to SONICS official: {n_matched:,}")
    print(f"  NOT matched (kept existing split): {n_unmatched:,}")
    if n_unmatched > 0:
        sample = (
            m.loc[sonics_mask & ~m["_stem"].isin(official_map), "_stem"]
            .head(10)
            .tolist()
        )
        print(f"  sample unmatched stems: {sample}")

    # Override split for SONICS rows that we could match.
    overridden_split = m.loc[sonics_mask & matched, "_stem"].map(official_map)
    m.loc[sonics_mask & matched, "split"] = overridden_split.values

    # Resolve any track_ids that now span >1 split (shouldn't happen, but verify).
    m, conflicted = _resolve_track_conflicts(m)
    if conflicted:
        print(f"  resolved {conflicted} track_id split conflicts by majority vote.")

    spans = m.groupby("track_id")["split"].nunique()
    if not (spans == 1).all():
        raise RuntimeError("Leakage check failed: some track_ids still span >1 split.")
    print(f"  leakage check (one split per track_id): True")

    m = m.drop(columns=["_stem"])

    # Summaries
    print("\n--- Counts by (source_dataset, split) ---")
    print(m.groupby(["source_dataset", "split"]).size().unstack(fill_value=0))
    print("\n--- SONICS counts by (class4_label, split) ---")
    son = m[m["source_dataset"] == "SONICS"]
    print(son.groupby(["class4_label", "split"]).size().unstack(fill_value=0))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(f"\nSaved aligned manifest -> {out}")


if __name__ == "__main__":
    main()
