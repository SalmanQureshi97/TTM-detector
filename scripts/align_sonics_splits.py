"""Override the SONICS rows in the master manifest to match SONICS' official splits.

Why: the SpecTTTra ALPHA-120s checkpoint was pretrained on SONICS using a specific
train/val/test partition. If we evaluate the model on any SONICS row it has already
seen during pretraining, that's leakage. This script aligns the ``split`` column
**at the file level** for every SONICS row in our manifest, looking up the
SONICS-official split by the basename of the audio file. FMA and FakeMusicCaps
rows are left untouched -- the SpecTTTra backbone never saw them, so our
group-aware splits are fine.

Encoded variants (``sonics_real_encoded``, ``sonics_fake_encoded``) inherit the
SONICS-official split through filename matching -- the stem of an encoded file is
identical to its source's stem.

Note on track_id grouping: SONICS may split variants of the same logical song
(e.g. ``fake_36081_udio_0`` vs ``fake_36081_udio_1``) into different splits. Our
original ``assign_group_splits`` grouped all variants by ``track_id``. After this
alignment, a SONICS ``track_id`` may span multiple splits -- this is by design,
because the correct leakage invariant for a SONICS-pretrained backbone is
**file-level** (every file in our val/test must be a file SpecTTTra never saw),
not track-level. FMA/FMC rows retain their track-grouped splits.

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


def _report_track_conflicts(df):
    """Count SONICS track_ids whose variants now land in >1 split.

    We **do not** resolve these. SONICS-official splits operate at the file
    level; forcing variants of a track into a single split (e.g. by majority
    vote) could move a file SpecTTTra saw during pretraining into our val/test
    -- the dangerous direction of leakage. Spans are logged for transparency.
    """
    son = df[df["source_dataset"] == "SONICS"]
    spans = son.groupby("track_id")["split"].nunique()
    return int((spans > 1).sum())


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

    # Override split for SONICS rows that we could match -- at the FILE level.
    overridden_split = m.loc[sonics_mask & matched, "_stem"].map(official_map)
    m.loc[sonics_mask & matched, "split"] = overridden_split.values

    # Information: how many SONICS track_ids now span more than one split.
    # We deliberately do NOT resolve this; see module docstring.
    sonics_spans = _report_track_conflicts(m)
    print(f"  SONICS track_ids spanning multiple splits (expected, file-level alignment): {sonics_spans}")

    # Verify SONICS rows are file-level aligned with the SONICS-official map.
    son = m[sonics_mask].copy()
    son["_official"] = son["_stem"].map(official_map)
    aligned_ok = bool((son["split"] == son["_official"]).all())
    if not aligned_ok:
        raise RuntimeError("SpecTTTra leakage check failed: a SONICS row's split "
                           "does not match the SONICS-official split.")
    print(f"  SpecTTTra leakage check (every SONICS file == SONICS-official split): True")

    # Verify FMA / FakeMusicCaps still respect track-grouped splits (their original invariant).
    non_son = m[~sonics_mask]
    non_son_spans = non_son.groupby("track_id")["split"].nunique()
    fma_fmc_ok = bool((non_son_spans == 1).all())
    if not fma_fmc_ok:
        print(f"  WARNING: {int((non_son_spans>1).sum())} non-SONICS track_ids span >1 split.")
    else:
        print(f"  FMA/FMC group invariant (one split per track_id outside SONICS): True")

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
