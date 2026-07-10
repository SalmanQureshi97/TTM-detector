"""Expand a master manifest into a chunked manifest: one row per 30 s chunk.

Every input row becomes N rows, where
  N = max(1, floor(track_duration_sec / chunk_seconds)).

Long tracks contribute all their non-overlapping chunks. Short tracks
(<chunk_seconds) contribute a single chunk that the training loader
zero-pads. Trailing remainder ( < chunk_seconds ) of long tracks is
dropped (see build_chunked_manifest.py:118). All manifest columns are
preserved; three new columns are added:
  * chunk_idx           (int)   0, 1, 2, ...
  * chunk_start_sec     (float) 0.0, 30.0, 60.0, ...
  * track_duration_sec  (float) from torchaudio.info()

Only track duration is read per file (no decode), so this is I/O-cheap
even for ~180k files.

Usage:
    python scripts/build_chunked_manifest.py \\
        --manifest /home/jovyan/.../master_manifest_with_splits.csv \\
        --output   /home/jovyan/.../master_manifest_with_splits_chunked.csv \\
        --chunk-seconds 30
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _worker_init():
    """Cap threads so N info() workers don't fight over BLAS."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _duration(filepath):
    """Return duration_sec, or None on any read failure."""
    try:
        info = torchaudio.info(filepath)
        return info.num_frames / info.sample_rate
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="Path to master_manifest_with_splits.csv")
    p.add_argument("--output", required=True,
                   help="Path to write chunked manifest CSV")
    p.add_argument("--chunk-seconds", type=float, default=30.0,
                   help="Chunk length in seconds (default 30).")
    p.add_argument("--num-workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--chunksize", type=int, default=32,
                   help="pool.imap_unordered chunksize. info() is fast, "
                        "so keep this high to amortise dispatch.")
    p.add_argument("--filepath-col", default="filepath")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("chunk_manifest")
    args = parse_args()

    log.info("Loading manifest %s", args.manifest)
    df = pd.read_csv(args.manifest, low_memory=False)
    log.info("Input rows: %d", len(df))

    filepaths = df[args.filepath_col].tolist()

    torch.set_num_threads(1)
    log.info("Reading durations with %d workers (chunksize=%d) ...",
             args.num_workers, args.chunksize)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_workers,
                  initializer=_worker_init) as pool:
        durations = list(tqdm(
            pool.imap(_duration, filepaths, chunksize=args.chunksize),
            total=len(filepaths), desc="info()", mininterval=1.0,
        ))

    df["track_duration_sec"] = durations
    n_bad = df["track_duration_sec"].isna().sum()
    if n_bad:
        log.warning("Dropping %d rows with unreadable duration.", n_bad)
        df = df.dropna(subset=["track_duration_sec"]).reset_index(drop=True)

    # Expand each row into N chunks.
    log.info("Expanding into chunks of %.1f s ...", args.chunk_seconds)
    chunk_sec = float(args.chunk_seconds)

    n_chunks = df["track_duration_sec"].apply(
        lambda t: max(1, int(t // chunk_sec)) if t >= chunk_sec else 1
    ).astype(int)

    # pandas.Index.repeat expands each row `n` times; we then attach chunk_idx.
    expanded = df.loc[df.index.repeat(n_chunks)].reset_index(drop=True)
    expanded["chunk_idx"] = _per_parent_chunk_idx(n_chunks)
    expanded["chunk_start_sec"] = expanded["chunk_idx"].astype(float) * chunk_sec

    log.info("Output rows: %d (avg %.2f chunks/track)",
             len(expanded), len(expanded) / max(len(df), 1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(out_path, index=False)
    log.info("Saved %s", out_path)

    # Per-class row-count summary if the manifest has a class4_label col.
    if "class4_label" in expanded.columns:
        counts = expanded.groupby("class4_label").size()
        log.info("Per-class chunked-row counts: %s", counts.to_dict())


def _per_parent_chunk_idx(n_chunks):
    """Given an array of chunk-counts per parent row, return an array of
    per-parent chunk indices (0..n-1 within each parent group)."""
    out = []
    for n in n_chunks.tolist():
        out.extend(range(int(n)))
    return out


if __name__ == "__main__":
    main()
