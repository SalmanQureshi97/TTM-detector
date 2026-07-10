"""Compute dataset-wide audio + spectrogram statistics over a manifest slice.

The output JSON is consumed by ``DeezerAmplitudeFrontend`` when its config sets
``stats_file:``. Same (mean, std) is applied to every input at both training
and inference so the model never sees an unfamiliar scale/offset.

Two input-manifest modes:
  * **Chunked manifest** (recommended, matches the new training pipeline):
    the manifest has a ``chunk_start_sec`` column emitted by
    ``scripts/build_chunked_manifest.py``. This script groups all rows by
    ``filepath`` so each file is decoded exactly once, then iterates every
    listed chunk within that file, accumulating stats across chunks. Matches
    what the training loader sees.
  * **Legacy** (no ``chunk_start_sec`` column): treat each row as a single
    "first ``max_seconds`` seconds" chunk. Equivalent to the pre-chunking
    training pipeline.

Preprocessing (per chunk) mirrors ``AudioManifestDataset._process_row`` and
the ``DeezerAmplitudeFrontend`` spectrogram op exactly.

Usage:
    python scripts/compute_dataset_stats.py \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_with_splits_chunked.csv \\
        --source SONICS --split train \\
        --sample-rate 44100 --n-fft 2048 --hop-length 512 --hf-cut 16000 \\
        --output outputs/dataset_stats/sonics_c1_chunked.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _process_file(filepath, chunk_starts, cfg, spec_op, to_db):
    """Process one file and return per-file running sums over all chunks.

    chunk_starts: list of chunk-start times in seconds. Empty list =>
    "first max_seconds only" (legacy manifest mode).
    Returns 6-tuple (audio_sum, audio_sum_sq, audio_n,
                     spec_sum,  spec_sum_sq,  spec_n)
    or None on any load failure.
    """
    try:
        sample_rate = cfg["sample_rate"]
        max_len = cfg["max_len"]
        hf_cut = cfg["hf_cut"]

        info = torchaudio.info(filepath)
        src_sr = info.sample_rate
        chunk_src_frames = int(max_len * src_sr / sample_rate)
        slack = int(0.1 * src_sr)

        if not chunk_starts:
            chunk_starts = [0.0]

        tot_a_s = tot_a_sq = 0.0; tot_a_n = 0
        tot_s_s = tot_s_sq = 0.0; tot_s_n = 0

        for start_sec in chunk_starts:
            frame_offset = int(float(start_sec) * src_sr)
            audio, sr = torchaudio.load(
                filepath,
                frame_offset=frame_offset,
                num_frames=chunk_src_frames + slack,
            )
            if sr != sample_rate:
                audio = torchaudio.functional.resample(audio, sr, sample_rate)
            audio = audio.mean(dim=0)  # mono
            if audio.numel() > max_len:
                audio = audio[:max_len]
            elif audio.numel() < max_len:
                audio = torch.nn.functional.pad(
                    audio, (0, max_len - audio.numel())
                )

            aud64 = audio.to(torch.float64)
            tot_a_s += float(aud64.sum().item())
            tot_a_sq += float((aud64 * aud64).sum().item())
            tot_a_n += int(aud64.numel())

            spec = to_db(spec_op(audio))
            hz_per_bin = sample_rate / 2.0 / spec.shape[-2]
            max_bin = int(hf_cut / hz_per_bin)
            spec = spec[:max_bin, :]
            spec64 = spec.to(torch.float64)
            tot_s_s += float(spec64.sum().item())
            tot_s_sq += float((spec64 * spec64).sum().item())
            tot_s_n += int(spec64.numel())

        return (tot_a_s, tot_a_sq, tot_a_n,
                tot_s_s, tot_s_sq, tot_s_n)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--source", default="SONICS",
                   help="Filter on manifest's source_dataset column.")
    p.add_argument("--split", default="train",
                   help="Manifest split (train / val / test).")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--max-seconds", type=float, default=30.0,
                   help="Truncate/pad every clip to this length before "
                        "accumulating stats. Matches the training loader.")
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--hf-cut", type=int, default=16000,
                   help="Upper frequency (Hz) to keep in the spectrogram, "
                        "mirroring DeezerAmplitudeFrontend.hf_cut.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("dataset_stats")
    args = parse_args()

    log.info("Loading manifest %s", args.manifest)
    df = pd.read_csv(args.manifest, low_memory=False)
    log.info("Manifest total rows: %d", len(df))

    df = df[df["source_dataset"] == args.source]
    df = df[df["split"] == args.split].reset_index(drop=True)
    log.info("Filtered to source=%s split=%s -> %d rows",
             args.source, args.split, len(df))
    if len(df) == 0:
        log.error("No rows to process -- aborting.")
        return

    # Group rows by filepath so each file is decoded exactly once, even when
    # the chunked manifest has multiple rows per file (one per chunk).
    if "chunk_start_sec" in df.columns:
        log.info("Chunked manifest detected (%d chunk rows over %d files, "
                 "avg %.2f chunks/file).",
                 len(df), df["filepath"].nunique(),
                 len(df) / max(df["filepath"].nunique(), 1))
        grouped = (df.groupby("filepath", sort=False)["chunk_start_sec"]
                     .apply(lambda s: sorted(float(x) for x in s.tolist())))
        tasks = [(fp, starts) for fp, starts in grouped.items()]
    else:
        log.info("Legacy manifest (no chunk_start_sec). Using first "
                 "%.1f s of every file.", args.max_seconds)
        tasks = [(fp, []) for fp in df["filepath"].tolist()]

    n_files = len(tasks)
    n_chunks_total = sum(max(1, len(s)) for _, s in tasks)

    # Running sums (float64).
    audio_sum = 0.0
    audio_sum_sq = 0.0
    audio_n = 0
    spec_sum = 0.0
    spec_sum_sq = 0.0
    spec_n = 0
    n_ok_files = 0
    n_failed_files = 0

    cfg = {
        "sample_rate": args.sample_rate,
        "max_len": int(args.sample_rate * args.max_seconds),
        "hf_cut": args.hf_cut,
    }
    spec_op = torchaudio.transforms.Spectrogram(
        n_fft=args.n_fft, hop_length=args.hop_length, power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB()

    log.info("Processing %d files / %d chunks ...", n_files, n_chunks_total)
    for filepath, chunk_starts in tqdm(tasks, desc="accumulate",
                                       mininterval=0.5):
        res = _process_file(filepath, chunk_starts, cfg, spec_op, to_db)
        if res is None:
            n_failed_files += 1
            continue
        a_s, a_sq, a_n, s_s, s_sq, s_n = res
        audio_sum += a_s
        audio_sum_sq += a_sq
        audio_n += a_n
        spec_sum += s_s
        spec_sum_sq += s_sq
        spec_n += s_n
        n_ok_files += 1

    if audio_n == 0 or spec_n == 0:
        log.error("No samples accumulated (n_ok_files=%d, n_failed_files=%d). "
                  "Aborting.", n_ok_files, n_failed_files)
        return

    audio_mean = audio_sum / audio_n
    audio_var = max(audio_sum_sq / audio_n - audio_mean * audio_mean, 0.0)
    audio_std = math.sqrt(audio_var)

    spec_mean = spec_sum / spec_n
    spec_var = max(spec_sum_sq / spec_n - spec_mean * spec_mean, 0.0)
    spec_std = math.sqrt(spec_var)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio":  {"mean": audio_mean, "std": audio_std,
                   "n_samples": audio_n},
        "spec":   {"mean": spec_mean,  "std": spec_std,
                   "n_samples": spec_n},
        "config": {
            "sample_rate": args.sample_rate,
            "max_seconds": args.max_seconds,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "hf_cut": args.hf_cut,
            "manifest": args.manifest,
            "source": args.source,
            "split": args.split,
            "n_files_processed": n_ok_files,
            "n_files_failed": n_failed_files,
            "n_chunks_total": n_chunks_total,
            "chunked_manifest": "chunk_start_sec" in df.columns,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out_path)
    log.info("  audio: mean=%.6f std=%.6f  (n=%d)", audio_mean, audio_std, audio_n)
    log.info("  spec:  mean=%.6f std=%.6f  (n=%d)", spec_mean,  spec_std,  spec_n)
    log.info("  files: %d ok, %d failed  |  chunks aggregated: %d",
             n_ok_files, n_failed_files, n_chunks_total)


if __name__ == "__main__":
    main()
