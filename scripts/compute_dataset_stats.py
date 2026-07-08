"""Compute dataset-wide audio + spectrogram statistics over a manifest slice.

The output JSON is consumed by ``DeezerAmplitudeFrontend`` when its config sets
``stats_file:``. Same (mean, std) is applied to every input at both training
and inference so the model never sees an unfamiliar scale/offset.

Preprocessing mirrors ``AudioManifestDataset._process_row`` + the
``DeezerAmplitudeFrontend`` spectrogram op exactly, so stats are computed on
the same distribution the model will see at forward-pass time.

Usage:
    python scripts/compute_dataset_stats.py \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_with_splits.csv \\
        --source SONICS --split train \\
        --sample-rate 44100 --n-fft 2048 --hop-length 512 --hf-cut 16000 \\
        --num-workers 4 \\
        --output outputs/dataset_stats/sonics_c1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
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


_WORKER_STATE = {}


def _worker_init(sample_rate, max_seconds, n_fft, hop_length, hf_cut):
    """Each worker: cap threads, pre-build the STFT modules once."""
    # Prevent every worker from spawning `nproc` BLAS threads. Without this,
    # N workers x M cores = N*M threads all fighting each other, which is why
    # the server slows to a crawl.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    _WORKER_STATE["sample_rate"] = int(sample_rate)
    _WORKER_STATE["max_len"] = int(sample_rate * max_seconds)
    _WORKER_STATE["hf_cut"] = int(hf_cut)
    _WORKER_STATE["spec"] = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    _WORKER_STATE["db"] = torchaudio.transforms.AmplitudeToDB()


def _worker(filepath):
    """Load one file, return per-file audio + spec running sums.

    Returns a 6-tuple: (audio_sum, audio_sum_sq, audio_n,
                        spec_sum,  spec_sum_sq,  spec_n).
    On any load / decode error returns None so the caller can skip.
    """
    try:
        sample_rate = _WORKER_STATE["sample_rate"]
        max_len = _WORKER_STATE["max_len"]
        hf_cut = _WORKER_STATE["hf_cut"]
        spec_op = _WORKER_STATE["spec"]
        to_db = _WORKER_STATE["db"]

        # Peek at the file's own sample rate so we can decode only the frames
        # we need. Decoding the whole track when we only keep 30 s is the #1
        # bottleneck for long SONICS clips.
        info = torchaudio.info(filepath)
        src_sr = info.sample_rate
        # +100 ms slack in case of resampling edge trimming.
        needed_src_frames = int(max_len * src_sr / sample_rate) + int(0.1 * src_sr)
        audio, sr = torchaudio.load(filepath, num_frames=needed_src_frames)

        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        audio = audio.mean(dim=0)  # mono

        if audio.numel() > max_len:
            audio = audio[:max_len]
        elif audio.numel() < max_len:
            audio = torch.nn.functional.pad(audio, (0, max_len - audio.numel()))

        # Audio-domain stats (float64 sums for numerical stability).
        aud64 = audio.to(torch.float64)
        audio_sum = float(aud64.sum().item())
        audio_sum_sq = float((aud64 * aud64).sum().item())
        audio_n = int(aud64.numel())

        # Spectrogram-domain stats: replicate DeezerAmplitudeFrontend exactly.
        spec = to_db(spec_op(audio))
        hz_per_bin = sample_rate / 2.0 / spec.shape[-2]
        max_bin = int(hf_cut / hz_per_bin)
        spec = spec[:max_bin, :]

        spec64 = spec.to(torch.float64)
        spec_sum = float(spec64.sum().item())
        spec_sum_sq = float((spec64 * spec64).sum().item())
        spec_n = int(spec64.numel())

        return (audio_sum, audio_sum_sq, audio_n,
                spec_sum, spec_sum_sq, spec_n)
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
    p.add_argument("--num-workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 1),
                   help="Default: cpu_count() - 1. Each worker uses a "
                        "single BLAS thread (see _worker_init).")
    p.add_argument("--chunksize", type=int, default=8,
                   help="pool.imap_unordered chunksize. 8-32 is a sweet "
                        "spot for I/O-heavy jobs; go higher if the pool "
                        "dispatcher is CPU-bound.")
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

    filepaths = df["filepath"].tolist()

    # Running sums (float64) aggregated across workers.
    audio_sum = 0.0
    audio_sum_sq = 0.0
    audio_n = 0
    spec_sum = 0.0
    spec_sum_sq = 0.0
    spec_n = 0
    n_ok = 0
    n_failed = 0

    # Cap the parent process's own threading too -- torch.load in the main
    # process (via manifest read etc.) can otherwise still spawn worker
    # BLAS threads that compete with the pool.
    torch.set_num_threads(1)

    log.info("Spawning %d workers over %d files (chunksize=%d) ...",
             args.num_workers, len(filepaths), args.chunksize)
    init_args = (args.sample_rate, args.max_seconds,
                 args.n_fft, args.hop_length, args.hf_cut)
    ctx = mp.get_context("spawn")  # safest for torch + macOS
    with ctx.Pool(processes=args.num_workers,
                  initializer=_worker_init,
                  initargs=init_args) as pool:
        # imap_unordered streams results back as they finish. chunksize > 1
        # amortises IPC / dispatch overhead across many small tasks -- with
        # ~180k files this is dramatically faster than submit()-per-file.
        it = pool.imap_unordered(_worker, filepaths, chunksize=args.chunksize)
        for res in tqdm(it, total=len(filepaths), desc="accumulate",
                        mininterval=1.0):
            if res is None:
                n_failed += 1
                continue
            a_s, a_sq, a_n, s_s, s_sq, s_n = res
            audio_sum += a_s
            audio_sum_sq += a_sq
            audio_n += a_n
            spec_sum += s_s
            spec_sum_sq += s_sq
            spec_n += s_n
            n_ok += 1

    if audio_n == 0 or spec_n == 0:
        log.error("No samples accumulated (n_ok=%d, n_failed=%d). Aborting.",
                  n_ok, n_failed)
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
            "n_files_processed": n_ok,
            "n_files_failed": n_failed,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out_path)
    log.info("  audio: mean=%.6f std=%.6f  (n=%d)", audio_mean, audio_std, audio_n)
    log.info("  spec:  mean=%.6f std=%.6f  (n=%d)", spec_mean,  spec_std,  spec_n)
    log.info("  files: %d ok, %d failed", n_ok, n_failed)


if __name__ == "__main__":
    main()
