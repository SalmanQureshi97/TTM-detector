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

    Decodes the file exactly ONCE (mp3 seek is fake: the decoder walks
    from position 0 to reach frame_offset, so per-chunk loads redo the
    same decode work). We load the whole file, resample once, and slice
    every chunk out in memory.

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

        audio, src_sr = torchaudio.load(filepath)
        if src_sr != sample_rate:
            audio = torchaudio.functional.resample(audio, src_sr, sample_rate)
        audio = audio.mean(dim=0)  # mono, shape [N_samples]

        if not chunk_starts:
            chunk_starts = [0.0]

        tot_a_s = tot_a_sq = 0.0; tot_a_n = 0
        tot_s_s = tot_s_sq = 0.0; tot_s_n = 0

        for start_sec in chunk_starts:
            start = int(float(start_sec) * sample_rate)
            end = start + max_len
            chunk = audio[start:end]
            if chunk.numel() < max_len:
                chunk = torch.nn.functional.pad(
                    chunk, (0, max_len - chunk.numel())
                )

            aud64 = chunk.to(torch.float64)
            tot_a_s += float(aud64.sum().item())
            tot_a_sq += float((aud64 * aud64).sum().item())
            tot_a_n += int(aud64.numel())

            spec = to_db(spec_op(chunk))
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
    p.add_argument("--resume", action="store_true",
                   help="Resume from <output>.checkpoint.json if present.")
    p.add_argument("--checkpoint-every", type=int, default=1000,
                   help="Persist running sums + index after every N files "
                        "(default 1000). Rename-atomic write so a crash "
                        "mid-checkpoint cannot corrupt the file.")
    return p.parse_args()


def _checkpoint_path(output_path):
    return Path(str(output_path) + ".checkpoint.json")


def _save_checkpoint(ckpt_path, state):
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    tmp.replace(ckpt_path)  # atomic on posix


def _load_checkpoint(ckpt_path):
    with open(ckpt_path) as f:
        return json.load(f)


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
    # Sort by filepath so the task order is fully deterministic -- required
    # for resume to skip the right prefix on restart.
    if "chunk_start_sec" in df.columns:
        log.info("Chunked manifest detected (%d chunk rows over %d files, "
                 "avg %.2f chunks/file).",
                 len(df), df["filepath"].nunique(),
                 len(df) / max(df["filepath"].nunique(), 1))
        grouped = (df.groupby("filepath", sort=True)["chunk_start_sec"]
                     .apply(lambda s: sorted(float(x) for x in s.tolist())))
        tasks = [(fp, starts) for fp, starts in grouped.items()]
    else:
        log.info("Legacy manifest (no chunk_start_sec). Using first "
                 "%.1f s of every file.", args.max_seconds)
        tasks = sorted(
            [(fp, []) for fp in df["filepath"].tolist()],
            key=lambda t: t[0],
        )

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
    start_index = 0

    out_path = Path(args.output)
    ckpt_path = _checkpoint_path(out_path)

    if args.resume and ckpt_path.exists():
        state = _load_checkpoint(ckpt_path)
        # Sanity check: the task-set must match what the checkpoint was
        # written against. If someone changes the manifest or the CLI args
        # between runs, silently resuming would produce garbage.
        if (state.get("n_files") != n_files
                or state.get("sample_rate") != args.sample_rate
                or state.get("max_seconds") != args.max_seconds
                or state.get("n_fft") != args.n_fft
                or state.get("hop_length") != args.hop_length
                or state.get("hf_cut") != args.hf_cut):
            log.error("Checkpoint %s doesn't match current CLI/manifest. "
                      "Delete it and re-run, or unset --resume.", ckpt_path)
            return
        audio_sum = state["audio_sum"]
        audio_sum_sq = state["audio_sum_sq"]
        audio_n = state["audio_n"]
        spec_sum = state["spec_sum"]
        spec_sum_sq = state["spec_sum_sq"]
        spec_n = state["spec_n"]
        n_ok_files = state["n_ok_files"]
        n_failed_files = state["n_failed_files"]
        start_index = state["next_index"]
        log.info("Resuming from checkpoint: %d/%d files already done.",
                 start_index, n_files)

    cfg = {
        "sample_rate": args.sample_rate,
        "max_len": int(args.sample_rate * args.max_seconds),
        "hf_cut": args.hf_cut,
    }
    spec_op = torchaudio.transforms.Spectrogram(
        n_fft=args.n_fft, hop_length=args.hop_length, power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB()

    def _write_checkpoint(next_index):
        _save_checkpoint(ckpt_path, {
            "n_files": n_files,
            "sample_rate": args.sample_rate,
            "max_seconds": args.max_seconds,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "hf_cut": args.hf_cut,
            "audio_sum": audio_sum,
            "audio_sum_sq": audio_sum_sq,
            "audio_n": audio_n,
            "spec_sum": spec_sum,
            "spec_sum_sq": spec_sum_sq,
            "spec_n": spec_n,
            "n_ok_files": n_ok_files,
            "n_failed_files": n_failed_files,
            "next_index": next_index,
        })

    log.info("Processing %d files / %d chunks (starting at index %d) ...",
             n_files, n_chunks_total, start_index)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    i = start_index  # so the KeyboardInterrupt handler can reference it
    try:
        for i in tqdm(range(start_index, n_files), desc="accumulate",
                      initial=start_index, total=n_files, mininterval=0.5):
            filepath, chunk_starts = tasks[i]
            res = _process_file(filepath, chunk_starts, cfg, spec_op, to_db)
            if res is None:
                n_failed_files += 1
            else:
                a_s, a_sq, a_n, s_s, s_sq, s_n = res
                audio_sum += a_s
                audio_sum_sq += a_sq
                audio_n += a_n
                spec_sum += s_s
                spec_sum_sq += s_sq
                spec_n += s_n
                n_ok_files += 1
            if (i + 1) % args.checkpoint_every == 0:
                _write_checkpoint(i + 1)
    except KeyboardInterrupt:
        _write_checkpoint(i + 1)
        log.warning("Interrupted; checkpoint saved at index %d. Re-run "
                    "with --resume to continue.", i + 1)
        raise

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
    # Successful full pass -- remove the checkpoint so a later --resume
    # doesn't accidentally restart against stale state.
    if ckpt_path.exists():
        ckpt_path.unlink()
    log.info("Saved %s", out_path)
    log.info("  audio: mean=%.6f std=%.6f  (n=%d)", audio_mean, audio_std, audio_n)
    log.info("  spec:  mean=%.6f std=%.6f  (n=%d)", spec_mean,  spec_std,  spec_n)
    log.info("  files: %d ok, %d failed  |  chunks aggregated: %d",
             n_ok_files, n_failed_files, n_chunks_total)


if __name__ == "__main__":
    main()
