"""Measure the exact audio lengths of the samples used by the Grad-CAM run.

Reproduces the per-slice sampling that ``run_gradcam_analysis.py`` does
(same SEED, same ``--n-aggregate``), then reads each file's duration via
``torchaudio.info()`` -- just the file header, no decoding. Produces:

  * a CSV with per-file durations under
    ``outputs/.../gradcam/audio_length_audit.csv``;
  * a summary table to stdout with mean / median / min / max duration and
    the actual audio-fraction inside each model's fixed input window.

Run separately per model (the two models have different sample rates and
segment windows, so the audio fraction differs):

    python scripts/audio_length_audit.py \\
        --model configs/models/deezer_speccnn_amplitude.yaml \\
        --task configs/tasks/four_class_flat.yaml \\
        --experiment configs/experiments/c1_sonics_only.yaml \\
        --runtime configs/runtime/train_default.yaml \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_sonics_aligned.csv \\
        --n-aggregate 50

For SpecTTTra, swap ``--model`` to
``configs/models/sonics_spectttra_alpha120s.yaml`` and ``--runtime`` to
``configs/runtime/train_spectttra.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torchaudio  # noqa: E402

from src.utils.config import load_yaml  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--n-aggregate", type=int, default=50,
                   help="Must match the value used by run_gradcam_analysis.py "
                        "so we sample the same files.")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the Grad-CAM run's seed.")
    return p.parse_args()


def file_duration_seconds(path):
    """Return audio duration in seconds via torchaudio.info() (header only)."""
    info = torchaudio.info(path)
    if info.num_frames <= 0 or info.sample_rate <= 0:
        return float("nan")
    return info.num_frames / info.sample_rate


def slice_sample(df, source, class4, n, seed, split):
    """Reproduce the slice sampling that run_gradcam_analysis.py uses."""
    sub = df[df["source_dataset"] == source]
    sub = sub[sub["split"] == split]
    if class4 is not None:
        sub = sub[sub["class4_label"] == class4]
    sub = sub.reset_index(drop=True)
    if len(sub) == 0:
        return sub
    take = min(n, len(sub))
    return sub.sample(n=take, random_state=seed).reset_index(drop=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("audit")

    args = parse_args()
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    exp_cfg = load_yaml(args.experiment)
    rt = load_yaml(args.runtime)

    sample_rate = model_cfg["frontend"]["sample_rate"]
    window_s = rt["segment_seconds"]
    log.info("Model: %s | sample_rate=%d Hz | window=%d s",
             model_cfg["name"], sample_rate, window_s)

    save_dir = (Path(rt["save_dir"]) / exp_cfg["name"]
                / model_cfg["name"] / task_cfg["name"] / "gradcam")
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "audio_length_audit.csv"

    df = pd.read_csv(args.manifest, low_memory=False)
    df["track_id"] = df["track_id"].astype(str)

    SLICES = [
        ("sonics_val",   "SONICS",        None, "val"),
        ("fma_test",     "FMA",           None, "test"),
        ("fmc_natural",  "FakeMusicCaps", 2,    "test"),
        ("fmc_encoded",  "FakeMusicCaps", 3,    "test"),
    ]

    rows = []
    summary = []

    for slice_name, source, class4, split in SLICES:
        sub = slice_sample(df, source, class4, args.n_aggregate, args.seed, split)
        if len(sub) == 0:
            log.warning("slice %s empty -- skipped", slice_name)
            continue
        log.info("Slice %s: %d files", slice_name, len(sub))

        durs = []
        for fp in sub["filepath"]:
            try:
                d = file_duration_seconds(fp)
                durs.append(d)
                rows.append({
                    "slice": slice_name,
                    "filepath": fp,
                    "duration_s": d,
                    "window_s": window_s,
                    "audio_fraction": min(d, window_s) / window_s,
                })
            except Exception as err:
                log.warning("  %s: failed (%s)", fp, err)

        durs = [d for d in durs if d == d]  # drop NaN
        if not durs:
            continue
        mean_d = statistics.mean(durs)
        median_d = statistics.median(durs)
        min_d = min(durs)
        max_d = max(durs)
        # The actual model input fraction that is real audio (after the
        # truncate-or-pad rule in AudioManifestDataset):
        audio_fracs = [min(d, window_s) / window_s for d in durs]
        mean_audio_frac = statistics.mean(audio_fracs)
        pct_short = sum(1 for d in durs if d < window_s) / len(durs)
        summary.append({
            "slice": slice_name,
            "n": len(durs),
            "mean_dur_s": mean_d,
            "median_dur_s": median_d,
            "min_dur_s": min_d,
            "max_dur_s": max_d,
            "pct_clips_below_window": pct_short,
            "mean_audio_fraction_in_window": mean_audio_frac,
            "mean_silence_fraction_in_window": 1.0 - mean_audio_frac,
        })

    # Write per-file CSV
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s (%d rows)", csv_path, len(rows))

    # Stdout summary
    print(f"\nModel: {model_cfg['name']}  |  window = {window_s} s @ {sample_rate} Hz\n")
    cols = ["slice", "n", "mean_dur_s", "median_dur_s", "min_dur_s", "max_dur_s",
            "pct_clips_below_window", "mean_audio_fraction_in_window",
            "mean_silence_fraction_in_window"]
    headers = ["slice", "n", "mean(s)", "median(s)", "min(s)", "max(s)",
               "%clips<window", "audio frac", "silence frac"]
    print("  ".join(f"{h:>14s}" for h in headers))
    for s in summary:
        vals = [
            s["slice"],
            s["n"],
            f"{s['mean_dur_s']:.2f}",
            f"{s['median_dur_s']:.2f}",
            f"{s['min_dur_s']:.2f}",
            f"{s['max_dur_s']:.2f}",
            f"{s['pct_clips_below_window']*100:.1f}%",
            f"{s['mean_audio_fraction_in_window']*100:.1f}%",
            f"{s['mean_silence_fraction_in_window']*100:.1f}%",
        ]
        print("  ".join(f"{str(v):>14s}" for v in vals))


if __name__ == "__main__":
    main()
