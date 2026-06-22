"""Per-class Grad-CAM attention histograms split by correct/incorrect, across
in-domain (SONICS val) and out-of-domain (FMA test, FMC test) slices.

For each (slice, true-class) bucket the script:
  1. Samples ``--n-per-class`` validation/test rows whose class4_label
     matches the bucket (FMA contributes only to classes 0/1, FMC only to
     classes 2/3 -- the script skips classes not present in the slice).
  2. Runs the model forward to obtain the predicted class per sample.
  3. Runs Grad-CAM with target = predicted class (so the heatmap shows
     what convinced the model of its actual decision, right or wrong).
  4. Reduces each heatmap to a frequency-axis profile (sum over time) and
     a time-axis profile (sum over freq).
  5. Buckets per-sample profiles into ``correct`` vs ``incorrect``.

Outputs land in ``outputs/<exp>/<model>/<task>/gradcam_per_class/``:

  * ``<slice>_freq_histograms.png`` -- 2x2 class panels, correct (green)
    overlaid with incorrect (red).
  * ``<slice>_time_histograms.png`` -- same layout, time axis.
  * ``per_class_counts.csv`` -- one row per (slice, class) with counts,
    accuracy, and the predicted-class confusion breakdown.
  * ``profiles.npz`` -- raw averaged profiles per (slice, class, bucket).

Usage (SpecCNN):
    python scripts/run_gradcam_per_class.py \\
        --model configs/models/deezer_speccnn_amplitude.yaml \\
        --task configs/tasks/four_class_flat.yaml \\
        --experiment configs/experiments/c1_sonics_only.yaml \\
        --runtime configs/runtime/train_default.yaml \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_sonics_aligned.csv \\
        --n-per-class 500
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import AudioManifestDataset  # noqa: E402
from src.models.model_factory import UnifiedAudioModel  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.visualization.gradcam import GradCAM  # noqa: E402


CLASS_NAMES = ["real", "real_enc", "fake", "fake_enc"]
COLOR_CORRECT = "#2ca02c"     # green
COLOR_INCORRECT = "#d62728"   # red

# Which (source_dataset, split, allowed-classes) slices to analyse.
# FMA only has real / real_enc rows; FMC only has fake / fake_enc rows.
SLICES = [
    {"name": "sonics_val", "source": "SONICS",        "split": "val",  "classes": [0, 1, 2, 3]},
    {"name": "fma_test",   "source": "FMA",           "split": "test", "classes": [0, 1]},
    {"name": "fmc_test",   "source": "FakeMusicCaps", "split": "test", "classes": [2, 3]},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--n-per-class", type=int, default=500,
                   help="Per-(slice, class) sample cap.")
    p.add_argument("--n-freq-bins", type=int, default=32)
    p.add_argument("--n-time-bins", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Ignore any cached slice data and recompute from scratch.")
    return p.parse_args()


def normalise(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    if s <= eps:
        return np.full_like(v, 1.0 / max(v.size, 1))
    return v / s


def bin_profile(prof, n_bins):
    """Bin a normalised 1-D profile into ``n_bins`` bands by **summing**.

    The raw profile sums to 1 (probability distribution over F raw bins).
    Summing within each band preserves the property: the binned histogram
    also sums to 1 (probability distribution over n_bins bands), and each
    bar reads as "fraction of total attention that landed in this band".
    That gives a consistent y-axis scale across models, axes and panels --
    uniform attention always yields bars at 1/n_bins regardless of how
    many raw bins fed into them.
    """
    prof = np.asarray(prof, dtype=np.float64)
    F = len(prof)
    if n_bins >= F:
        return prof.copy()
    edges = np.linspace(0, F, n_bins + 1).astype(int)
    binned = np.zeros(n_bins)
    for i in range(n_bins):
        lo = edges[i]
        hi = max(edges[i] + 1, edges[i + 1])
        binned[i] = prof[lo:hi].sum()
    return binned


def analyse_class(model, cam, manifest_path, task_cfg, source, split, class4,
                  sample_rate, max_seconds, n_take_max, seed, device, log):
    """Run Grad-CAM over up to ``n_take_max`` samples of one true class
    inside one (source, split) slice. Returns bucketed mean profiles plus
    counts."""
    ds = AudioManifestDataset(
        manifest_path, split=split, task_cfg=task_cfg,
        dataset_filter=[source], sample_rate=sample_rate, max_seconds=max_seconds,
    )
    ds.df = ds.df[ds.df["class4_label"] == class4].reset_index(drop=True)
    n_available = len(ds.df)
    if n_available == 0:
        log.info("    class %d not present in %s/%s -- skipping", class4, source, split)
        return None
    n_take = min(n_take_max, n_available)
    ds.df = ds.df.sample(n=n_take, random_state=seed).reset_index(drop=True)
    log.info("    sampling %d / %d available rows", n_take, n_available)

    correct_freq, correct_time = [], []
    incorrect_freq, incorrect_time = [], []
    confusion_to = np.zeros((4,), dtype=int)
    failures = 0

    for i in range(n_take):
        try:
            item = ds[i]
        except Exception as err:
            log.warning("    load %d failed: %s", i, err)
            failures += 1
            continue
        audio = item["audio"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(audio)
            pred = int(logits.argmax(dim=1).item())
        confusion_to[pred] += 1

        try:
            heatmap = cam.compute(audio, target_class=pred)
        except Exception as err:
            log.warning("    cam %d failed: %s", i, err)
            failures += 1
            continue

        heatmap = np.asarray(heatmap, dtype=np.float64)
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-12:
            heatmap = (heatmap - hmin) / (hmax - hmin)

        freq_prof = normalise(heatmap.sum(axis=1))
        time_prof = normalise(heatmap.sum(axis=0))

        if pred == class4:
            correct_freq.append(freq_prof)
            correct_time.append(time_prof)
        else:
            incorrect_freq.append(freq_prof)
            incorrect_time.append(time_prof)

        if (i + 1) % 100 == 0:
            log.info("    processed %d/%d (correct: %d)",
                     i + 1, n_take, len(correct_freq))

    n_correct = len(correct_freq)
    n_incorrect = len(incorrect_freq)
    log.info("    correct: %d  incorrect: %d  failures: %d  acc: %.2f%%",
             n_correct, n_incorrect, failures,
             100.0 * n_correct / max(n_correct + n_incorrect, 1))

    return {
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "confusion_to": confusion_to.tolist(),
        "correct_freq": np.stack(correct_freq).mean(axis=0) if correct_freq else None,
        "correct_time": np.stack(correct_time).mean(axis=0) if correct_time else None,
        "incorrect_freq": np.stack(incorrect_freq).mean(axis=0) if incorrect_freq else None,
        "incorrect_time": np.stack(incorrect_time).mean(axis=0) if incorrect_time else None,
    }


def plot_slice(per_class, slice_name, axis, n_bins, x_min, x_max, x_label,
               model_name, out_path, y_max=None):
    """One 2x2 figure for a given slice and axis (freq or time)."""
    correct_key = f"correct_{axis}"
    incorrect_key = f"incorrect_{axis}"
    bar_width = (x_max - x_min) / max(n_bins, 1)
    x = np.linspace(x_min, x_max, n_bins)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for class4, ax in zip(range(4), axes.flat):
        d = per_class.get(class4)
        if d is None:
            ax.text(0.5, 0.5,
                    f"class {class4} ({CLASS_NAMES[class4]})\n"
                    f"not present in {slice_name}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"true class {class4}: {CLASS_NAMES[class4]}")
            continue
        if d[correct_key] is not None:
            ax.bar(x, bin_profile(d[correct_key], n_bins),
                   width=bar_width, color=COLOR_CORRECT, alpha=0.55,
                   label=f"correct (n={d['n_correct']})")
        if d[incorrect_key] is not None:
            ax.bar(x, bin_profile(d[incorrect_key], n_bins),
                   width=bar_width, color=COLOR_INCORRECT, alpha=0.55,
                   label=f"incorrect (n={d['n_incorrect']})")
        if d[correct_key] is None and d[incorrect_key] is None:
            ax.text(0.5, 0.5, f"no samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#888888")
        ax.set_title(f"true class {class4}: {CLASS_NAMES[class4]}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean normalised attention")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        if y_max is not None:
            ax.set_ylim(0, y_max)
    fig.suptitle(
        f"{model_name} -- {axis}-axis Grad-CAM histograms on {slice_name}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("gradcam_per_class")

    args = parse_args()
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    exp_cfg = load_yaml(args.experiment)
    rt = load_yaml(args.runtime)

    sample_rate = model_cfg["frontend"]["sample_rate"]
    max_seconds = rt["segment_seconds"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | sample_rate=%d Hz | window=%d s", DEVICE, sample_rate, max_seconds)

    save_dir = (Path(rt["save_dir"]) / exp_cfg["name"]
                / model_cfg["name"] / task_cfg["name"])
    out_dir = save_dir / "gradcam_per_class"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model + best.pt -----------------------------------------------
    model = UnifiedAudioModel(model_cfg, task_cfg).to(DEVICE)
    ck = torch.load(save_dir / "best.pt", map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()  # SpecAugment / dropout off; Grad-CAM gets grads via input.requires_grad_
    log.info("Loaded best.pt | epoch %s | val_loss %.4f",
             ck.get("epoch"), ck.get("val_loss", float("nan")))
    cam = GradCAM(model, model_cfg)

    # --- Frequency axis range ----------------------------------------------
    fe = model_cfg["frontend"]
    if fe["type"] == "logmel":
        freq_min_hz = fe.get("f_min", 0)
        freq_max_hz = fe.get("f_max", sample_rate // 2)
    elif fe["type"] == "deezer_amplitude":
        freq_min_hz = 0
        freq_max_hz = fe.get("hf_cut", sample_rate // 2)
    else:
        freq_min_hz, freq_max_hz = 0, sample_rate // 2

    # --- Iterate slices -----------------------------------------------------
    all_slices_data = {}     # slice_name -> {class4 -> bucket dict or None}
    csv_rows = []

    for slice_def in SLICES:
        slice_name = slice_def["name"]
        source = slice_def["source"]
        split = slice_def["split"]
        allowed_classes = set(slice_def["classes"])

        log.info("========== SLICE: %s (source=%s, split=%s) ==========",
                 slice_name, source, split)

        cache_path = out_dir / f"{slice_name}_cache.pkl"
        if cache_path.exists() and not args.force:
            log.info("  cache found at %s -- loading and skipping Grad-CAM",
                     cache_path)
            with open(cache_path, "rb") as f:
                per_class = pickle.load(f)
        else:
            per_class = {}
            for class4 in range(4):
                if class4 not in allowed_classes:
                    per_class[class4] = None
                    continue
                log.info("  --- class %d (%s) ---", class4, CLASS_NAMES[class4])
                per_class[class4] = analyse_class(
                    model=model, cam=cam, manifest_path=args.manifest,
                    task_cfg=task_cfg, source=source, split=split, class4=class4,
                    sample_rate=sample_rate, max_seconds=max_seconds,
                    n_take_max=args.n_per_class, seed=args.seed,
                    device=DEVICE, log=log,
                )
            with open(cache_path, "wb") as f:
                pickle.dump(per_class, f)
            log.info("  cached slice data to %s", cache_path)

        all_slices_data[slice_name] = per_class

        # --- CSV rows for this slice --------------------------------------
        for class4 in range(4):
            d = per_class[class4]
            if d is None:
                continue
            total = max(d["n_correct"] + d["n_incorrect"], 1)
            row = {
                "slice": slice_name,
                "class4_label": class4,
                "class_name": CLASS_NAMES[class4],
                "n_correct": d["n_correct"],
                "n_incorrect": d["n_incorrect"],
                "per_class_accuracy": d["n_correct"] / total,
            }
            for j in range(4):
                row[f"predicted_as_{CLASS_NAMES[j]}"] = int(d["confusion_to"][j])
            csv_rows.append(row)

    # --- Compute shared y-axis maxima across every slice/class/bucket -----
    def _axis_max(axis_name, n_bins):
        vmax = 0.0
        for per_class in all_slices_data.values():
            for d in per_class.values():
                if d is None:
                    continue
                for k in (f"correct_{axis_name}", f"incorrect_{axis_name}"):
                    arr = d.get(k)
                    if arr is None:
                        continue
                    binned = bin_profile(arr, n_bins)
                    vmax = max(vmax, float(binned.max()))
        return vmax * 1.05  # 5% headroom above the tallest bar

    y_max_freq = _axis_max("freq", args.n_freq_bins)
    y_max_time = _axis_max("time", args.n_time_bins)
    # Single shared y-axis ceiling across every histogram (freq + time).
    y_max_shared = max(y_max_freq, y_max_time)
    y_max_freq = y_max_shared
    y_max_time = y_max_shared
    log.info("Shared y-axis max across freq + time: %.4f", y_max_shared)

    # --- Plot every slice with the shared y-axis --------------------------
    for slice_name, per_class in all_slices_data.items():
        plot_slice(
            per_class, slice_name, axis="freq",
            n_bins=args.n_freq_bins, x_min=freq_min_hz, x_max=freq_max_hz,
            x_label="Frequency (Hz)",
            model_name=model_cfg["name"],
            out_path=out_dir / f"{slice_name}_freq_histograms.png",
            y_max=y_max_freq,
        )
        log.info("Saved %s", out_dir / f"{slice_name}_freq_histograms.png")
        plot_slice(
            per_class, slice_name, axis="time",
            n_bins=args.n_time_bins, x_min=0, x_max=max_seconds,
            x_label="Time (s)",
            model_name=model_cfg["name"],
            out_path=out_dir / f"{slice_name}_time_histograms.png",
            y_max=y_max_time,
        )
        log.info("Saved %s", out_dir / f"{slice_name}_time_histograms.png")

    # --- Final CSV + npz ----------------------------------------------------
    pd.DataFrame(csv_rows).to_csv(out_dir / "per_class_counts.csv", index=False)
    log.info("Saved %s", out_dir / "per_class_counts.csv")

    save_dict = {}
    for slice_name, per_class in all_slices_data.items():
        for class4 in range(4):
            d = per_class[class4]
            if d is None:
                continue
            for k in ("correct_freq", "correct_time",
                      "incorrect_freq", "incorrect_time"):
                if d[k] is not None:
                    save_dict[f"{slice_name}_c{class4}_{k}"] = d[k]
    np.savez(out_dir / "profiles.npz", **save_dict)
    log.info("Saved %s", out_dir / "profiles.npz")

    log.info("Done. Artefacts under %s", out_dir)


if __name__ == "__main__":
    main()
