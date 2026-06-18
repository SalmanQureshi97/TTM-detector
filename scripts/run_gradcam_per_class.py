"""Per-class Grad-CAM attention histograms, split by correct vs incorrect predictions.

For each of the four classes (real, real_enc, fake, fake_enc):
  1. Sample ``--n-per-class`` validation rows whose ``class4_label`` matches.
  2. Run the model forward to obtain the predicted class for each sample.
  3. Run Grad-CAM with target = predicted class (the prediction the model
     committed to -- so we visualise what *drove* the actual decision,
     right or wrong).
  4. Reduce each 2D heatmap to a frequency-axis profile (sum over time) and
     a time-axis profile (sum over freq).
  5. Bucket the per-sample profiles by ``correct`` (pred == true) vs
     ``incorrect``.

The script writes:
  * ``gradcam_per_class/freq_histograms.png`` -- 2x2 panels (one per class).
    Each panel overlays the mean correct-prediction frequency histogram
    (green) against the mean incorrect-prediction histogram (red).
  * ``gradcam_per_class/time_histograms.png`` -- same layout, time axis.
  * ``gradcam_per_class/per_class_counts.csv`` -- correct/incorrect counts
    and per-class accuracy.
  * ``gradcam_per_class/profiles.npz`` -- raw averaged profiles per bucket
    for downstream analysis.

Usage (SpecCNN):
    python scripts/run_gradcam_per_class.py \\
        --model configs/models/deezer_speccnn_amplitude.yaml \\
        --task configs/tasks/four_class_flat.yaml \\
        --experiment configs/experiments/c1_sonics_only.yaml \\
        --runtime configs/runtime/train_default.yaml \\
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_sonics_aligned.csv \\
        --n-per-class 500

For SpecTTTra: swap --model to sonics_spectttra_alpha120s.yaml and
--runtime to train_spectttra.yaml.
"""

from __future__ import annotations

import argparse
import logging
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--n-per-class", type=int, default=500,
                   help="Validation rows sampled per true class.")
    p.add_argument("--n-freq-bins", type=int, default=32,
                   help="Number of bars in the frequency histogram per class.")
    p.add_argument("--n-time-bins", type=int, default=24,
                   help="Number of bars in the time histogram per class.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalise(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    if s <= eps:
        return np.full_like(v, 1.0 / max(v.size, 1))
    return v / s


def bin_profile(prof, n_bins):
    """Reduce a 1-D normalised profile to ``n_bins`` averaged bins."""
    prof = np.asarray(prof, dtype=np.float64)
    F = len(prof)
    if n_bins >= F:
        return prof.copy()
    edges = np.linspace(0, F, n_bins + 1).astype(int)
    binned = np.zeros(n_bins)
    for i in range(n_bins):
        lo = edges[i]
        hi = max(edges[i] + 1, edges[i + 1])
        binned[i] = prof[lo:hi].mean()
    return binned


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
    # eval() forces SpecAugment / dropout off; Grad-CAM still gets grads via
    # input.requires_grad_ inside GradCAM.compute().
    model.eval()
    log.info("Loaded best.pt | epoch %s | val_loss %.4f",
             ck.get("epoch"), ck.get("val_loss", float("nan")))

    cam = GradCAM(model, model_cfg)

    # --- Frequency axis range (frontend-specific) ---------------------------
    fe = model_cfg["frontend"]
    if fe["type"] == "logmel":
        freq_min_hz = fe.get("f_min", 0)
        freq_max_hz = fe.get("f_max", sample_rate // 2)
    elif fe["type"] == "deezer_amplitude":
        freq_min_hz = 0
        freq_max_hz = fe.get("hf_cut", sample_rate // 2)
    else:
        freq_min_hz, freq_max_hz = 0, sample_rate // 2

    # --- Per-class Grad-CAM pass --------------------------------------------
    per_class = {}

    for class4 in range(4):
        log.info("=== class %d (%s) ===", class4, CLASS_NAMES[class4])
        ds = AudioManifestDataset(
            args.manifest, split="val", task_cfg=task_cfg,
            dataset_filter=exp_cfg["val_datasets"],
            sample_rate=sample_rate, max_seconds=max_seconds,
        )
        ds.df = ds.df[ds.df["class4_label"] == class4].reset_index(drop=True)
        n_available = len(ds.df)
        n_take = min(args.n_per_class, n_available)
        ds.df = ds.df.sample(n=n_take, random_state=args.seed).reset_index(drop=True)
        log.info("  sampling %d / %d available val rows", n_take, n_available)

        correct_freq, correct_time = [], []
        incorrect_freq, incorrect_time = [], []
        confusion = np.zeros((4,), dtype=int)  # what the model predicted
        failures = 0

        for i in range(n_take):
            try:
                item = ds[i]
            except Exception as err:
                log.warning("  load %d failed: %s", i, err)
                failures += 1
                continue
            audio = item["audio"].unsqueeze(0).to(DEVICE)

            # Predicted class (no Grad-CAM yet -- cheap forward pass)
            with torch.no_grad():
                logits = model(audio)
                pred = int(logits.argmax(dim=1).item())
            confusion[pred] += 1

            # Grad-CAM with predicted class as target -- shows what drove the decision
            try:
                heatmap = cam.compute(audio, target_class=pred)
            except Exception as err:
                log.warning("  cam %d failed: %s", i, err)
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
                log.info("  processed %d/%d (correct so far: %d)",
                         i + 1, n_take, len(correct_freq))

        n_correct = len(correct_freq)
        n_incorrect = len(incorrect_freq)
        log.info("  correct: %d  incorrect: %d  failures: %d  acc: %.2f%%",
                 n_correct, n_incorrect, failures,
                 100.0 * n_correct / max(n_correct + n_incorrect, 1))

        per_class[class4] = {
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "confusion_to": confusion.tolist(),
            "correct_freq": np.stack(correct_freq).mean(axis=0) if correct_freq else None,
            "correct_time": np.stack(correct_time).mean(axis=0) if correct_time else None,
            "incorrect_freq": np.stack(incorrect_freq).mean(axis=0) if incorrect_freq else None,
            "incorrect_time": np.stack(incorrect_time).mean(axis=0) if incorrect_time else None,
        }

    # --- Frequency histograms (4 panels, one per true class) ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_width = (freq_max_hz - freq_min_hz) / max(args.n_freq_bins, 1)
    for class4, ax in zip(range(4), axes.flat):
        d = per_class[class4]
        x = np.linspace(freq_min_hz, freq_max_hz, args.n_freq_bins)
        if d["correct_freq"] is not None:
            ax.bar(x, bin_profile(d["correct_freq"], args.n_freq_bins),
                   width=bar_width, color=COLOR_CORRECT, alpha=0.55,
                   label=f"correct (n={d['n_correct']})")
        if d["incorrect_freq"] is not None:
            ax.bar(x, bin_profile(d["incorrect_freq"], args.n_freq_bins),
                   width=bar_width, color=COLOR_INCORRECT, alpha=0.55,
                   label=f"incorrect (n={d['n_incorrect']})")
        ax.set_title(f"true class {class4}: {CLASS_NAMES[class4]}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean normalised attention")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        f"{model_cfg['name']} -- frequency-axis Grad-CAM histograms by true class",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "freq_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_dir / "freq_histograms.png")

    # --- Time histograms ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_width_t = max_seconds / max(args.n_time_bins, 1)
    for class4, ax in zip(range(4), axes.flat):
        d = per_class[class4]
        x = np.linspace(0, max_seconds, args.n_time_bins)
        if d["correct_time"] is not None:
            ax.bar(x, bin_profile(d["correct_time"], args.n_time_bins),
                   width=bar_width_t, color=COLOR_CORRECT, alpha=0.55,
                   label=f"correct (n={d['n_correct']})")
        if d["incorrect_time"] is not None:
            ax.bar(x, bin_profile(d["incorrect_time"], args.n_time_bins),
                   width=bar_width_t, color=COLOR_INCORRECT, alpha=0.55,
                   label=f"incorrect (n={d['n_incorrect']})")
        ax.set_title(f"true class {class4}: {CLASS_NAMES[class4]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean normalised attention")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        f"{model_cfg['name']} -- time-axis Grad-CAM histograms by true class",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "time_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_dir / "time_histograms.png")

    # --- CSV: counts + confusion (where did the model send incorrect samples?)
    rows = []
    for class4 in range(4):
        d = per_class[class4]
        total = max(d["n_correct"] + d["n_incorrect"], 1)
        row = {
            "class4_label": class4,
            "class_name": CLASS_NAMES[class4],
            "n_correct": d["n_correct"],
            "n_incorrect": d["n_incorrect"],
            "per_class_accuracy": d["n_correct"] / total,
        }
        for j in range(4):
            row[f"predicted_as_{CLASS_NAMES[j]}"] = int(d["confusion_to"][j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "per_class_counts.csv", index=False)
    log.info("Saved %s", out_dir / "per_class_counts.csv")

    # --- npz of averaged profiles ------------------------------------------
    save_dict = {}
    for class4 in range(4):
        d = per_class[class4]
        for k in ("correct_freq", "correct_time", "incorrect_freq", "incorrect_time"):
            if d[k] is not None:
                save_dict[f"c{class4}_{k}"] = d[k]
    np.savez(out_dir / "profiles.npz", **save_dict)
    log.info("Saved %s", out_dir / "profiles.npz")

    log.info("Done. Artefacts under %s", out_dir)


if __name__ == "__main__":
    main()
