"""Grad-CAM analysis for a trained model.

Produces, for each of four evaluation slices (SONICS val, FMA test, FMC
fake_natural, FMC fake_encoded):
  * PNG overlays of the heatmap on the input log-mel spectrogram for the
    first ``--n-images`` samples (saved under ``gradcam/images/``).
  * Aggregated frequency- and time-axis attention profiles averaged over
    ``--n-aggregate`` samples (saved under ``gradcam/profiles/``).
  * Per-slice quantitative metrics (frequency entropy, peak frequency in
    Hz, time-axis entropy, time centre-of-mass, Jensen-Shannon divergence
    vs the in-domain SONICS-val reference) written to
    ``gradcam/metrics.csv``.
  * A 2x2 summary figure overlaying the four slice profiles
    (``gradcam/summary.png``).

Usage example -- SpecCNN run (44.1 kHz, 30 s segments):
    python scripts/run_gradcam_analysis.py \
        --model configs/models/deezer_speccnn_amplitude.yaml \
        --task configs/tasks/four_class_flat.yaml \
        --experiment configs/experiments/c1_sonics_only.yaml \
        --runtime configs/runtime/train_default.yaml \
        --manifest /home/jovyan/Thesis/Code/data/manifests/master_manifest_sonics_aligned.csv \
        --n-aggregate 50 --n-images 4

For SpecTTTra (16 kHz, 120 s), swap ``--model`` to
``configs/models/sonics_spectttra_alpha120s.yaml`` and ``--runtime`` to
``configs/runtime/train_spectttra.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manifest_dataset import AudioManifestDataset  # noqa: E402
from src.models.model_factory import UnifiedAudioModel  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402
from src.visualization.gradcam import GradCAM  # noqa: E402
from src.visualization.overlay import (  # noqa: E402
    get_spectrogram_from_model,
    overlay_heatmap_on_spectrogram,
)


CLASS_NAMES = ["real", "real_enc", "fake", "fake_enc"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--n-aggregate", type=int, default=50,
                   help="Samples per slice used for the averaged profiles.")
    p.add_argument("--n-images", type=int, default=4,
                   help="Per-slice PNG overlays to save (subset of --n-aggregate).")
    p.add_argument("--device", default=None,
                   help="cuda|cpu (default: cuda if available).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalise(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= eps:
        return np.full_like(p, 1.0 / max(p.size, 1))
    return p / s


def entropy(p, eps=1e-12):
    p = normalise(p, eps)
    return float(-(p * np.log(p + eps)).sum())


def jensen_shannon(p, q, eps=1e-12):
    """Symmetric JSD (in nats), squared distance returned by scipy if you prefer."""
    p = normalise(p, eps)
    q = normalise(q, eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum()
    kl_qm = (q * (np.log(q + eps) - np.log(m + eps))).sum()
    return float(0.5 * (kl_pm + kl_qm))


def center_of_mass(p, axis_values):
    p = normalise(p)
    return float((p * axis_values).sum())


def mel_bin_to_hz(bin_idx, n_mels, f_min, f_max):
    """Approximate mel-bin -> Hz using a linear mel-Hz mapping.

    The exact mapping inside torchaudio is non-linear (HTK mel), but for a
    diagnostic peak-frequency report a linear approximation between f_min
    and f_max is close enough at the mel resolutions we use.
    """
    if n_mels <= 1:
        return float(f_min)
    return float(f_min + (f_max - f_min) * (bin_idx / (n_mels - 1)))


def slice_dataset(manifest, task_cfg, source_dataset, class4, sample_rate,
                  max_seconds, n, seed):
    ds = AudioManifestDataset(
        manifest_path=manifest,
        split="test" if source_dataset != "SONICS" else "val",
        task_cfg=task_cfg,
        dataset_filter=[source_dataset],
        sample_rate=sample_rate,
        max_seconds=max_seconds,
    )
    if class4 is not None:
        ds.df = ds.df[ds.df["class4_label"] == class4].copy()
    ds.df = ds.df.reset_index(drop=True)
    if len(ds.df) == 0:
        return ds
    take = min(n, len(ds.df))
    ds.df = ds.df.sample(n=take, random_state=seed).reset_index(drop=True)
    return ds


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("gradcam")

    args = parse_args()
    model_cfg = load_yaml(args.model)
    task_cfg = load_yaml(args.task)
    exp_cfg = load_yaml(args.experiment)
    rt = load_yaml(args.runtime)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    save_dir = (Path(rt["save_dir"]) / exp_cfg["name"] / model_cfg["name"]
                / task_cfg["name"])
    gradcam_dir = save_dir / "gradcam"
    images_dir = gradcam_dir / "images"
    profiles_dir = gradcam_dir / "profiles"
    images_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model + best checkpoint ----------------------------------------
    model = UnifiedAudioModel(model_cfg, task_cfg).to(device)
    ck_path = save_dir / "best.pt"
    if not ck_path.exists():
        raise FileNotFoundError(f"No best.pt at {ck_path}")
    state = torch.load(ck_path, map_location=device)
    model.load_state_dict(state["model_state"])
    log.info(
        "Loaded %s | epoch %s | val_loss %.4f",
        ck_path.name, state.get("epoch"), state.get("val_loss", float("nan")),
    )

    cam = GradCAM(model, model_cfg)

    # --- Audio/frontend params ----------------------------------------------
    fe = model_cfg["frontend"]
    sample_rate = fe["sample_rate"]
    hop_length = fe.get("hop_length", 512)
    max_seconds = rt.get("segment_seconds", 30)

    # The heatmap's frequency dimension comes from the actual spectrogram
    # shape, which depends on the frontend type (mel banks vs. STFT bins
    # capped at hf_cut). f_min/f_max are taken per-frontend:
    if fe["type"] == "logmel":
        freq_min_hz = fe.get("f_min", 0)
        freq_max_hz = fe.get("f_max", sample_rate // 2)
    elif fe["type"] == "deezer_amplitude":
        freq_min_hz = 0
        freq_max_hz = fe.get("hf_cut", sample_rate // 2)
    else:
        freq_min_hz = 0
        freq_max_hz = sample_rate // 2

    # The actual F (number of frequency bins in the spectrogram / heatmap) is
    # determined at runtime from the first successful sample's heatmap shape.
    # We'll build the Hz axis lazily once we know F.

    # --- Slices --------------------------------------------------------------
    # (label, source_dataset, class4, target_class_for_cam)
    SLICES = [
        ("sonics_val",   "SONICS",         None, None),  # mixed; use predicted class
        ("fma_test",     "FMA",            None, None),
        ("fmc_natural",  "FakeMusicCaps",  2,    2),
        ("fmc_encoded",  "FakeMusicCaps",  3,    3),
    ]

    per_slice_freq_profiles = {}
    per_slice_time_profiles = {}
    rows = []

    for slice_name, source, class4, target_class in SLICES:
        ds = slice_dataset(
            manifest=args.manifest,
            task_cfg=task_cfg,
            source_dataset=source,
            class4=class4,
            sample_rate=sample_rate,
            max_seconds=max_seconds,
            n=args.n_aggregate,
            seed=args.seed,
        )
        if len(ds.df) == 0:
            log.warning("slice %s is empty -- skipping", slice_name)
            continue
        log.info("Slice %s: %d samples", slice_name, len(ds.df))

        freq_profs = []
        time_profs = []
        for i in range(len(ds.df)):
            item = ds[i]
            audio = item["audio"].unsqueeze(0).to(device)

            try:
                heatmap = cam.compute(audio, target_class=target_class)
            except Exception as err:  # pragma: no cover -- diagnostic
                log.warning("  sample %d failed: %s", i, err)
                continue

            heatmap = np.asarray(heatmap, dtype=np.float64)
            # Normalise to [0, 1]
            hmin, hmax = heatmap.min(), heatmap.max()
            if hmax - hmin > 1e-12:
                heatmap = (heatmap - hmin) / (hmax - hmin)

            # Marginal profiles (mass over the rejected axis)
            freq_prof = heatmap.sum(axis=1)   # along time -> per-frequency mass
            time_prof = heatmap.sum(axis=0)   # along freq -> per-time mass

            freq_profs.append(normalise(freq_prof))
            time_profs.append(normalise(time_prof))

            if i < args.n_images:
                spec = get_spectrogram_from_model(model, audio)
                true_lbl = CLASS_NAMES[int(item["meta"]["class4_label"])]
                title = (
                    f"{slice_name} | sample {i} | true={true_lbl} | "
                    f"target_class={target_class if target_class is not None else 'argmax'}"
                )
                out_png = images_dir / f"{slice_name}_idx{i:02d}.png"
                overlay_heatmap_on_spectrogram(
                    spec, heatmap, title=title,
                    output_path=out_png,
                    hop_length=hop_length, sample_rate=sample_rate,
                )

        if not freq_profs:
            log.warning("  no successful samples for %s", slice_name)
            continue

        freq_arr = np.stack(freq_profs).mean(axis=0)
        time_arr = np.stack(time_profs).mean(axis=0)
        per_slice_freq_profiles[slice_name] = freq_arr
        per_slice_time_profiles[slice_name] = time_arr

        # Build the Hz axis from the actual frequency dimension F of this
        # slice's averaged profile. F equals the spectrogram's frequency
        # axis (e.g. 128 for log-mel, ~743 for Deezer-amplitude at 16 kHz
        # hf_cut with 44.1 kHz / n_fft=2048).
        F = len(freq_arr)
        hz_axis = np.linspace(freq_min_hz, freq_max_hz, F)

        peak_bin = int(freq_arr.argmax())
        peak_hz = float(hz_axis[peak_bin])
        rows.append({
            "slice": slice_name,
            "n_samples": len(freq_profs),
            "H_freq_nats": entropy(freq_arr),
            "H_freq_pct_of_max": entropy(freq_arr) / np.log(F),
            "peak_freq_bin": peak_bin,
            "peak_freq_hz": peak_hz,
            "freq_center_of_mass_hz": center_of_mass(freq_arr, hz_axis),
            "H_time_nats": entropy(time_arr),
            "H_time_pct_of_max": entropy(time_arr) / np.log(len(time_arr)),
            "time_center_of_mass_frame": center_of_mass(
                time_arr, np.arange(len(time_arr))
            ),
        })

    # --- JSD vs SONICS-val reference ----------------------------------------
    if "sonics_val" in per_slice_freq_profiles:
        ref_f = per_slice_freq_profiles["sonics_val"]
        ref_t = per_slice_time_profiles["sonics_val"]
        for row in rows:
            other = row["slice"]
            row["JSD_freq_vs_sonics_val"] = jensen_shannon(
                ref_f, per_slice_freq_profiles[other])
            row["JSD_time_vs_sonics_val"] = jensen_shannon(
                ref_t, per_slice_time_profiles[other])
    else:
        log.warning("No sonics_val profile -- skipping JSD reference comparison.")

    # --- Write CSV ----------------------------------------------------------
    if rows:
        cols = list(rows[0].keys())
        with open(gradcam_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s", gradcam_dir / "metrics.csv")

    # --- Save raw profiles --------------------------------------------------
    np.savez(profiles_dir / "freq_profiles.npz", **per_slice_freq_profiles)
    np.savez(profiles_dir / "time_profiles.npz", **per_slice_time_profiles)
    log.info("Saved averaged profiles to %s", profiles_dir)

    # --- Summary plot (frequency profiles + time profiles, side-by-side) ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for name, prof in per_slice_freq_profiles.items():
        hz_axis = np.linspace(freq_min_hz, freq_max_hz, len(prof))
        axes[0].plot(hz_axis, prof, label=name)
    axes[0].set_xlabel("Approx. frequency (Hz)")
    axes[0].set_ylabel("Normalised attention")
    axes[0].set_title("Grad-CAM frequency profile, averaged per slice")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for name, prof in per_slice_time_profiles.items():
        t_axis = np.arange(len(prof))
        axes[1].plot(t_axis, prof, label=name)
    axes[1].set_xlabel("Time frame index")
    axes[1].set_ylabel("Normalised attention")
    axes[1].set_title("Grad-CAM time profile, averaged per slice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{model_cfg['name']} | best.pt epoch {state.get('epoch')} | "
        f"val_loss {state.get('val_loss', float('nan')):.4f}"
    )
    fig.tight_layout()
    fig.savefig(gradcam_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved summary plot to %s", gradcam_dir / "summary.png")

    log.info("Done. Artefacts under %s", gradcam_dir)


if __name__ == "__main__":
    main()
