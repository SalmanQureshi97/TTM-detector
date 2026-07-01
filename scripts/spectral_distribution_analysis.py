"""Spectral distribution analysis across SONICS / FMA / FMC slices.

This script supports three claims for the paper's main finding:

  H1: distribution-shift hypothesis -- compute Jensen-Shannon divergence
      between each slice's mean PSD and SONICS-train, then correlate the
      JSD with the model's auth-detection error rate per slice. Strong
      monotonic correlation => cross-domain failure is driven by content-
      distribution shift, not by artefact absence in the target domain.

  H2: codec-peak hypothesis (Afchar et al. 2025) -- compare mean PSDs
      across the 2x2 factorial (auth x encoded). Codec peaks should
      appear in any *_encoded slice independent of authenticity;
      generator peaks should appear in any *_fake slice independent of
      encoding. The factorial design lets us isolate which peaks are
      codec-attributable vs generator-attributable.

  H3 (visual companion using existing Grad-CAM data): the model's
      Grad-CAM frequency attention does NOT concentrate at the
      identified peaks -- i.e., the model has access to the artefact
      signature but learned content-statistics instead.

Independent of any model: PSD computations are over raw audio. The
optional ``--confusion-dirs`` flag joins with cross-domain confusion
matrices to draw the JSD-vs-error correlation per model.

Outputs (under outputs/c1_sonics_only/spectral_analysis/):
  mean_spectrograms/<slice>.png      per-slice mean mel-spectrogram
  psd_overlay.png                    all-slice PSD overlay (log-y)
  factorial_2x2.png                  FMA real / FMA enc / FMC nat / FMC enc
  peaks.csv                          per-slice top peaks (freq_Hz, prominence)
  jsd_matrix.csv                     pairwise JSD between slices
  jsd_vs_sonics.csv                  one-column JSD vs SONICS train
  jsd_vs_error_<model>.png           (optional) correlation scatter per model
  jsd_vs_error_<model>.csv           (optional) per-slice JSD + auth-error rate
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# (name, source_dataset, split, class4 filter or None, role tag)
SLICES = [
    ("sonics_train_mixed",  "SONICS",        "train", None, "reference"),
    ("sonics_val_real",     "SONICS",        "val",   0,    "in_domain"),
    ("sonics_val_real_enc", "SONICS",        "val",   1,    "in_domain"),
    ("sonics_val_fake",     "SONICS",        "val",   2,    "in_domain"),
    ("sonics_val_fake_enc", "SONICS",        "val",   3,    "in_domain"),
    ("fma_real",            "FMA",           "test",  0,    "ood_real_natural"),
    ("fma_real_encoded",    "FMA",           "test",  1,    "ood_real_codec"),
    ("fmc_fake_natural",    "FakeMusicCaps", "test",  2,    "ood_fake_natural"),
    ("fmc_fake_encoded",    "FakeMusicCaps", "test",  3,    "ood_fake_codec"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", default="outputs/c1_sonics_only/spectral_analysis")
    p.add_argument("--n-per-slice", type=int, default=500)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--n-fft", type=int, default=4096,
                   help="Larger -> finer freq resolution (~SR/n_fft Hz/bin).")
    p.add_argument("--hop-length", type=int, default=1024)
    p.add_argument("--max-seconds", type=float, default=30.0,
                   help="Per-clip cap. We do NOT zero-pad below this.")
    p.add_argument("--peak-prominence", type=float, default=0.20,
                   help="Min prominence above detrended PSD baseline (~1.0).")
    p.add_argument("--peak-top-k", type=int, default=20,
                   help="Report at most this many peaks per slice.")
    p.add_argument("--smoothing-sigma", type=float, default=20.0,
                   help="Gaussian smoothing sigma (bins) for envelope detrend.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confusion-dirs", nargs="*", default=None,
                   help="Optional model output dirs to pull cross-domain confusion "
                        "CSVs from (computes JSD-vs-error correlation per model).")
    return p.parse_args()


# ----- core spectral utilities -------------------------------------------------

def load_audio(path, sample_rate, max_seconds):
    """Load, mono-mix, resample. Truncate to ``max_seconds`` if longer; do NOT
    zero-pad if shorter -- length difference is OK for PSD estimation as long
    as we have a few seconds."""
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    audio = audio.mean(dim=0)
    max_len = int(sample_rate * max_seconds)
    if audio.numel() > max_len:
        audio = audio[:max_len]
    return audio


def mean_power_spectrum(audio, n_fft, hop_length):
    """STFT -> magnitude^2 -> mean across time. Returns 1-D array of length
    n_fft//2 + 1."""
    if audio.numel() < n_fft:
        # Too short to STFT -- zero-pad just to satisfy frame size.
        audio = torch.nn.functional.pad(audio, (0, n_fft - audio.numel()))
    window = torch.hann_window(n_fft)
    spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)
    power = spec.abs().pow(2)            # [freq, time]
    return power.mean(dim=-1).cpu().numpy()


def normalise(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    return p / s if s > eps else np.full_like(p, 1.0 / max(len(p), 1))


def jsd(p, q, eps=1e-12):
    """Symmetric Jensen-Shannon divergence in nats."""
    p, q = normalise(p, eps), normalise(q, eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * (np.log(a + eps) - np.log(b + eps))).sum())
    return 0.5 * (kl(p, m) + kl(q, m))


def detrend(psd, sigma_bins):
    """Divide PSD by Gaussian-smoothed envelope to expose narrow peaks."""
    smooth = gaussian_filter1d(psd, sigma=sigma_bins)
    return psd / np.maximum(smooth, 1e-12)


def detect_peaks(psd, sample_rate, n_fft, sigma_bins, prominence, top_k):
    """Returns list of (freq_hz, prominence_value), sorted by prominence desc."""
    detrended = detrend(psd, sigma_bins)
    peaks, props = find_peaks(detrended, prominence=prominence)
    if len(peaks) == 0:
        return []
    proms = props["prominences"]
    order = np.argsort(-proms)[:top_k]
    peaks = peaks[order]
    proms = proms[order]
    hz_per_bin = sample_rate / n_fft
    return [(float(b * hz_per_bin), float(p)) for b, p in zip(peaks, proms)]


# ----- per-slice processing ---------------------------------------------------

def process_slice(manifest_df, source, split, class4, n, seed, sample_rate,
                  max_seconds, n_fft, hop_length, log):
    """Sample rows for one slice, compute mean PSD across samples, return:
       dict(name, n_used, mean_psd, mean_spec_db, hz_axis)."""
    sub = manifest_df[manifest_df["source_dataset"] == source]
    sub = sub[sub["split"] == split]
    if class4 is not None:
        sub = sub[sub["class4_label"] == class4]
    if len(sub) == 0:
        log.warning("  no rows in slice (source=%s, split=%s, class4=%s)",
                    source, split, class4)
        return None
    take = min(n, len(sub))
    sub = sub.sample(n=take, random_state=seed).reset_index(drop=True)
    log.info("  sampling %d / %d rows", take, len(sub))

    psds = []
    mel_specs = []
    mel_xform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=128, f_min=0, f_max=sample_rate // 2,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    failures = 0
    for i, row in sub.iterrows():
        try:
            audio = load_audio(row["filepath"], sample_rate, max_seconds)
            psd = mean_power_spectrum(audio, n_fft, hop_length)
            psds.append(psd)
            mel = to_db(mel_xform(audio)).cpu().numpy()
            mel_specs.append(mel)
        except Exception as err:
            log.warning("  sample %d failed: %s", i, err)
            failures += 1
        if (i + 1) % 100 == 0:
            log.info("    processed %d / %d", i + 1, take)
    if not psds:
        return None
    mean_psd = np.stack(psds).mean(axis=0)
    # Variable-length mel specs -> pad / trim to a common time length for the mean
    target_T = int(np.median([m.shape[1] for m in mel_specs]))
    aligned = []
    for m in mel_specs:
        if m.shape[1] >= target_T:
            aligned.append(m[:, :target_T])
        else:
            pad = np.full((m.shape[0], target_T - m.shape[1]),
                          float(m.min()), dtype=m.dtype)
            aligned.append(np.concatenate([m, pad], axis=1))
    mean_spec_db = np.stack(aligned).mean(axis=0)
    hz_axis = np.linspace(0, sample_rate / 2, len(mean_psd))
    log.info("  finished slice: n_used=%d, failures=%d", len(psds), failures)
    return {
        "n_used": len(psds),
        "mean_psd": mean_psd,
        "mean_spec_db": mean_spec_db,
        "hz_axis": hz_axis,
    }


# ----- plotting --------------------------------------------------------------

def plot_mean_spec(spec_db, slice_name, out_path, sample_rate, hop_length):
    fig, ax = plt.subplots(figsize=(8, 4))
    extent = [0, spec_db.shape[1] * hop_length / sample_rate, 0, sample_rate / 2 / 1000]
    im = ax.imshow(spec_db, origin="lower", aspect="auto", cmap="magma", extent=extent)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel-band freq (kHz, approx)")
    ax.set_title(f"Mean log-mel spectrogram: {slice_name}")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_psd_overlay(slices, out_path):
    """All slices' PSDs on one log-y plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, d in slices.items():
        psd = normalise(d["mean_psd"])
        ax.semilogy(d["hz_axis"], psd, label=name, linewidth=0.9, alpha=0.85)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised mean power (log scale)")
    ax.set_title("Mean PSD overlay -- all slices")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_factorial_2x2(slices, peaks_by_slice, out_path):
    """2x2 panels for the FMA real / FMA enc / FMC nat / FMC enc factorial.
    Each panel shows the slice's detrended PSD with detected peaks marked."""
    panels = [
        ("fma_real",          "Real, no codec (FMA real)"),
        ("fma_real_encoded",  "Real + codec (FMA real_encoded)"),
        ("fmc_fake_natural",  "Fake, no codec (FMC fake_natural)"),
        ("fmc_fake_encoded",  "Fake + codec (FMC fake_encoded)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for (key, title), ax in zip(panels, axes.flat):
        if key not in slices:
            ax.text(0.5, 0.5, f"{key}: no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(title); continue
        d = slices[key]
        detrended = detrend(d["mean_psd"], sigma_bins=20)
        ax.semilogy(d["hz_axis"], detrended, color="black", linewidth=0.8)
        for f_hz, prom in peaks_by_slice.get(key, [])[:10]:
            ax.axvline(f_hz, color="crimson", alpha=0.45, linewidth=0.9)
            ax.text(f_hz, ax.get_ylim()[1] * 0.7,
                    f"{int(f_hz)} Hz", fontsize=7, rotation=90,
                    ha="right", va="top", color="crimson")
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Detrended PSD (a.u., log scale)")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "2x2 factorial: which peaks are codec-attributable vs generator-attributable?",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----- model-side: auth error rates from existing confusion CSVs --------------

def auth_error_rates_from_dir(model_dir):
    """Return {slice_name -> auth_error_rate} based on the cross-domain
    confusion CSVs in ``model_dir``."""
    model_dir = Path(model_dir)
    out = {}
    # FMA: confusion_fma_test.csv has rows for class 0 (real) and class 1 (real_enc)
    fp = model_dir / "confusion_fma_test.csv"
    if fp.exists():
        df = pd.read_csv(fp, index_col=0)
        # auth correct = predicted as real or real_enc (cols 0+1)
        for class4, slice_key in [(0, "fma_real"), (1, "fma_real_encoded")]:
            row_name = ["real", "real_enc", "fake", "fake_enc"][class4]
            row = df.loc[row_name]
            total = row.sum()
            correct = row.iloc[0] + row.iloc[1]
            if total > 0:
                out[slice_key] = 1.0 - (correct / total)
    # FMC natural / encoded
    for fname, slice_key, class4 in [
        ("confusion_fmc_natural_test.csv", "fmc_fake_natural", 2),
        ("confusion_fmc_encoded_test.csv", "fmc_fake_encoded", 3),
    ]:
        fp = model_dir / fname
        if not fp.exists():
            continue
        df = pd.read_csv(fp, index_col=0)
        row_name = ["real", "real_enc", "fake", "fake_enc"][class4]
        row = df.loc[row_name]
        total = row.sum()
        correct = row.iloc[2] + row.iloc[3]   # auth fake = cols 2+3
        if total > 0:
            out[slice_key] = 1.0 - (correct / total)
    # In-domain SONICS val: auth error rate from confusion_val.csv across rows
    fp = model_dir / "confusion_val.csv"
    if fp.exists():
        df = pd.read_csv(fp, index_col=0)
        names = ["real", "real_enc", "fake", "fake_enc"]
        for class4, slice_key in enumerate(
                ["sonics_val_real", "sonics_val_real_enc",
                 "sonics_val_fake", "sonics_val_fake_enc"]):
            row = df.loc[names[class4]]
            total = row.sum()
            if total == 0:
                continue
            if class4 < 2:
                correct = row.iloc[0] + row.iloc[1]
            else:
                correct = row.iloc[2] + row.iloc[3]
            out[slice_key] = 1.0 - (correct / total)
    return out


def plot_jsd_vs_error(model_name, jsd_per_slice, err_per_slice, out_path):
    rows = []
    for slice_key, jsd_val in jsd_per_slice.items():
        if slice_key in err_per_slice:
            rows.append((slice_key, jsd_val, err_per_slice[slice_key]))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[1])
    names, xs, ys = zip(*rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=80, color="steelblue", edgecolor="black", zorder=3)
    for n, x, y in rows:
        ax.annotate(n, (x, y), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        ax.set_title(
            f"{model_name}: JSD vs auth error rate "
            f"(Spearman rho={rho:.2f}, p={p:.3g})"
        )
    except Exception:
        ax.set_title(f"{model_name}: JSD vs auth error rate")
    ax.set_xlabel("JSD(slice PSD || SONICS train PSD)  -- nats")
    ax.set_ylabel("Auth error rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return rows


# ----- main ------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("spectral")
    args = parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "mean_spectrograms").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    df["track_id"] = df["track_id"].astype(str)
    log.info("Loaded manifest with %d rows", len(df))

    slices = {}
    peaks_by_slice = {}
    for name, source, split, class4, role in SLICES:
        log.info("---- slice %s (%s/%s class4=%s, role=%s) ----",
                 name, source, split, class4, role)
        result = process_slice(
            df, source, split, class4, args.n_per_slice, args.seed,
            args.sample_rate, args.max_seconds, args.n_fft, args.hop_length, log,
        )
        if result is None:
            continue
        slices[name] = result
        plot_mean_spec(
            result["mean_spec_db"], name,
            out_dir / "mean_spectrograms" / f"{name}.png",
            args.sample_rate, args.hop_length,
        )
        peaks = detect_peaks(
            result["mean_psd"], args.sample_rate, args.n_fft,
            args.smoothing_sigma, args.peak_prominence, args.peak_top_k,
        )
        peaks_by_slice[name] = peaks
        log.info("  detected %d peaks; top 5: %s",
                 len(peaks), peaks[:5])

    if not slices:
        log.error("No slices produced data -- aborting.")
        return

    # --- overlay + factorial figures ----------------------------------------
    plot_psd_overlay(slices, out_dir / "psd_overlay.png")
    log.info("Saved %s", out_dir / "psd_overlay.png")
    plot_factorial_2x2(slices, peaks_by_slice, out_dir / "factorial_2x2.png")
    log.info("Saved %s", out_dir / "factorial_2x2.png")

    # --- peaks CSV ----------------------------------------------------------
    with open(out_dir / "peaks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice", "rank", "freq_Hz", "prominence"])
        for name, peaks in peaks_by_slice.items():
            for i, (f_hz, prom) in enumerate(peaks):
                w.writerow([name, i, f"{f_hz:.2f}", f"{prom:.4f}"])
    log.info("Saved %s", out_dir / "peaks.csv")

    # --- pairwise JSD matrix + JSD vs SONICS train --------------------------
    names = list(slices.keys())
    jsd_matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            jsd_matrix[i, j] = jsd(slices[a]["mean_psd"], slices[b]["mean_psd"])
    pd.DataFrame(jsd_matrix, index=names, columns=names).to_csv(
        out_dir / "jsd_matrix.csv")
    log.info("Saved %s", out_dir / "jsd_matrix.csv")

    reference = "sonics_train_mixed"
    if reference in slices:
        ref_psd = slices[reference]["mean_psd"]
        jsd_vs_ref = {name: jsd(ref_psd, slices[name]["mean_psd"])
                      for name in names}
        with open(out_dir / "jsd_vs_sonics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slice", "JSD_vs_sonics_train"])
            for name, val in sorted(jsd_vs_ref.items(), key=lambda kv: kv[1]):
                w.writerow([name, f"{val:.6f}"])
        log.info("Saved %s", out_dir / "jsd_vs_sonics.csv")
    else:
        log.warning("No 'sonics_train_mixed' slice -- skipping JSD-vs-reference")
        jsd_vs_ref = None

    # --- optional: correlation with per-model auth error rates --------------
    if args.confusion_dirs and jsd_vs_ref:
        for model_dir in args.confusion_dirs:
            model_dir = Path(model_dir)
            err_per_slice = auth_error_rates_from_dir(model_dir)
            if not err_per_slice:
                log.warning("  no confusion CSVs in %s -- skipping", model_dir)
                continue
            model_name = model_dir.parent.name  # e.g. deezer_speccnn_amplitude
            out_png = out_dir / f"jsd_vs_error_{model_name}.png"
            rows = plot_jsd_vs_error(model_name, jsd_vs_ref, err_per_slice, out_png)
            if rows:
                pd.DataFrame(rows, columns=["slice", "JSD", "auth_error_rate"]).to_csv(
                    out_dir / f"jsd_vs_error_{model_name}.csv", index=False)
                log.info("Saved %s", out_png)

    log.info("Done. Artefacts under %s", out_dir)


if __name__ == "__main__":
    main()
