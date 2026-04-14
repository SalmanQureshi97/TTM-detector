"""Spectrogram overlay rendering for attribution / attention heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def get_spectrogram_from_model(
    model: nn.Module,
    audio: torch.Tensor,
) -> np.ndarray:
    """Run the model frontend to obtain the spectrogram as a numpy array.

    Parameters
    ----------
    model : UnifiedAudioModel
        Must have a ``.frontend`` attribute (LogMelFrontend or DeezerAmplitudeFrontend).
    audio : Tensor
        Raw waveform, shape ``[B, num_samples]`` or ``[num_samples]``.

    Returns
    -------
    np.ndarray
        Spectrogram of shape ``[H, W]`` (first item of the batch).
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    with torch.no_grad():
        spec = model.frontend(audio)  # [B, H, W]
    return spec[0].cpu().numpy()


def overlay_heatmap_on_spectrogram(
    spectrogram: np.ndarray,
    heatmap: np.ndarray,
    title: str = "",
    colormap: str = "jet",
    alpha: float = 0.5,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 4),
    hop_length: int = 512,
    sample_rate: int = 44100,
) -> Figure:
    """Overlay an attribution heatmap on the log-mel spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray [H, W]
        The base spectrogram (log-mel or amplitude).
    heatmap : np.ndarray [H, W]
        Attribution map.  Will be resized to match *spectrogram* if needed.
    title : str
        Figure title.
    colormap : str
        Matplotlib colormap for the heatmap overlay.
    alpha : float
        Transparency of the heatmap layer.
    output_path : Path, optional
        If given, save figure to this path.
    figsize : tuple
        Figure dimensions (width, height) in inches.
    hop_length : int
        Hop length used by the frontend (for the time axis label).
    sample_rate : int
        Sample rate (for the time axis label).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from skimage.transform import resize as sk_resize  # lazy import

    # Resize heatmap to spectrogram shape if needed
    if heatmap.shape != spectrogram.shape:
        heatmap = sk_resize(
            heatmap, spectrogram.shape, order=1, preserve_range=True
        ).astype(np.float32)

    # Normalise heatmap to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 1e-8:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Base spectrogram
    ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
    )

    # Heatmap overlay
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        cmap=colormap,
        alpha=alpha,
        interpolation="nearest",
    )

    # Time axis: approximate seconds
    n_frames = spectrogram.shape[1]
    duration = n_frames * hop_length / sample_rate
    tick_positions = np.linspace(0, n_frames - 1, num=6)
    tick_labels = [f"{t:.1f}" for t in np.linspace(0, duration, num=6)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Attribution")
    ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_side_by_side(
    spectrogram: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    title: str = "",
    output_path: Optional[Path] = None,
    figsize_per_panel: tuple[float, float] = (6, 3),
) -> Figure:
    """Plot the spectrogram alongside multiple attribution maps.

    Parameters
    ----------
    spectrogram : np.ndarray [H, W]
    heatmaps : dict mapping method name → heatmap [H, W]
    title : str
    output_path : Path, optional
    figsize_per_panel : tuple
        Size of each sub-panel.
    """
    from skimage.transform import resize as sk_resize

    n = 1 + len(heatmaps)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1])
    )
    if n == 1:
        axes = [axes]

    # Base spectrogram
    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Spectrogram")
    axes[0].set_ylabel("Mel bin")

    for ax, (name, hmap) in zip(axes[1:], heatmaps.items()):
        if hmap.shape != spectrogram.shape:
            hmap = sk_resize(hmap, spectrogram.shape, order=1, preserve_range=True)
        hmin, hmax = hmap.min(), hmap.max()
        if hmax - hmin > 1e-8:
            hmap = (hmap - hmin) / (hmax - hmin)
        ax.imshow(hmap, aspect="auto", origin="lower", cmap="jet")
        ax.set_title(name)

    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
