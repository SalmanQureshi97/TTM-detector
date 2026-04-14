"""Map token indices back to spectrogram (frequency, time) regions.

ViT and SpecTTTra tokenize the spectrogram differently.  This module
provides mapper classes that convert per-token scalar weights into
2-D heatmaps aligned with the original spectrogram dimensions.
"""

from __future__ import annotations

import numpy as np


class PatchMapper:
    """Map ViT patch indices to spectrogram regions.

    ViT divides the input ``[H, W]`` spectrogram into a regular grid of
    non-overlapping ``patch_size × patch_size`` patches.  Patch *i* maps
    to row ``i // n_cols`` and column ``i % n_cols`` of that grid.

    Parameters
    ----------
    input_shape : tuple[int, int]
        (H, W) of the resized spectrogram fed to the ViT (e.g. [128, 768]).
    patch_size : int
        Side length of each square patch (e.g. 16).
    """

    def __init__(self, input_shape: tuple[int, int], patch_size: int) -> None:
        self.H, self.W = input_shape
        self.ps = patch_size
        self.n_rows = self.H // patch_size
        self.n_cols = self.W // patch_size
        self.num_tokens = self.n_rows * self.n_cols

    def token_to_region(self, token_idx: int) -> tuple[int, int, int, int]:
        """Return (freq_start, freq_end, time_start, time_end) for *token_idx*."""
        row = token_idx // self.n_cols
        col = token_idx % self.n_cols
        return (
            row * self.ps,
            (row + 1) * self.ps,
            col * self.ps,
            (col + 1) * self.ps,
        )

    def tokens_to_heatmap(self, weights: np.ndarray) -> np.ndarray:
        """Convert per-token weights ``[N]`` to a ``[H, W]`` heatmap.

        Each patch region is filled with the corresponding token weight.
        """
        weights = np.asarray(weights).ravel()
        assert weights.shape[0] == self.num_tokens, (
            f"Expected {self.num_tokens} weights, got {weights.shape[0]}"
        )
        heatmap = np.zeros((self.H, self.W), dtype=np.float32)
        for i, w in enumerate(weights):
            f0, f1, t0, t1 = self.token_to_region(i)
            heatmap[f0:f1, t0:t1] = w
        return heatmap


class STTokenMapper:
    """Map SpecTTTra token indices to spectrogram regions.

    SpecTTTra's ``STTokenizer`` produces two kinds of token:

    * **Temporal tokens** (indices ``0 .. num_temporal-1``):  each covers
      a narrow time slice of width ``t_clip`` spanning the full frequency
      axis.
    * **Spectral tokens** (indices ``num_temporal .. num_temporal+num_spectral-1``):
      each covers a narrow frequency band of height ``f_clip`` spanning the
      full time axis.

    Parameters
    ----------
    input_shape : tuple[int, int]
        (H, W) of the resized spectrogram (e.g. [128, 768]).
    t_clip : int
        Temporal clip size used by the STTokenizer (e.g. 3).
    f_clip : int
        Frequency clip size used by the STTokenizer (e.g. 5).
    """

    def __init__(
        self, input_shape: tuple[int, int], t_clip: int, f_clip: int
    ) -> None:
        self.H, self.W = input_shape
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.num_temporal = (self.W - t_clip) // t_clip + 1
        self.num_spectral = (self.H - f_clip) // f_clip + 1
        self.num_tokens = self.num_temporal + self.num_spectral

    def token_to_region(self, token_idx: int) -> tuple[int, int, int, int]:
        """Return (freq_start, freq_end, time_start, time_end)."""
        if token_idx < self.num_temporal:
            t0 = token_idx * self.t_clip
            return (0, self.H, t0, t0 + self.t_clip)
        else:
            j = token_idx - self.num_temporal
            f0 = j * self.f_clip
            return (f0, f0 + self.f_clip, 0, self.W)

    def tokens_to_heatmap(self, weights: np.ndarray) -> np.ndarray:
        """Convert per-token weights ``[N]`` to a ``[H, W]`` heatmap.

        Temporal and spectral contributions are accumulated additively
        and then normalised to [0, 1].
        """
        weights = np.asarray(weights).ravel()
        assert weights.shape[0] == self.num_tokens, (
            f"Expected {self.num_tokens} weights, got {weights.shape[0]}"
        )
        heatmap = np.zeros((self.H, self.W), dtype=np.float64)
        count = np.zeros((self.H, self.W), dtype=np.float64)

        for i, w in enumerate(weights):
            f0, f1, t0, t1 = self.token_to_region(i)
            heatmap[f0:f1, t0:t1] += w
            count[f0:f1, t0:t1] += 1.0

        # Average where multiple tokens overlap
        mask = count > 0
        heatmap[mask] /= count[mask]

        # Normalise to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-8:
            heatmap = (heatmap - hmin) / (hmax - hmin)

        return heatmap.astype(np.float32)


def get_mapper(model_cfg: dict):
    """Factory: return the right mapper for the configured backbone.

    Parameters
    ----------
    model_cfg : dict
        The full model config (with ``backbone.type``, ``backbone.input_shape``,
        ``backbone.patch_size``, ``backbone.t_clip``, ``backbone.f_clip``).

    Returns
    -------
    PatchMapper | STTokenMapper | None
        ``None`` for CNN backbones that do not use token-based representations.
    """
    bb = model_cfg["backbone"]
    bb_type = bb["type"]
    if bb_type == "vit":
        return PatchMapper(
            input_shape=tuple(bb["input_shape"]),
            patch_size=bb["patch_size"],
        )
    elif bb_type == "spectttra":
        return STTokenMapper(
            input_shape=tuple(bb["input_shape"]),
            t_clip=bb["t_clip"],
            f_clip=bb["f_clip"],
        )
    return None
