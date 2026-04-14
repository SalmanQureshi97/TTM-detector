"""Per-frequency sensitivity analysis.

Provides two complementary methods:
1. **Band occlusion**: zero out contiguous mel-frequency bands and measure
   the change in model output.
2. **Gradient frequency profile**: compute the gradient of the output w.r.t.
   each frequency bin of the spectrogram.

Both methods reveal which spectral regions the model relies on for detection,
which can be compared against the theoretically predicted locations of
deconvolution-induced spectral peaks (Afchar et al., 2025).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class FrequencySensitivity:
    """Analyse frequency-band dependence of a ``UnifiedAudioModel``.

    Parameters
    ----------
    model : UnifiedAudioModel
    device : torch.device
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    # ------------------------------------------------------------------
    # Internal: replicate the forward pipeline with an injected spec
    # ------------------------------------------------------------------
    def _forward_from_spec(
        self,
        spec: torch.Tensor,
        task_output_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Run backbone + head from a spectrogram tensor ``[B, H, W]``.

        Replicates the ``UnifiedAudioModel.forward`` path *after* the
        frontend, so we can inject a modified spectrogram.
        """
        x = spec.unsqueeze(1)  # [B, 1, H, W]
        if self.model.resize_to is not None:
            x = F.interpolate(x, size=self.model.resize_to, mode="bilinear")
        embedding = self.model.backbone(x)
        output = self.model.head(embedding)

        if isinstance(output, dict):
            key = task_output_key or "auth"
            return torch.sigmoid(output[key])
        elif output.dim() == 1:
            return torch.sigmoid(output)
        else:
            return torch.softmax(output, dim=1)

    # ------------------------------------------------------------------
    # Band occlusion
    # ------------------------------------------------------------------
    def band_occlusion(
        self,
        audio: torch.Tensor,
        n_bands: int = 16,
        task_output_key: Optional[str] = None,
    ) -> dict:
        """Zero out mel-frequency bands and measure output change.

        Parameters
        ----------
        audio : Tensor
            ``[1, num_samples]`` or ``[num_samples]``.
        n_bands : int
            Number of equal-width frequency bands to occlude.
        task_output_key : str, optional
            ``"auth"`` or ``"enc"`` for multitask models.

        Returns
        -------
        dict with keys:
            ``band_edges``  – list of ``(start_bin, end_bin)`` tuples
            ``sensitivity`` – np.ndarray ``[n_bands]`` of absolute output deltas
            ``baseline``    – float, baseline output value
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        self.model.eval()
        with torch.no_grad():
            spec = self.model.frontend(audio)  # [B, H, W]
            baseline_out = self._forward_from_spec(spec, task_output_key)
            baseline_val = baseline_out.mean().item()

        H = spec.shape[1]
        band_size = H // n_bands
        band_edges = []
        sensitivity = []

        for i in range(n_bands):
            start = i * band_size
            end = min((i + 1) * band_size, H)
            band_edges.append((start, end))

            masked_spec = spec.clone()
            masked_spec[:, start:end, :] = 0.0

            with torch.no_grad():
                masked_out = self._forward_from_spec(masked_spec, task_output_key)
                delta = abs(masked_out.mean().item() - baseline_val)
                sensitivity.append(delta)

        return {
            "band_edges": band_edges,
            "sensitivity": np.array(sensitivity),
            "baseline": baseline_val,
        }

    # ------------------------------------------------------------------
    # Gradient frequency profile
    # ------------------------------------------------------------------
    def gradient_frequency_profile(
        self,
        audio: torch.Tensor,
        task_output_key: Optional[str] = None,
    ) -> dict:
        """Compute gradient of the output w.r.t. each frequency bin.

        Parameters
        ----------
        audio : Tensor
            ``[1, num_samples]`` or ``[num_samples]``.
        task_output_key : str, optional

        Returns
        -------
        dict with keys:
            ``freq_bins``          – np.ndarray ``[H]`` bin indices
            ``gradient_magnitude`` – np.ndarray ``[H]`` mean |grad| per freq row
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        self.model.eval()

        # Compute spectrogram and enable grad
        with torch.no_grad():
            spec = self.model.frontend(audio)  # [B, H, W]
        spec = spec.detach().requires_grad_(True)

        # Forward from spec (with grad)
        x = spec.unsqueeze(1)
        if self.model.resize_to is not None:
            x = F.interpolate(x, size=self.model.resize_to, mode="bilinear")
        embedding = self.model.backbone(x)
        output = self.model.head(embedding)

        if isinstance(output, dict):
            key = task_output_key or "auth"
            scalar = output[key].sum()
        elif output.dim() == 1:
            scalar = output.sum()
        else:
            scalar = output.max(dim=1).values.sum()

        scalar.backward()

        grad = spec.grad[0]  # [H, W]
        # Mean absolute gradient per frequency bin
        grad_mag = grad.abs().mean(dim=1).cpu().numpy()  # [H]

        return {
            "freq_bins": np.arange(grad_mag.shape[0]),
            "gradient_magnitude": grad_mag,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_sensitivity(
        self,
        result: dict,
        method: str,
        output_path: Optional[Path] = None,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (10, 4),
    ) -> Figure:
        """Plot frequency sensitivity as a bar/line chart.

        Parameters
        ----------
        result : dict
            Output from ``band_occlusion`` or ``gradient_frequency_profile``.
        method : str
            ``"occlusion"`` or ``"gradient"``.
        output_path : Path, optional
        title : str, optional
        """
        fig, ax = plt.subplots(figsize=figsize)

        if method == "occlusion":
            edges = result["band_edges"]
            vals = result["sensitivity"]
            labels = [f"{s}-{e}" for s, e in edges]
            x = np.arange(len(vals))
            ax.bar(x, vals, color="steelblue", edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("Mel bin range")
            ax.set_ylabel("|Output delta|")
            ax.set_title(title or "Frequency band occlusion sensitivity")
        elif method == "gradient":
            bins = result["freq_bins"]
            mags = result["gradient_magnitude"]
            ax.barh(bins, mags, color="coral", edgecolor="white", height=0.8)
            ax.set_ylabel("Mel bin")
            ax.set_xlabel("Mean |gradient|")
            ax.set_title(title or "Gradient frequency profile")
            ax.invert_yaxis()
        else:
            raise ValueError(f"Unknown method: {method}")

        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig
