"""Gradient-weighted Class Activation Mapping (Grad-CAM).

Supports all three backbone families:
- **SpecCNN**: standard Grad-CAM on the last convolutional feature map.
- **ViT / SpecTTTra**: gradient × activation on the last transformer
  block output, mapped back to the spectrogram via token mappers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.visualization.hooks import (
    FeatureExtractor,
    GradientExtractor,
    find_last_conv2d,
)
from src.visualization.overlay import get_spectrogram_from_model
from src.visualization.token_mapping import get_mapper


class GradCAM:
    """Compute Grad-CAM heatmaps for a ``UnifiedAudioModel``.

    Parameters
    ----------
    model : UnifiedAudioModel
        The full audio model (frontend → backbone → head).
    model_cfg : dict
        Model config dict (needed to resolve backbone type and token mapper).
    target_layer : str, optional
        Explicit layer name to target.  If *None*, auto-selected based on
        the backbone type.
    """

    def __init__(
        self,
        model: nn.Module,
        model_cfg: dict,
        target_layer: Optional[str] = None,
    ) -> None:
        self.model = model
        self.model_cfg = model_cfg
        self.backbone_type = model_cfg["backbone"]["type"]
        self.target_layer = target_layer or self._auto_select_layer()
        self.mapper = get_mapper(model_cfg)

    # ------------------------------------------------------------------
    def _auto_select_layer(self) -> str:
        if self.backbone_type == "speccnn":
            # Last conv block inside backbone.features
            n_blocks = len(list(self.model.backbone.features.children()))
            return f"backbone.features.{n_blocks - 1}"
        elif self.backbone_type in ("vit", "spectttra"):
            # Last transformer block
            n_blocks = len(list(self.model.backbone.encoder.transformer.blocks))
            return f"backbone.encoder.transformer.blocks.{n_blocks - 1}"
        raise ValueError(f"Unknown backbone type: {self.backbone_type}")

    # ------------------------------------------------------------------
    def compute(
        self,
        audio: torch.Tensor,
        target_class: Optional[int] = None,
        task_output_key: Optional[str] = None,
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap for one audio sample.

        Parameters
        ----------
        audio : Tensor
            Raw waveform, shape ``[1, num_samples]`` or ``[num_samples]``.
        target_class : int, optional
            Class index for multiclass tasks.  ``None`` → use predicted class.
        task_output_key : str, optional
            ``"auth"`` or ``"enc"`` for multitask models.

        Returns
        -------
        np.ndarray
            Heatmap of shape matching the *spectrogram* dimensions ``[H, W]``.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = next(self.model.parameters()).device
        audio = audio.to(device)

        self.model.zero_grad()
        grad_ext = GradientExtractor(self.model, [self.target_layer])

        # Forward
        output = self.model(audio)

        # Select scalar to backprop from
        scalar = self._select_scalar(output, target_class, task_output_key)
        scalar.backward(retain_graph=False)

        activations = grad_ext.get_activations()[self.target_layer]
        gradients = grad_ext.get()[self.target_layer]
        grad_ext.remove()

        if self.backbone_type == "speccnn":
            heatmap = self._cam_cnn(activations, gradients)
        else:
            heatmap = self._cam_transformer(activations, gradients)

        # Get the spectrogram shape for resizing
        spec = get_spectrogram_from_model(self.model, audio)
        if heatmap.shape != spec.shape:
            heatmap = _resize(heatmap, spec.shape)

        return heatmap

    # ------------------------------------------------------------------
    @staticmethod
    def _select_scalar(output, target_class, task_output_key):
        """Pick a single scalar from model output to backprop from."""
        if isinstance(output, dict):
            key = task_output_key or "auth"
            logit = output[key]
        elif output.dim() == 1:
            logit = output  # binary: [B]
        else:
            logit = output  # multiclass: [B, C]

        if logit.dim() == 1:
            return logit.sum()
        else:
            # Multiclass
            if target_class is None:
                target_class = logit.argmax(dim=1).item()
            return logit[0, target_class]

    # ------------------------------------------------------------------
    @staticmethod
    def _cam_cnn(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """Standard Grad-CAM for CNN feature maps ``[B, C, H, W]``."""
        # Global-average-pool the gradients to get per-channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        # Normalise
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        return cam

    # ------------------------------------------------------------------
    def _cam_transformer(
        self, activations: torch.Tensor, gradients: torch.Tensor
    ) -> np.ndarray:
        """Grad-CAM for transformer outputs ``[B, N, D]``.

        Per-token importance = mean(|gradient × activation|) over the
        embedding dimension, producing a weight per token.
        """
        # activations, gradients: [B, N, D]
        importance = (gradients * activations).abs().mean(dim=-1)  # [B, N]
        weights = importance[0].detach().cpu().numpy()

        if self.mapper is not None:
            # Truncate or pad if token count doesn't match exactly
            expected = self.mapper.num_tokens
            if weights.shape[0] > expected:
                weights = weights[:expected]
            elif weights.shape[0] < expected:
                weights = np.pad(weights, (0, expected - weights.shape[0]))
            return self.mapper.tokens_to_heatmap(weights)
        else:
            # Fallback: reshape to square-ish and resize later
            n = weights.shape[0]
            side = int(np.ceil(np.sqrt(n)))
            padded = np.zeros(side * side)
            padded[:n] = weights
            return padded.reshape(side, side)


def _resize(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize heatmap using bilinear interpolation."""
    t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()
