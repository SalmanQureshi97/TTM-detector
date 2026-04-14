"""Attention rollout for transformer-based backbones (ViT, SpecTTTra).

Implements the method of Abnar & Zuidema (2020):
  "Quantifying Attention Flow in Transformers."

Since both ViT and SpecTTTra use ``.mean(dim=1)`` pooling (not a CLS token),
the final rollout is averaged across all rows to represent the aggregated
attention from the output back to each input token.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.visualization.hooks import AttentionExtractor
from src.visualization.overlay import get_spectrogram_from_model
from src.visualization.token_mapping import get_mapper


def _fuse_heads(attn: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Reduce multi-head attention ``[B, H, N, N]`` → ``[B, N, N]``.

    Parameters
    ----------
    attn : Tensor [B, H, N, N]
    method : str
        ``"mean"`` | ``"max"`` | ``"min"`` across the head dimension.
    """
    if method == "mean":
        return attn.mean(dim=1)
    elif method == "max":
        return attn.max(dim=1).values
    elif method == "min":
        return attn.min(dim=1).values
    raise ValueError(f"Unknown head fusion method: {method}")


def _add_residual_and_renorm(attn: torch.Tensor) -> torch.Tensor:
    """Add identity (residual connection) and re-normalise rows.

    Parameters
    ----------
    attn : Tensor [B, N, N]
    """
    eye = torch.eye(attn.size(-1), device=attn.device, dtype=attn.dtype)
    attn = attn + eye.unsqueeze(0)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


class AttentionRollout:
    """Compute attention rollout heatmaps for ViT / SpecTTTra.

    Parameters
    ----------
    model : UnifiedAudioModel
    model_cfg : dict
        Model configuration (needed for token mapper).
    head_fusion : str
        How to fuse multi-head attention: ``"mean"`` | ``"max"`` | ``"min"``.
    """

    def __init__(
        self,
        model: nn.Module,
        model_cfg: dict,
        head_fusion: str = "mean",
    ) -> None:
        bb_type = model_cfg["backbone"]["type"]
        if bb_type == "speccnn":
            raise ValueError(
                "AttentionRollout is only supported for transformer backbones "
                "(vit, spectttra), not for speccnn."
            )
        self.model = model
        self.model_cfg = model_cfg
        self.head_fusion = head_fusion
        self.mapper = get_mapper(model_cfg)

    def compute(self, audio: torch.Tensor) -> np.ndarray:
        """Compute rollout heatmap for a single audio sample.

        Parameters
        ----------
        audio : Tensor
            Raw waveform ``[1, num_samples]`` or ``[num_samples]``.

        Returns
        -------
        np.ndarray
            Heatmap ``[H, W]`` matching the spectrogram dimensions.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = next(self.model.parameters()).device
        audio = audio.to(device)

        extractor = AttentionExtractor(self.model)
        with torch.no_grad():
            self.model(audio)
        attn_layers = extractor.get()  # list[Tensor[B, H, N, N]]
        extractor.remove()

        if not attn_layers:
            raise RuntimeError("No attention weights captured. Is this a transformer model?")

        # Rollout: multiply attention matrices across layers
        rollout = None
        for attn in attn_layers:
            attn = _fuse_heads(attn, self.head_fusion)  # [B, N, N]
            attn = _add_residual_and_renorm(attn)
            if rollout is None:
                rollout = attn
            else:
                rollout = rollout @ attn

        # Since pooling is mean over all tokens, average all rows
        # to get per-token importance from the perspective of the pooled output
        token_importance = rollout[0].mean(dim=0).cpu().numpy()  # [N]

        # Map to spectrogram
        if self.mapper is not None:
            expected = self.mapper.num_tokens
            if token_importance.shape[0] > expected:
                token_importance = token_importance[:expected]
            elif token_importance.shape[0] < expected:
                token_importance = np.pad(
                    token_importance, (0, expected - token_importance.shape[0])
                )
            heatmap = self.mapper.tokens_to_heatmap(token_importance)
        else:
            n = token_importance.shape[0]
            side = int(np.ceil(np.sqrt(n)))
            padded = np.zeros(side * side)
            padded[:n] = token_importance
            heatmap = padded.reshape(side, side)

        # Resize to match spectrogram
        spec = get_spectrogram_from_model(self.model, audio)
        if heatmap.shape != spec.shape:
            t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=spec.shape, mode="bilinear", align_corners=False)
            heatmap = t[0, 0].numpy()

        return heatmap

    def compute_per_layer(self, audio: torch.Tensor) -> list[np.ndarray]:
        """Return per-layer attention heatmaps (for diagnostic comparison).

        Each heatmap is the fused single-layer attention (not rollout).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = next(self.model.parameters()).device
        audio = audio.to(device)

        extractor = AttentionExtractor(self.model)
        with torch.no_grad():
            self.model(audio)
        attn_layers = extractor.get()
        extractor.remove()

        spec = get_spectrogram_from_model(self.model, audio)
        heatmaps = []

        for attn in attn_layers:
            fused = _fuse_heads(attn, self.head_fusion)  # [B, N, N]
            token_importance = fused[0].mean(dim=0).cpu().numpy()

            if self.mapper is not None:
                expected = self.mapper.num_tokens
                if token_importance.shape[0] > expected:
                    token_importance = token_importance[:expected]
                elif token_importance.shape[0] < expected:
                    token_importance = np.pad(
                        token_importance, (0, expected - token_importance.shape[0])
                    )
                hmap = self.mapper.tokens_to_heatmap(token_importance)
            else:
                n = token_importance.shape[0]
                side = int(np.ceil(np.sqrt(n)))
                padded = np.zeros(side * side)
                padded[:n] = token_importance
                hmap = padded.reshape(side, side)

            if hmap.shape != spec.shape:
                t = torch.from_numpy(hmap).float().unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=spec.shape, mode="bilinear", align_corners=False)
                hmap = t[0, 0].numpy()

            heatmaps.append(hmap)

        return heatmaps
