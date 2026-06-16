"""SpecAugment (Park et al., 2019).

Applies frequency masking and time masking to a spectrogram. Time warping
from the original paper is omitted: the paper's ablation (Table 4) shows
warping contributes the least of the three operations and the modern
practice is to drop it for simplicity. Mixup is a separate augmentation
that operates on (input, label) pairs and is not handled here.

The module is a no-op outside ``training`` mode, so you can leave it
attached during evaluation without affecting val/test results.

Reference:
    Park et al., "SpecAugment: A Simple Data Augmentation Method for
    Automatic Speech Recognition", Interspeech 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpecAugment(nn.Module):
    """Apply frequency + time masking to a spectrogram tensor.

    Parameters
    ----------
    freq_mask_param : int
        Maximum width of each frequency mask (in mel bins / freq bins).
        Each mask is sampled uniformly in ``[0, freq_mask_param]``.
    n_freq_masks : int
        Number of frequency masks to draw per sample.
    time_mask_param : int
        Maximum width of each time mask (in time frames).
    n_time_masks : int
        Number of time masks to draw per sample.
    p : float
        Probability of applying SpecAugment for a given training forward
        pass (whole-batch coin flip; default 1.0 = always apply during
        training).
    mask_value : float
        Value to fill masked regions with. Default ``0.0`` is appropriate
        for spectrograms that have been mean-std normalised (zero is the
        mean) or for any zero-centred representation.

    Notes
    -----
    Input shape ``[B, C, F, T]`` is required when ``n>=1`` masks are
    requested and per-sample (iid) masking is desired. ``[B, F, T]`` is
    also accepted -- in that case the *same* mask is used across the
    batch, which is the older torchaudio behaviour.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        n_freq_masks: int = 2,
        time_mask_param: int = 100,
        n_time_masks: int = 2,
        p: float = 1.0,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.mask_value = float(mask_value)
        self.n_freq_masks = int(n_freq_masks)
        self.n_time_masks = int(n_time_masks)

        # iid_masks=True requires 4D input. Per-sample mask is the modern
        # default and noticeably more effective than a single batch mask.
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
            for _ in range(self.n_freq_masks)
        ])
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)
            for _ in range(self.n_time_masks)
        ])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        if self.p < 1.0 and torch.rand(1, device=spec.device).item() > self.p:
            return spec

        # Both torchaudio masks expect the last two dims to be (freq, time).
        # If we got [B, F, T], add a channel dim for iid_masks compatibility,
        # then restore the original rank on the way out.
        added_channel = False
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
            added_channel = True

        for m in self.freq_masks:
            spec = m(spec, mask_value=self.mask_value)
        for m in self.time_masks:
            spec = m(spec, mask_value=self.mask_value)

        if added_channel:
            spec = spec.squeeze(1)
        return spec
