from __future__ import annotations

import torch
import torchaudio


class LogMelFrontend(torch.nn.Module):
    """Log-mel spectrogram frontend.

    Default behaviour (backwards compatible with the SpecCNN config) is a
    plain ``MelSpectrogram`` -> ``AmplitudeToDB``. The extra knobs
    (``win_length``, ``power``, ``top_db``, ``norm``) let the same module
    reproduce the SONICS SpecTTTra preprocessing: per-sample ``(mean, std)``
    normalisation on a dB-mel computed with explicit window length and
    ``top_db`` clip.
    """

    def __init__(
        self,
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
        f_min=0,
        f_max=16000,
        win_length=None,
        power=2.0,
        top_db=80.0,
        norm=None,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        if norm not in (None, "mean_std"):
            raise ValueError(f"Unsupported logmel norm={norm!r}")
        self.norm = norm

    def forward(self, audio):
        spec = self.db(self.mel(audio))
        if self.norm == "mean_std":
            # Per-sample (mean, std) on the (mel, time) plane.
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            spec = (spec - mean) / std
        return spec
