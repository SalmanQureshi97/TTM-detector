from __future__ import annotations

import torch
import torchaudio


class DeezerAmplitudeFrontend(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, hf_cut, normalize_mean, normalize_std):
        super().__init__()
        self.sample_rate = sample_rate
        self.hf_cut = hf_cut
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, audio):
        spec = self.db(self.spec(audio))
        hz_per_bin = self.sample_rate / 2 / spec.shape[-2]
        max_bin = int(self.hf_cut / hz_per_bin)
        spec = spec[:, :max_bin, :]
        return (spec - self.normalize_mean) / self.normalize_std
