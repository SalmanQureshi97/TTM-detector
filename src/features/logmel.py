from __future__ import annotations

import torch
import torchaudio


class LogMelFrontend(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min=0, f_max=16000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, audio):
        spec = self.mel(audio)
        return self.db(spec)
