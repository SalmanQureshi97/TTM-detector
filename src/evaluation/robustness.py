from __future__ import annotations

import torch
import torchaudio.functional as AF


def apply_bandpass(audio, sample_rate, low_hz=None, high_hz=None):
    if low_hz is not None:
        audio = AF.highpass_biquad(audio, sample_rate, low_hz)
    if high_hz is not None:
        audio = AF.lowpass_biquad(audio, sample_rate, high_hz)
    return audio
