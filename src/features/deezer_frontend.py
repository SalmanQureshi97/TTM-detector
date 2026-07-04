from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torchaudio

log = logging.getLogger(__name__)


class DeezerAmplitudeFrontend(torch.nn.Module):
    """Amplitude-spectrogram frontend used by the SpecCNN model.

    Backwards-compatible default: raw power spectrogram in dB, cropped at
    ``hf_cut`` and standardised with the caller-provided constants
    ``normalize_mean`` / ``normalize_std``.

    When ``stats_file`` is given, the constants are replaced by
    dataset-wide (mean, std) values computed offline via
    ``scripts/compute_dataset_stats.py``. The same (mean, std) applies to
    every input at both training and inference. Two normalisation steps
    are performed in that mode:

    1. **Audio-domain** standardisation of the raw waveform:
       ``(audio - audio_mean) / audio_std``.
    2. **Spectrogram-domain** standardisation of the dB spectrogram:
       ``(spec  - spec_mean)  / spec_std``.

    Both normalisation tensors are registered as buffers so they follow
    the model to GPU and are saved with ``state_dict``.
    """

    def __init__(
        self,
        sample_rate,
        n_fft,
        hop_length,
        hf_cut,
        normalize_mean,
        normalize_std,
        stats_file: str | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hf_cut = hf_cut
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

        # Stats-file mode. Load once at construction; register as buffers.
        self.use_stats_file = stats_file is not None
        if self.use_stats_file:
            path = Path(stats_file)
            if not path.exists():
                # Try repo-root relative (mirror the SpecTTTra pretrained-weights lookup).
                repo_root = Path(__file__).resolve().parents[2]
                candidate = repo_root / stats_file
                if candidate.exists():
                    path = candidate
                else:
                    raise FileNotFoundError(
                        f"stats_file not found: {stats_file} (also tried {candidate})"
                    )
            with open(path) as f:
                payload = json.load(f)
            self.register_buffer(
                "audio_mean", torch.tensor(float(payload["audio"]["mean"])),
                persistent=True,
            )
            self.register_buffer(
                "audio_std", torch.tensor(float(payload["audio"]["std"])),
                persistent=True,
            )
            self.register_buffer(
                "spec_mean", torch.tensor(float(payload["spec"]["mean"])),
                persistent=True,
            )
            self.register_buffer(
                "spec_std", torch.tensor(float(payload["spec"]["std"])),
                persistent=True,
            )
            log.info(
                "DeezerAmplitudeFrontend using stats_file=%s: "
                "audio(mean=%.4f, std=%.4f) spec(mean=%.4f, std=%.4f)",
                path.name,
                payload["audio"]["mean"], payload["audio"]["std"],
                payload["spec"]["mean"], payload["spec"]["std"],
            )
        else:
            # Legacy constant-normalisation path. Store on self so the same
            # attributes work for both branches of forward().
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std

    def forward(self, audio):
        if self.use_stats_file:
            audio = (audio - self.audio_mean) / self.audio_std.clamp(min=1e-6)
        spec = self.db(self.spec(audio))
        hz_per_bin = self.sample_rate / 2 / spec.shape[-2]
        max_bin = int(self.hf_cut / hz_per_bin)
        spec = spec[:, :max_bin, :]
        if self.use_stats_file:
            return (spec - self.spec_mean) / self.spec_std.clamp(min=1e-6)
        return (spec - self.normalize_mean) / self.normalize_std
