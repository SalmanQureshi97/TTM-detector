from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from src.datasets.label_builders import build_target

log = logging.getLogger(__name__)


class AudioManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        task_cfg,
        dataset_filter=None,
        sample_rate=44100,
        max_seconds=30,
    ):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].copy()
        if dataset_filter:
            self.df = self.df[self.df["source_dataset"].isin(dataset_filter)].copy()

        self.task_cfg = task_cfg
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

        if task_cfg["type"] == "binary":
            self.df["target_value"] = self.df[task_cfg["target"]]

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _process_row(self, row):
        filepath = Path(row["filepath"])

        # Chunked-manifest mode: if the row carries a chunk_start_sec, decode
        # only the frames for that 30 s window (torchaudio.load supports
        # frame_offset + num_frames, so this is cheap even for long files).
        # Falls back to the legacy first-max_seconds behavior otherwise, so
        # older manifests still work unchanged.
        chunk_start = row.get("chunk_start_sec") if hasattr(row, "get") else None
        if chunk_start is not None and pd.notna(chunk_start):
            info = torchaudio.info(filepath)
            src_sr = info.sample_rate
            frame_offset = int(float(chunk_start) * src_sr)
            # +100 ms slack in case resampling trims edges.
            needed_src_frames = (
                int(self.max_seconds * src_sr) + int(0.1 * src_sr)
            )
            audio, sr = torchaudio.load(
                filepath,
                frame_offset=frame_offset,
                num_frames=needed_src_frames,
            )
        else:
            audio, sr = torchaudio.load(filepath)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.mean(dim=0)

        max_len = int(self.sample_rate * self.max_seconds)
        if audio.numel() > max_len:
            audio = audio[:max_len]
        elif audio.numel() < max_len:
            audio = torch.nn.functional.pad(audio, (0, max_len - audio.numel()))

        target = build_target(row, self.task_cfg["type"])
        return {
            "audio": audio,
            "target": target,
            "meta": row.to_dict(),
        }

    def __getitem__(self, idx):
        # Some source files are corrupt / unreadable (e.g. ffmpeg "Invalid
        # argument"). A single bad file must not crash the whole run, so on a
        # load failure we log it and substitute another random sample. With a
        # handful of bad files out of ~180k, the label distribution is
        # unaffected.
        cur = idx
        for _ in range(10):
            row = self.df.iloc[cur]
            try:
                return self._process_row(row)
            except Exception as err:  # noqa: BLE001 - torchaudio raises RuntimeError/OSError
                log.warning("Skipping unreadable file %s (%s)", row["filepath"], err)
                cur = random.randrange(len(self.df))
        raise RuntimeError(
            f"Failed to load a valid sample after 10 attempts (started at idx {idx})"
        )
