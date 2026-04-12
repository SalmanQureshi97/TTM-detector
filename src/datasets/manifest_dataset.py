from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from src.datasets.label_builders import build_target


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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = torchaudio.load(Path(row["filepath"]))
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
