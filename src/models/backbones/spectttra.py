from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
SONICS_ROOT = WORKSPACE_ROOT / "sonics"
if str(SONICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SONICS_ROOT))

from sonics.models.spectttra import SpecTTTra  # noqa: E402


class SpectTTTraBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SpecTTTra(
            input_spec_dim=cfg["input_shape"][0],
            input_temp_dim=cfg["input_shape"][1],
            embed_dim=cfg["embed_dim"],
            t_clip=cfg["t_clip"],
            f_clip=cfg["f_clip"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            pre_norm=cfg.get("pre_norm", False),
            pe_learnable=cfg.get("pe_learnable", False),
        )
        self.embed_dim = cfg["embed_dim"]

    def forward(self, x):
        return self.encoder(x).mean(dim=1)
