from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
SONICS_ROOT = WORKSPACE_ROOT / "sonics"
if str(SONICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SONICS_ROOT))

from sonics.models.vit import ViT  # noqa: E402


class ViTBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ViT(
            image_size=cfg["input_shape"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            pe_learnable=cfg.get("pe_learnable", False),
        )
        self.embed_dim = cfg["embed_dim"]

    def forward(self, x):
        return self.encoder(x).mean(dim=1)
