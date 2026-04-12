from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.deezer_frontend import DeezerAmplitudeFrontend
from src.features.logmel import LogMelFrontend
from src.models.heads.binary_head import BinaryHead
from src.models.heads.four_class_head import FourClassHead
from src.models.heads.multitask_head import MultiTaskHead


class UnifiedAudioModel(nn.Module):
    def __init__(self, model_cfg, task_cfg):
        super().__init__()
        self.task_cfg = task_cfg
        frontend_cfg = model_cfg["frontend"]
        if frontend_cfg["type"] == "logmel":
            self.frontend = LogMelFrontend(
                sample_rate=frontend_cfg["sample_rate"],
                n_fft=frontend_cfg["n_fft"],
                hop_length=frontend_cfg["hop_length"],
                n_mels=frontend_cfg["n_mels"],
                f_min=frontend_cfg.get("f_min", 0),
                f_max=frontend_cfg.get("f_max", 16000),
            )
        elif frontend_cfg["type"] == "deezer_amplitude":
            self.frontend = DeezerAmplitudeFrontend(
                sample_rate=frontend_cfg["sample_rate"],
                n_fft=frontend_cfg["n_fft"],
                hop_length=frontend_cfg["hop_length"],
                hf_cut=frontend_cfg["hf_cut"],
                normalize_mean=frontend_cfg["normalize_mean"],
                normalize_std=frontend_cfg["normalize_std"],
            )
        else:
            raise ValueError(f"Unsupported frontend={frontend_cfg['type']}")

        backbone_cfg = model_cfg["backbone"]
        if backbone_cfg["type"] == "speccnn":
            from src.models.backbones.speccnn import SpecCNNBackbone

            self.backbone = SpecCNNBackbone(
                channels=backbone_cfg["channels"],
                strides=backbone_cfg["strides"],
                kernel_sizes=backbone_cfg["kernel_sizes"],
                head_dim=backbone_cfg["head_dim"],
            )
            out_dim = backbone_cfg["head_dim"]
            self.resize_to = None
        elif backbone_cfg["type"] == "vit":
            from src.models.backbones.vit import ViTBackbone

            self.backbone = ViTBackbone(backbone_cfg)
            out_dim = self.backbone.embed_dim
            self.resize_to = tuple(backbone_cfg["input_shape"])
        elif backbone_cfg["type"] == "spectttra":
            from src.models.backbones.spectttra import SpectTTTraBackbone

            self.backbone = SpectTTTraBackbone(backbone_cfg)
            out_dim = self.backbone.embed_dim
            self.resize_to = tuple(backbone_cfg["input_shape"])
        else:
            raise ValueError(f"Unsupported backbone={backbone_cfg['type']}")

        if task_cfg["type"] == "binary":
            self.head = BinaryHead(out_dim)
        elif task_cfg["type"] == "multiclass":
            self.head = FourClassHead(out_dim, task_cfg.get("num_classes", 4))
        elif task_cfg["type"] in {"multitask", "hierarchical"}:
            self.head = MultiTaskHead(out_dim)
        else:
            raise ValueError(f"Unsupported task_type={task_cfg['type']}")

    def forward(self, audio):
        spec = self.frontend(audio)
        spec = spec.unsqueeze(1)
        if self.resize_to is not None:
            spec = F.interpolate(spec, size=self.resize_to, mode="bilinear")
        embedding = self.backbone(spec)
        return self.head(embedding)
