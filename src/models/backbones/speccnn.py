from __future__ import annotations

import torch
import torch.nn as nn


class SpecCNNBackbone(nn.Module):
    def __init__(self, channels, strides, kernel_sizes, head_dim):
        super().__init__()
        layers = []
        in_ch = 1
        for ch, stride, kernel in zip(channels, strides, kernel_sizes):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_ch, head_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)
