from __future__ import annotations

import torch.nn as nn


class FourClassHead(nn.Module):
    def __init__(self, in_dim, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
