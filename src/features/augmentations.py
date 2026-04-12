from __future__ import annotations

import torch


class IdentityAugment(torch.nn.Module):
    def forward(self, x):
        return x
