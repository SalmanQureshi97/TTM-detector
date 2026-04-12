from __future__ import annotations

import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.auth = nn.Linear(in_dim, 1)
        self.enc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return {
            "auth": self.auth(x).squeeze(-1),
            "enc": self.enc(x).squeeze(-1),
        }
