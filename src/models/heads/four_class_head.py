from __future__ import annotations

import torch.nn as nn


class FourClassHead(nn.Module):
    """Classification head over the shared encoder embedding.

    Default (``hidden_dim=None``): a single linear layer ``in_dim -> num_classes``
    (1,540 params for a 384-dim backbone -> 4 classes). With ``hidden_dim``
    set, becomes a 2-layer MLP ``in_dim -> hidden_dim -> num_classes`` with
    optional dropout in between -- handy when probing a frozen backbone and
    a linear projection isn't expressive enough.
    """

    def __init__(self, in_dim, num_classes=4, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x):
        return self.net(x)
