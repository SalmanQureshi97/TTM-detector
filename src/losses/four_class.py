from __future__ import annotations

import torch.nn.functional as F


def four_class_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)
