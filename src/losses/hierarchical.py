from __future__ import annotations

from src.losses.multitask import multitask_loss


def hierarchical_loss(outputs, targets, auth_weight=1.0, enc_weight=1.0):
    return multitask_loss(outputs, targets, auth_weight=auth_weight, enc_weight=enc_weight)
