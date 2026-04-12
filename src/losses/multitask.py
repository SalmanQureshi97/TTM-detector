from __future__ import annotations

import torch.nn.functional as F


def multitask_loss(outputs, targets, auth_weight=1.0, enc_weight=1.0):
    return (
        auth_weight * F.binary_cross_entropy_with_logits(outputs["auth"], targets["auth"])
        + enc_weight * F.binary_cross_entropy_with_logits(outputs["enc"], targets["enc"])
    )
