from __future__ import annotations

import torch


CLASS4_TO_AUTH = {0: 0, 1: 0, 2: 1, 3: 1}
CLASS4_TO_ENC = {0: 0, 1: 1, 2: 0, 3: 1}


def build_target(row, task_type: str):
    if task_type == "binary":
        return torch.tensor(float(row["target_value"]), dtype=torch.float32)

    if task_type == "multiclass":
        return torch.tensor(int(row["class4_label"]), dtype=torch.long)

    if task_type in {"multitask", "hierarchical"}:
        return {
            "auth": torch.tensor(float(row["auth_label"]), dtype=torch.float32),
            "enc": torch.tensor(float(row["enc_label"]), dtype=torch.float32),
        }

    raise ValueError(f"Unsupported task_type={task_type}")
