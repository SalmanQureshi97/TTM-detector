from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(*dicts: dict) -> dict:
    out = {}
    for d in dicts:
        out = _merge_two(out, d)
    return out


def _merge_two(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_two(out[key], value)
        else:
            out[key] = value
    return out
