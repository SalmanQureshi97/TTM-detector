from __future__ import annotations

import json
from pathlib import Path
from typing import Union


def save_report(report: dict, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
