"""Camera jitter perturbation: updates ctx['noise']['camera_jitter']."""
from __future__ import annotations

from typing import Dict, Optional


def apply(ctx: Dict, level: Optional[float] = 0.0) -> None:
    if level is None:
        level = 0.0
    ctx.setdefault("noise", {})["camera_jitter"] = float(level)

