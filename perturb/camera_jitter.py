"""Camera jitter perturbation: updates ctx['noise']['camera_jitter']."""
from __future__ import annotations

from typing import Dict


def apply(ctx: Dict, level: float) -> None:
    ctx.setdefault("noise", {})["camera_jitter"] = float(level)

