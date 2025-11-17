"""Lighting perturbation: updates ctx['noise']['lighting']."""
from __future__ import annotations

from typing import Dict, Optional


def apply(ctx: Dict, level: Optional[float] = 0.0) -> None:
    if level is None:
        level = 0.0
    ctx.setdefault("noise", {})["lighting"] = float(level)

