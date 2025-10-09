"""Lighting perturbation: updates ctx['noise']['lighting']."""
from __future__ import annotations

from typing import Dict


def apply(ctx: Dict, level: float) -> None:
    ctx.setdefault("noise", {})["lighting"] = float(level)

