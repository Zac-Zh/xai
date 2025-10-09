"""Overlap perturbation: updates ctx['noise']['overlap'."""
from __future__ import annotations

from typing import Dict


def apply(ctx: Dict, level: float) -> None:
    ctx.setdefault("noise", {})["overlap"] = float(level)

