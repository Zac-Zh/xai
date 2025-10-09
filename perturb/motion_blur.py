"""Motion blur perturbation: updates ctx['noise']['motion_blur']."""
from __future__ import annotations

from typing import Dict


def apply(ctx: Dict, level: float) -> None:
    ctx.setdefault("noise", {})["motion_blur"] = float(level)

