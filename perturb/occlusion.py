"""Occlusion perturbation: updates ctx['noise'] and adds obstacles."""
from __future__ import annotations

from typing import Dict, Optional


def apply(ctx: Dict, level: Optional[float] = 0.0) -> None:
    if level is None:
        level = 0.0
    ctx.setdefault("noise", {})["occlusion"] = float(level)
    # Add circular obstacles proportional to level
    obs = ctx.setdefault("obstacles", [])
    obs.clear()
    n = int(5 * level)
    # Pack obstacles near the center area
    for i in range(n):
        # positions spread around 0.5,0.5 with small radius
        x = 0.4 + 0.2 * (i % 5) / max(1, n)
        y = 0.4 + 0.2 * (i // 5) / max(1, n)
        r = 0.05
        obs.append((x, y, r))

