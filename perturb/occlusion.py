"""Occlusion perturbation: simulates partial and complete object occlusion."""
from __future__ import annotations

from typing import Dict
import numpy as np


def apply(ctx: Dict, level: float) -> None:
    """
    Apply occlusion perturbation.

    Args:
        ctx: Context dictionary containing RNG and occlusion specifications
        level: Occlusion severity [0.0-1.0]
            0.0 = no occlusion
            0.3 = partial occlusion (30% coverage)
            0.6 = significant occlusion (60% coverage)
            1.0 = near-complete occlusion (90% coverage)
    """
    rng = ctx.get("rng", np.random.default_rng(0))

    ctx.setdefault("noise", {})["occlusion"] = float(level)

    # Occlusion types
    occlusion_config = ctx.setdefault("occlusion_config", {})
    occlusion_config["coverage_ratio"] = float(level)

    # Add circular obstacles proportional to level
    obs = ctx.setdefault("obstacles", [])
    obs.clear()

    # Number and size of occluders based on level
    if level > 0.0:
        # Dynamic occluders (e.g., other objects, people)
        num_dynamic = int(level * 8)
        for i in range(num_dynamic):
            # Random positions with bias toward target region
            if rng.random() < 0.7:
                # Near center (target likely location)
                x = float(rng.uniform(0.3, 0.7))
                y = float(rng.uniform(0.3, 0.7))
            else:
                # Anywhere
                x = float(rng.uniform(0.1, 0.9))
                y = float(rng.uniform(0.1, 0.9))

            # Radius increases with level
            r = float(rng.uniform(0.03, 0.08 * (1 + level)))
            obs.append((x, y, r))

        # Static occluders (e.g., walls, furniture)
        if level > 0.4:
            num_static = int((level - 0.4) * 5)
            for i in range(num_static):
                x = float(rng.uniform(0.2, 0.8))
                y = float(rng.uniform(0.2, 0.8))
                r = float(rng.uniform(0.05, 0.12))
                obs.append((x, y, r))

    # Occlusion patterns
    if level > 0.5:
        occlusion_config["partial_occlusion"] = True
        occlusion_config["edge_occlusion_prob"] = float((level - 0.5) * 0.8)

    if level > 0.7:
        occlusion_config["intermittent_occlusion"] = True
        occlusion_config["occlusion_frequency"] = float((level - 0.7) * 5.0)  # Hz

