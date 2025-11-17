"""Module failure perturbation: simulates component failures in the perception-planning-control pipeline."""
from __future__ import annotations

from typing import Dict
import numpy as np


def apply(ctx: Dict, level: float) -> None:
    """
    Apply module failure perturbation.

    Args:
        ctx: Context dictionary containing RNG and module failure specifications
        level: Failure severity [0.0-1.0]
            0.0 = no failures
            0.3 = minor degradation
            0.6 = moderate failures
            1.0 = critical failures
    """
    rng = ctx.get("rng", np.random.default_rng(0))

    # Initialize failure modes
    failures = ctx.setdefault("failures", {})

    # Perception module failures
    if level > 0.2:
        # Detector confidence degradation
        failures["detector_dropout"] = float(min(level * 0.7, 0.9))
        # Segmentation accuracy reduction
        failures["segmentation_noise"] = float(level * 0.6)

    # Geometry module failures
    if level > 0.3:
        # PnP solver failures
        failures["pnp_failure_rate"] = float(min(level * 0.5, 0.8))
        # Pose estimation error amplification
        failures["pose_error_scale"] = float(1.0 + level * 3.0)

    # Planning module failures
    if level > 0.4:
        # Path planner timeout/failure
        failures["planner_failure_rate"] = float(min(level * 0.4, 0.7))
        # Suboptimal path generation
        failures["path_quality_degradation"] = float(level * 0.5)

    # Control module failures
    if level > 0.5:
        # Controller gain reduction
        failures["control_gain_factor"] = float(max(1.0 - level, 0.1))
        # Actuator lag
        failures["actuator_delay"] = float(level * 0.3)

    # Communication failures
    if level > 0.6:
        # Inter-module communication drops
        failures["comm_dropout_rate"] = float(min((level - 0.6) * 2.0, 0.5))

    # Add noise proportional to failure level
    ctx.setdefault("noise", {})["module_failure"] = float(level)
