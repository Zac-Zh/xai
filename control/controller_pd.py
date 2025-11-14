"""PD tracking simulation with noise injected by perturbations."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def track(path: List[List[float]], ctx: Dict) -> Dict[str, object]:
    """Simulate tracking of a 2D path.

    Returns:
        {'track_rmse': float or None, 'overshoot': float or None, 'oscillation': bool}
    """
    if not path:
        return {"track_rmse": None, "overshoot": None, "oscillation": False}
    rng: np.random.Generator = ctx.get("rng") or np.random.default_rng(0)
    noise = ctx.get("noise", {})
    mblur = float(noise.get("motion_blur", 0.0))
    jitter = float(noise.get("camera_jitter", 0.0))
    base_sigma = 0.002
    sigma = base_sigma * (1.0 + 3.0 * (0.5 * mblur + 0.5 * jitter))
    pts = np.asarray(path, dtype=float)
    err = rng.normal(0.0, sigma, size=pts.shape)
    tracked = np.clip(pts + err, 0.0, 1.0)
    rmse = float(np.sqrt(np.mean(np.sum((tracked - pts) ** 2, axis=1))))
    overshoot = float(np.clip(rng.normal(0.02 * (mblur + jitter), 0.005), 0.0, 0.2))
    oscillation = bool(rng.random() < 0.1 * (mblur + jitter))
    return {"track_rmse": rmse, "overshoot": overshoot, "oscillation": oscillation}


# Unified interface (compatible with real implementations)
def track_path(path: List[np.ndarray], perturbation_level: float = 0.0) -> Dict[str, object]:
    """
    Unified tracking interface (compatible with controller_real.py).

    Args:
        path: Path waypoints (list of numpy arrays or lists)
        perturbation_level: Perturbation level (0.0 to 1.0)

    Returns:
        Tracking result dictionary
    """
    # Convert path to list of lists if needed
    path_lists = []
    for waypoint in path:
        if isinstance(waypoint, np.ndarray):
            path_lists.append(waypoint[:2].tolist() if len(waypoint) > 2 else waypoint.tolist())
        else:
            path_lists.append(list(waypoint[:2]) if len(waypoint) > 2 else list(waypoint))

    ctx = {
        "rng": np.random.default_rng(0),
        "noise": {
            "motion_blur": perturbation_level,
            "camera_jitter": perturbation_level * 0.5,
        },
    }

    result = track(path_lists, ctx)

    # Add success field for compatibility
    result["success"] = result["track_rmse"] is not None and result["track_rmse"] < 0.1

    return result

