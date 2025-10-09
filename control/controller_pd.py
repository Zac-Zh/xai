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

