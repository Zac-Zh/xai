"""Geometry stub: simulate 2D pose estimation and PnP quality."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def estimate_pose(detection: Dict[str, object], ctx: Dict) -> Dict[str, object]:
    """Simulate PnP pose estimate and reprojection RMSE.

    Success depends on detection and segmentation quality and noise.
    Returns:
        {'pnp_success': bool, 'pnp_rmse': float or None, 'pose_estimate': [x,y,theta]}
    """
    rng: np.random.Generator = ctx.get("rng") or np.random.default_rng(0)
    noise = ctx.get("noise", {})
    seg_iou = ctx.get("perception", {}).get("seg_iou")  # if caller set
    detected = bool(detection.get("detected", False))

    base_ok = 1.0 if detected else 0.3
    if seg_iou is None:
        seg_factor = 0.5
    else:
        seg_factor = float(np.clip(seg_iou, 0.0, 1.0))

    occl = float(noise.get("occlusion", 0.0))
    light = float(noise.get("lighting", 0.0))
    mblur = float(noise.get("motion_blur", 0.0))
    penalty = 1.0 - (0.5 * occl + 0.2 * light + 0.2 * mblur)
    quality = np.clip(base_ok * seg_factor * penalty, 0.0, 1.0)
    p_success = 0.2 + 0.8 * quality
    pnp_success = bool(rng.random() < p_success)
    # RMSE lower when success; higher with noise
    rmse = None  # type: Optional[float]
    if pnp_success:
        rmse = float(np.clip(5.0 * (1.0 - quality) + rng.normal(0, 0.5), 0.0, 10.0))
    else:
        rmse = float(np.clip(10.0 * (1.0 - quality) + rng.normal(0, 1.0), 5.0, 20.0))

    gt = ctx.get("gt", {})
    tx = float(gt.get("target_x", 0.5))
    ty = float(gt.get("target_y", 0.5))
    theta = float(rng.normal(0.0, 0.1 * (1.0 - quality + 0.1)))
    noise_xy = rng.normal(0, 0.05 * (1.0 - quality + 0.1), size=2)
    pose = [float(np.clip(tx + noise_xy[0], 0.0, 1.0)), float(np.clip(ty + noise_xy[1], 0.0, 1.0)), theta]
    return {"pnp_success": pnp_success, "pnp_rmse": rmse, "pose_estimate": pose}

