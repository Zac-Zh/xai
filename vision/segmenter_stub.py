"""Synthetic segmenter stub returning a binary mask and a simulated IoU."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _gt_mask(h: int, w: int, tx: float, ty: float, radius_px: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cx = int(np.clip(tx * w, 0, w - 1))
    cy = int(np.clip((1.0 - ty) * h, 0, h - 1))
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2


def segment_target(image: np.ndarray, ctx: Dict) -> Dict[str, object]:
    """Generate a plausible mask and IoU against synthetic GT mask.

    Returns:
        {'mask': ndarray(bool), 'seg_iou': float or None}
    """
    rng: np.random.Generator = ctx.get("rng") or np.random.default_rng(0)
    noise = ctx.get("noise", {})
    h, w = int(image.shape[0]), int(image.shape[1])
    gt = ctx.get("gt", {})
    tx = float(gt.get("target_x", 0.5))
    ty = float(gt.get("target_y", 0.5))
    base_radius = max(4, w // 80)
    gt_m = _gt_mask(h, w, tx, ty, base_radius)

    # Predicted mask perturbed by occlusion/lighting
    occl = float(noise.get("occlusion", 0.0))
    light = float(noise.get("lighting", 0.0))
    mblur = float(noise.get("motion_blur", 0.0))
    # Dilate/erode synthetic mask by noise
    pred_radius = int(np.clip(base_radius * (1.0 + 0.8 * mblur - 0.6 * light), 2, base_radius * 3))
    pred_mask = _gt_mask(h, w, tx + rng.normal(0, 0.01 * (1 + mblur)), ty + rng.normal(0, 0.01 * (1 + mblur)), pred_radius)
    # Randomly occlude pixels
    if occl > 0:
        drop = rng.random((h, w)) < (0.2 * occl)
        pred_mask = np.logical_and(pred_mask, ~drop)

    inter = np.logical_and(gt_m, pred_mask).sum()
    union = np.logical_or(gt_m, pred_mask).sum()
    iou: Optional[float] = float(inter / union) if union > 0 else None
    return {"mask": pred_mask.astype(bool), "seg_iou": iou}

