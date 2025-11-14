"""Synthetic detector stub that simulates target detection and confidence.

The behavior degrades with perturbation levels found in ctx['noise'] and uses
ctx['rng'] for deterministic randomness.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _base_conf(ctx: Dict) -> float:
    noise = ctx.get("noise", {})
    occl = float(noise.get("occlusion", 0.0))
    light = float(noise.get("lighting", 0.0))
    mblur = float(noise.get("motion_blur", 0.0))
    overlap = float(noise.get("overlap", 0.0))
    jitter = float(noise.get("camera_jitter", 0.0))
    # Combine as multiplicative penalty in [0,1]
    penalty = 1.0 - (0.6 * occl + 0.3 * light + 0.25 * mblur + 0.2 * overlap + 0.15 * jitter)
    return float(np.clip(0.85 * penalty, 0.0, 1.0))


def detect_target(image: np.ndarray, ctx: Dict) -> Dict[str, object]:
    """Simulate detection and confidence.

    Returns:
        {
          'detected': bool,
          'avg_conf': float,
          'bbox': [x1,y1,x2,y2]
        }
    """
    rng: np.random.Generator = ctx.get("rng") or np.random.default_rng(0)
    h, w = int(image.shape[0]), int(image.shape[1])
    conf = _base_conf(ctx)
    detected = bool(rng.random() < conf)

    # Use GT target if provided to anchor bbox
    gt = ctx.get("gt", {})
    tx = float(gt.get("target_x", 0.5))
    ty = float(gt.get("target_y", 0.5))
    cx = int(np.clip(tx * w + rng.normal(0, w * 0.01), 0, w - 1))
    cy = int(np.clip((1.0 - ty) * h + rng.normal(0, h * 0.01), 0, h - 1))
    bw = max(10, int(w * 0.1 * (1.0 + rng.normal(0, 0.05))))
    bh = max(10, int(h * 0.1 * (1.0 + rng.normal(0, 0.05))))
    x1 = int(np.clip(cx - bw // 2, 0, w - 1))
    y1 = int(np.clip(cy - bh // 2, 0, h - 1))
    x2 = int(np.clip(x1 + bw, 0, w - 1))
    y2 = int(np.clip(y1 + bh, 0, h - 1))
    bbox: List[int] = [x1, y1, x2, y2]

    if not detected:
        conf *= 0.5  # low confidence on miss
    return {"detected": detected, "avg_conf": float(conf), "bbox": bbox}


# Unified interface (compatible with real implementations)
def detect(img: np.ndarray, perturbation_level: float = 0.0) -> Dict[str, object]:
    """
    Unified detection interface (compatible with detector_real.py).

    Args:
        img: RGB image
        perturbation_level: Perturbation level (0.0 to 1.0)

    Returns:
        Detection dictionary
    """
    ctx = {
        "noise": {"occlusion": perturbation_level},
        "rng": np.random.default_rng(0),
        "gt": {"target_x": 0.5, "target_y": 0.5},
    }
    return detect_target(img, ctx)

