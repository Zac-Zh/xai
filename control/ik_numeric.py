"""Numeric approximate IK for a 2-link planar arm (placeholder)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def ik_solve(target_xy: np.ndarray) -> Dict[str, object]:
    """Approximate IK for 2-link arm with unit link lengths.

    Args:
        target_xy: Target in plane (x,y) in [0,1]^2 scaled to reachable space.
    Returns:
        {'ik_ok': bool, 'q': ndarray shape (2,)}
    """
    x, y = float(target_xy[0] * 1.5), float(target_xy[1] * 1.5)
    r2 = x * x + y * y
    l1 = l2 = 1.0
    cos_q2 = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if cos_q2 < -1 or cos_q2 > 1:
        return {"ik_ok": False, "q": np.array([0.0, 0.0], dtype=float)}
    q2 = float(np.arccos(np.clip(cos_q2, -1.0, 1.0)))
    k1 = l1 + l2 * np.cos(q2)
    k2 = l2 * np.sin(q2)
    q1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
    return {"ik_ok": True, "q": np.array([q1, q2], dtype=float)}

