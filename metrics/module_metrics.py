"""Utilities for module-level metrics (placeholder for extensibility)."""
from __future__ import annotations

from typing import Dict, Optional


def perception_ok(avg_conf: float, seg_iou: Optional[float], th: Dict) -> bool:
    tau_c = float(th["perception"]["conf_tau"])
    tau_i = float(th["perception"]["seg_iou_tau"])
    return (avg_conf >= tau_c) and (seg_iou is None or seg_iou >= tau_i)


def geometry_ok(pnp_success: bool, pnp_rmse: Optional[float], th: Dict) -> bool:
    if not pnp_success:
        return False
    tau = float(th["geometry"]["pnp_rmse_tau"])
    return pnp_rmse is not None and pnp_rmse <= tau


def planning_ok(success: bool, collisions: int, th: Dict) -> bool:
    max_c = int(th["planning"]["max_collisions"])
    return bool(success) and int(collisions) <= max_c


def control_ok(track_rmse: Optional[float], th: Dict) -> bool:
    if track_rmse is None:
        return False
    tau = float(th["control"]["track_rmse_tau"])
    return float(track_rmse) <= tau

