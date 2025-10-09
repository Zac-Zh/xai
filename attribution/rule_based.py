"""Rule-based layered failure attribution for R2."""
from __future__ import annotations

from typing import Dict, List, Tuple


def _root_cause_from_scenario(s: str) -> str:
    mapping = {
        "occlusion": "Occlusion",
        "lighting": "Lighting",
        "motion_blur": "MotionBlur",
        "overlap": "Overlap",
        "camera_jitter": "CameraJitter",
    }
    return mapping.get(s, "Unknown")


def attribute_failure(log: Dict, th: Dict) -> Tuple[List[Tuple[str, str]], str]:
    """Apply layered checks against thresholds to attribute failures.

    Returns list of (Module, error_code) in order and a root-cause label.
    """
    errs: List[Tuple[str, str]] = []
    # Perception
    p = log["perception"]
    if not bool(p.get("detected", False)):
        errs.append(("Perception", "miss_detection"))
    if float(p.get("avg_conf", 0.0)) < float(th["perception"]["conf_tau"]):
        errs.append(("Perception", "low_conf"))
    seg_iou = p.get("seg_iou", None)
    if seg_iou is not None and float(seg_iou) < float(th["perception"]["seg_iou_tau"]):
        errs.append(("Perception", "seg_iou_below_tau"))

    # Geometry
    g = log["geometry"]
    if not bool(g.get("pnp_success", False)):
        errs.append(("Geometry", "align_fail"))
    else:
        pr = g.get("pnp_rmse", None)
        if pr is None or float(pr) > float(th["geometry"]["pnp_rmse_tau"]):
            errs.append(("Geometry", "pnp_reproj_high"))

    # Planning
    pl = log["planning"]
    if not bool(pl.get("success", False)):
        errs.append(("Planning", "no_path"))
    if int(pl.get("collisions", 0)) > int(th["planning"]["max_collisions"]):
        errs.append(("Planning", "collision_pred"))
    pc = pl.get("path_cost", None)
    if pc is not None and float(pc) > 1.5:  # heuristic threshold
        errs.append(("Planning", "excess_cost"))

    # Control
    c = log["control"]
    tr = c.get("track_rmse", None)
    if tr is None or float(tr) > float(th["control"]["track_rmse_tau"]):
        errs.append(("Control", "tracking_rmse_high"))
    if float(c.get("overshoot", 0.0)) > 0.05:
        errs.append(("Control", "overshoot_high"))
    if bool(c.get("oscillation", False)):
        errs.append(("Control", "oscillation"))

    root = _root_cause_from_scenario(str(log["meta"]["scenario"]))
    return errs, root

