from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _assert_type(name: str, val: Any, typ) -> None:
    assert isinstance(val, typ), f"{name} should be {typ}, got {type(val)}"


def _validate_log(log: Dict[str, Any]) -> None:
    # Top-level keys
    for k in [
        "run_id",
        "meta",
        "perception",
        "geometry",
        "planning",
        "control",
        "system",
        "artifacts",
    ]:
        assert k in log, f"Missing key: {k}"
    _assert_type("run_id", log["run_id"], str)

    meta = log["meta"]
    for k, t in {
        "task": str,
        "robot": str,
        "seed": int,
        "scenario": str,
        "level": (int, float),
        "steps": int,
        "utc_time": str,
    }.items():
        assert k in meta, f"meta.{k} missing"
        _assert_type(f"meta.{k}", meta[k], t)

    p = log["perception"]
    for k, t in {
        "avg_conf": (int, float),
        "detected": bool,
        "seg_iou": (type(None), int, float),
        "bbox": list,
    }.items():
        assert k in p, f"perception.{k} missing"
        _assert_type(f"perception.{k}", p[k], t)
    assert len(p["bbox"]) == 4
    assert all(isinstance(v, int) for v in p["bbox"])

    g = log["geometry"]
    for k, t in {
        "pnp_success": bool,
        "pnp_rmse": (type(None), int, float),
        "pose_estimate": list,
    }.items():
        assert k in g, f"geometry.{k} missing"
        _assert_type(f"geometry.{k}", g[k], t)
    assert len(g["pose_estimate"]) == 3
    assert all(isinstance(v, (int, float)) for v in g["pose_estimate"])

    pl = log["planning"]
    for k, t in {
        "success": bool,
        "path_cost": (type(None), int, float),
        "collisions": int,
        "planner": str,
    }.items():
        assert k in pl, f"planning.{k} missing"
        _assert_type(f"planning.{k}", pl[k], t)

    c = log["control"]
    for k, t in {
        "track_rmse": (type(None), int, float),
        "overshoot": (type(None), int, float),
        "oscillation": bool,
    }.items():
        assert k in c, f"control.{k} missing"
        _assert_type(f"control.{k}", c[k], t)

    s = log["system"]
    for k, t in {
        "success": bool,
        "final_dist_to_goal": (int, float),
        "stop_reason": (type(None), str),
    }.items():
        assert k in s, f"system.{k} missing"
        _assert_type(f"system.{k}", s[k], t)

    a = log["artifacts"]
    for k, t in {
        "rgb_path": str,
        "mask_path": (type(None), str),
        "path_plot": (type(None), str),
    }.items():
        assert k in a, f"artifacts.{k} missing"
        _assert_type(f"artifacts.{k}", a[k], t)


def main() -> None:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "fixtures", "sample_log_ok.json")
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    _validate_log(log)
    # Simple success
    assert log["system"]["success"] is True


if __name__ == "__main__":
    main()

