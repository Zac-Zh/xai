from __future__ import annotations

from typing import Dict, List, Tuple

from attribution.rule_based import attribute_failure


TH = {
    "perception": {"conf_tau": 0.35, "seg_iou_tau": 0.5},
    "geometry": {"pnp_rmse_tau": 3.0},
    "planning": {"max_collisions": 0},
    "control": {"track_rmse_tau": 0.02},
    "system": {"success_dist_tau": 0.03},
}


def _base_log() -> Dict:
    return {
        "meta": {"scenario": "occlusion"},
        "perception": {"avg_conf": 0.8, "detected": True, "seg_iou": 0.7},
        "geometry": {"pnp_success": True, "pnp_rmse": 1.0},
        "planning": {"success": True, "path_cost": 0.5, "collisions": 0},
        "control": {"track_rmse": 0.01, "overshoot": 0.01, "oscillation": False},
    }


def main() -> None:
    # Perception failure: miss detection
    log = _base_log()
    log["perception"]["detected"] = False
    errs, root = attribute_failure(log, TH)
    assert ("Perception", "miss_detection") in errs
    assert root == "Occlusion"

    # Geometry failure: high RMSE
    log = _base_log()
    log["geometry"]["pnp_rmse"] = 10.0
    errs, _ = attribute_failure(log, TH)
    assert ("Geometry", "pnp_reproj_high") in errs

    # Planning failure: no path
    log = _base_log()
    log["planning"]["success"] = False
    errs, _ = attribute_failure(log, TH)
    assert ("Planning", "no_path") in errs

    # Control failure: tracking rmse high
    log = _base_log()
    log["control"]["track_rmse"] = 0.2
    errs, _ = attribute_failure(log, TH)
    assert ("Control", "tracking_rmse_high") in errs

    # Another scenario mapping
    log = _base_log()
    log["meta"]["scenario"] = "lighting"
    errs, root = attribute_failure(log, TH)
    assert root == "Lighting"


if __name__ == "__main__":
    main()

