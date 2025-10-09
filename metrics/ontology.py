ERROR_ONTOLOGY = {
  "Perception": ["miss_detection", "low_conf", "seg_iou_below_tau"],
  "Geometry":   ["pnp_reproj_high", "align_fail", "pose_nan"],
  "Planning":   ["no_path", "collision_pred", "excess_cost"],
  "Control":    ["tracking_rmse_high", "overshoot_high", "oscillation"]
}

