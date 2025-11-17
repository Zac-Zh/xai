"""
Oracle Attribution System for Robo-Oracle

This module provides programmatic, causal failure labels from the deterministic
classical pipeline. These labels serve as ground-truth supervision for training
diagnostic models on opaque end-to-end policies.

The attribution system analyzes failures across the perception-planning-control
pipeline and returns structured, serializable failure labels.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json


@dataclass
class FailureLabel:
    """
    Structured failure label with causal attribution.

    This is the key output of the Oracle - a programmatic, verifiable
    description of why a failure occurred.
    """
    # Primary failure classification
    failure_occurred: bool
    failure_module: Optional[str] = None  # "PERCEPTION", "GEOMETRY", "PLANNING", "CONTROL", "SYSTEM"
    failure_reason: Optional[str] = None  # Specific failure type

    # Detailed metrics that caused the failure
    threshold_violated: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None

    # Cascading failure information
    is_cascading: bool = False
    root_cause_module: Optional[str] = None

    # Natural language description (for VLM training)
    natural_language_description: Optional[str] = None

    # Additional context
    severity: Optional[str] = None  # "CRITICAL", "MODERATE", "MINOR"
    recoverable: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_natural_language(self) -> str:
        """
        Convert programmatic label to natural language description.

        This is critical for VLM training in Module 4.
        """
        if not self.failure_occurred:
            return "The task completed successfully with no failures."

        # Build natural language description
        base = f"The task failed due to a {self.severity or 'significant'} error in the {self.failure_module} module."

        # Add specific reason
        if self.failure_reason:
            reason_descriptions = {
                # Perception failures
                "DETECTION_FAILURE": "The vision system failed to detect the target object in the scene.",
                "LOW_CONFIDENCE": "The object detector had insufficient confidence in its detection.",
                "SEGMENTATION_FAILURE": "The segmentation mask quality was too poor (low IoU).",

                # Geometry failures
                "PNP_FAILURE": "The PnP solver failed to compute a valid pose estimate.",
                "PNP_RMSE_VIOLATION": "The pose estimation had excessive error (high RMSE).",
                "POSE_ESTIMATION_ERROR": "The 3D pose estimation was inaccurate.",

                # Planning failures
                "PLANNING_FAILURE": "The path planner failed to find a collision-free path.",
                "EXCESSIVE_COLLISIONS": "The planned path had too many collisions with obstacles.",
                "PATH_QUALITY_POOR": "The planned path quality was insufficient (high cost).",

                # Control failures
                "TRACKING_ERROR": "The controller failed to accurately follow the planned path.",
                "OSCILLATION": "The controller exhibited unstable oscillatory behavior.",
                "OVERSHOOT": "The controller significantly overshot the target position.",

                # System failures
                "GOAL_NOT_REACHED": "The system failed to reach the goal position within tolerance.",
                "TIMEOUT": "The task exceeded the maximum time limit.",
            }

            reason_desc = reason_descriptions.get(
                self.failure_reason,
                f"The failure was due to: {self.failure_reason}."
            )
            base += f" Specifically, {reason_desc}"

        # Add metric details
        if self.threshold_violated and self.threshold_value is not None and self.actual_value is not None:
            base += f" The {self.threshold_violated} threshold was {self.threshold_value}, but the actual value was {self.actual_value:.3f}."

        # Add cascading failure info
        if self.is_cascading and self.root_cause_module:
            base += f" This failure cascaded from an earlier failure in the {self.root_cause_module} module."

        # Add recoverability
        if self.recoverable is not None:
            if self.recoverable:
                base += " This type of failure may be recoverable with an appropriate fallback strategy."
            else:
                base += " This failure is critical and cannot be recovered from."

        return base


def attribute_failure(
    run_log: Dict[str, Any],
    thresholds: Dict[str, Any]
) -> FailureLabel:
    """
    Perform causal failure attribution on a single run log.

    This is the core Oracle function that analyzes a classical pipeline
    execution and returns a programmatic failure label.

    Args:
        run_log: Complete execution log from run_suite.py
        thresholds: Threshold configuration from configs/thresholds.yaml

    Returns:
        FailureLabel with causal attribution
    """
    # Start with system success
    system_success = run_log["system"]["success"]

    if system_success:
        return FailureLabel(
            failure_occurred=False,
            natural_language_description="The task completed successfully with no failures."
        )

    # System failed - perform attribution
    # Attribution order: Perception -> Geometry -> Planning -> Control -> System
    # This follows the pipeline dependency structure

    # Check Perception
    perception = run_log["perception"]
    perc_thresh = thresholds.get("perception", {})

    if not perception["detected"]:
        return FailureLabel(
            failure_occurred=True,
            failure_module="PERCEPTION",
            failure_reason="DETECTION_FAILURE",
            severity="CRITICAL",
            recoverable=False,
            natural_language_description="The vision system failed to detect the target object."
        )

    if perception["avg_conf"] < perc_thresh.get("min_confidence", 0.0):
        return FailureLabel(
            failure_occurred=True,
            failure_module="PERCEPTION",
            failure_reason="LOW_CONFIDENCE",
            threshold_violated="min_confidence",
            threshold_value=perc_thresh.get("min_confidence"),
            actual_value=perception["avg_conf"],
            severity="CRITICAL",
            recoverable=False
        )

    if perception["seg_iou"] is not None and perception["seg_iou"] < perc_thresh.get("min_seg_iou", 0.0):
        return FailureLabel(
            failure_occurred=True,
            failure_module="PERCEPTION",
            failure_reason="SEGMENTATION_FAILURE",
            threshold_violated="min_seg_iou",
            threshold_value=perc_thresh.get("min_seg_iou"),
            actual_value=perception["seg_iou"],
            severity="MODERATE",
            recoverable=True
        )

    # Check Geometry
    geometry = run_log["geometry"]
    geom_thresh = thresholds.get("geometry", {})

    if not geometry["pnp_success"]:
        # Check if this cascaded from perception
        is_cascading = perception["avg_conf"] < 0.7  # Heuristic

        return FailureLabel(
            failure_occurred=True,
            failure_module="GEOMETRY",
            failure_reason="PNP_FAILURE",
            severity="CRITICAL",
            recoverable=False,
            is_cascading=is_cascading,
            root_cause_module="PERCEPTION" if is_cascading else None
        )

    if geometry["pnp_rmse"] is not None and geometry["pnp_rmse"] > geom_thresh.get("max_pnp_rmse", float('inf')):
        return FailureLabel(
            failure_occurred=True,
            failure_module="GEOMETRY",
            failure_reason="PNP_RMSE_VIOLATION",
            threshold_violated="max_pnp_rmse",
            threshold_value=geom_thresh.get("max_pnp_rmse"),
            actual_value=geometry["pnp_rmse"],
            severity="MODERATE",
            recoverable=True
        )

    # Check Planning
    planning = run_log["planning"]
    plan_thresh = thresholds.get("planning", {})

    if not planning["success"]:
        # Check if this cascaded from geometry
        is_cascading = geometry.get("pnp_rmse", 0) > 0.5  # Heuristic

        return FailureLabel(
            failure_occurred=True,
            failure_module="PLANNING",
            failure_reason="PLANNING_FAILURE",
            severity="CRITICAL",
            recoverable=True,  # RRT* fallback may help
            is_cascading=is_cascading,
            root_cause_module="GEOMETRY" if is_cascading else None
        )

    if planning["collisions"] > plan_thresh.get("max_collisions", float('inf')):
        return FailureLabel(
            failure_occurred=True,
            failure_module="PLANNING",
            failure_reason="EXCESSIVE_COLLISIONS",
            threshold_violated="max_collisions",
            threshold_value=plan_thresh.get("max_collisions"),
            actual_value=planning["collisions"],
            severity="MODERATE",
            recoverable=True
        )

    # Check Control
    control = run_log["control"]
    ctrl_thresh = thresholds.get("control", {})

    if control["oscillation"]:
        return FailureLabel(
            failure_occurred=True,
            failure_module="CONTROL",
            failure_reason="OSCILLATION",
            severity="MODERATE",
            recoverable=True
        )

    if control["track_rmse"] is not None and control["track_rmse"] > ctrl_thresh.get("max_track_rmse", float('inf')):
        return FailureLabel(
            failure_occurred=True,
            failure_module="CONTROL",
            failure_reason="TRACKING_ERROR",
            threshold_violated="max_track_rmse",
            threshold_value=ctrl_thresh.get("max_track_rmse"),
            actual_value=control["track_rmse"],
            severity="MODERATE",
            recoverable=True
        )

    # System-level failure (goal not reached)
    system = run_log["system"]
    sys_thresh = thresholds.get("system", {})

    return FailureLabel(
        failure_occurred=True,
        failure_module="SYSTEM",
        failure_reason="GOAL_NOT_REACHED",
        threshold_violated="success_dist_tau",
        threshold_value=sys_thresh.get("success_dist_tau"),
        actual_value=system["final_dist_to_goal"],
        severity="MODERATE",
        recoverable=False
    )


def analyze_failure_distribution(
    logs: List[Dict[str, Any]],
    thresholds: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze distribution of failures across multiple runs.

    This is useful for understanding which modules are most vulnerable
    and for creating balanced training datasets for the diagnostic model.

    Args:
        logs: List of run logs
        thresholds: Threshold configuration

    Returns:
        Statistical summary of failure distribution
    """
    from collections import Counter

    failure_modules = []
    failure_reasons = []
    cascading_failures = 0
    total_failures = 0

    for log in logs:
        label = attribute_failure(log, thresholds)
        if label.failure_occurred:
            total_failures += 1
            failure_modules.append(label.failure_module)
            failure_reasons.append(label.failure_reason)
            if label.is_cascading:
                cascading_failures += 1

    return {
        "total_runs": len(logs),
        "total_failures": total_failures,
        "failure_rate": total_failures / len(logs) if logs else 0,
        "module_distribution": dict(Counter(failure_modules)),
        "reason_distribution": dict(Counter(failure_reasons)),
        "cascading_failure_rate": cascading_failures / total_failures if total_failures > 0 else 0
    }
