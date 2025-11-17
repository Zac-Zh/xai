"""
Classical Oracle Interface - Deterministic Pipeline with Causal Failure Attribution

This module provides a clean interface to the classical robotics pipeline that can be
called programmatically to generate ground-truth failure labels.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add repo root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.yload import load as yload
from simulators.synth_env import SynthLiftEnv
from vision.detector_stub import detect_target
from vision.segmenter_stub import segment_target
from geometry.pose_pnp_stub import estimate_pose
from planning.rrt_star_fallback import plan
from control.controller_pd import track
from metrics.system_metrics import success as sys_success
from attribution.rule_based import attribute_failure


@dataclass
class OracleResult:
    """Result from running the classical oracle pipeline."""

    success: bool
    final_distance: float
    failure_label: Optional[Dict[str, Any]]
    log_data: Dict[str, Any]
    rollout_frames: List[np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_distance": self.final_distance,
            "failure_label": self.failure_label,
            "log_data": self.log_data,
            "num_frames": len(self.rollout_frames)
        }


class ClassicalOracle:
    """
    The Classical Oracle - A deterministic, white-box robotics pipeline.

    This class provides the ground-truth "supervisor" for the Robo-Oracle system.
    It runs the full classical pipeline (Vision → Geometry → Planning → Control)
    and provides programmatic, causal failure attribution.
    """

    def __init__(self, config_path: str, thresholds_path: str):
        """
        Initialize the Classical Oracle.

        Args:
            config_path: Path to the configuration YAML file
            thresholds_path: Path to the thresholds YAML file
        """
        self.cfg = yload(config_path)
        self.thresholds = yload(thresholds_path)

    def run_single(
        self,
        scenario: str,
        perturbation_level: float,
        seed: int,
        save_artifacts: bool = False,
        artifact_dir: Optional[str] = None
    ) -> OracleResult:
        """
        Run the classical pipeline once and return the result with failure attribution.

        Args:
            scenario: Perturbation scenario (e.g., "occlusion", "lighting")
            perturbation_level: Perturbation strength (0.0 to 1.0)
            seed: Random seed for reproducibility
            save_artifacts: Whether to save visualization artifacts
            artifact_dir: Directory to save artifacts (required if save_artifacts=True)

        Returns:
            OracleResult containing success status, failure labels, and rollout data
        """
        # Import perturbation module dynamically
        import importlib
        perturb_mod = importlib.import_module(f"perturb.{scenario}")

        # Initialize context
        ctx: Dict[str, Any] = {
            "noise": {},
            "obstacles": [],
            "meta": {
                "task": self.cfg["task"],
                "robot": self.cfg["robot"],
                "camera": self.cfg["camera"]
            }
        }
        rng = np.random.default_rng(seed)
        ctx["rng"] = rng

        # Apply perturbation
        perturb_mod.apply(ctx, float(perturbation_level))

        # Initialize environment
        env = SynthLiftEnv(
            seed=seed,
            max_steps=int(self.cfg["max_steps"]),
            camera=self.cfg["camera"]
        )
        env.reset()
        state = env.get_state()
        ctx["gt"] = {
            "target_x": state["target_x"],
            "target_y": state["target_y"]
        }

        # Collect rollout frames
        rollout_frames = []

        # Run Vision module
        rgb = env.render_rgb()
        rollout_frames.append(rgb.copy())
        det = detect_target(rgb, ctx)
        seg = segment_target(rgb, ctx)
        ctx.setdefault("perception", {})["seg_iou"] = seg.get("seg_iou")

        # Run Geometry module
        geom = estimate_pose(det, ctx)

        # Run Planning module
        start = np.array([state["agent_x"], state["agent_y"]], dtype=float)
        goal = np.array([state["target_x"], state["target_y"]], dtype=float)
        plan_out = plan(start, goal, ctx)

        # Run Control module
        ctrl = track(plan_out.get("path", []), ctx)

        # Calculate final distance
        if plan_out.get("path"):
            final_pos = np.array(plan_out["path"][-1], dtype=float)
            final_dist = float(
                np.linalg.norm(final_pos - goal) + (ctrl["track_rmse"] or 0.0)
            )
        else:
            final_dist = float(np.linalg.norm(start - goal))

        # Determine system success
        sys_ok = sys_success(
            final_dist,
            float(self.thresholds["system"]["success_dist_tau"])
        )

        # Build log object
        log_data = {
            "meta": {
                "task": self.cfg["task"],
                "robot": self.cfg["robot"],
                "seed": int(seed),
                "scenario": str(scenario),
                "level": float(perturbation_level),
                "steps": int(self.cfg["max_steps"]),
            },
            "perception": {
                "avg_conf": float(det["avg_conf"]),
                "detected": bool(det["detected"]),
                "seg_iou": None if seg.get("seg_iou") is None else float(seg["seg_iou"]),
                "bbox": [int(x) for x in det["bbox"]],
            },
            "geometry": {
                "pnp_success": bool(geom["pnp_success"]),
                "pnp_rmse": None if geom.get("pnp_rmse") is None else float(geom["pnp_rmse"]),
                "pose_estimate": [float(v) for v in geom["pose_estimate"]],
            },
            "planning": {
                "success": bool(plan_out["success"]),
                "path_cost": None if plan_out.get("path_cost") is None else float(plan_out["path_cost"]),
                "collisions": int(plan_out["collisions"]),
                "planner": "RRTstar",
            },
            "control": {
                "track_rmse": None if ctrl.get("track_rmse") is None else float(ctrl["track_rmse"]),
                "overshoot": None if ctrl.get("overshoot") is None else float(ctrl["overshoot"]),
                "oscillation": bool(ctrl.get("oscillation", False)),
            },
            "system": {
                "success": bool(sys_ok),
                "final_dist_to_goal": float(final_dist),
                "stop_reason": None,
            },
        }

        # Generate failure label if system failed
        failure_label = None
        if not sys_ok:
            errors, root_cause = attribute_failure(log_data, self.thresholds)
            failure_label = {
                "failure_detected": True,
                "errors": [
                    {"module": mod, "error_code": err}
                    for mod, err in errors
                ],
                "root_cause": root_cause,
                "primary_failure_module": errors[0][0] if errors else "Unknown",
                "primary_error_code": errors[0][1] if errors else "unknown",
            }

        # Save artifacts if requested
        if save_artifacts and artifact_dir:
            self._save_artifacts(
                rgb, seg["mask"], plan_out, start, goal,
                scenario, perturbation_level, seed, artifact_dir
            )

        return OracleResult(
            success=sys_ok,
            final_distance=final_dist,
            failure_label=failure_label,
            log_data=log_data,
            rollout_frames=rollout_frames
        )

    def _save_artifacts(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        plan_out: Dict,
        start: np.ndarray,
        goal: np.ndarray,
        scenario: str,
        level: float,
        seed: int,
        artifact_dir: str
    ) -> None:
        """Save visualization artifacts."""
        os.makedirs(artifact_dir, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            run_tag = f"{scenario}_{level}_seed{seed}"

            # Save RGB
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(rgb)
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(artifact_dir, f"{run_tag}_rgb.png"), dpi=150)
            plt.close(fig)

            # Save mask if available
            if mask is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(mask, cmap='gray')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(artifact_dir, f"{run_tag}_mask.png"), dpi=150)
                plt.close(fig)

            # Save path plot
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.plot([start[0]], [start[1]], marker='o', label='start')
            ax.plot([goal[0]], [goal[1]], marker='x', label='goal')
            if plan_out.get("path"):
                pts = np.asarray(plan_out["path"], dtype=float)
                ax.plot(pts[:, 0], pts[:, 1], '-')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(artifact_dir, f"{run_tag}_path.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not save artifacts: {e}")


def run_classical_oracle(
    config_path: str,
    thresholds_path: str,
    scenario: str,
    perturbation_level: float,
    seed: int
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Convenience function to run the classical oracle and get results.

    Args:
        config_path: Path to configuration file
        thresholds_path: Path to thresholds file
        scenario: Perturbation scenario
        perturbation_level: Perturbation strength (0.0 to 1.0)
        seed: Random seed

    Returns:
        Tuple of (success: bool, failure_label: Optional[Dict])
    """
    oracle = ClassicalOracle(config_path, thresholds_path)
    result = oracle.run_single(scenario, perturbation_level, seed)
    return result.success, result.failure_label
