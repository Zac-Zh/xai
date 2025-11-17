"""
Oracle Interface for Robo-Oracle

This module provides a clean, callable interface to the classical pipeline
that serves as the ground-truth "Oracle" for failure attribution.

The Oracle can:
1. Execute the classical pipeline on any configuration
2. Return programmatic failure labels
3. Generate expert demonstrations (successful trajectories)
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
from utils.yload import load as yload

# Import classical pipeline components
from simulators.synth_env import SynthLiftEnv
from vision.detector_stub import detect_target
from vision.segmenter_stub import segment_target
from vision.viz_overlay import overlay_detection
from geometry.pose_pnp_stub import estimate_pose
from planning.rrt_star_fallback import plan
from control.controller_pd import track
from metrics.system_metrics import success as sys_success

# Import attribution system
from attribution.oracle_attribution import attribute_failure, FailureLabel


class OracleResult:
    """Result from Oracle execution."""

    def __init__(
        self,
        success: bool,
        failure_label: Optional[FailureLabel],
        run_log: Dict[str, Any],
        trajectory_data: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.failure_label = failure_label
        self.run_log = run_log
        self.trajectory_data = trajectory_data  # For expert demonstrations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "failure_label": self.failure_label.to_dict() if self.failure_label else None,
            "run_log": self.run_log,
            "has_trajectory": self.trajectory_data is not None
        }


class Oracle:
    """
    The Oracle: A deterministic, attributable classical pipeline.

    This is the core asset for Robo-Oracle. It provides ground-truth
    causal labels for failures in opaque end-to-end policies.
    """

    def __init__(
        self,
        config_path: str = "configs/robosuite_grasp.yaml",
        thresholds_path: str = "configs/thresholds.yaml"
    ):
        """
        Initialize the Oracle.

        Args:
            config_path: Path to experiment configuration
            thresholds_path: Path to threshold configuration
        """
        self.config = yload(config_path)
        self.thresholds = yload(thresholds_path)

    def run_single(
        self,
        scenario: str,
        level: float,
        seed: int = 0,
        collect_trajectory: bool = False
    ) -> OracleResult:
        """
        Execute the Oracle on a single configuration.

        This is the key interface function - it runs the classical pipeline
        and returns both success/failure and causal attribution.

        Args:
            scenario: Perturbation scenario name (e.g., "occlusion")
            level: Perturbation level [0.0-1.0]
            seed: Random seed for reproducibility
            collect_trajectory: If True, collect full state/action/image trajectory

        Returns:
            OracleResult with success, failure label, and optional trajectory
        """
        # Prepare perturbation module
        import importlib
        perturb_mod = importlib.import_module(f"perturb.{scenario}")

        # Create context
        ctx: Dict[str, Any] = {
            "noise": {},
            "obstacles": [],
            "meta": {
                "task": self.config["task"],
                "robot": self.config["robot"],
                "camera": self.config["camera"]
            }
        }

        rng = np.random.default_rng(seed)
        ctx["rng"] = rng

        # Apply perturbation
        perturb_mod.apply(ctx, float(level))

        # Initialize environment
        env = SynthLiftEnv(
            seed=seed,
            max_steps=int(self.config["max_steps"]),
            camera=self.config["camera"]
        )
        env.reset()
        state = env.get_state()
        ctx["gt"] = {
            "target_x": state["target_x"],
            "target_y": state["target_y"]
        }

        # Trajectory collection
        trajectory = {
            "states": [],
            "actions": [],
            "images": []
        } if collect_trajectory else None

        # === Execute Classical Pipeline ===

        # 1. Vision (Perception)
        rgb = env.render_rgb()
        if collect_trajectory:
            trajectory["states"].append(state.copy())
            trajectory["images"].append(rgb.copy())

        det = detect_target(rgb, ctx)
        seg = segment_target(rgb, ctx)
        ctx.setdefault("perception", {})["seg_iou"] = seg.get("seg_iou")

        # 2. Geometry (Pose Estimation)
        geom = estimate_pose(det, ctx)

        # 3. Planning
        start = np.array([state["agent_x"], state["agent_y"]], dtype=float)
        goal = np.array([state["target_x"], state["target_y"]], dtype=float)
        plan_out = plan(start, goal, ctx)

        if collect_trajectory and plan_out.get("path"):
            # Store planned actions
            trajectory["actions"] = plan_out["path"]

        # 4. Control
        ctrl = track(plan_out.get("path", []), ctx)

        # 5. System-level evaluation
        if plan_out.get("path"):
            final_pos = np.array(plan_out["path"][-1], dtype=float)
            final_dist = float(np.linalg.norm(final_pos - goal) + (ctrl["track_rmse"] or 0.0))
        else:
            final_dist = float(np.linalg.norm(start - goal))

        sys_ok = sys_success(final_dist, float(self.thresholds["system"]["success_dist_tau"]))

        # === Build Run Log ===
        run_log: Dict[str, Any] = {
            "run_id": f"oracle_{scenario}_{level}_{seed}",
            "meta": {
                "task": self.config["task"],
                "robot": self.config["robot"],
                "seed": int(seed),
                "scenario": str(scenario),
                "level": float(level),
                "steps": int(self.config["max_steps"])
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
            }
        }

        # === Perform Attribution ===
        failure_label = attribute_failure(run_log, self.thresholds)

        # Auto-generate natural language description
        if failure_label.natural_language_description is None:
            failure_label.natural_language_description = failure_label.to_natural_language()

        return OracleResult(
            success=sys_ok,
            failure_label=failure_label,
            run_log=run_log,
            trajectory_data=trajectory
        )

    def generate_expert_demonstrations(
        self,
        num_demos: int = 1000,
        output_dir: str = "data/expert_demonstrations",
        max_perturbation: float = 0.1
    ) -> str:
        """
        Generate expert demonstration dataset for training opaque policies.

        This creates a dataset of successful trajectories from the Oracle
        that can be used to train a Diffusion Policy via imitation learning.

        Args:
            num_demos: Number of demonstrations to generate
            output_dir: Where to save the dataset
            max_perturbation: Maximum perturbation level (keep low for success)

        Returns:
            Path to the saved dataset
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        demonstrations = []
        scenarios = ["occlusion", "lighting", "motion_blur"]

        print(f"Generating {num_demos} expert demonstrations...")

        successful_count = 0
        attempts = 0
        max_attempts = num_demos * 3  # Try up to 3x to get successful demos

        while successful_count < num_demos and attempts < max_attempts:
            attempts += 1

            # Sample random configuration
            scenario = np.random.choice(scenarios)
            level = np.random.uniform(0.0, max_perturbation)
            seed = np.random.randint(0, 100000)

            # Run Oracle
            result = self.run_single(
                scenario=scenario,
                level=level,
                seed=seed,
                collect_trajectory=True
            )

            # Only keep successful demonstrations
            if result.success and result.trajectory_data:
                demo_entry = {
                    "demo_id": successful_count,
                    "scenario": scenario,
                    "level": level,
                    "seed": seed,
                    "trajectory": result.trajectory_data,
                    "run_log": result.run_log
                }
                demonstrations.append(demo_entry)
                successful_count += 1

                if successful_count % 100 == 0:
                    print(f"  Generated {successful_count}/{num_demos} successful demos...")

        # Save dataset
        output_path = os.path.join(output_dir, "expert_demos.jsonl")

        # Helper to convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        with open(output_path, 'w') as f:
            for demo in demonstrations:
                demo_serializable = convert_numpy(demo)
                f.write(json.dumps(demo_serializable) + '\n')

        print(f"✓ Saved {successful_count} expert demonstrations to {output_path}")
        print(f"  Success rate: {successful_count/attempts*100:.1f}%")

        return output_path

    def batch_execute(
        self,
        scenarios: List[str],
        levels: List[float],
        runs_per_condition: int = 10
    ) -> List[OracleResult]:
        """
        Execute Oracle across multiple scenarios and levels.

        Args:
            scenarios: List of scenario names
            levels: List of perturbation levels
            runs_per_condition: Number of runs per (scenario, level) pair

        Returns:
            List of OracleResults
        """
        results = []

        for scenario in scenarios:
            for level in levels:
                for run_idx in range(runs_per_condition):
                    seed = run_idx
                    result = self.run_single(scenario, level, seed)
                    results.append(result)

        return results


# Convenience functions for backward compatibility and ease of use

def run_classical_oracle(
    scenario: str,
    level: float,
    seed: int = 0,
    config_path: str = "configs/robosuite_grasp.yaml",
    thresholds_path: str = "configs/thresholds.yaml"
) -> Tuple[bool, Optional[FailureLabel], Dict[str, Any]]:
    """
    Simplified interface: Run Oracle and return (success, failure_label, run_log).

    This is the primary interface mentioned in the Robo-Oracle roadmap.

    Args:
        scenario: Perturbation scenario
        level: Perturbation level
        seed: Random seed
        config_path: Path to config
        thresholds_path: Path to thresholds

    Returns:
        Tuple of (success, failure_label, run_log)
    """
    oracle = Oracle(config_path, thresholds_path)
    result = oracle.run_single(scenario, level, seed)

    return (
        result.success,
        result.failure_label,
        result.run_log
    )
