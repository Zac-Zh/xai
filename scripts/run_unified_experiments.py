"""
Unified experiment orchestrator for 2D (synthetic) and 3D (PyBullet) with real implementations.
Supports multiple tasks: Lift, PickPlace, Push, Stack
Supports all perturbations with both stub and real implementations.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Import environments
from simulators.synth_env import SynthEnv
try:
    from simulators.pybullet_env import PyBulletEnv
    PYBULLET_AVAILABLE = True
except:
    PYBULLET_AVAILABLE = False
    print("Warning: PyBullet not available. 3D experiments will be skipped.")

# Import real implementations
try:
    from vision.detector_real import YOLODetector
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available, using stubs")

try:
    from vision.segmenter_real import MaskRCNNSegmenter
    MASKRCNN_AVAILABLE = True
except:
    MASKRCNN_AVAILABLE = False
    print("Warning: Mask R-CNN not available, using stubs")

try:
    from geometry.pose_pnp_real import PnPPoseEstimator
    PNP_AVAILABLE = True
except:
    PNP_AVAILABLE = False
    print("Warning: Real PnP not available, using stubs")

try:
    from planning.rrt_star_real import RRTStarPlanner
    RRTSTAR_AVAILABLE = True
except:
    RRTSTAR_AVAILABLE = False
    print("Warning: Real RRT* not available, using stubs")

try:
    from control.controller_real import TrajectoryTracker
    CONTROLLER_AVAILABLE = True
except:
    CONTROLLER_AVAILABLE = False
    print("Warning: Real controller not available, using stubs")

# Import stubs as fallbacks
from vision import detector_stub, segmenter_stub
from geometry import pose_pnp_stub
from planning import rrt_star_fallback
from control import controller_pd

# Import perturbations
from perturb import occlusion, lighting, motion_blur, camera_jitter, overlap

# Import utilities
from utils.yload import load as yload


class UnifiedExperimentRunner:
    """Unified experiment runner for all scenarios, tasks, and implementations."""

    def __init__(
        self,
        mode: str = "2d",  # "2d" or "3d"
        use_real_models: bool = True,
        task: str = "Lift",
        seed: int = 0,
    ):
        """
        Initialize unified runner.

        Args:
            mode: "2d" (synthetic) or "3d" (PyBullet)
            use_real_models: Use real DL models vs stubs
            task: Task name (Lift, PickPlace, Push, Stack)
            seed: Random seed
        """
        self.mode = mode
        self.use_real_models = use_real_models
        self.task = task
        self.seed = seed

        # Initialize environment
        if mode == "3d" and PYBULLET_AVAILABLE:
            self.env = PyBulletEnv(task=task, gui=False, seed=seed)
        else:
            self.env = SynthEnv(seed=seed)

        # Initialize models
        if use_real_models:
            self.detector = YOLODetector() if YOLO_AVAILABLE else None
            self.segmenter = MaskRCNNSegmenter() if MASKRCNN_AVAILABLE else None
            self.pose_estimator = PnPPoseEstimator() if PNP_AVAILABLE else None
            self.planner = RRTStarPlanner() if RRTSTAR_AVAILABLE else None
            self.controller = TrajectoryTracker() if CONTROLLER_AVAILABLE else None
        else:
            self.detector = None
            self.segmenter = None
            self.pose_estimator = None
            self.planner = None
            self.controller = None

    def run_experiment(
        self,
        scenario: str,
        level: float,
        thresholds: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run single experiment.

        Args:
            scenario: Perturbation scenario name
            level: Perturbation level
            thresholds: Success thresholds

        Returns:
            Experiment results dictionary
        """
        # Reset environment
        self.env.reset()

        # Get camera image
        img = self.env.get_camera_image()

        # Apply perturbation
        img = self._apply_perturbation(img, scenario, level)

        # Get depth if available
        if hasattr(self.env, 'get_depth_image'):
            depth_img = self.env.get_depth_image()
        else:
            depth_img = None

        # PERCEPTION: Detection
        if self.use_real_models and self.detector is not None:
            detection = self.detector.detect(img)
        else:
            detection = detector_stub.detect(img, level)

        # PERCEPTION: Segmentation
        bbox = detection.get("bbox", [0, 0, 0, 0])
        if self.use_real_models and self.segmenter is not None:
            segmentation = self.segmenter.segment(img, bbox)
        else:
            segmentation = segmenter_stub.segment(img, bbox, level)

        # GEOMETRY: Pose estimation
        mask = segmentation.get("mask", np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8))
        if self.use_real_models and self.pose_estimator is not None:
            camera_intrinsics = self.env.get_camera_intrinsics() if hasattr(self.env, 'get_camera_intrinsics') else None
            pose = self.pose_estimator.estimate_pose_from_mask(mask, depth_img)
        else:
            pose = pose_pnp_stub.estimate_pose(level)

        # PLANNING: Path planning
        if hasattr(self.env, 'get_target_position'):
            target_pos = self.env.get_target_position()
            goal_pos = self.env.goal_position if hasattr(self.env, 'goal_position') else target_pos + np.array([0, 0, 0.3])
        else:
            target_pos = np.array(pose.get("pose_estimate", [0.5, 0.5, 0.05]))
            goal_pos = target_pos + np.array([0, 0, 0.3])

        # Get obstacles
        obstacles = self._get_obstacles(scenario, level)

        if self.use_real_models and self.planner is not None:
            planning_result = self.planner.plan(target_pos, goal_pos, obstacles)
        else:
            planning_result = rrt_star_fallback.plan_path(target_pos, goal_pos, obstacles, level)

        # CONTROL: Trajectory tracking
        path = planning_result.get("path", [target_pos, goal_pos])
        if self.use_real_models and self.controller is not None:
            control_result = self.controller.track_trajectory(path, target_pos)
        else:
            control_result = controller_pd.track_path(path, level)

        # SYSTEM: Check success
        if hasattr(self.env, 'check_success'):
            system_success = self.env.check_success()
        else:
            # Fallback success check
            final_pos = path[-1] if len(path) > 0 else target_pos
            final_dist = np.linalg.norm(final_pos - goal_pos)
            system_success = final_dist < thresholds.get("system", {}).get("max_final_distance", 0.05)

        # Compile results
        run_id = f"{datetime.utcnow().isoformat()}_{self.seed}"

        result = {
            "run_id": run_id,
            "meta": {
                "mode": self.mode,
                "task": self.task,
                "robot": "Panda" if self.mode == "3d" else "Synthetic",
                "seed": self.seed,
                "scenario": scenario,
                "level": level,
                "use_real_models": self.use_real_models,
                "utc_time": datetime.utcnow().isoformat(),
            },
            "perception": {
                "avg_conf": detection.get("avg_conf", 0.0),
                "detected": detection.get("detected", False),
                "seg_iou": segmentation.get("seg_iou", 0.0),
                "bbox": bbox,
            },
            "geometry": {
                "pnp_success": pose.get("pnp_success", False),
                "pnp_rmse": pose.get("pnp_rmse", np.inf),
                "pose_estimate": pose.get("pose_estimate", [0.0, 0.0, 0.0]),
            },
            "planning": {
                "success": planning_result.get("success", False),
                "path_cost": planning_result.get("path_cost", 0.0),
                "collisions": planning_result.get("collisions", 0),
                "planner": planning_result.get("planner", "Unknown"),
            },
            "control": {
                "track_rmse": control_result.get("track_rmse", 0.0),
                "overshoot": control_result.get("overshoot", 0.0),
                "oscillation": control_result.get("oscillation", False),
            },
            "system": {
                "success": system_success,
                "final_dist_to_goal": np.linalg.norm(path[-1] - goal_pos) if len(path) > 0 else np.inf,
                "stop_reason": None,
            },
        }

        return result

    def _apply_perturbation(self, img: np.ndarray, scenario: str, level: float) -> np.ndarray:
        """Apply perturbation to image."""
        if scenario == "occlusion":
            return occlusion.apply_occlusion(img, level)
        elif scenario == "lighting":
            return lighting.apply_lighting(img, level)
        elif scenario == "motion_blur":
            return motion_blur.apply_motion_blur(img, level)
        elif scenario == "camera_jitter" and hasattr(self.env, 'apply_camera_jitter'):
            self.env.apply_camera_jitter(level)
            return self.env.get_camera_image()
        elif scenario == "overlap":
            return overlap.apply_overlap(img, level)
        else:
            return img

    def _get_obstacles(self, scenario: str, level: float) -> List[Dict]:
        """Get obstacles for planning."""
        obstacles = []

        if scenario in ["occlusion", "overlap"]:
            # Add obstacles based on perturbation
            num_obstacles = int(level * 5)
            for _ in range(num_obstacles):
                obstacles.append({
                    "center": np.random.uniform([0.3, -0.2, 0], [0.7, 0.2, 0.3]),
                    "radius": 0.05,
                })

        return obstacles

    def close(self) -> None:
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both",
                        help="Simulation mode")
    parser.add_argument("--use_real_models", action="store_true",
                        help="Use real DL models (requires GPU)")
    parser.add_argument("--task", choices=["Lift", "PickPlace", "Push", "Stack", "all"], default="all",
                        help="Task to run")
    parser.add_argument("--scenario", required=True, help="Perturbation scenario")
    parser.add_argument("--levels", nargs="+", type=float, required=True,
                        help="Perturbation levels")
    parser.add_argument("--runs", type=int, default=3, help="Runs per level")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml",
                        help="Thresholds config")
    parser.add_argument("--output", required=True, help="Output JSONL file")

    args = parser.parse_args()

    # Load thresholds
    thresholds = yload(args.thresholds)

    # Determine modes and tasks
    modes = ["2d", "3d"] if args.mode == "both" else [args.mode]
    if "3d" in modes and not PYBULLET_AVAILABLE:
        modes.remove("3d")
        print("Skipping 3D mode (PyBullet not available)")

    tasks = ["Lift", "PickPlace", "Push", "Stack"] if args.task == "all" else [args.task]

    # Run experiments
    results = []

    for mode in modes:
        for task in tasks:
            print(f"\nRunning {mode.upper()} - {task}")

            for level in args.levels:
                print(f"  Level {level}:")

                for run_idx in range(args.runs):
                    seed = run_idx

                    # Initialize runner
                    runner = UnifiedExperimentRunner(
                        mode=mode,
                        use_real_models=args.use_real_models,
                        task=task,
                        seed=seed,
                    )

                    # Run experiment
                    result = runner.run_experiment(args.scenario, level, thresholds)
                    results.append(result)

                    # Close runner
                    runner.close()

                    print(f"    Run {run_idx+1}/{args.runs}: Success={result['system']['success']}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n✓ Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
