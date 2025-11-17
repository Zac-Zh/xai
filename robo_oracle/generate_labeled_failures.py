"""
Robo-Oracle: Generate Labeled Failure Dataset

This is the core methodological contribution of the Robo-Oracle system.
It runs the Opaque Policy (Diffusion Policy) and uses the Classical Oracle
to provide ground-truth, causal failure labels.

This solves the key bottleneck in SOTA failure diagnosis systems like "AHA"
by providing programmatic, causal labels instead of correlational labels.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Install for progress bars: pip install tqdm")

from oracle.classical_oracle import ClassicalOracle
from opaque.opaque_policy_interface import OpaquePolicy


class RoboOracleDataGenerator:
    """
    Robo-Oracle Data Generator

    This class orchestrates the generation of the labeled failure dataset
    by running both the Opaque Policy and the Classical Oracle.
    """

    def __init__(
        self,
        opaque_model_checkpoint: str,
        config_path: str,
        thresholds_path: str,
        output_dir: str
    ):
        """
        Initialize the Robo-Oracle Data Generator.

        Args:
            opaque_model_checkpoint: Path to trained Diffusion Policy
            config_path: Path to environment configuration
            thresholds_path: Path to thresholds for attribution
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create output subdirectories
        self.videos_dir = os.path.join(output_dir, "failure_videos")
        self.frames_dir = os.path.join(output_dir, "failure_frames")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        # Initialize policies
        print("Initializing Robo-Oracle Data Generator...")
        print(f"Loading Opaque Policy...")
        self.opaque_policy = OpaquePolicy(
            model_checkpoint=opaque_model_checkpoint,
            config_path=config_path
        )

        print(f"Loading Classical Oracle...")
        self.oracle = ClassicalOracle(config_path, thresholds_path)

        print("Initialization complete!")

    def generate_labeled_dataset(
        self,
        scenarios: List[str],
        perturbation_levels: List[float],
        num_seeds: int,
        save_videos: bool = True
    ) -> str:
        """
        Generate the labeled failure dataset.

        This implements the core Robo-Oracle algorithm:
        1. Run Opaque Policy on perturbed scenarios
        2. For each failure, run Classical Oracle on same scenario
        3. Use Oracle's causal label as ground truth

        Args:
            scenarios: List of perturbation scenarios
            perturbation_levels: List of perturbation levels to test
            num_seeds: Number of random seeds per scenario/level
            save_videos: Whether to save failure videos

        Returns:
            Path to the generated dataset JSON file
        """
        dataset_records = []
        stats = {
            "total_runs": 0,
            "opaque_failures": 0,
            "oracle_failures": 0,
            "both_failed": 0,
            "only_opaque_failed": 0,
            "failure_categories": {}
        }

        print("\n" + "="*60)
        print("Generating Robo-Oracle Labeled Failure Dataset")
        print("="*60)
        print(f"Scenarios: {scenarios}")
        print(f"Perturbation levels: {perturbation_levels}")
        print(f"Seeds per config: {num_seeds}")
        print(f"Total configurations: {len(scenarios) * len(perturbation_levels) * num_seeds}")
        print("="*60 + "\n")

        for scenario in scenarios:
            for level in perturbation_levels:
                # Create iterator
                seed_range = range(num_seeds)
                if HAS_TQDM:
                    seed_range = tqdm(
                        seed_range,
                        desc=f"{scenario} @ {level:.2f}",
                        leave=False
                    )

                for seed in seed_range:
                    stats["total_runs"] += 1

                    try:
                        # Run Opaque Policy
                        opaque_result = self.opaque_policy.run_single(
                            scenario=scenario,
                            perturbation_level=level,
                            seed=seed
                        )

                        # Check if Opaque Policy failed
                        if not opaque_result.success:
                            stats["opaque_failures"] += 1

                            # Run Classical Oracle on same scenario to get ground-truth label
                            oracle_result = self.oracle.run_single(
                                scenario=scenario,
                                perturbation_level=level,
                                seed=seed
                            )

                            if not oracle_result.success:
                                stats["oracle_failures"] += 1
                                stats["both_failed"] += 1
                            else:
                                stats["only_opaque_failed"] += 1

                            # Create failure record
                            record = self._create_failure_record(
                                scenario=scenario,
                                perturbation_level=level,
                                seed=seed,
                                opaque_result=opaque_result,
                                oracle_result=oracle_result,
                                save_video=save_videos
                            )

                            dataset_records.append(record)

                            # Update category stats
                            if oracle_result.failure_label:
                                category = oracle_result.failure_label["primary_failure_module"]
                                stats["failure_categories"][category] = \
                                    stats["failure_categories"].get(category, 0) + 1

                    except Exception as e:
                        print(f"\nError processing {scenario}_{level}_{seed}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        # Save dataset
        dataset_path = self._save_dataset(dataset_records, stats)

        # Print summary
        self._print_summary(stats, dataset_path)

        return dataset_path

    def _create_failure_record(
        self,
        scenario: str,
        perturbation_level: float,
        seed: int,
        opaque_result: Any,
        oracle_result: Any,
        save_video: bool
    ) -> Dict[str, Any]:
        """Create a failure record with opaque rollout and oracle label."""
        failure_id = f"{scenario}_{perturbation_level}_{seed}"

        # Save video frames
        video_paths = []
        if save_video and opaque_result.rollout_frames:
            failure_frames_dir = os.path.join(self.frames_dir, failure_id)
            os.makedirs(failure_frames_dir, exist_ok=True)

            for frame_idx, frame in enumerate(opaque_result.rollout_frames):
                frame_path = os.path.join(
                    failure_frames_dir,
                    f"frame_{frame_idx:04d}.npy"
                )
                np.save(frame_path, frame)
                video_paths.append(frame_path)

        # Create record
        record = {
            "failure_id": failure_id,
            "scenario": scenario,
            "perturbation_level": perturbation_level,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),

            # Opaque policy data (the "mystery")
            "opaque_policy": {
                "success": opaque_result.success,
                "final_distance": opaque_result.final_distance,
                "num_frames": len(opaque_result.rollout_frames),
                "video_paths": video_paths
            },

            # Oracle label (the "ground truth")
            "oracle_label": {
                "success": oracle_result.success,
                "final_distance": oracle_result.final_distance,
                "failure_label": oracle_result.failure_label,
                "log_data": oracle_result.log_data
            },

            # Metadata
            "metadata": {
                "oracle_also_failed": not oracle_result.success,
                "primary_failure_module": oracle_result.failure_label["primary_failure_module"]
                    if oracle_result.failure_label else "None",
                "primary_error_code": oracle_result.failure_label["primary_error_code"]
                    if oracle_result.failure_label else "None"
            }
        }

        return record

    def _save_dataset(
        self,
        records: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> str:
        """Save the dataset to JSON."""
        dataset = {
            "metadata": {
                "generator": "Robo-Oracle",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_failures": len(records),
                "statistics": stats
            },
            "failures": records
        }

        dataset_path = os.path.join(self.output_dir, "labeled_failures.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

        return dataset_path

    def _print_summary(self, stats: Dict[str, Any], dataset_path: str):
        """Print generation summary."""
        print("\n" + "="*60)
        print("Robo-Oracle Dataset Generation Complete!")
        print("="*60)
        print(f"Total runs: {stats['total_runs']}")
        print(f"Opaque policy failures: {stats['opaque_failures']}")
        print(f"  - Both policies failed: {stats['both_failed']}")
        print(f"  - Only opaque failed: {stats['only_opaque_failed']}")
        print(f"\nFailure categories (from Oracle labels):")
        for category, count in sorted(
            stats['failure_categories'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / stats['opaque_failures'] * 100) if stats['opaque_failures'] > 0 else 0
            print(f"  - {category}: {count} ({percentage:.1f}%)")
        print(f"\nDataset saved to: {dataset_path}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Robo-Oracle labeled failure dataset"
    )
    parser.add_argument(
        "--opaque-model",
        required=True,
        help="Path to trained Diffusion Policy checkpoint"
    )
    parser.add_argument(
        "--cfg",
        default="configs/robosuite_grasp.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--thresholds",
        default="configs/thresholds.yaml",
        help="Path to thresholds file"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["occlusion", "lighting", "motion_blur", "overlap", "camera_jitter"],
        help="Perturbation scenarios to test"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Perturbation levels (higher levels cause more failures)"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of random seeds per scenario/level"
    )
    parser.add_argument(
        "--output-dir",
        default="results/robo_oracle_dataset",
        help="Directory to save generated dataset"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip saving video frames"
    )

    args = parser.parse_args()

    # Create generator
    generator = RoboOracleDataGenerator(
        opaque_model_checkpoint=args.opaque_model,
        config_path=args.cfg,
        thresholds_path=args.thresholds,
        output_dir=args.output_dir
    )

    # Generate dataset
    generator.generate_labeled_dataset(
        scenarios=args.scenarios,
        perturbation_levels=args.levels,
        num_seeds=args.num_seeds,
        save_videos=not args.no_videos
    )


if __name__ == "__main__":
    main()
