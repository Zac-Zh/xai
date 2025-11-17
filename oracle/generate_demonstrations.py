"""
Generate Expert Demonstrations from the Classical Oracle

This script runs the classical pipeline with zero or low perturbations to collect
successful demonstrations for training the opaque end-to-end policy.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from oracle.classical_oracle import ClassicalOracle, OracleResult
from tqdm import tqdm


def generate_demonstrations(
    config_path: str,
    thresholds_path: str,
    scenarios: List[str],
    perturbation_levels: List[float],
    num_seeds: int,
    output_dir: str,
    save_videos: bool = True
) -> str:
    """
    Generate expert demonstrations from successful Oracle runs.

    Args:
        config_path: Path to configuration file
        thresholds_path: Path to thresholds file
        scenarios: List of perturbation scenarios to include
        perturbation_levels: List of perturbation levels (use [0.0] for clean demos)
        num_seeds: Number of random seeds to try per scenario/level
        output_dir: Directory to save demonstrations
        save_videos: Whether to save video frames

    Returns:
        Path to the dataset JSON file
    """
    oracle = ClassicalOracle(config_path, thresholds_path)

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames") if save_videos else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    demonstrations = []
    success_count = 0
    total_count = 0

    print(f"Generating expert demonstrations...")
    print(f"Scenarios: {scenarios}")
    print(f"Perturbation levels: {perturbation_levels}")
    print(f"Seeds per config: {num_seeds}")

    for scenario in scenarios:
        for level in perturbation_levels:
            for seed in tqdm(
                range(num_seeds),
                desc=f"{scenario} @ {level:.2f}"
            ):
                total_count += 1

                try:
                    result = oracle.run_single(
                        scenario=scenario,
                        perturbation_level=level,
                        seed=seed,
                        save_artifacts=False
                    )

                    if result.success:
                        success_count += 1

                        # Save frames if requested
                        frames_paths = []
                        if save_videos and frames_dir:
                            demo_id = f"{scenario}_{level}_{seed}"
                            demo_frames_dir = os.path.join(frames_dir, demo_id)
                            os.makedirs(demo_frames_dir, exist_ok=True)

                            for frame_idx, frame in enumerate(result.rollout_frames):
                                frame_path = os.path.join(
                                    demo_frames_dir,
                                    f"frame_{frame_idx:04d}.npy"
                                )
                                np.save(frame_path, frame)
                                frames_paths.append(frame_path)

                        # Create demonstration record
                        demo_record = {
                            "demo_id": f"{scenario}_{level}_{seed}",
                            "scenario": scenario,
                            "perturbation_level": level,
                            "seed": seed,
                            "success": True,
                            "final_distance": result.final_distance,
                            "log_data": result.log_data,
                            "frames_paths": frames_paths if save_videos else [],
                            "num_frames": len(result.rollout_frames)
                        }

                        demonstrations.append(demo_record)

                except Exception as e:
                    print(f"\nError in {scenario}_{level}_{seed}: {e}")
                    continue

    # Save dataset
    dataset_path = os.path.join(output_dir, "expert_demonstrations.json")
    with open(dataset_path, "w") as f:
        json.dump({
            "metadata": {
                "total_runs": total_count,
                "successful_runs": success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "scenarios": scenarios,
                "perturbation_levels": perturbation_levels,
                "num_seeds": num_seeds
            },
            "demonstrations": demonstrations
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Expert Demonstrations Generated!")
    print(f"{'='*60}")
    print(f"Total runs: {total_count}")
    print(f"Successful runs: {success_count}")
    print(f"Success rate: {success_count / total_count * 100:.1f}%")
    print(f"Dataset saved to: {dataset_path}")
    print(f"{'='*60}")

    return dataset_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate expert demonstrations from the Classical Oracle"
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
        default=["occlusion", "lighting", "motion_blur"],
        help="Perturbation scenarios to include"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2],
        help="Perturbation levels (use 0.0 for clean demonstrations)"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of random seeds per scenario/level combination"
    )
    parser.add_argument(
        "--output-dir",
        default="results/expert_demos",
        help="Directory to save demonstrations"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip saving video frames (only save metadata)"
    )

    args = parser.parse_args()

    generate_demonstrations(
        config_path=args.cfg,
        thresholds_path=args.thresholds,
        scenarios=args.scenarios,
        perturbation_levels=args.levels,
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        save_videos=not args.no_videos
    )


if __name__ == "__main__":
    main()
