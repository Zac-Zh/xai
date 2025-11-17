"""
Robo-Oracle Data Generation Pipeline

This module implements Module 3 of the Robo-Oracle roadmap:
Building the labeled failure dataset by running both the Oracle (classical
pipeline) and an opaque E2E policy, then labeling opaque failures with
Oracle attributions.

This is the core methodological contribution - creating a dataset of
(opaque_policy_failure_video, causal_oracle_label) pairs that doesn't
exist anywhere else in the literature.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from utils.yload import load as yload

from oracle.oracle_interface import Oracle, OracleResult


class RoboOracleDataGenerator:
    """
    Master data generation pipeline for Robo-Oracle.

    This orchestrates both the Oracle (classical pipeline) and the opaque
    policy to create the world's first causally-labeled failure dataset
    for opaque visuomotor policies.
    """

    def __init__(
        self,
        oracle_config: str = "configs/robosuite_grasp.yaml",
        oracle_thresholds: str = "configs/thresholds.yaml",
        perturbations_config: str = "configs/perturbations.yaml",
        opaque_policy_path: Optional[str] = None
    ):
        """
        Initialize the data generator.

        Args:
            oracle_config: Oracle configuration
            oracle_thresholds: Oracle thresholds
            perturbations_config: Perturbation scenarios
            opaque_policy_path: Path to trained opaque policy (e.g., Diffusion Policy)
        """
        # Initialize Oracle
        self.oracle = Oracle(oracle_config, oracle_thresholds)

        # Load perturbation config
        self.pert_config = yload(perturbations_config)

        # Opaque policy (will be implemented in Module 2)
        self.opaque_policy_path = opaque_policy_path
        self.opaque_policy = None  # Placeholder for Module 2

        # Statistics
        self.stats = {
            "total_runs": 0,
            "opaque_failures": 0,
            "oracle_failures": 0,
            "labeled_failures": 0,
            "failure_modules": defaultdict(int)
        }

    def load_opaque_policy(self, policy_path: str):
        """
        Load a trained opaque policy.

        This will be implemented in Module 2 after the Diffusion Policy
        is trained.

        Args:
            policy_path: Path to policy checkpoint
        """
        # TODO: Implement in Module 2
        # For now, this is a placeholder
        self.opaque_policy_path = policy_path
        print(f"⚠ Opaque policy loading not yet implemented (Module 2)")
        print(f"  Policy path: {policy_path}")

    def run_opaque_policy(
        self,
        scenario: str,
        level: float,
        seed: int
    ) -> Dict[str, Any]:
        """
        Run the opaque policy on a configuration.

        This is a stub for Module 2. When implemented, it will:
        1. Load the configuration
        2. Run the Diffusion Policy
        3. Return success/failure and trajectory video

        Args:
            scenario: Perturbation scenario
            level: Perturbation level
            seed: Random seed

        Returns:
            Dictionary with 'success' and 'trajectory_data'
        """
        # TODO: Implement in Module 2
        # For now, return a placeholder that simulates failures
        # This allows testing the pipeline structure

        # Simulate: higher perturbation = more failures
        failure_prob = min(level, 0.9)
        success = np.random.rand() > failure_prob

        return {
            "success": success,
            "trajectory_data": {
                "scenario": scenario,
                "level": level,
                "seed": seed,
                "rollout_video": None,  # TODO: Actual video in Module 2
                "states": None,  # TODO: Actual states in Module 2
            },
            "placeholder": True  # Flag to indicate this is a stub
        }

    def generate_labeled_failure_dataset(
        self,
        output_dir: str = "data/robo_oracle_failures",
        num_samples: int = 1000,
        target_scenarios: Optional[List[str]] = None,
        min_level: float = 0.2,
        max_level: float = 0.8
    ) -> str:
        """
        Generate the core Robo-Oracle dataset.

        This is the implementation of Task 3.2 from the roadmap:
        For each configuration:
        1. Run opaque policy
        2. If it fails, run Oracle on same config
        3. Get Oracle's causal label
        4. Save (opaque_failure, oracle_label) pair

        Args:
            output_dir: Where to save the dataset
            num_samples: Target number of labeled failures
            target_scenarios: Scenarios to use (None = use all from config)
            min_level: Minimum perturbation level
            max_level: Maximum perturbation level

        Returns:
            Path to generated dataset
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Prepare output files
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dataset_path = os.path.join(output_dir, f"labeled_failures_{timestamp}.jsonl")
        stats_path = os.path.join(output_dir, f"generation_stats_{timestamp}.json")

        # Determine scenarios
        if target_scenarios is None:
            # Use main 5 scenarios from publication suite
            target_scenarios = [
                "occlusion",
                "module_failure",
                "data_corruption",
                "noise_injection",
                "adversarial_patches"
            ]

        print(f"{'='*70}")
        print(f"Robo-Oracle Data Generation")
        print(f"{'='*70}")
        print(f"Target samples: {num_samples}")
        print(f"Scenarios: {target_scenarios}")
        print(f"Perturbation range: [{min_level}, {max_level}]")
        print(f"Output: {dataset_path}")
        print(f"{'='*70}\n")

        labeled_failures = []
        attempts = 0
        max_attempts = num_samples * 5  # Try up to 5x to get target samples

        start_time = time.time()

        while len(labeled_failures) < num_samples and attempts < max_attempts:
            attempts += 1
            self.stats["total_runs"] += 1

            # Sample random configuration
            scenario = np.random.choice(target_scenarios)
            level = np.random.uniform(min_level, max_level)
            seed = np.random.randint(0, 100000)

            # === Step 1: Run Opaque Policy ===
            opaque_result = self.run_opaque_policy(scenario, level, seed)

            # === Step 2: Check for Failure ===
            if not opaque_result["success"]:
                self.stats["opaque_failures"] += 1

                # === Step 3: Run Oracle on Same Configuration ===
                oracle_result = self.oracle.run_single(
                    scenario=scenario,
                    level=level,
                    seed=seed,
                    collect_trajectory=False  # Don't need full trajectory for labeling
                )

                self.stats["oracle_failures"] += int(not oracle_result.success)

                # === Step 4: Create Labeled Entry ===
                # The key insight: We have an opaque failure with a causal Oracle label
                labeled_entry = {
                    "sample_id": len(labeled_failures),
                    "timestamp": datetime.now(timezone.utc).isoformat(),

                    # Configuration
                    "config": {
                        "scenario": scenario,
                        "level": level,
                        "seed": seed
                    },

                    # Opaque policy result (the "mystery")
                    "opaque_policy": {
                        "success": False,
                        "trajectory": opaque_result["trajectory_data"],
                        "is_placeholder": opaque_result.get("placeholder", False)
                    },

                    # Oracle attribution (the "answer")
                    "oracle_label": oracle_result.failure_label.to_dict(),

                    # Natural language description for VLM training
                    "natural_language": oracle_result.failure_label.natural_language_description
                }

                labeled_failures.append(labeled_entry)
                self.stats["labeled_failures"] += 1

                # Track failure distribution
                if oracle_result.failure_label.failure_module:
                    self.stats["failure_modules"][oracle_result.failure_label.failure_module] += 1

                # Progress reporting
                if len(labeled_failures) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = len(labeled_failures) / elapsed
                    eta = (num_samples - len(labeled_failures)) / rate if rate > 0 else 0

                    print(f"Progress: {len(labeled_failures)}/{num_samples} labeled failures")
                    print(f"  Attempts: {attempts} (success rate: {len(labeled_failures)/attempts*100:.1f}%)")
                    print(f"  Rate: {rate:.1f} samples/sec")
                    print(f"  ETA: {eta/60:.1f} minutes\n")

        # === Save Dataset ===
        print(f"\nSaving dataset...")
        with open(dataset_path, 'w') as f:
            for entry in labeled_failures:
                f.write(json.dumps(entry) + '\n')

        # === Save Statistics ===
        final_stats = {
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": dataset_path,
            "configuration": {
                "target_samples": num_samples,
                "actual_samples": len(labeled_failures),
                "scenarios": target_scenarios,
                "perturbation_range": [min_level, max_level]
            },
            "statistics": dict(self.stats),
            "failure_distribution": dict(self.stats["failure_modules"]),
            "generation_time_seconds": time.time() - start_time
        }

        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)

        # === Summary ===
        print(f"\n{'='*70}")
        print(f"✓ Dataset Generation Complete")
        print(f"{'='*70}")
        print(f"Labeled failures: {len(labeled_failures)}")
        print(f"Total attempts: {attempts}")
        print(f"Success rate: {len(labeled_failures)/attempts*100:.1f}%")
        print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"\nFailure distribution:")
        for module, count in sorted(self.stats["failure_modules"].items(), key=lambda x: -x[1]):
            pct = 100 * count / len(labeled_failures)
            print(f"  {module}: {count} ({pct:.1f}%)")
        print(f"\nDataset: {dataset_path}")
        print(f"Statistics: {stats_path}")
        print(f"{'='*70}\n")

        return dataset_path

    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze a generated dataset for balance and quality.

        Args:
            dataset_path: Path to dataset JSONL file

        Returns:
            Analysis dictionary
        """
        # Load dataset
        samples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Analyze
        failure_modules = defaultdict(int)
        failure_reasons = defaultdict(int)
        cascading_count = 0
        scenarios = defaultdict(int)

        for sample in samples:
            label = sample["oracle_label"]

            if label["failure_module"]:
                failure_modules[label["failure_module"]] += 1

            if label["failure_reason"]:
                failure_reasons[label["failure_reason"]] += 1

            if label.get("is_cascading"):
                cascading_count += 1

            scenarios[sample["config"]["scenario"]] += 1

        analysis = {
            "total_samples": len(samples),
            "failure_module_distribution": dict(failure_modules),
            "failure_reason_distribution": dict(failure_reasons),
            "scenario_distribution": dict(scenarios),
            "cascading_failures": cascading_count,
            "cascading_rate": cascading_count / len(samples) if samples else 0
        }

        # Check for balance
        module_counts = list(failure_modules.values())
        if module_counts:
            balance_ratio = min(module_counts) / max(module_counts) if max(module_counts) > 0 else 0
            analysis["balance_ratio"] = balance_ratio
            analysis["is_balanced"] = balance_ratio > 0.5  # Heuristic

        return analysis


def main():
    """Command-line interface for data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Robo-Oracle labeled failure dataset")
    parser.add_argument("--output-dir", default="data/robo_oracle_failures", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of labeled failures")
    parser.add_argument("--scenarios", nargs="+", help="Scenarios to use")
    parser.add_argument("--min-level", type=float, default=0.2, help="Min perturbation level")
    parser.add_argument("--max-level", type=float, default=0.8, help="Max perturbation level")
    parser.add_argument("--opaque-policy", help="Path to opaque policy (Module 2)")

    args = parser.parse_args()

    # Create generator
    generator = RoboOracleDataGenerator(
        opaque_policy_path=args.opaque_policy
    )

    # Generate dataset
    dataset_path = generator.generate_labeled_failure_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_scenarios=args.scenarios,
        min_level=args.min_level,
        max_level=args.max_level
    )

    # Analyze
    print("\nAnalyzing dataset...")
    analysis = generator.analyze_dataset(dataset_path)

    print(f"\nDataset Analysis:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Balanced: {'Yes' if analysis.get('is_balanced') else 'No'}")
    print(f"  Balance ratio: {analysis.get('balance_ratio', 0):.2f}")
    print(f"  Cascading failures: {analysis['cascading_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
