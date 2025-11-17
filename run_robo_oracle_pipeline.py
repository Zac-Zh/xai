#!/usr/bin/env python3
"""
Robo-Oracle Master Pipeline Script

This script orchestrates the complete Robo-Oracle pipeline from start to finish:
1. Generate expert demonstrations
2. Train Diffusion Policy
3. Generate labeled failure dataset
4. Prepare VLM training data
5. Train diagnostic VLM
6. Evaluate diagnostic model

Usage:
  python run_robo_oracle_pipeline.py --mode full
  python run_robo_oracle_pipeline.py --mode quick-test  # Smaller dataset for testing
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


class RoboOraclePipeline:
    """Master pipeline orchestrator for Robo-Oracle."""

    def __init__(
        self,
        mode: str = "full",
        output_base: str = "results/robo_oracle_pipeline",
        config_path: str = "configs/robosuite_grasp.yaml",
        thresholds_path: str = "configs/thresholds.yaml"
    ):
        self.mode = mode
        self.output_base = output_base
        self.config_path = config_path
        self.thresholds_path = thresholds_path

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_base, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Set parameters based on mode
        if mode == "quick-test":
            self.params = {
                "demo_scenarios": ["occlusion", "lighting"],
                "demo_levels": [0.0, 0.1],
                "demo_seeds": 50,
                "failure_scenarios": ["occlusion", "lighting"],
                "failure_levels": [0.4, 0.6],
                "failure_seeds": 50,
                "diffusion_epochs": 20,
                "diffusion_batch_size": 16,
                "vlm_epochs": 20,
                "vlm_batch_size": 8
            }
        else:  # full mode
            self.params = {
                "demo_scenarios": ["occlusion", "lighting", "motion_blur"],
                "demo_levels": [0.0, 0.1, 0.2],
                "demo_seeds": 100,
                "failure_scenarios": ["occlusion", "lighting", "motion_blur", "overlap", "camera_jitter"],
                "failure_levels": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "failure_seeds": 100,
                "diffusion_epochs": 100,
                "diffusion_batch_size": 32,
                "vlm_epochs": 50,
                "vlm_batch_size": 16
            }

        # Output directories
        self.dirs = {
            "demos": os.path.join(self.run_dir, "expert_demos"),
            "diffusion": os.path.join(self.run_dir, "diffusion_policy"),
            "failures": os.path.join(self.run_dir, "labeled_failures"),
            "vlm_data": os.path.join(self.run_dir, "vlm_dataset"),
            "vlm_model": os.path.join(self.run_dir, "diagnostic_vlm"),
            "evaluation": os.path.join(self.run_dir, "evaluation")
        }

        print(f"\n{'='*70}")
        print(f"Robo-Oracle Pipeline - Mode: {mode.upper()}")
        print(f"{'='*70}")
        print(f"Output directory: {self.run_dir}")
        print(f"{'='*70}\n")

    def run_command(self, cmd: list, step_name: str) -> bool:
        """Run a shell command and handle errors."""
        print(f"\n{'='*70}")
        print(f"STEP: {step_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            print(f"\n✓ {step_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {step_name} failed with error:")
            print(f"  {e}")
            return False
        except Exception as e:
            print(f"\n✗ {step_name} failed with unexpected error:")
            print(f"  {e}")
            return False

    def step1_generate_demonstrations(self) -> bool:
        """Step 1: Generate expert demonstrations from Classical Oracle."""
        cmd = [
            "python", "oracle/generate_demonstrations.py",
            "--cfg", self.config_path,
            "--thresholds", self.thresholds_path,
            "--scenarios", *self.params["demo_scenarios"],
            "--levels", *[str(l) for l in self.params["demo_levels"]],
            "--num-seeds", str(self.params["demo_seeds"]),
            "--output-dir", self.dirs["demos"]
        ]
        return self.run_command(cmd, "Generate Expert Demonstrations")

    def step2_train_diffusion_policy(self) -> bool:
        """Step 2: Train Diffusion Policy on expert demonstrations."""
        demos_path = os.path.join(self.dirs["demos"], "expert_demonstrations.json")

        if not os.path.exists(demos_path):
            print(f"Error: Expert demonstrations not found at {demos_path}")
            return False

        cmd = [
            "python", "opaque/train_diffusion_policy.py",
            "--demonstrations", demos_path,
            "--output-dir", self.dirs["diffusion"],
            "--epochs", str(self.params["diffusion_epochs"]),
            "--batch-size", str(self.params["diffusion_batch_size"])
        ]
        return self.run_command(cmd, "Train Diffusion Policy")

    def step3_generate_labeled_failures(self) -> bool:
        """Step 3: Generate labeled failure dataset using Robo-Oracle."""
        model_path = os.path.join(self.dirs["diffusion"], "best_policy.pth")

        if not os.path.exists(model_path):
            print(f"Error: Trained Diffusion Policy not found at {model_path}")
            return False

        cmd = [
            "python", "robo_oracle/generate_labeled_failures.py",
            "--opaque-model", model_path,
            "--cfg", self.config_path,
            "--thresholds", self.thresholds_path,
            "--scenarios", *self.params["failure_scenarios"],
            "--levels", *[str(l) for l in self.params["failure_levels"]],
            "--num-seeds", str(self.params["failure_seeds"]),
            "--output-dir", self.dirs["failures"]
        ]
        return self.run_command(cmd, "Generate Labeled Failure Dataset")

    def step4_prepare_vlm_dataset(self) -> bool:
        """Step 4: Prepare VLM training dataset."""
        failures_path = os.path.join(self.dirs["failures"], "labeled_failures.json")

        if not os.path.exists(failures_path):
            print(f"Error: Labeled failures not found at {failures_path}")
            return False

        cmd = [
            "python", "diagnostic/prepare_vlm_dataset.py",
            "--labeled-failures", failures_path,
            "--output-dir", self.dirs["vlm_data"],
            "--instruction-types", "diagnosis", "classification"
        ]
        return self.run_command(cmd, "Prepare VLM Training Dataset")

    def step5_train_diagnostic_vlm(self) -> bool:
        """Step 5: Train diagnostic VLM."""
        train_path = os.path.join(self.dirs["vlm_data"], "train_vlm_dataset.json")
        val_path = os.path.join(self.dirs["vlm_data"], "val_vlm_dataset.json")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print(f"Error: VLM datasets not found")
            return False

        cmd = [
            "python", "diagnostic/train_diagnostic_vlm.py",
            "--train-dataset", train_path,
            "--val-dataset", val_path,
            "--output-dir", self.dirs["vlm_model"],
            "--epochs", str(self.params["vlm_epochs"]),
            "--batch-size", str(self.params["vlm_batch_size"])
        ]
        return self.run_command(cmd, "Train Diagnostic VLM")

    def step6_evaluate(self) -> bool:
        """Step 6: Evaluate diagnostic VLM."""
        model_path = os.path.join(self.dirs["vlm_model"], "best_diagnostic_vlm.pth")
        val_path = os.path.join(self.dirs["vlm_data"], "val_vlm_dataset.json")

        if not os.path.exists(model_path):
            print(f"Error: Trained VLM not found at {model_path}")
            return False

        cmd = [
            "python", "diagnostic/evaluate.py",
            "--model", model_path,
            "--test-dataset", val_path,
            "--output-dir", self.dirs["evaluation"]
        ]
        return self.run_command(cmd, "Evaluate Diagnostic VLM")

    def save_pipeline_metadata(self, success: bool):
        """Save pipeline metadata."""
        metadata = {
            "mode": self.mode,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "parameters": self.params,
            "directories": self.dirs,
            "config_path": self.config_path,
            "thresholds_path": self.thresholds_path
        }

        metadata_path = os.path.join(self.run_dir, "pipeline_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nPipeline metadata saved to: {metadata_path}")

    def run(self) -> bool:
        """Run the complete pipeline."""
        steps = [
            ("Step 1/6: Generate Expert Demonstrations", self.step1_generate_demonstrations),
            ("Step 2/6: Train Diffusion Policy", self.step2_train_diffusion_policy),
            ("Step 3/6: Generate Labeled Failures", self.step3_generate_labeled_failures),
            ("Step 4/6: Prepare VLM Dataset", self.step4_prepare_vlm_dataset),
            ("Step 5/6: Train Diagnostic VLM", self.step5_train_diagnostic_vlm),
            ("Step 6/6: Evaluate Diagnostic Model", self.step6_evaluate)
        ]

        success = True
        for step_name, step_func in steps:
            print(f"\n\n{'#'*70}")
            print(f"# {step_name}")
            print(f"{'#'*70}\n")

            if not step_func():
                print(f"\n✗ Pipeline failed at: {step_name}")
                success = False
                break

        # Save metadata
        self.save_pipeline_metadata(success)

        # Print final summary
        print(f"\n\n{'='*70}")
        if success:
            print("✓ ROBO-ORACLE PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print("✗ ROBO-ORACLE PIPELINE FAILED")
        print(f"{'='*70}")
        print(f"\nResults saved to: {self.run_dir}")

        if success:
            print("\nKey outputs:")
            print(f"  - Expert Demonstrations: {self.dirs['demos']}/expert_demonstrations.json")
            print(f"  - Trained Diffusion Policy: {self.dirs['diffusion']}/best_policy.pth")
            print(f"  - Labeled Failures: {self.dirs['failures']}/labeled_failures.json")
            print(f"  - Trained Diagnostic VLM: {self.dirs['vlm_model']}/best_diagnostic_vlm.pth")
            print(f"  - Evaluation Results: {self.dirs['evaluation']}/evaluation_results.json")

        print(f"{'='*70}\n")

        return success


def main():
    parser = argparse.ArgumentParser(
        description="Robo-Oracle Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (production)
  python run_robo_oracle_pipeline.py --mode full

  # Run quick test (for development/debugging)
  python run_robo_oracle_pipeline.py --mode quick-test

  # Custom output directory
  python run_robo_oracle_pipeline.py --mode full --output results/my_experiment
        """
    )

    parser.add_argument(
        "--mode",
        choices=["full", "quick-test"],
        default="full",
        help="Pipeline mode: 'full' for production, 'quick-test' for development"
    )
    parser.add_argument(
        "--output",
        default="results/robo_oracle_pipeline",
        help="Base output directory"
    )
    parser.add_argument(
        "--cfg",
        default="configs/robosuite_grasp.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--thresholds",
        default="configs/thresholds.yaml",
        help="Thresholds file path"
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = RoboOraclePipeline(
        mode=args.mode,
        output_base=args.output,
        config_path=args.cfg,
        thresholds_path=args.thresholds
    )

    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
