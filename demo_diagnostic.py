#!/usr/bin/env python3
"""
Robo-Oracle Diagnostic Demo

This script demonstrates the trained diagnostic VLM by diagnosing
a sample failure and comparing it with the Oracle's ground truth.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diagnostic.diagnostic_interface import DiagnosticInterface
from diagnostic.label_to_text import LabelToTextConverter


def demo_single_failure(
    model_checkpoint: str,
    labeled_failures_json: str,
    failure_idx: int = 0
):
    """
    Demonstrate diagnosis on a single failure.

    Args:
        model_checkpoint: Path to trained diagnostic VLM
        labeled_failures_json: Path to labeled_failures.json
        failure_idx: Index of failure to diagnose
    """
    print("\n" + "="*70)
    print("Robo-Oracle Diagnostic Demo")
    print("="*70)

    # Load diagnostic interface
    print("\nLoading diagnostic model...")
    diagnostic = DiagnosticInterface(model_checkpoint)

    # Load labeled failures
    print("Loading labeled failure dataset...")
    with open(labeled_failures_json, "r") as f:
        data = json.load(f)

    failures = data["failures"]

    if failure_idx >= len(failures):
        print(f"Error: failure_idx {failure_idx} out of range (max: {len(failures)-1})")
        return

    failure = failures[failure_idx]

    print(f"\nAnalyzing failure: {failure['failure_id']}")
    print(f"Scenario: {failure['scenario']}")
    print(f"Perturbation level: {failure['perturbation_level']}")

    # Load video frames
    video_paths = failure["opaque_policy"]["video_paths"]

    if not video_paths:
        print("Error: No video frames available for this failure")
        return

    print(f"Loading {len(video_paths)} video frames...")
    frames = []
    for path in video_paths[:10]:  # Limit to first 10 frames
        if os.path.exists(path):
            frame = np.load(path)
            frames.append(frame)

    if not frames:
        print("Error: Could not load any frames")
        return

    # Get Oracle ground truth
    oracle_label = failure["oracle_label"]["failure_label"]
    oracle_module = oracle_label["primary_failure_module"] if oracle_label else "Unknown"
    oracle_error = oracle_label["primary_error_code"] if oracle_label else "unknown"

    # Diagnose with VLM
    print("\nRunning diagnostic model...")
    diagnosis = diagnostic.diagnose_failure(frames, return_confidence=True)

    # Convert oracle label to text
    converter = LabelToTextConverter()
    oracle_explanation = converter.convert_to_text(oracle_label) if oracle_label else "N/A"

    # Display results
    print("\n" + "="*70)
    print("DIAGNOSTIC RESULTS")
    print("="*70)

    print("\n📹 OBSERVED FAILURE (from Opaque Policy):")
    print(f"  - Final Distance: {failure['opaque_policy']['final_distance']:.4f}")
    print(f"  - Video Frames: {len(video_paths)}")

    print("\n🔍 ORACLE GROUND TRUTH (from Classical Pipeline):")
    print(f"  - Primary Module: {oracle_module}")
    print(f"  - Error Code: {oracle_error}")
    print(f"  - Explanation: {oracle_explanation[:200]}...")

    print("\n🤖 DIAGNOSTIC VLM PREDICTION:")
    print(f"  - Predicted Module: {diagnosis['predicted_module']}")
    print(f"  - Confidence: {diagnosis['confidence']*100:.1f}%")
    print(f"  - Explanation: {diagnosis['explanation'][:200]}...")

    print("\n📊 ALL CONFIDENCES:")
    for module, conf in sorted(
        diagnosis['all_confidences'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(conf * 50)
        print(f"  {module:12s} [{conf*100:5.1f}%] {bar}")

    print("\n" + "="*70)
    match = (oracle_module == diagnosis['predicted_module'])
    if match:
        print("✅ CORRECT PREDICTION!")
    else:
        print("❌ INCORRECT PREDICTION")
        print(f"   Expected: {oracle_module}")
        print(f"   Got: {diagnosis['predicted_module']}")
    print("="*70 + "\n")


def demo_multiple_failures(
    model_checkpoint: str,
    labeled_failures_json: str,
    num_samples: int = 5
):
    """Demonstrate diagnosis on multiple random failures."""
    import random

    print("\n" + "="*70)
    print(f"Robo-Oracle Diagnostic Demo - {num_samples} Random Samples")
    print("="*70)

    diagnostic = DiagnosticInterface(model_checkpoint)

    with open(labeled_failures_json, "r") as f:
        data = json.load(f)

    failures = data["failures"]

    # Random sample
    random.seed(42)
    samples = random.sample(failures, min(num_samples, len(failures)))

    results = []

    for i, failure in enumerate(samples, 1):
        print(f"\n[{i}/{num_samples}] {failure['failure_id']}")

        video_paths = failure["opaque_policy"]["video_paths"]
        if not video_paths:
            continue

        # Load frames
        frames = []
        for path in video_paths[:10]:
            if os.path.exists(path):
                frames.append(np.load(path))

        if not frames:
            continue

        # Diagnose
        try:
            diagnosis = diagnostic.diagnose_failure(frames)
            oracle_label = failure["oracle_label"]["failure_label"]
            oracle_module = oracle_label["primary_failure_module"] if oracle_label else "Unknown"

            match = (oracle_module == diagnosis['predicted_module'])

            results.append({
                "failure_id": failure['failure_id'],
                "oracle": oracle_module,
                "predicted": diagnosis['predicted_module'],
                "confidence": diagnosis['confidence'],
                "match": match
            })

            status = "✅" if match else "❌"
            print(f"  Oracle: {oracle_module:12s} | Predicted: {diagnosis['predicted_module']:12s} | "
                  f"Conf: {diagnosis['confidence']*100:5.1f}% {status}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        correct = sum(1 for r in results if r["match"])
        total = len(results)
        accuracy = correct / total * 100

        print(f"Samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1f}%")

        # Per-module breakdown
        per_module = {}
        for r in results:
            mod = r["oracle"]
            if mod not in per_module:
                per_module[mod] = {"correct": 0, "total": 0}
            per_module[mod]["total"] += 1
            if r["match"]:
                per_module[mod]["correct"] += 1

        print("\nPer-Module Accuracy:")
        for mod, stats in sorted(per_module.items()):
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {mod:12s}: {acc:5.1f}% ({stats['correct']}/{stats['total']})")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Robo-Oracle Diagnostic Demo")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained diagnostic VLM checkpoint"
    )
    parser.add_argument(
        "--failures",
        required=True,
        help="Path to labeled_failures.json"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multiple"],
        default="single",
        help="Demo mode"
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Failure index for single mode"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of samples for multiple mode"
    )

    args = parser.parse_args()

    if args.mode == "single":
        demo_single_failure(args.model, args.failures, args.idx)
    else:
        demo_multiple_failures(args.model, args.failures, args.num)


if __name__ == "__main__":
    main()
