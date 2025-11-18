"""
Evaluation Module for Robo-Oracle Diagnostic System

This module provides comprehensive evaluation metrics for the diagnostic VLM,
including accuracy, confusion matrices, per-category performance, and
qualitative analysis.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Missing dependencies")

from diagnostic.diagnostic_interface import DiagnosticInterface


class DiagnosticEvaluator:
    """Evaluator for the diagnostic VLM."""

    def __init__(
        self,
        model_checkpoint: str,
        test_dataset_json: str,
        output_dir: str
    ):
        """
        Initialize the evaluator.

        Args:
            model_checkpoint: Path to trained diagnostic model
            test_dataset_json: Path to test/validation dataset
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load diagnostic interface
        print("Loading diagnostic model...")
        self.diagnostic = DiagnosticInterface(model_checkpoint)

        # Load test dataset
        print(f"Loading test dataset from {test_dataset_json}...")
        with open(test_dataset_json, "r") as f:
            data = json.load(f)
        self.test_samples = data["samples"]

        print(f"Loaded {len(self.test_samples)} test samples")

    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.

        Returns:
            Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("Running Diagnostic VLM Evaluation")
        print("="*60)

        all_predictions = []
        all_labels = []
        all_confidences = []
        per_category_results = defaultdict(lambda: {"correct": 0, "total": 0})

        # Evaluate each sample
        for sample in tqdm(self.test_samples, desc="Evaluating"):
            # Get ground truth
            gt_module = sample["metadata"]["failure_module"]

            # Diagnose
            try:
                video_path = sample["video"]
                frames = self._load_frames(video_path)

                diagnosis = self.diagnostic.diagnose_failure(frames)

                pred_module = diagnosis["predicted_module"]
                confidence = diagnosis["confidence"]

                # Record results
                all_predictions.append(pred_module)
                all_labels.append(gt_module)
                all_confidences.append(confidence)

                # Per-category
                per_category_results[gt_module]["total"] += 1
                if pred_module == gt_module:
                    per_category_results[gt_module]["correct"] += 1

            except Exception as e:
                print(f"\nError evaluating sample {sample.get('failure_id', 'unknown')}: {e}")
                continue

        # Compute metrics
        results = self._compute_metrics(
            all_predictions,
            all_labels,
            all_confidences,
            per_category_results
        )

        # Save results
        self._save_results(results)

        # Generate visualizations
        self._generate_visualizations(
            all_predictions,
            all_labels,
            all_confidences
        )

        # Print summary
        self._print_summary(results)

        return results

    def _load_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video path."""
        if video_path.endswith('.npy') or video_path.endswith('.npz'):
            # Single frame - handle both .npy and .npz
            loaded = np.load(video_path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                # Extract first array from .npz archive
                keys = list(loaded.keys())
                if not keys:
                    raise ValueError(f"Empty .npz file: {video_path}")
                frame = loaded[keys[0]]
                loaded.close()
                return [frame]
            else:
                return [loaded]
        elif video_path.endswith('.gif'):
            # Load GIF frames
            from PIL import Image
            img = Image.open(video_path)
            frames = []
            try:
                while True:
                    frames.append(np.array(img.convert('RGB')))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            return frames
        elif video_path.endswith('.mp4') or video_path.endswith('.avi'):
            # Load video frames using OpenCV
            try:
                import cv2
            except ImportError:
                raise ImportError("OpenCV (cv2) is required to load video files. Install with: pip install opencv-python")

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()

            if not frames:
                raise ValueError(f"No frames loaded from video: {video_path}")
            return frames
        else:
            # Try single image
            from PIL import Image
            return [np.array(Image.open(video_path))]

    def _compute_metrics(
        self,
        predictions: List[str],
        labels: List[str],
        confidences: List[float],
        per_category: Dict
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        # Overall accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        # Per-category accuracy
        category_accuracy = {}
        for cat, stats in per_category.items():
            cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            category_accuracy[cat] = {
                "accuracy": cat_acc,
                "correct": stats["correct"],
                "total": stats["total"]
            }

        # Confusion matrix
        categories = sorted(set(labels))
        confusion = self._compute_confusion_matrix(predictions, labels, categories)

        # Average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        confidence_correct = np.mean([
            c for c, p, l in zip(confidences, predictions, labels) if p == l
        ]) if confidences else 0.0
        confidence_incorrect = np.mean([
            c for c, p, l in zip(confidences, predictions, labels) if p != l
        ]) if confidences else 0.0

        return {
            "overall_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_category_accuracy": category_accuracy,
            "confusion_matrix": confusion,
            "categories": categories,
            "avg_confidence": avg_confidence,
            "avg_confidence_correct": confidence_correct,
            "avg_confidence_incorrect": confidence_incorrect
        }

    def _compute_confusion_matrix(
        self,
        predictions: List[str],
        labels: List[str],
        categories: List[str]
    ) -> List[List[int]]:
        """Compute confusion matrix."""
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        n = len(categories)
        matrix = [[0] * n for _ in range(n)]

        for pred, label in zip(predictions, labels):
            if pred in cat_to_idx and label in cat_to_idx:
                i = cat_to_idx[label]
                j = cat_to_idx[pred]
                matrix[i][j] += 1

        return matrix

    def _generate_visualizations(
        self,
        predictions: List[str],
        labels: List[str],
        confidences: List[float]
    ):
        """Generate evaluation visualizations."""
        # Check if we have any valid results
        if not predictions or not labels:
            print("\nNo valid predictions - skipping visualizations")
            return

        # Confusion matrix heatmap
        categories = sorted(set(labels))
        if not categories:
            print("\nNo categories found - skipping visualizations")
            return

        confusion = self._compute_confusion_matrix(predictions, labels, categories)

        plt.figure(figsize=(10, 8))
        plt.imshow(confusion, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16)
        plt.colorbar()

        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=45)
        plt.yticks(tick_marks, categories)

        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(categories)):
                plt.text(j, i, str(confusion[i][j]),
                        ha="center", va="center",
                        color="white" if confusion[i][j] > max(max(row) for row in confusion) / 2 else "black")

        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

        # Confidence distribution
        if not confidences:
            print("\nNo confidence scores - skipping confidence distribution")
            return

        plt.figure(figsize=(10, 6))
        correct_confidences = [c for c, p, l in zip(confidences, predictions, labels) if p == l]
        incorrect_confidences = [c for c, p, l in zip(confidences, predictions, labels) if p != l]

        if correct_confidences or incorrect_confidences:
            if correct_confidences:
                plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
            if incorrect_confidences:
                plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
            plt.xlabel('Confidence', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Confidence Distribution', fontsize=16)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'confidence_distribution.png'), dpi=150)
            plt.close()

        print(f"\nVisualizations saved to {self.output_dir}")

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON."""
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')

        # Convert numpy types for JSON serialization
        serializable_results = {
            "overall_accuracy": float(results["overall_accuracy"]),
            "correct": int(results["correct"]),
            "total": int(results["total"]),
            "per_category_accuracy": results["per_category_accuracy"],
            "confusion_matrix": results["confusion_matrix"],
            "categories": results["categories"],
            "avg_confidence": float(results["avg_confidence"]),
            "avg_confidence_correct": float(results["avg_confidence_correct"]),
            "avg_confidence_incorrect": float(results["avg_confidence_incorrect"])
        }

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {results_path}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("Evaluation Results Summary")
        print("="*60)
        print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
        print(f"Correct: {results['correct']}/{results['total']}")
        print(f"\nAverage Confidence: {results['avg_confidence']*100:.1f}%")
        print(f"  - On correct predictions: {results['avg_confidence_correct']*100:.1f}%")
        print(f"  - On incorrect predictions: {results['avg_confidence_incorrect']*100:.1f}%")

        print("\nPer-Category Accuracy:")
        for cat, stats in sorted(
            results['per_category_accuracy'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        ):
            acc = stats['accuracy'] * 100
            correct = stats['correct']
            total = stats['total']
            print(f"  {cat:12s}: {acc:5.1f}% ({correct}/{total})")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Diagnostic VLM"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained diagnostic model checkpoint"
    )
    parser.add_argument(
        "--test-dataset",
        required=True,
        help="Path to test/validation VLM dataset JSON"
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = DiagnosticEvaluator(
        model_checkpoint=args.model,
        test_dataset_json=args.test_dataset,
        output_dir=args.output_dir
    )

    results = evaluator.evaluate()


if __name__ == "__main__":
    main()
