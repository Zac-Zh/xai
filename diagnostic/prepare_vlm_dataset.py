"""
Prepare VLM Training Dataset

This script converts the Robo-Oracle labeled failure dataset into a format
suitable for training Vision-Language Models.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from diagnostic.label_to_text import (
    LabelToTextConverter,
    create_instruction_tuning_example
)

try:
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Missing dependencies. Install with: pip install numpy pillow tqdm")


def frames_to_video(
    frame_paths: List[str],
    output_path: str,
    fps: int = 10
) -> str:
    """
    Convert frames to video.

    Args:
        frame_paths: List of paths to frame files (.npy)
        output_path: Output video path
        fps: Frames per second

    Returns:
        Path to created video
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    if not has_cv2:
        # Fallback: create a GIF
        try:
            frames = []
            for path in frame_paths:
                frame = np.load(path)
                frames.append(Image.fromarray(frame.astype(np.uint8)))

            # Save as GIF
            gif_path = output_path.replace('.mp4', '.gif')
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000/fps),
                loop=0
            )
            return gif_path
        except Exception as e:
            print(f"Error creating GIF: {e}")
            return frame_paths[0] if frame_paths else ""

    # Use OpenCV to create video
    frames = []
    for path in frame_paths:
        frame = np.load(path)
        frames.append(frame)

    if not frames:
        return ""

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path


def prepare_vlm_dataset(
    labeled_failures_json: str,
    output_dir: str,
    create_videos: bool = True,
    instruction_types: List[str] = ["diagnosis"],
    train_split: float = 0.8,
    max_samples: int = None
):
    """
    Prepare VLM training dataset from Robo-Oracle labeled failures.

    Args:
        labeled_failures_json: Path to labeled_failures.json
        output_dir: Directory to save VLM dataset
        create_videos: Whether to convert frames to videos
        instruction_types: Types of instructions to generate
        train_split: Fraction of data for training (rest for validation)
        max_samples: Maximum number of samples (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)

    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print("="*60)
    print("Preparing VLM Training Dataset")
    print("="*60)

    # Load labeled failures
    print(f"\nLoading labeled failures from {labeled_failures_json}...")
    with open(labeled_failures_json, "r") as f:
        data = json.load(f)

    failures = data["failures"]
    metadata = data["metadata"]

    print(f"Total failures: {len(failures)}")
    print(f"Statistics: {metadata.get('statistics', {})}")

    # Limit samples if specified
    if max_samples and max_samples < len(failures):
        print(f"Limiting to {max_samples} samples for testing")
        import random
        random.seed(42)
        failures = random.sample(failures, max_samples)

    # Process failures
    converter = LabelToTextConverter()
    vlm_samples = []

    print("\nProcessing failures...")
    for failure in tqdm(failures) if HAS_DEPS else failures:
        failure_id = failure["failure_id"]
        oracle_label = failure["oracle_label"]["failure_label"]

        # Skip if oracle also succeeded (edge case)
        if not oracle_label:
            continue

        # Create video from frames
        video_paths = failure["opaque_policy"]["video_paths"]

        if create_videos and video_paths:
            video_path = os.path.join(videos_dir, f"{failure_id}.mp4")
            if not os.path.exists(video_path):
                video_path = frames_to_video(video_paths, video_path)
        else:
            # Use first frame as static image
            video_path = video_paths[0] if video_paths else None

        if not video_path:
            continue

        # Create instruction-tuning examples
        for inst_type in instruction_types:
            example = create_instruction_tuning_example(
                video_path=video_path,
                failure_label=oracle_label,
                instruction_type=inst_type
            )

            # Add additional context
            example["failure_id"] = failure_id
            example["scenario"] = failure["scenario"]
            example["perturbation_level"] = failure["perturbation_level"]

            vlm_samples.append(example)

    print(f"\nGenerated {len(vlm_samples)} VLM training samples")

    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(vlm_samples)

    split_idx = int(len(vlm_samples) * train_split)
    train_samples = vlm_samples[:split_idx]
    val_samples = vlm_samples[split_idx:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Save datasets
    train_path = os.path.join(output_dir, "train_vlm_dataset.json")
    val_path = os.path.join(output_dir, "val_vlm_dataset.json")

    with open(train_path, "w") as f:
        json.dump({
            "metadata": {
                "source": "Robo-Oracle",
                "version": "1.0",
                "num_samples": len(train_samples),
                "instruction_types": instruction_types
            },
            "samples": train_samples
        }, f, indent=2)

    with open(val_path, "w") as f:
        json.dump({
            "metadata": {
                "source": "Robo-Oracle",
                "version": "1.0",
                "num_samples": len(val_samples),
                "instruction_types": instruction_types
            },
            "samples": val_samples
        }, f, indent=2)

    # Create category distribution report
    category_dist = {}
    for sample in train_samples:
        cat = sample["metadata"]["failure_module"]
        category_dist[cat] = category_dist.get(cat, 0) + 1

    print("\nTraining set failure category distribution:")
    for cat, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(train_samples) * 100) if train_samples else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print("\n" + "="*60)
    print("VLM Dataset Preparation Complete!")
    print("="*60)
    print(f"Training dataset: {train_path}")
    print(f"Validation dataset: {val_path}")
    print("="*60)

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VLM training dataset from Robo-Oracle labeled failures"
    )
    parser.add_argument(
        "--labeled-failures",
        required=True,
        help="Path to labeled_failures.json from Robo-Oracle"
    )
    parser.add_argument(
        "--output-dir",
        default="results/vlm_dataset",
        help="Directory to save VLM dataset"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip creating videos (use static frames)"
    )
    parser.add_argument(
        "--instruction-types",
        nargs="+",
        default=["diagnosis"],
        choices=["diagnosis", "classification", "recovery"],
        help="Types of instructions to generate"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (for testing)"
    )

    args = parser.parse_args()

    prepare_vlm_dataset(
        labeled_failures_json=args.labeled_failures,
        output_dir=args.output_dir,
        create_videos=not args.no_videos,
        instruction_types=args.instruction_types,
        train_split=args.train_split,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
