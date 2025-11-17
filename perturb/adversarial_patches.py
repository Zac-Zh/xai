"""Adversarial patches perturbation: adds adversarial visual patterns to fool perception."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def apply(ctx: Dict, level: float) -> None:
    """
    Apply adversarial patches perturbation.

    Args:
        ctx: Context dictionary containing RNG and patch specifications
        level: Adversarial strength [0.0-1.0]
            0.0 = no patches
            0.3 = single small patch
            0.6 = multiple patches
            1.0 = large/optimized patches
    """
    rng = ctx.get("rng", np.random.default_rng(0))

    # Initialize adversarial patch configuration
    adv_config = ctx.setdefault("adversarial", {})

    # Number of patches increases with level
    if level < 0.3:
        num_patches = 0
    elif level < 0.6:
        num_patches = int(1 + (level - 0.3) * 10)  # 1-3 patches
    else:
        num_patches = int(3 + (level - 0.6) * 15)  # 3-9 patches

    adv_config["num_patches"] = num_patches

    # Patch properties
    patches: List[Dict] = []

    for i in range(num_patches):
        patch = {}

        # Patch size (increases with level)
        min_size = 0.05  # 5% of image
        max_size = 0.25  # 25% of image
        patch["size"] = float(min_size + (max_size - min_size) * level)

        # Patch position (random but can target specific regions)
        if level > 0.7:
            # Target likely object locations (center region)
            patch["x"] = float(rng.uniform(0.3, 0.7))
            patch["y"] = float(rng.uniform(0.3, 0.7))
        else:
            # Random placement
            patch["x"] = float(rng.uniform(0.1, 0.9))
            patch["y"] = float(rng.uniform(0.1, 0.9))

        # Patch type
        patch_types = ["checkerboard", "gradient", "noise", "optimized"]
        if level < 0.4:
            patch["type"] = "checkerboard"  # Simple patterns
        elif level < 0.7:
            patch["type"] = rng.choice(["checkerboard", "gradient", "noise"])
        else:
            patch["type"] = "optimized"  # Simulated optimized adversarial patch

        # Patch intensity (how visible/disruptive)
        patch["intensity"] = float(level)

        # Patch orientation
        patch["rotation"] = float(rng.uniform(0, 360))  # degrees

        # For optimized patches, add target class
        if patch["type"] == "optimized":
            # Target misclassification
            patch["target_class"] = rng.choice(["background", "distractor", "wrong_object"])
            # Optimization strength
            patch["perturbation_budget"] = float(level * 0.3)  # L_inf bound

        patches.append(patch)

    adv_config["patches"] = patches

    # Physical patch properties (if 3D)
    if level > 0.5:
        adv_config["physical_3d"] = True
        adv_config["patch_texture_resolution"] = int(256 * (1 + level))  # Higher res for stronger attacks
        adv_config["lighting_invariance"] = level > 0.7  # Optimized for different lighting
        adv_config["viewpoint_invariance"] = level > 0.8  # Optimized for different angles

    # Attack strategy
    if level > 0.6:
        # Targeted attack
        adv_config["attack_type"] = "targeted"
        adv_config["target_behavior"] = rng.choice([
            "misdetection",  # Fail to detect object
            "mislocalization",  # Wrong bounding box
            "misclassification",  # Wrong class
            "multi_object_confusion"  # Hallucinate multiple objects
        ])
    else:
        # Untargeted attack
        adv_config["attack_type"] = "untargeted"

    # Patch visibility/stealthiness trade-off
    adv_config["stealth_factor"] = float(max(1.0 - level, 0.2))  # Lower = more obvious

    # Add to noise marker
    ctx.setdefault("noise", {})["adversarial_patches"] = float(level)

    # For planning, add patch obstacles that might interfere
    if level > 0.4:
        obs = ctx.setdefault("obstacles", [])
        # Add visual distraction zones
        for patch in patches:
            # Small circular obstacle at patch location to simulate confusion
            x, y = patch["x"], patch["y"]
            r = patch["size"] * 0.5
            obs.append((x, y, r))
