"""Data corruption perturbation: corrupts sensor data and internal state representations."""
from __future__ import annotations

from typing import Dict
import numpy as np


def apply(ctx: Dict, level: float) -> None:
    """
    Apply data corruption perturbation.

    Args:
        ctx: Context dictionary containing RNG and corruption specifications
        level: Corruption severity [0.0-1.0]
            0.0 = no corruption
            0.3 = minor bit flips
            0.6 = moderate data loss
            1.0 = severe corruption
    """
    rng = ctx.get("rng", np.random.default_rng(0))

    # Initialize corruption modes
    corruption = ctx.setdefault("corruption", {})

    # Image sensor corruption
    # Pixel dropout (dead pixels)
    corruption["pixel_dropout_rate"] = float(level * 0.15)

    # Bit flip errors in image data
    corruption["bit_flip_rate"] = float(level * 0.001)  # bits per pixel

    # Salt and pepper noise
    corruption["salt_pepper_prob"] = float(level * 0.1)

    # Color channel corruption
    if level > 0.3:
        # Random channel drops (R/G/B)
        corruption["channel_dropout_prob"] = float((level - 0.3) * 0.3)

    # State corruption (pose, velocity estimates)
    if level > 0.4:
        # State vector corruption
        corruption["state_noise_scale"] = float(level * 0.5)
        # Timestamp corruption
        corruption["timestamp_jitter"] = float(level * 0.1)  # seconds

    # Memory corruption
    if level > 0.5:
        # Map/obstacle data corruption
        corruption["map_corruption_rate"] = float((level - 0.5) * 0.4)
        # Feature descriptor corruption
        corruption["feature_corruption_rate"] = float((level - 0.5) * 0.6)

    # Calibration data corruption
    if level > 0.6:
        # Camera intrinsics corruption
        corruption["intrinsics_error"] = float((level - 0.6) * 2.0)
        # Extrinsics corruption
        corruption["extrinsics_error"] = float((level - 0.6) * 1.5)

    # Data structure corruption
    if level > 0.7:
        # Buffer overflow/underflow simulation
        corruption["buffer_error_rate"] = float((level - 0.7) * 0.3)
        # Pointer corruption
        corruption["pointer_error_rate"] = float((level - 0.7) * 0.2)

    # Add noise marker
    ctx.setdefault("noise", {})["data_corruption"] = float(level)

    # Add corrupted image noise parameters
    noise_params = ctx.setdefault("noise", {})
    noise_params["gaussian_std"] = float(level * 50.0)  # 0-50 pixel value std
    noise_params["uniform_range"] = float(level * 100.0)  # ±0-100 pixel values
