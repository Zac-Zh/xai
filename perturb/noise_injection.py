"""Noise injection perturbation: adds various types of noise to sensor and processing pipeline."""
from __future__ import annotations

from typing import Dict
import numpy as np


def apply(ctx: Dict, level: float) -> None:
    """
    Apply noise injection perturbation.

    Args:
        ctx: Context dictionary containing RNG and noise specifications
        level: Noise intensity [0.0-1.0]
            0.0 = no noise
            0.3 = low noise (realistic)
            0.6 = moderate noise
            1.0 = high noise (stress test)
    """
    rng = ctx.get("rng", np.random.default_rng(0))

    # Initialize noise dictionary
    noise = ctx.setdefault("noise", {})
    noise["noise_injection"] = float(level)

    # Sensor noise
    # Gaussian sensor noise (thermal, shot noise)
    noise["sensor_gaussian_std"] = float(level * 25.0)  # pixel intensity std

    # Photon shot noise (Poisson)
    noise["shot_noise_lambda"] = float(level * 10.0)

    # Dark current noise
    noise["dark_current"] = float(level * 5.0)

    # Read noise
    noise["read_noise_std"] = float(level * 3.0)

    # Environmental noise
    # Motion blur from vibration
    noise["vibration_blur"] = float(level * 0.5)

    # Atmospheric turbulence
    noise["atmospheric_distortion"] = float(level * 0.3)

    # EMI (electromagnetic interference)
    noise["emi_amplitude"] = float(level * 20.0)

    # Processing noise
    # Quantization noise
    noise["quantization_bits"] = int(max(8 - level * 4, 4))  # 8-bit to 4-bit

    # Compression artifacts
    noise["compression_quality"] = int(100 - level * 60)  # 100% to 40% quality

    # Discretization errors
    noise["discretization_error"] = float(level * 0.01)

    # State estimation noise
    # IMU noise
    noise["imu_accel_std"] = float(level * 0.5)  # m/s^2
    noise["imu_gyro_std"] = float(level * 0.1)  # rad/s

    # Odometry noise
    noise["odometry_drift"] = float(level * 0.05)  # 5% max drift

    # GPS noise (if applicable)
    noise["gps_position_std"] = float(level * 2.0)  # meters

    # Temporal noise
    # Frame jitter
    noise["frame_jitter_std"] = float(level * 0.05)  # seconds

    # Processing latency variance
    noise["latency_jitter"] = float(level * 0.1)  # seconds

    # Network noise (if multi-agent)
    if level > 0.5:
        noise["network_latency_mean"] = float(level * 0.2)  # seconds
        noise["network_latency_std"] = float(level * 0.1)
        noise["packet_loss_rate"] = float((level - 0.5) * 0.2)  # up to 10%

    # Stochastic computational noise
    if level > 0.6:
        # Floating point errors
        noise["numerical_precision_loss"] = float((level - 0.6) * 1e-6)
        # Parallel processing non-determinism
        noise["thread_synchronization_jitter"] = float((level - 0.6) * 0.01)
