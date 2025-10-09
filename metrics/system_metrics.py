"""System-level metrics utilities."""
from __future__ import annotations


def success(final_dist: float, tau: float) -> bool:
    """Return True if final distance is within threshold."""
    return float(final_dist) <= float(tau)

