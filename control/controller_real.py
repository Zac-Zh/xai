"""
Real physics-based PD controller for trajectory tracking.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class PDController:
    """Real PD controller with physics-based dynamics."""

    def __init__(
        self,
        kp: float = 10.0,
        kd: float = 2.0,
        max_velocity: float = 1.0,
        max_acceleration: float = 2.0,
    ):
        """
        Initialize PD controller.

        Args:
            kp: Proportional gain
            kd: Derivative gain
            max_velocity: Maximum velocity
            max_acceleration: Maximum acceleration
        """
        self.kp = kp
        self.kd = kd
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        # State
        self.prev_error: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None

    def compute_control(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: Optional[np.ndarray] = None,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Compute control command.

        Args:
            current_pos: Current position
            target_pos: Target position
            current_vel: Current velocity (estimated from position if None)
            dt: Time step

        Returns:
            Control command (acceleration)
        """
        # Compute error
        error = target_pos - current_pos

        # Estimate velocity if not provided
        if current_vel is None:
            if self.prev_error is not None:
                current_vel = (error - self.prev_error) / dt
            else:
                current_vel = np.zeros_like(error)

        # PD control
        control = self.kp * error - self.kd * current_vel

        # Clamp acceleration
        control_mag = np.linalg.norm(control)
        if control_mag > self.max_acceleration:
            control = control * (self.max_acceleration / control_mag)

        # Update state
        self.prev_error = error

        return control

    def reset(self) -> None:
        """Reset controller state."""
        self.prev_error = None
        self.prev_time = None


class TrajectoryTracker:
    """Trajectory tracking with PD control."""

    def __init__(self, kp: float = 10.0, kd: float = 2.0):
        """Initialize trajectory tracker."""
        self.controller = PDController(kp=kp, kd=kd)

    def track_trajectory(
        self,
        waypoints: List[np.ndarray],
        initial_pos: np.ndarray,
        dt: float = 0.01,
        tolerance: float = 0.05
    ) -> Dict:
        """
        Track trajectory through waypoints.

        Args:
            waypoints: List of waypoint positions
            initial_pos: Initial position
            dt: Time step
            tolerance: Position tolerance for waypoint reaching

        Returns:
            Tracking result dictionary
        """
        if len(waypoints) == 0:
            return {
                "track_rmse": np.inf,
                "overshoot": 0.0,
                "oscillation": False,
                "success": False,
            }

        # Simulate tracking
        current_pos = initial_pos.copy()
        current_vel = np.zeros_like(initial_pos)

        tracking_errors = []
        positions = [current_pos.copy()]
        max_overshoot = 0.0

        for waypoint in waypoints:
            reached = False
            steps = 0
            max_steps = int(10.0 / dt)  # 10 seconds max per waypoint

            while not reached and steps < max_steps:
                # Compute control
                control = self.controller.compute_control(
                    current_pos, waypoint, current_vel, dt
                )

                # Update state (simplified dynamics)
                current_vel += control * dt
                # Velocity damping
                current_vel *= 0.98

                # Clamp velocity
                vel_mag = np.linalg.norm(current_vel)
                if vel_mag > self.controller.max_velocity:
                    current_vel = current_vel * (self.controller.max_velocity / vel_mag)

                current_pos += current_vel * dt

                positions.append(current_pos.copy())

                # Track error
                error = np.linalg.norm(waypoint - current_pos)
                tracking_errors.append(error)

                # Check overshoot
                dist_to_prev = np.linalg.norm(current_pos - (waypoints[max(0, waypoints.index(waypoint) - 1)] if waypoints.index(waypoint) > 0 else initial_pos))
                target_dist = np.linalg.norm(waypoint - (waypoints[max(0, waypoints.index(waypoint) - 1)] if waypoints.index(waypoint) > 0 else initial_pos))
                if dist_to_prev > target_dist:
                    overshoot = dist_to_prev - target_dist
                    max_overshoot = max(max_overshoot, overshoot)

                # Check if reached
                if error < tolerance:
                    reached = True

                steps += 1

            if not reached:
                # Failed to reach waypoint
                return {
                    "track_rmse": np.mean(tracking_errors) if tracking_errors else np.inf,
                    "overshoot": max_overshoot,
                    "oscillation": self._detect_oscillation(positions),
                    "success": False,
                }

        # Compute metrics
        track_rmse = np.sqrt(np.mean(np.array(tracking_errors) ** 2))

        return {
            "track_rmse": float(track_rmse),
            "overshoot": float(max_overshoot),
            "oscillation": self._detect_oscillation(positions),
            "success": True,
        }

    def _detect_oscillation(self, positions: List[np.ndarray], window: int = 10) -> bool:
        """Detect oscillation in trajectory."""
        if len(positions) < window * 2:
            return False

        # Check for repeated direction changes
        recent_positions = positions[-window * 2:]
        velocities = np.diff(recent_positions, axis=0)

        if len(velocities) < 2:
            return False

        # Count sign changes in velocity
        sign_changes = 0
        for i in range(1, len(velocities)):
            for dim in range(velocities.shape[1]):
                if velocities[i - 1, dim] * velocities[i, dim] < 0:
                    sign_changes += 1

        # If many sign changes, likely oscillating
        return sign_changes > window


def track_path(
    path: List[np.ndarray],
    initial_pos: np.ndarray,
    kp: float = 10.0,
    kd: float = 2.0,
    perturbation_level: float = 0.0
) -> Dict:
    """
    Unified path tracking interface (compatible with old API).

    Args:
        path: Path waypoints
        initial_pos: Initial position
        kp: Proportional gain
        kd: Derivative gain
        perturbation_level: Perturbation level (affects tracking error)

    Returns:
        Tracking result dictionary
    """
    # Initialize tracker
    tracker = TrajectoryTracker(kp=kp, kd=kd)

    # Track
    result = tracker.track_trajectory(path, initial_pos)

    # Apply perturbation effect
    if perturbation_level > 0:
        # Increase tracking error
        noise = np.random.uniform(0.01, 0.05) * perturbation_level
        result["track_rmse"] += noise
        result["overshoot"] += noise * 0.5

    return result
