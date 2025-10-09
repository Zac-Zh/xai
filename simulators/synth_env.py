"""Synthetic 2D Lift environment for offline experiments.

The agent and target live on a unit square [0,1]x[0,1]. The agent tries to
reach and "lift" the target (toggle a lifted flag) if distance is below a
success threshold provided externally for system success assessment.

This simulator is deterministic for a given seed via NumPy RNG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SynthLiftEnv:
    """Lightweight synthetic Lift task with simple kinematics and rendering.

    Attributes:
        seed: RNG seed for determinism.
        max_steps: Maximum steps per episode.
        camera: Dict with keys name,width,height,fps for render size.
    """

    seed: int
    max_steps: int
    camera: Dict[str, int]

    def __init__(self, seed: int, max_steps: int, camera: Dict[str, int]):
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.camera = dict(camera)
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.agent_xy = np.zeros(2, dtype=float)
        self.target_xy = np.zeros(2, dtype=float)
        self.lifted = False

    def reset(self) -> Dict[str, float]:
        """Reset environment to a deterministic initial state for the seed.

        Returns a dictionary state snapshot.
        """
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        # Target near center; agent near a corner for non-trivial path
        self.target_xy = self.rng.uniform(0.35, 0.65, size=2)
        self.agent_xy = self.rng.uniform(0.05, 0.15, size=2)
        self.lifted = False
        return self.get_state()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, float], float, bool, Dict]:
        """Advance the environment by one step.

        Args:
            action: Delta in XY to move the agent by (clipped).

        Returns: (state, reward, done, info)
        """
        self.step_count += 1
        delta = np.asarray(action, dtype=float).reshape(2)
        delta = np.clip(delta, -0.05, 0.05)
        self.agent_xy = np.clip(self.agent_xy + delta, 0.0, 1.0)
        dist = float(np.linalg.norm(self.agent_xy - self.target_xy))
        reward = -dist
        if dist < 0.03:  # nominal success distance (system has its own tau)
            self.lifted = True
        done = self.step_count >= self.max_steps or self.lifted
        return self.get_state(), reward, done, {}

    def render_rgb(self) -> np.ndarray:
        """Render a simple RGB image marking agent (blue) and target (red)."""
        h = int(self.camera.get("height", 480))
        w = int(self.camera.get("width", 640))
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        ax = int(np.clip(self.agent_xy[0] * w, 0, w - 1))
        ay = int(np.clip((1.0 - self.agent_xy[1]) * h, 0, h - 1))
        tx = int(np.clip(self.target_xy[0] * w, 0, w - 1))
        ty = int(np.clip((1.0 - self.target_xy[1]) * h, 0, h - 1))

        def draw_disk(cx: int, cy: int, radius: int, color: Tuple[int, int, int]) -> None:
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            img[mask] = color

        draw_disk(tx, ty, max(3, w // 80), (200, 30, 30))
        draw_disk(ax, ay, max(3, w // 90), (30, 30, 200))
        return img

    def get_state(self) -> Dict[str, float]:
        """Return ground-truth positions and lift flag."""
        return {
            "agent_x": float(self.agent_xy[0]),
            "agent_y": float(self.agent_xy[1]),
            "target_x": float(self.target_xy[0]),
            "target_y": float(self.target_xy[1]),
            "lifted": bool(self.lifted),
            "step": int(self.step_count),
        }

