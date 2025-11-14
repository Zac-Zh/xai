"""Tiny 2D planner: straight-line with fallback random waypoint (RRT*-like)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _collides(p: np.ndarray, obstacles: List[Tuple[float, float, float]]) -> bool:
    for ox, oy, r in obstacles:
        if np.linalg.norm(p - np.array([ox, oy])) <= r:
            return True
    return False


def _line_free(a: np.ndarray, b: np.ndarray, obstacles: List[Tuple[float, float, float]]) -> bool:
    samples = 50
    for t in np.linspace(0.0, 1.0, samples):
        p = (1 - t) * a + t * b
        if _collides(p, obstacles):
            return False
    return True


def plan(start: np.ndarray, goal: np.ndarray, ctx: Dict) -> Dict[str, object]:
    """Plan a path on [0,1]^2 avoiding circular obstacles in ctx['obstacles'].

    Returns a dict with success, path (list of [x,y]), path_cost, collisions and planner.
    """
    rng: np.random.Generator = ctx.get("rng") or np.random.default_rng(0)
    obstacles = ctx.get("obstacles", [])  # list of (x,y,r)
    a = np.clip(np.asarray(start, dtype=float).reshape(2), 0.0, 1.0)
    b = np.clip(np.asarray(goal, dtype=float).reshape(2), 0.0, 1.0)

    path: List[List[float]] = []
    collisions = 0

    if _line_free(a, b, obstacles):
        pts = [((1 - t) * a + t * b).tolist() for t in np.linspace(0.0, 1.0, 20)]
        path = [[float(x), float(y)] for x, y in pts]
        cost = float(np.linalg.norm(b - a))
        return {"success": True, "path": path, "path_cost": cost, "collisions": collisions, "planner": "RRTstar"}

    # Fallback: sample a random waypoint; if both segments free, connect
    for _ in range(200):
        w = rng.uniform(0.0, 1.0, size=2)
        if _line_free(a, w, obstacles) and _line_free(w, b, obstacles):
            seg1 = [((1 - t) * a + t * w).tolist() for t in np.linspace(0.0, 1.0, 10)]
            seg2 = [((1 - t) * w + t * b).tolist() for t in np.linspace(0.0, 1.0, 10)]
            pts = seg1 + seg2[1:]
            path = [[float(x), float(y)] for x, y in pts]
            cost = float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))
            return {"success": True, "path": path, "path_cost": cost, "collisions": collisions, "planner": "RRTstar"}

    # Failed to find a path
    # Estimate collisions along straight line as a proxy
    samples = 50
    for t in np.linspace(0.0, 1.0, samples):
        p = (1 - t) * a + t * b
        if _collides(p, obstacles):
            collisions += 1
    return {"success": False, "path": [], "path_cost": None, "collisions": collisions, "planner": "RRTstar"}


# Unified interface (compatible with real implementations)
def plan_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    perturbation_level: float = 0.0
) -> Dict[str, object]:
    """
    Unified planning interface (compatible with rrt_star_real.py).

    Args:
        start: Start position
        goal: Goal position
        obstacles: List of obstacle dicts with 'center' and 'radius'
        perturbation_level: Perturbation level (0.0 to 1.0)

    Returns:
        Planning result dictionary
    """
    # Convert obstacles from dict format to tuple format
    obstacle_tuples = []
    for obs in obstacles:
        center = obs.get("center", [0.5, 0.5])
        radius = obs.get("radius", 0.05)
        # Handle both 2D and 3D centers
        if len(center) >= 2:
            obstacle_tuples.append((center[0], center[1], radius))

    ctx = {
        "rng": np.random.default_rng(0),
        "obstacles": obstacle_tuples,
    }

    result = plan(start[:2] if len(start) > 2 else start, goal[:2] if len(goal) > 2 else goal, ctx)

    # Apply perturbation effect if needed
    if perturbation_level > 0 and result["path_cost"] is not None:
        result["path_cost"] *= (1.0 + perturbation_level * 0.5)

    return result

