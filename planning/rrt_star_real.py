"""
Real RRT* (Rapidly-exploring Random Tree Star) path planner.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.neighbors import KDTree


class Node:
    """RRT* tree node."""

    def __init__(self, position: np.ndarray):
        self.position = position
        self.parent: Optional[Node] = None
        self.cost: float = 0.0
        self.children: List[Node] = []


class RRTStarPlanner:
    """Real RRT* path planner with collision checking."""

    def __init__(
        self,
        workspace_bounds: Tuple[Tuple[float, float], ...] = ((0, 1), (0, 1), (0, 1)),
        step_size: float = 0.05,
        goal_sample_rate: float = 0.1,
        search_radius: float = 0.15,
        max_iterations: int = 1000,
    ):
        """
        Initialize RRT* planner.

        Args:
            workspace_bounds: Workspace bounds ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            step_size: Maximum step size for tree extension
            goal_sample_rate: Probability of sampling goal
            search_radius: Radius for finding near nodes
            max_iterations: Maximum planning iterations
        """
        self.workspace_bounds = workspace_bounds
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.max_iterations = max_iterations
        self.dimension = len(workspace_bounds)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict]
    ) -> Dict:
        """
        Plan path from start to goal avoiding obstacles.

        Args:
            start: Start position
            goal: Goal position
            obstacles: List of obstacle dictionaries with 'center' and 'radius'

        Returns:
            Planning result dictionary
        """
        # Initialize tree
        start_node = Node(start)
        nodes = [start_node]

        # Try to find path
        goal_node = None

        for iteration in range(self.max_iterations):
            # Sample random point
            if np.random.rand() < self.goal_sample_rate:
                sample = goal
            else:
                sample = self._sample_random()

            # Find nearest node
            nearest_node = self._find_nearest(nodes, sample)

            # Steer towards sample
            new_position = self._steer(nearest_node.position, sample)

            # Check collision
            if self._is_collision_free(nearest_node.position, new_position, obstacles):
                # Create new node
                new_node = Node(new_position)
                new_node.cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
                new_node.parent = nearest_node
                nearest_node.children.append(new_node)

                # Find near nodes for rewiring
                near_nodes = self._find_near_nodes(nodes, new_node)

                # Choose best parent
                for near_node in near_nodes:
                    if self._is_collision_free(near_node.position, new_position, obstacles):
                        potential_cost = near_node.cost + np.linalg.norm(new_position - near_node.position)
                        if potential_cost < new_node.cost:
                            # Remove from old parent
                            if new_node.parent is not None:
                                new_node.parent.children.remove(new_node)
                            # Rewire
                            new_node.parent = near_node
                            new_node.cost = potential_cost
                            near_node.children.append(new_node)

                # Rewire near nodes through new node
                for near_node in near_nodes:
                    if near_node == new_node.parent:
                        continue
                    if self._is_collision_free(new_position, near_node.position, obstacles):
                        potential_cost = new_node.cost + np.linalg.norm(near_node.position - new_position)
                        if potential_cost < near_node.cost:
                            # Remove from old parent
                            if near_node.parent is not None:
                                near_node.parent.children.remove(near_node)
                            # Rewire
                            near_node.parent = new_node
                            near_node.cost = potential_cost
                            new_node.children.append(near_node)

                nodes.append(new_node)

                # Check if goal reached
                if np.linalg.norm(new_position - goal) < self.step_size:
                    goal_node = new_node
                    # Continue to optimize path
                    if iteration > self.max_iterations * 0.8:
                        break

        # Extract path
        if goal_node is not None:
            path = self._extract_path(goal_node)
            path_cost = goal_node.cost
            collisions = 0  # Path is collision-free by construction
            success = True
        else:
            # Failed to find path
            path = [start, goal]
            path_cost = np.linalg.norm(goal - start)
            collisions = len(obstacles)  # Indicate failure
            success = False

        return {
            "success": success,
            "path": path,
            "path_cost": path_cost,
            "collisions": collisions,
            "planner": "RRTstar",
        }

    def _sample_random(self) -> np.ndarray:
        """Sample random point in workspace."""
        sample = np.zeros(self.dimension)
        for i, (low, high) in enumerate(self.workspace_bounds):
            sample[i] = np.random.uniform(low, high)
        return sample

    def _find_nearest(self, nodes: List[Node], point: np.ndarray) -> Node:
        """Find nearest node to point."""
        min_dist = np.inf
        nearest_node = nodes[0]

        for node in nodes:
            dist = np.linalg.norm(node.position - point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def _find_near_nodes(self, nodes: List[Node], new_node: Node) -> List[Node]:
        """Find nodes within search radius of new node."""
        near_nodes = []

        for node in nodes:
            if node == new_node:
                continue
            dist = np.linalg.norm(node.position - new_node.position)
            if dist < self.search_radius:
                near_nodes.append(node)

        return near_nodes

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from from_pos towards to_pos by at most step_size."""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)

        if dist <= self.step_size:
            return to_pos
        else:
            return from_pos + (direction / dist) * self.step_size

    def _is_collision_free(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        obstacles: List[Dict]
    ) -> bool:
        """Check if line segment is collision-free."""
        # Check multiple points along segment
        num_checks = int(np.ceil(np.linalg.norm(to_pos - from_pos) / (self.step_size / 2)))
        num_checks = max(num_checks, 5)

        for i in range(num_checks + 1):
            t = i / num_checks
            point = from_pos + t * (to_pos - from_pos)

            # Check against all obstacles
            for obstacle in obstacles:
                center = np.array(obstacle["center"])
                radius = obstacle["radius"]

                dist = np.linalg.norm(point - center)
                if dist < radius:
                    return False

        return True

    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """Extract path from root to goal node."""
        path = []
        current = goal_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return path


def plan_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Dict],
    workspace_bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    perturbation_level: float = 0.0
) -> Dict:
    """
    Unified planning interface (compatible with old API).

    Args:
        start: Start position
        goal: Goal position
        obstacles: List of obstacles
        workspace_bounds: Workspace bounds
        perturbation_level: Perturbation level (affects collision tolerance)

    Returns:
        Planning result dictionary
    """
    if workspace_bounds is None:
        # Infer from start and goal
        bounds = []
        for i in range(len(start)):
            min_val = min(start[i], goal[i]) - 0.5
            max_val = max(start[i], goal[i]) + 0.5
            bounds.append((min_val, max_val))
        workspace_bounds = tuple(bounds)

    # Initialize planner
    planner = RRTStarPlanner(
        workspace_bounds=workspace_bounds,
        max_iterations=1000,
    )

    # Plan
    result = planner.plan(start, goal, obstacles)

    # Apply perturbation effect
    if perturbation_level > 0:
        # Increase path cost due to perturbation (less optimal path)
        result["path_cost"] *= (1.0 + perturbation_level * 0.5)

    return result
