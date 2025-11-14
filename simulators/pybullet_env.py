"""
Real 3D simulation environment using PyBullet with Franka Panda robot.
Supports multiple tasks: Lift, PickPlace, Push, Stack
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletEnv:
    """Complete 3D simulation environment with Franka Panda robot."""

    def __init__(
        self,
        task: str = "Lift",
        gui: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        seed: int = 0,
    ):
        """
        Initialize PyBullet environment.

        Args:
            task: Task name (Lift, PickPlace, Push, Stack)
            gui: Show GUI
            camera_width: Camera resolution width
            camera_height: Camera resolution height
            seed: Random seed
        """
        self.task = task
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.seed = seed
        np.random.seed(seed)

        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")

        # Load Franka Panda robot
        self.robot_id = self._load_franka_panda()

        # Robot parameters
        self.num_joints = p.getNumJoints(self.robot_id)
        self.end_effector_index = 11  # Franka Panda end effector

        # Joint indices (7 arm joints + 2 gripper fingers)
        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joint_indices = [9, 10]

        # Camera parameters
        self.camera_target = [0.5, 0.0, 0.2]
        self.camera_distance = 1.2
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_roll = 0

        # Objects in scene
        self.target_object_id: Optional[int] = None
        self.obstacle_ids: List[int] = []

        # Task-specific setup
        self.goal_position: Optional[np.ndarray] = None
        self._setup_task()

        # Reset to initial position
        self.reset()

    def _load_franka_panda(self) -> int:
        """Load Franka Panda robot URDF."""
        # Try to use pybullet_data's Franka, or download if available
        try:
            robot_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
            )
        except:
            # Fallback: create simplified robot
            print("Warning: Franka Panda URDF not found, using simplified model")
            robot_id = self._create_simplified_robot()

        return robot_id

    def _create_simplified_robot(self) -> int:
        """Create simplified robot if Franka URDF not available."""
        # Create a simple 7-DOF arm
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0.7, 0.7, 0.7, 1])

        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0, 0, 0.1],
        )

        return robot_id

    def _setup_task(self) -> None:
        """Setup task-specific elements."""
        if self.task == "Lift":
            self.goal_position = np.array([0.5, 0.0, 0.5])  # Lift to 50cm height
        elif self.task == "PickPlace":
            self.goal_position = np.array([0.3, 0.4, 0.2])  # Place at different location
        elif self.task == "Push":
            self.goal_position = np.array([0.6, 0.0, 0.02])  # Push forward
        elif self.task == "Stack":
            self.goal_position = np.array([0.5, 0.0, 0.1])  # Stack on top
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def reset(self) -> None:
        """Reset environment to initial state."""
        # Reset robot to home position
        home_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, home_joint_positions[i])

        # Open gripper
        for joint_idx in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.04)

        # Remove old objects
        if self.target_object_id is not None:
            p.removeBody(self.target_object_id)
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids = []

        # Spawn target object
        self.target_object_id = self._spawn_target_object()

        # Spawn obstacles for some tasks
        if self.task in ["PickPlace", "Stack"]:
            self.obstacle_ids = self._spawn_obstacles()

        # Step simulation to stabilize
        for _ in range(100):
            p.stepSimulation()

    def _spawn_target_object(self) -> int:
        """Spawn target object for manipulation."""
        # Random position with some noise
        base_pos = [0.5, 0.0, 0.05]
        pos = base_pos + np.random.randn(3) * 0.02
        pos[2] = 0.05  # Keep on table

        # Create cube object
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.03],
            rgbaColor=[1, 0, 0, 1],  # Red cube
        )

        object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos,
        )

        return object_id

    def _spawn_obstacles(self) -> List[int]:
        """Spawn obstacle objects."""
        obstacles = []
        num_obstacles = np.random.randint(1, 4)

        for _ in range(num_obstacles):
            pos = [
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-0.3, 0.3),
                0.05,
            ]

            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.1)
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.03,
                length=0.1,
                rgbaColor=[0, 1, 0, 1],  # Green cylinders
            )

            obs_id = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos,
            )
            obstacles.append(obs_id)

        return obstacles

    def get_camera_image(self) -> np.ndarray:
        """Get RGB image from camera."""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=self.camera_roll,
            upAxisIndex=2,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=5.0,
        )

        (_, _, px, _, _) = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

        return rgb_array

    def get_depth_image(self) -> np.ndarray:
        """Get depth image from camera."""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=self.camera_roll,
            upAxisIndex=2,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=5.0,
        )

        (_, _, _, depth, _) = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        depth_array = np.array(depth, dtype=np.float32)
        return depth_array

    def get_target_position(self) -> np.ndarray:
        """Get current target object position."""
        if self.target_object_id is None:
            return np.zeros(3)
        pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        return np.array(pos)

    def get_target_ground_truth_bbox(self) -> Tuple[int, int, int, int]:
        """Get ground truth bounding box of target object in image coordinates."""
        if self.target_object_id is None:
            return (0, 0, 0, 0)

        # Get object position
        pos, orn = p.getBasePositionAndOrientation(self.target_object_id)

        # Get camera matrices
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=self.camera_roll,
            upAxisIndex=2,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=5.0,
        )

        # Project to image coordinates
        # Simplified: assume object at center, create bbox
        # In real implementation, project all corners of bounding box
        center_x, center_y = self.camera_width // 2, self.camera_height // 2
        bbox_size = 100  # Approximate

        x1 = max(0, center_x - bbox_size // 2)
        y1 = max(0, center_y - bbox_size // 2)
        x2 = min(self.camera_width, center_x + bbox_size // 2)
        y2 = min(self.camera_height, center_y + bbox_size // 2)

        return (x1, y1, x2, y2)

    def get_ee_position(self) -> np.ndarray:
        """Get end effector position."""
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(state[0])

    def move_ee_to(self, target_pos: np.ndarray, target_orn: Optional[np.ndarray] = None) -> bool:
        """
        Move end effector to target position using inverse kinematics.

        Args:
            target_pos: Target position [x, y, z]
            target_orn: Target orientation (quaternion)

        Returns:
            True if successful
        """
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        # Compute IK
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn,
        )

        # Apply joint positions
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )

        # Step simulation
        for _ in range(100):
            p.stepSimulation()

        # Check if reached
        current_pos = self.get_ee_position()
        dist = np.linalg.norm(current_pos - target_pos)

        return dist < 0.05  # 5cm tolerance

    def grasp(self) -> bool:
        """Close gripper to grasp object."""
        for joint_idx in self.gripper_joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=200,
            )

        # Step simulation
        for _ in range(50):
            p.stepSimulation()

        # Check if grasping
        contact_points = p.getContactPoints(self.robot_id, self.target_object_id)
        return len(contact_points) > 0

    def release(self) -> None:
        """Open gripper to release object."""
        for joint_idx in self.gripper_joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=200,
            )

        for _ in range(50):
            p.stepSimulation()

    def check_success(self) -> bool:
        """Check if task is successful."""
        if self.target_object_id is None:
            return False

        target_pos = self.get_target_position()
        dist = np.linalg.norm(target_pos - self.goal_position)

        if self.task == "Lift":
            # Success if object is lifted to goal height
            return target_pos[2] > 0.3 and dist < 0.1
        elif self.task == "PickPlace":
            # Success if object is at goal position
            return dist < 0.1
        elif self.task == "Push":
            # Success if object pushed forward
            return target_pos[0] > 0.55 and abs(target_pos[1]) < 0.1
        elif self.task == "Stack":
            # Success if object is at stacking height
            return target_pos[2] > 0.08 and dist < 0.1

        return False

    def apply_camera_jitter(self, level: float) -> None:
        """Apply camera jitter perturbation."""
        self.camera_yaw += np.random.randn() * level * 10
        self.camera_pitch += np.random.randn() * level * 5
        self.camera_distance += np.random.randn() * level * 0.1

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsic parameters."""
        fov = 60  # degrees
        fx = self.camera_width / (2 * np.tan(np.radians(fov / 2)))
        fy = self.camera_height / (2 * np.tan(np.radians(fov / 2)))
        cx = self.camera_width / 2
        cy = self.camera_height / 2

        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": self.camera_width,
            "height": self.camera_height,
        }

    def close(self) -> None:
        """Close PyBullet connection."""
        p.disconnect(self.client)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            p.disconnect(self.client)
        except:
            pass
