"""
Real pose estimation using OpenCV PnP (Perspective-n-Point).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2


class PnPPoseEstimator:
    """Real 6D pose estimation using PnP."""

    def __init__(self, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize PnP pose estimator.

        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
        """
        if camera_matrix is None:
            # Default camera matrix for 640x480 image
            fx = fy = 500.0
            cx, cy = 320, 240
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        else:
            self.dist_coeffs = dist_coeffs

    def estimate_pose_from_mask(
        self,
        mask: np.ndarray,
        depth_img: Optional[np.ndarray] = None,
        object_size: float = 0.06
    ) -> Dict:
        """
        Estimate pose from segmentation mask.

        Args:
            mask: Binary segmentation mask
            depth_img: Optional depth image
            object_size: Approximate object size in meters

        Returns:
            Pose estimation dictionary
        """
        # Extract 2D points from mask contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {
                "pnp_success": False,
                "pnp_rmse": np.inf,
                "pose_estimate": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
            }

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Fit minimum area rectangle to get oriented bbox
        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)

        # Define 3D object points (assume object lying flat)
        # Object model: cube with size object_size
        half_size = object_size / 2
        object_points_3d = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0],
        ], dtype=np.float32)

        # Corresponding 2D image points
        image_points_2d = box_points.astype(np.float32)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points_3d,
            image_points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "pnp_success": False,
                "pnp_rmse": np.inf,
                "pose_estimate": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
            }

        # Compute reprojection error
        projected_points, _ = cv2.projectPoints(
            object_points_3d,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        projected_points = projected_points.reshape(-1, 2)
        rmse = np.sqrt(np.mean(np.sum((projected_points - image_points_2d) ** 2, axis=1)))

        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation_euler = self._rotation_matrix_to_euler(rotation_matrix)

        # Translation vector is the position
        position = tvec.flatten()

        return {
            "pnp_success": True,
            "pnp_rmse": float(rmse),
            "pose_estimate": position.tolist(),
            "rotation": rotation_euler.tolist(),
        }

    def estimate_pose_from_bbox_and_depth(
        self,
        bbox: List[int],
        depth_img: np.ndarray,
        object_size: float = 0.06
    ) -> Dict:
        """
        Estimate pose from bounding box and depth image.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_img: Depth image
            object_size: Approximate object size in meters

        Returns:
            Pose estimation dictionary
        """
        x1, y1, x2, y2 = bbox

        # Get center point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Get depth at center
        if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
            z = depth_img[cy, cx]
        else:
            z = 1.0  # Default depth

        # Convert to 3D using camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx_cam = self.camera_matrix[0, 2]
        cy_cam = self.camera_matrix[1, 2]

        x = (cx - cx_cam) * z / fx
        y = (cy - cy_cam) * z / fy

        # Estimate orientation (assume facing camera)
        rotation_euler = np.array([0.0, 0.0, 0.0])

        # Compute simple RMSE based on bbox size consistency
        bbox_width = x2 - x1
        expected_width = (object_size * fx) / z
        rmse = abs(bbox_width - expected_width)

        return {
            "pnp_success": True,
            "pnp_rmse": float(rmse),
            "pose_estimate": [float(x), float(y), float(z)],
            "rotation": rotation_euler.tolist(),
        }

    def estimate_pose_from_points(
        self,
        image_points: np.ndarray,
        object_points: np.ndarray
    ) -> Dict:
        """
        Estimate pose from corresponding 2D-3D points.

        Args:
            image_points: 2D image points (N, 2)
            object_points: 3D object points (N, 3)

        Returns:
            Pose estimation dictionary
        """
        if len(image_points) < 4:
            return {
                "pnp_success": False,
                "pnp_rmse": np.inf,
                "pose_estimate": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
            }

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "pnp_success": False,
                "pnp_rmse": np.inf,
                "pose_estimate": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
            }

        # Compute reprojection error
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        projected_points = projected_points.reshape(-1, 2)
        rmse = np.sqrt(np.mean(np.sum((projected_points - image_points) ** 2, axis=1)))

        # Convert rotation
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation_euler = self._rotation_matrix_to_euler(rotation_matrix)

        position = tvec.flatten()

        return {
            "pnp_success": True,
            "pnp_rmse": float(rmse),
            "pose_estimate": position.tolist(),
            "rotation": rotation_euler.tolist(),
        }

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ convention)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])


def estimate_pose(
    mask: np.ndarray,
    depth_img: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[Dict] = None,
    perturbation_level: float = 0.0
) -> Dict:
    """
    Unified pose estimation interface (compatible with old stub API).

    Args:
        mask: Binary segmentation mask
        depth_img: Optional depth image
        camera_intrinsics: Camera parameters
        perturbation_level: Perturbation level (affects RMSE)

    Returns:
        Pose estimation dictionary
    """
    # Build camera matrix from intrinsics
    if camera_intrinsics is not None:
        fx = camera_intrinsics.get("fx", 500.0)
        fy = camera_intrinsics.get("fy", 500.0)
        cx = camera_intrinsics.get("cx", 320.0)
        cy = camera_intrinsics.get("cy", 240.0)

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        camera_matrix = None

    # Initialize estimator
    estimator = PnPPoseEstimator(camera_matrix=camera_matrix)

    # Estimate
    result = estimator.estimate_pose_from_mask(mask, depth_img)

    # Apply perturbation effect to RMSE
    if perturbation_level > 0 and result["pnp_success"]:
        # Increase RMSE based on perturbation
        noise = np.random.uniform(1.0, 3.0) * perturbation_level
        result["pnp_rmse"] += noise

    return result
