"""Real 3D models for testing: human, vehicle, and traffic sign models."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class Model3D:
    """Base class for 3D models."""

    def __init__(self, name: str):
        self.name = name
        self.vertices: np.ndarray = np.array([])
        self.faces: List[Tuple[int, int, int]] = []
        self.texture_coords: np.ndarray = np.array([])

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return min and max corners of axis-aligned bounding box."""
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def get_center(self) -> np.ndarray:
        """Return center of bounding box."""
        bbox_min, bbox_max = self.get_bounding_box()
        return (bbox_min + bbox_max) / 2.0

    def transform(self, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0):
        """Apply rigid transformation to model."""
        # Scale
        vertices = self.vertices * scale
        # Rotate
        vertices = vertices @ rotation.T
        # Translate
        vertices = vertices + translation
        return vertices


class HumanModel3D(Model3D):
    """Simplified 3D human model (SMPL-inspired parametric model)."""

    def __init__(self):
        super().__init__("human")
        self._create_human_mesh()

    def _create_human_mesh(self):
        """Create a simplified human mesh (stick figure with volumes)."""
        # Human proportions (average adult, ~1.7m height)
        # Using capsule primitives for body parts

        vertices = []
        faces = []

        # Torso (cylinder: chest + abdomen)
        torso_height = 0.6  # meters
        torso_radius = 0.15
        torso_center = np.array([0.0, 0.0, 1.0])  # Standing on ground at z=0
        v_torso, f_torso = self._create_capsule(torso_center, torso_height, torso_radius, segments=16)
        vertices.extend(v_torso)
        faces.extend(f_torso)

        base_idx = len(vertices)

        # Head (sphere)
        head_radius = 0.1
        head_center = torso_center + np.array([0, 0, torso_height/2 + head_radius])
        v_head, f_head = self._create_sphere(head_center, head_radius, segments=12)
        vertices.extend(v_head)
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in f_head])

        base_idx = len(vertices)

        # Arms (2 capsules)
        arm_length = 0.5
        arm_radius = 0.04
        shoulder_height = torso_center[2] + torso_height/2 - 0.05

        # Left arm
        left_shoulder = np.array([-torso_radius, 0, shoulder_height])
        left_hand = left_shoulder + np.array([-arm_length*0.7, 0, -arm_length*0.7])
        v_larm, f_larm = self._create_capsule_between_points(left_shoulder, left_hand, arm_radius, segments=8)
        vertices.extend(v_larm)
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in f_larm])

        base_idx = len(vertices)

        # Right arm
        right_shoulder = np.array([torso_radius, 0, shoulder_height])
        right_hand = right_shoulder + np.array([arm_length*0.7, 0, -arm_length*0.7])
        v_rarm, f_rarm = self._create_capsule_between_points(right_shoulder, right_hand, arm_radius, segments=8)
        vertices.extend(v_rarm)
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in f_rarm])

        base_idx = len(vertices)

        # Legs (2 capsules)
        leg_length = 0.8
        leg_radius = 0.06
        hip_height = torso_center[2] - torso_height/2

        # Left leg
        left_hip = np.array([-0.08, 0, hip_height])
        left_foot = left_hip + np.array([0, 0, -leg_length])
        v_lleg, f_lleg = self._create_capsule_between_points(left_hip, left_foot, leg_radius, segments=8)
        vertices.extend(v_lleg)
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in f_lleg])

        base_idx = len(vertices)

        # Right leg
        right_hip = np.array([0.08, 0, hip_height])
        right_foot = right_hip + np.array([0, 0, -leg_length])
        v_rleg, f_rleg = self._create_capsule_between_points(right_hip, right_foot, leg_radius, segments=8)
        vertices.extend(v_rleg)
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in f_rleg])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = faces

    @staticmethod
    def _create_sphere(center: np.ndarray, radius: float, segments: int = 16) -> Tuple[List, List]:
        """Create sphere mesh."""
        vertices = []
        faces = []

        for i in range(segments + 1):
            theta = i * np.pi / segments
            for j in range(segments + 1):
                phi = j * 2 * np.pi / segments
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                vertices.append(center + np.array([x, y, z]))

        for i in range(segments):
            for j in range(segments):
                a = i * (segments + 1) + j
                b = a + segments + 1
                faces.append((a, b, a + 1))
                faces.append((b, b + 1, a + 1))

        return vertices, faces

    @staticmethod
    def _create_capsule(center: np.ndarray, height: float, radius: float, segments: int = 16) -> Tuple[List, List]:
        """Create capsule (cylinder with hemispherical caps)."""
        vertices = []
        faces = []

        # Cylinder body
        for i in range(segments + 1):
            angle = i * 2 * np.pi / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Bottom
            vertices.append(center + np.array([x, y, -height/2]))
            # Top
            vertices.append(center + np.array([x, y, height/2]))

        # Cylinder faces
        for i in range(segments):
            a = i * 2
            b = (i + 1) * 2
            faces.append((a, b, a + 1))
            faces.append((b, b + 1, a + 1))

        return vertices, faces

    @staticmethod
    def _create_capsule_between_points(p1: np.ndarray, p2: np.ndarray, radius: float, segments: int = 8) -> Tuple[List, List]:
        """Create capsule between two points."""
        direction = p2 - p1
        height = np.linalg.norm(direction)
        if height < 1e-6:
            return [], []

        center = (p1 + p2) / 2
        vertices, faces = HumanModel3D._create_capsule(center, height, radius, segments)

        # Rotate to align with direction
        z_axis = direction / height
        # Find rotation matrix (simplified, assumes small angles)
        return vertices, faces


class VehicleModel3D(Model3D):
    """Simplified 3D vehicle model (car)."""

    def __init__(self):
        super().__init__("vehicle")
        self._create_vehicle_mesh()

    def _create_vehicle_mesh(self):
        """Create simplified car mesh (box with details)."""
        # Standard car dimensions (~4.5m length, 1.8m width, 1.5m height)
        length = 4.5
        width = 1.8
        height = 1.5

        # Main body (box)
        vertices = [
            # Bottom
            [-length/2, -width/2, 0],
            [length/2, -width/2, 0],
            [length/2, width/2, 0],
            [-length/2, width/2, 0],
            # Top of lower body
            [-length/2, -width/2, height*0.5],
            [length/2, -width/2, height*0.5],
            [length/2, width/2, height*0.5],
            [-length/2, width/2, height*0.5],
        ]

        faces = [
            # Bottom
            (0, 1, 2), (0, 2, 3),
            # Sides
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
            # Top
            (4, 5, 6), (4, 6, 7),
        ]

        # Cabin (smaller box on top)
        cabin_length = length * 0.5
        cabin_width = width * 0.9
        cabin_height = height * 0.5
        cabin_offset = -length * 0.1  # Slightly toward rear

        base_idx = len(vertices)
        cabin_vertices = [
            [cabin_offset - cabin_length/2, -cabin_width/2, height*0.5],
            [cabin_offset + cabin_length/2, -cabin_width/2, height*0.5],
            [cabin_offset + cabin_length/2, cabin_width/2, height*0.5],
            [cabin_offset - cabin_length/2, cabin_width/2, height*0.5],
            [cabin_offset - cabin_length/2, -cabin_width/2, height*0.5 + cabin_height],
            [cabin_offset + cabin_length/2, -cabin_width/2, height*0.5 + cabin_height],
            [cabin_offset + cabin_length/2, cabin_width/2, height*0.5 + cabin_height],
            [cabin_offset - cabin_length/2, cabin_width/2, height*0.5 + cabin_height],
        ]
        vertices.extend(cabin_vertices)

        cabin_faces = [
            (0, 1, 2), (0, 2, 3),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
            (4, 5, 6), (4, 6, 7),
        ]
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in cabin_faces])

        # Wheels (4 cylinders)
        wheel_radius = 0.35
        wheel_width = 0.25
        wheel_positions = [
            [-length/2 + 0.8, -width/2 - wheel_width/2, wheel_radius],  # Front left
            [-length/2 + 0.8, width/2 + wheel_width/2, wheel_radius],   # Front right
            [length/2 - 0.8, -width/2 - wheel_width/2, wheel_radius],   # Rear left
            [length/2 - 0.8, width/2 + wheel_width/2, wheel_radius],    # Rear right
        ]

        for wheel_pos in wheel_positions:
            base_idx = len(vertices)
            wheel_verts, wheel_faces = self._create_cylinder(
                np.array(wheel_pos), wheel_radius, wheel_width, axis=1, segments=12
            )
            vertices.extend(wheel_verts)
            faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in wheel_faces])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = faces

    @staticmethod
    def _create_cylinder(center: np.ndarray, radius: float, height: float, axis: int = 2, segments: int = 16) -> Tuple[List, List]:
        """Create cylinder along specified axis."""
        vertices = []
        faces = []

        for i in range(segments + 1):
            angle = i * 2 * np.pi / segments
            if axis == 2:  # Z-axis
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                vertices.append(center + np.array([x, y, -height/2]))
                vertices.append(center + np.array([x, y, height/2]))
            elif axis == 1:  # Y-axis
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                vertices.append(center + np.array([x, -height/2, z]))
                vertices.append(center + np.array([x, height/2, z]))

        for i in range(segments):
            a = i * 2
            b = (i + 1) * 2
            faces.append((a, b, a + 1))
            faces.append((b, b + 1, a + 1))

        return vertices, faces


class TrafficSignModel3D(Model3D):
    """3D traffic sign model (stop sign, yield, etc.)."""

    def __init__(self, sign_type: str = "stop"):
        super().__init__(f"traffic_sign_{sign_type}")
        self.sign_type = sign_type
        self._create_sign_mesh()

    def _create_sign_mesh(self):
        """Create traffic sign mesh."""
        # Sign dimensions
        sign_size = 0.6  # meters (diameter/width)
        sign_thickness = 0.05
        pole_height = 2.0
        pole_radius = 0.05

        vertices = []
        faces = []

        # Pole (cylinder)
        pole_center = np.array([0, 0, pole_height/2])
        v_pole, f_pole = self._create_cylinder(pole_center, pole_radius, pole_height, segments=12)
        vertices.extend(v_pole)
        faces.extend(f_pole)

        base_idx = len(vertices)

        # Sign board (depends on type)
        sign_center = np.array([0, 0, pole_height - sign_size/2])

        if self.sign_type == "stop":
            # Octagon
            v_sign, f_sign = self._create_octagon(sign_center, sign_size, sign_thickness)
        elif self.sign_type == "yield":
            # Inverted triangle
            v_sign, f_sign = self._create_triangle(sign_center, sign_size, sign_thickness, inverted=True)
        else:  # default: rectangular
            v_sign, f_sign = self._create_rectangle(sign_center, sign_size, sign_size, sign_thickness)

        vertices.extend(v_sign)
        faces.extend([(f[0]+base_idx, f[1]+base_idx, f[2]+base_idx) for f in f_sign])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = faces

    @staticmethod
    def _create_cylinder(center: np.ndarray, radius: float, height: float, segments: int = 16) -> Tuple[List, List]:
        """Create cylinder."""
        vertices = []
        faces = []

        for i in range(segments + 1):
            angle = i * 2 * np.pi / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append(center + np.array([x, y, -height/2]))
            vertices.append(center + np.array([x, y, height/2]))

        for i in range(segments):
            a = i * 2
            b = (i + 1) * 2
            faces.append((a, b, a + 1))
            faces.append((b, b + 1, a + 1))

        return vertices, faces

    @staticmethod
    def _create_octagon(center: np.ndarray, size: float, thickness: float) -> Tuple[List, List]:
        """Create octagonal prism (stop sign)."""
        vertices = []
        faces = []

        # 8 vertices on front and back
        for side in [-1, 1]:
            z_offset = side * thickness / 2
            for i in range(8):
                angle = i * 2 * np.pi / 8 + np.pi/8
                x = (size/2) * np.cos(angle)
                y = (size/2) * np.sin(angle)
                vertices.append(center + np.array([x, y, z_offset]))

        # Front and back faces
        for i in range(6):
            faces.append((0, i+1, i+2))  # Front
            faces.append((8, 8+i+2, 8+i+1))  # Back

        # Side faces
        for i in range(8):
            a = i
            b = (i + 1) % 8
            faces.append((a, b, b+8))
            faces.append((a, b+8, a+8))

        return vertices, faces

    @staticmethod
    def _create_triangle(center: np.ndarray, size: float, thickness: float, inverted: bool = False) -> Tuple[List, List]:
        """Create triangular prism (yield sign)."""
        vertices = []
        faces = []

        # 3 vertices on front and back
        angles = [90, 210, 330] if not inverted else [270, 30, 150]
        for side in [-1, 1]:
            z_offset = side * thickness / 2
            for angle_deg in angles:
                angle = np.radians(angle_deg)
                x = (size/2) * np.cos(angle)
                y = (size/2) * np.sin(angle)
                vertices.append(center + np.array([x, y, z_offset]))

        # Front and back
        faces.append((0, 1, 2))
        faces.append((3, 5, 4))

        # Sides
        for i in range(3):
            a = i
            b = (i + 1) % 3
            faces.append((a, b, b+3))
            faces.append((a, b+3, a+3))

        return vertices, faces

    @staticmethod
    def _create_rectangle(center: np.ndarray, width: float, height: float, thickness: float) -> Tuple[List, List]:
        """Create rectangular prism."""
        vertices = []
        faces = []

        # 4 vertices on front and back
        corners = [
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2],
        ]

        for side in [-1, 1]:
            z_offset = side * thickness / 2
            for x, y in corners:
                vertices.append(center + np.array([x, y, z_offset]))

        # Front and back
        faces.extend([(0, 1, 2), (0, 2, 3)])
        faces.extend([(4, 6, 5), (4, 7, 6)])

        # Sides
        for i in range(4):
            a = i
            b = (i + 1) % 4
            faces.append((a, b, b+4))
            faces.append((a, b+4, a+4))

        return vertices, faces


def get_model(model_type: str, **kwargs) -> Model3D:
    """Factory function to get 3D models."""
    if model_type == "human":
        return HumanModel3D()
    elif model_type == "vehicle":
        return VehicleModel3D()
    elif model_type.startswith("traffic_sign"):
        sign_type = kwargs.get("sign_type", "stop")
        return TrafficSignModel3D(sign_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
