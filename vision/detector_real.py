"""
Real object detector using YOLOv8.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


class YOLODetector:
    """Real object detector using YOLOv8."""

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.25):
        """
        Initialize YOLO detector.

        Args:
            model_size: Model size (n, s, m, l, x) - n is fastest
            conf_threshold: Confidence threshold for detections
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.conf_threshold = conf_threshold

        # Load YOLOv8 model
        model_name = f"yolov8{model_size}.pt"
        self.model = YOLO(model_name)

        # Target classes (COCO dataset)
        # We're interested in objects that could be manipulated
        self.target_classes = [
            39,  # bottle
            41,  # cup
            42,  # fork
            43,  # knife
            44,  # spoon
            45,  # bowl
            46,  # banana
            47,  # apple
            48,  # sandwich
            49,  # orange
            73,  # book
            74,  # clock
            75,  # vase
        ]

    def detect(
        self,
        img: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict:
        """
        Detect objects in image.

        Args:
            img: RGB image (H, W, 3)
            target_class: Optional specific class to detect

        Returns:
            Detection results dictionary
        """
        # Run inference
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return {
                "detected": False,
                "avg_conf": 0.0,
                "bbox": [0, 0, 0, 0],
                "class_id": -1,
                "class_name": "none",
            }

        # Get detections
        boxes = results[0].boxes

        # Filter by target classes if specified
        if target_class is not None:
            mask = boxes.cls == target_class
        else:
            # Filter to manipulable objects
            mask = np.isin(boxes.cls.cpu().numpy(), self.target_classes)

        if not mask.any():
            # If no target objects, take highest confidence detection
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            box = boxes[best_idx]
        else:
            # Take highest confidence target object
            filtered_boxes = boxes[mask]
            confidences = filtered_boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            box = filtered_boxes[best_idx]

        # Extract bbox (xyxy format)
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = self.model.names[class_id]

        return {
            "detected": True,
            "avg_conf": confidence,
            "bbox": bbox.tolist(),  # [x1, y1, x2, y2]
            "class_id": class_id,
            "class_name": class_name,
        }

    def detect_all(self, img: np.ndarray) -> List[Dict]:
        """
        Detect all objects in image.

        Args:
            img: RGB image (H, W, 3)

        Returns:
            List of detection dictionaries
        """
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        detections = []

        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            detections.append({
                "detected": True,
                "avg_conf": confidence,
                "bbox": bbox.tolist(),
                "class_id": class_id,
                "class_name": class_name,
            })

        return detections

    def visualize(self, img: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Visualize detection on image.

        Args:
            img: RGB image
            detection: Detection dictionary

        Returns:
            Image with detection visualized
        """
        import cv2

        img_vis = img.copy()

        if not detection["detected"]:
            return img_vis

        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox

        # Draw bbox
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{detection['class_name']}: {detection['avg_conf']:.2f}"
        cv2.putText(
            img_vis,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        return img_vis


def detect(img: np.ndarray, perturbation_level: float = 0.0) -> Dict:
    """
    Unified detection interface (compatible with old stub API).

    Args:
        img: RGB image
        perturbation_level: Perturbation level (affects confidence)

    Returns:
        Detection dictionary
    """
    # Initialize detector (cached in production)
    detector = YOLODetector(model_size="n")

    # Detect
    result = detector.detect(img)

    # Apply perturbation effect to confidence
    if perturbation_level > 0 and result["detected"]:
        # Degrade confidence based on perturbation
        degradation = np.random.uniform(0.5, 1.0) * perturbation_level
        result["avg_conf"] *= (1.0 - degradation)
        result["avg_conf"] = max(0.0, result["avg_conf"])

    return result
