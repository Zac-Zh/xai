"""
Diagnostic Interface

This module provides a unified interface for using the trained diagnostic VLM
to predict failure causes from video observations.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch required for diagnostic interface")

from diagnostic.train_diagnostic_vlm import SimpleDiagnosticVLM
from diagnostic.label_to_text import LabelToTextConverter


class DiagnosticInterface:
    """
    Interface for diagnosing robotic failures using the trained VLM.

    This is the final product of the Robo-Oracle system - a model that can
    look at a video of an opaque policy failing and predict the causal reason.
    """

    def __init__(
        self,
        model_checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 224
    ):
        """
        Initialize the diagnostic interface.

        Args:
            model_checkpoint: Path to trained diagnostic VLM checkpoint
            device: Device to run inference on
            image_size: Image size for preprocessing
        """
        self.device = torch.device(device)
        self.image_size = image_size

        # Load model
        print(f"Loading diagnostic model from {model_checkpoint}...")
        self.model = SimpleDiagnosticVLM()

        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.converter = LabelToTextConverter()

        print("Diagnostic interface ready!")

    @torch.no_grad()
    def diagnose_failure(
        self,
        video_frames: List[np.ndarray],
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Diagnose a failure from video frames.

        Args:
            video_frames: List of video frames as numpy arrays (H, W, 3)
            return_confidence: Whether to return confidence scores

        Returns:
            Diagnosis dictionary with predicted failure module and explanation
        """
        # Preprocess frames (use middle frame or average representation)
        if len(video_frames) == 0:
            raise ValueError("No video frames provided")

        # Use middle frame
        middle_idx = len(video_frames) // 2
        frame = video_frames[middle_idx]

        # Preprocess
        image_tensor = self._preprocess_image(frame).unsqueeze(0).to(self.device)

        # Predict
        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        # Get prediction
        pred_idx = torch.argmax(probabilities).item()
        predicted_module = self.model.idx_to_module[pred_idx]
        confidence = probabilities[pred_idx].item()

        # Generate explanation
        explanation = self._generate_explanation(
            predicted_module,
            confidence
        )

        result = {
            "predicted_module": predicted_module,
            "confidence": confidence,
            "explanation": explanation,
            "video_frames_analyzed": len(video_frames)
        }

        if return_confidence:
            result["all_confidences"] = {
                self.model.idx_to_module[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }

        return result

    def diagnose_from_paths(
        self,
        frame_paths: List[str],
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Diagnose from a list of frame file paths.

        Args:
            frame_paths: List of paths to frame files (.npy or .png)
            return_confidence: Whether to return confidence scores

        Returns:
            Diagnosis dictionary
        """
        frames = []
        for path in frame_paths:
            if path.endswith('.npy'):
                frame = np.load(path)
            else:
                frame = np.array(Image.open(path))
            frames.append(frame)

        return self.diagnose_failure(frames, return_confidence)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for the model."""
        # Convert to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Resize
        pil_image = pil_image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        # To array and normalize
        image_array = np.array(pil_image).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        # To tensor
        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1)
        ).float()

        return image_tensor

    def _generate_explanation(
        self,
        predicted_module: str,
        confidence: float
    ) -> str:
        """Generate natural language explanation."""
        confidence_pct = confidence * 100

        explanations = {
            "Perception": (
                f"The failure is attributed to the Vision/Perception module "
                f"(confidence: {confidence_pct:.1f}%). "
                f"This suggests issues with object detection, segmentation, or "
                f"visual feature extraction. Possible causes include occlusion, "
                f"poor lighting, motion blur, or low detection confidence."
            ),
            "Geometry": (
                f"The failure is attributed to the Geometry/Pose Estimation module "
                f"(confidence: {confidence_pct:.1f}%). "
                f"This suggests issues with estimating the 3D pose of objects. "
                f"Possible causes include PnP algorithm failure, high reprojection "
                f"error, or misalignment between the object model and visual observations."
            ),
            "Planning": (
                f"The failure is attributed to the Motion Planning module "
                f"(confidence: {confidence_pct:.1f}%). "
                f"This suggests the planner could not find a valid path to the target. "
                f"Possible causes include obstacles in the workspace, collision "
                f"predictions, or excessive path cost due to geometric constraints."
            ),
            "Control": (
                f"The failure is attributed to the Control/Trajectory Tracking module "
                f"(confidence: {confidence_pct:.1f}%). "
                f"This suggests issues with executing the planned trajectory. "
                f"Possible causes include high tracking error, overshoot, oscillation, "
                f"or poor controller tuning."
            )
        }

        return explanations.get(
            predicted_module,
            f"Unknown failure module: {predicted_module}"
        )

    def compare_with_oracle(
        self,
        video_frames: List[np.ndarray],
        oracle_label: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Diagnose and compare with Oracle ground truth.

        Useful for evaluation and analysis.

        Args:
            video_frames: Video frames
            oracle_label: Ground truth label from Classical Oracle

        Returns:
            Comparison results
        """
        diagnosis = self.diagnose_failure(video_frames)

        oracle_module = oracle_label.get("primary_failure_module", "Unknown")
        predicted_module = diagnosis["predicted_module"]

        match = (oracle_module == predicted_module)

        return {
            "diagnosis": diagnosis,
            "oracle_label": oracle_label,
            "match": match,
            "oracle_module": oracle_module,
            "predicted_module": predicted_module,
            "oracle_explanation": self.converter.convert_to_text(oracle_label)
        }


def diagnose_failure(
    model_checkpoint: str,
    frame_paths: List[str]
) -> Dict[str, Any]:
    """
    Convenience function to diagnose a failure.

    Args:
        model_checkpoint: Path to trained model
        frame_paths: List of frame file paths

    Returns:
        Diagnosis results
    """
    interface = DiagnosticInterface(model_checkpoint)
    return interface.diagnose_from_paths(frame_paths)
