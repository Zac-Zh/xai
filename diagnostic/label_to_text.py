"""
Label to Text Conversion

This module converts programmatic Oracle failure labels into natural language
descriptions for VLM training.
"""
from __future__ import annotations

from typing import Dict, Any, List


class LabelToTextConverter:
    """
    Converts programmatic failure labels to natural language descriptions.

    This is crucial for training the diagnostic VLM - it transforms the
    Oracle's structured labels into training data for language understanding.
    """

    # Error code descriptions
    ERROR_DESCRIPTIONS = {
        # Perception errors
        "miss_detection": "the target object was not detected in the visual input",
        "low_conf": "the object detection confidence was below the acceptable threshold",
        "seg_iou_below_tau": "the segmentation quality (IoU) was insufficient",

        # Geometry errors
        "align_fail": "the pose estimation algorithm failed to align the object model",
        "pnp_reproj_high": "the PnP reprojection error exceeded acceptable limits, indicating inaccurate pose estimation",

        # Planning errors
        "no_path": "the motion planner (RRT*) could not find a collision-free path to the target",
        "collision_pred": "the planned path contained predicted collisions with obstacles",
        "excess_cost": "the path cost was excessively high, suggesting a suboptimal or difficult trajectory",

        # Control errors
        "tracking_rmse_high": "the trajectory tracking error (RMSE) was too high, indicating poor controller performance",
        "overshoot_high": "the controller exhibited significant overshoot beyond the target",
        "oscillation": "the controller exhibited oscillatory behavior around the target"
    }

    # Module-level descriptions
    MODULE_DESCRIPTIONS = {
        "Perception": "the Vision/Perception module",
        "Geometry": "the Geometry/Pose Estimation module",
        "Planning": "the Motion Planning module",
        "Control": "the Control/Trajectory Tracking module"
    }

    def convert_to_text(
        self,
        failure_label: Dict[str, Any],
        include_technical_details: bool = True,
        include_all_errors: bool = False
    ) -> str:
        """
        Convert a failure label to natural language.

        Args:
            failure_label: The Oracle's programmatic failure label
            include_technical_details: Whether to include technical metrics
            include_all_errors: Whether to describe all errors or just primary

        Returns:
            Natural language description of the failure
        """
        if not failure_label or not failure_label.get("failure_detected"):
            return "The policy executed successfully with no detected failures."

        primary_module = failure_label.get("primary_failure_module", "Unknown")
        primary_error = failure_label.get("primary_error_code", "unknown")
        errors = failure_label.get("errors", [])
        root_cause = failure_label.get("root_cause", "Unknown")

        # Start with the primary failure
        text_parts = []

        # Opening statement
        text_parts.append(
            f"The robotic manipulation task failed. "
            f"The root cause analysis attributes this failure to "
            f"{self.MODULE_DESCRIPTIONS.get(primary_module, primary_module)}."
        )

        # Describe the primary error
        if primary_error in self.ERROR_DESCRIPTIONS:
            text_parts.append(
                f"Specifically, {self.ERROR_DESCRIPTIONS[primary_error]}."
            )

        # Add environmental context
        if root_cause and root_cause != "Unknown":
            text_parts.append(
                f"This failure occurred under {root_cause} perturbation conditions."
            )

        # Include all errors if requested
        if include_all_errors and len(errors) > 1:
            text_parts.append("\nAdditional contributing factors include:")
            for i, error in enumerate(errors[1:], 1):  # Skip first (primary)
                module = error.get("module", "Unknown")
                error_code = error.get("error_code", "unknown")
                if error_code in self.ERROR_DESCRIPTIONS:
                    text_parts.append(
                        f"{i}. In {self.MODULE_DESCRIPTIONS.get(module, module)}, "
                        f"{self.ERROR_DESCRIPTIONS[error_code]}."
                    )

        # Add technical summary if requested
        if include_technical_details:
            text_parts.append(
                f"\n[Technical Summary: Module={primary_module}, "
                f"ErrorCode={primary_error}, RootCause={root_cause}]"
            )

        return " ".join(text_parts)

    def convert_to_short_form(self, failure_label: Dict[str, Any]) -> str:
        """
        Convert to a short, concise description.

        Useful for training compact representations.
        """
        if not failure_label or not failure_label.get("failure_detected"):
            return "Success"

        primary_module = failure_label.get("primary_failure_module", "Unknown")
        primary_error = failure_label.get("primary_error_code", "unknown")

        return f"{primary_module} failure: {primary_error}"

    def convert_to_structured_json(self, failure_label: Dict[str, Any]) -> str:
        """
        Convert to a structured JSON-like format for instruction tuning.

        This format is useful for training VLMs with structured outputs.
        """
        import json

        if not failure_label or not failure_label.get("failure_detected"):
            return json.dumps({
                "status": "success",
                "failure_detected": False
            }, indent=2)

        primary_module = failure_label.get("primary_failure_module", "Unknown")
        primary_error = failure_label.get("primary_error_code", "unknown")
        errors = failure_label.get("errors", [])

        output = {
            "status": "failure",
            "failure_detected": True,
            "primary_failure": {
                "module": primary_module,
                "error_code": primary_error,
                "description": self.ERROR_DESCRIPTIONS.get(primary_error, "Unknown error")
            },
            "all_errors": [
                {
                    "module": err.get("module"),
                    "error_code": err.get("error_code"),
                    "description": self.ERROR_DESCRIPTIONS.get(err.get("error_code"), "Unknown")
                }
                for err in errors
            ],
            "root_cause": failure_label.get("root_cause", "Unknown")
        }

        return json.dumps(output, indent=2)

    def convert_batch(
        self,
        failure_labels: List[Dict[str, Any]],
        format: str = "long"
    ) -> List[str]:
        """
        Convert a batch of failure labels to text.

        Args:
            failure_labels: List of failure labels
            format: One of "long", "short", or "json"

        Returns:
            List of text descriptions
        """
        results = []
        for label in failure_labels:
            if format == "long":
                text = self.convert_to_text(label)
            elif format == "short":
                text = self.convert_to_short_form(label)
            elif format == "json":
                text = self.convert_to_structured_json(label)
            else:
                raise ValueError(f"Unknown format: {format}")
            results.append(text)
        return results


# Example usage and templates
INSTRUCTION_TEMPLATES = {
    "diagnosis": [
        "Analyze this robotic manipulation failure and explain what went wrong:",
        "What caused this robot to fail the task? Provide a detailed diagnosis:",
        "Watch this failed robotic manipulation. Identify the failure mode:",
        "Diagnose the failure in this robotic task execution:"
    ],
    "classification": [
        "Classify the primary failure module (Perception/Geometry/Planning/Control):",
        "Which component failed in this robotic system?",
        "Identify the failing module:"
    ],
    "recovery": [
        "Given this failure, suggest a recovery strategy:",
        "How should the robot recover from this failure?",
        "What corrective action is needed?"
    ]
}


def create_instruction_tuning_example(
    video_path: str,
    failure_label: Dict[str, Any],
    instruction_type: str = "diagnosis"
) -> Dict[str, Any]:
    """
    Create an instruction-tuning example for VLM training.

    Args:
        video_path: Path to the failure video
        failure_label: Oracle's failure label
        instruction_type: Type of instruction ("diagnosis", "classification", etc.)

    Returns:
        Instruction-tuning example
    """
    import random

    converter = LabelToTextConverter()

    # Random instruction from template
    templates = INSTRUCTION_TEMPLATES.get(instruction_type, INSTRUCTION_TEMPLATES["diagnosis"])
    instruction = random.choice(templates)

    # Generate response
    response = converter.convert_to_text(failure_label, include_technical_details=False)

    return {
        "video": video_path,
        "instruction": instruction,
        "response": response,
        "metadata": {
            "failure_module": failure_label.get("primary_failure_module"),
            "error_code": failure_label.get("primary_error_code")
        }
    }
