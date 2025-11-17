"""
Opaque Policy Interface

This module provides a unified interface for running the end-to-end Diffusion Policy
with the same API as the Classical Oracle, enabling direct comparison.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Opaque Policy will not work.")

from utils.yload import load as yload
from simulators.synth_env import SynthLiftEnv
from opaque.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig


@dataclass
class OpaquePolicyResult:
    """Result from running the opaque policy."""

    success: bool
    final_distance: float
    rollout_frames: List[np.ndarray]
    predicted_actions: List[np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_distance": self.final_distance,
            "num_frames": len(self.rollout_frames),
            "num_actions": len(self.predicted_actions)
        }


class OpaquePolicy:
    """
    Opaque End-to-End Policy Interface

    This class wraps the trained Diffusion Policy and provides an interface
    matching the Classical Oracle for direct comparison.
    """

    def __init__(
        self,
        model_checkpoint: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Opaque Policy.

        Args:
            model_checkpoint: Path to the trained Diffusion Policy checkpoint
            config_path: Path to the environment configuration YAML
            device: Device to run inference on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use the Opaque Policy")

        self.device = torch.device(device)
        self.cfg = yload(config_path)

        # Load policy
        print(f"Loading Opaque Policy from {model_checkpoint}...")
        checkpoint = torch.load(model_checkpoint, map_location=self.device)

        # Reconstruct config
        if 'config' in checkpoint:
            policy_config_dict = checkpoint['config']
            self.policy_config = DiffusionPolicyConfig(**policy_config_dict)
        else:
            # Use default config if not saved
            self.policy_config = DiffusionPolicyConfig(device=device)

        # Create and load policy
        self.policy = DiffusionPolicy(self.policy_config)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.to(self.device)
        self.policy.eval()

        print(f"Opaque Policy loaded successfully")

    @torch.no_grad()
    def run_single(
        self,
        scenario: str,
        perturbation_level: float,
        seed: int,
        success_distance_threshold: float = 0.03
    ) -> OpaquePolicyResult:
        """
        Run the opaque policy once.

        Args:
            scenario: Perturbation scenario
            perturbation_level: Perturbation strength (0.0 to 1.0)
            seed: Random seed
            success_distance_threshold: Distance threshold for success

        Returns:
            OpaquePolicyResult containing success status and rollout data
        """
        # Import perturbation module
        import importlib
        perturb_mod = importlib.import_module(f"perturb.{scenario}")

        # Initialize context
        ctx: Dict[str, Any] = {
            "noise": {},
            "obstacles": [],
            "meta": {
                "task": self.cfg["task"],
                "robot": self.cfg["robot"],
                "camera": self.cfg["camera"]
            }
        }
        rng = np.random.default_rng(seed)
        ctx["rng"] = rng

        # Apply perturbation
        perturb_mod.apply(ctx, float(perturbation_level))

        # Initialize environment
        env = SynthLiftEnv(
            seed=seed,
            max_steps=int(self.cfg["max_steps"]),
            camera=self.cfg["camera"]
        )
        env.reset()
        state = env.get_state()

        # Get goal
        goal = np.array([state["target_x"], state["target_y"]], dtype=float)

        # Collect rollout
        rollout_frames = []
        predicted_actions_list = []
        current_position = np.array([state["agent_x"], state["agent_y"]], dtype=float)

        # Observation buffer for obs_horizon
        obs_buffer = []

        max_steps = int(self.cfg["max_steps"])

        for step in range(max_steps):
            # Render current observation
            rgb = env.render_rgb()
            rollout_frames.append(rgb.copy())

            # Prepare observation
            obs_frame = self._preprocess_image(rgb)
            obs_buffer.append(obs_frame)

            # Keep only obs_horizon frames
            if len(obs_buffer) > self.policy_config.obs_horizon:
                obs_buffer.pop(0)

            # Pad if needed
            while len(obs_buffer) < self.policy_config.obs_horizon:
                obs_buffer.insert(0, obs_buffer[0] if obs_buffer else obs_frame)

            # Stack observations
            obs_tensor = torch.stack(obs_buffer, dim=0).unsqueeze(0)  # (1, obs_horizon, 3, H, W)
            obs_tensor = obs_tensor.to(self.device)

            # Predict action sequence
            predicted_actions = self.policy.predict_action(obs_tensor)  # (1, action_horizon, action_dim)
            predicted_actions = predicted_actions[0].cpu().numpy()  # (action_horizon, action_dim)

            # Take the first action (predicted absolute position)
            predicted_position = predicted_actions[0]
            predicted_actions_list.append(predicted_position.copy())

            # Calculate delta (action for env.step expects delta, not absolute position)
            delta = predicted_position - current_position

            # Execute action in environment
            new_state, reward, done, info = env.step(delta)

            # Update current position from environment
            current_position = np.array([new_state["agent_x"], new_state["agent_y"]], dtype=float)

            # Check if reached goal or episode done
            distance_to_goal = float(np.linalg.norm(current_position - goal))

            if distance_to_goal < success_distance_threshold or done:
                break

        # Calculate final distance
        final_distance = float(np.linalg.norm(current_position - goal))
        success = final_distance < success_distance_threshold

        return OpaquePolicyResult(
            success=success,
            final_distance=final_distance,
            rollout_frames=rollout_frames,
            predicted_actions=predicted_actions_list
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for the policy.

        Args:
            image: (H, W, 3) numpy array

        Returns:
            Preprocessed tensor (3, H, W)
        """
        # Resize to 64x64
        try:
            from PIL import Image
            img = Image.fromarray(image.astype(np.uint8))
            img = img.resize((64, 64), Image.BILINEAR)
            image_resized = np.array(img)
        except ImportError:
            # Fallback
            image_resized = image[:64, :64]

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Transpose to (3, H, W)
        image_transposed = np.transpose(image_normalized, (2, 0, 1))

        # Convert to tensor
        tensor = torch.from_numpy(image_transposed).float()

        return tensor


def run_opaque_policy(
    model_checkpoint: str,
    config_path: str,
    scenario: str,
    perturbation_level: float,
    seed: int,
    success_distance_threshold: float = 0.03
) -> Tuple[bool, float, List[np.ndarray]]:
    """
    Convenience function to run the opaque policy.

    Args:
        model_checkpoint: Path to trained model
        config_path: Path to configuration file
        scenario: Perturbation scenario
        perturbation_level: Perturbation strength
        seed: Random seed
        success_distance_threshold: Distance threshold for success

    Returns:
        Tuple of (success, final_distance, rollout_frames)
    """
    policy = OpaquePolicy(model_checkpoint, config_path)
    result = policy.run_single(
        scenario,
        perturbation_level,
        seed,
        success_distance_threshold
    )
    return result.success, result.final_distance, result.rollout_frames
