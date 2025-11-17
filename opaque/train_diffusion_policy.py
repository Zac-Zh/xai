"""
Train Diffusion Policy on Expert Demonstrations

This script loads expert demonstrations from the Classical Oracle and trains
an end-to-end Diffusion Policy for visuomotor control.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    from tqdm import tqdm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required to train the Diffusion Policy. Install with: pip install torch torchvision")

from opaque.diffusion_policy import DiffusionPolicy, DiffusionPolicyConfig, DiffusionPolicyTrainer


class ExpertDemonstrationDataset(Dataset):
    """Dataset for expert demonstrations."""

    def __init__(
        self,
        demonstrations_json: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        image_size: Tuple[int, int] = (64, 64),
        normalize: bool = True
    ):
        """
        Load expert demonstrations.

        Args:
            demonstrations_json: Path to expert_demonstrations.json
            obs_horizon: Number of past observations to include
            action_horizon: Number of future actions to predict
            image_size: Target image size (H, W)
            normalize: Whether to normalize images to [0, 1]
        """
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.normalize = normalize

        # Load demonstrations
        with open(demonstrations_json, "r") as f:
            data = json.load(f)

        self.demonstrations = data["demonstrations"]
        self.metadata = data["metadata"]

        print(f"Loaded {len(self.demonstrations)} demonstrations")
        print(f"Success rate: {self.metadata['success_rate'] * 100:.1f}%")

        # Prepare trajectories
        self.trajectories = self._prepare_trajectories()

    def _prepare_trajectories(self) -> List[Dict]:
        """
        Prepare trajectories from demonstrations.

        Each trajectory contains:
        - observations: List of frames (images)
        - actions: List of (x, y) positions in the planned path
        """
        trajectories = []

        for demo in self.demonstrations:
            # Load frames
            frames = []
            for frame_path in demo["frames_paths"]:
                if os.path.exists(frame_path):
                    frame = np.load(frame_path)
                    # Resize if needed
                    if frame.shape[:2] != self.image_size:
                        frame = self._resize_image(frame)
                    frames.append(frame)

            # Extract action sequence from the planned path
            # The planning module generates a path (list of waypoints)
            planning_data = demo["log_data"]["planning"]

            # For the synthetic environment, we use a simplified action space:
            # actions are waypoints in 2D space
            # In a real implementation, actions would be robot joint commands

            # Create a simple action sequence from start to goal
            geometry_data = demo["log_data"]["geometry"]
            pose_estimate = geometry_data["pose_estimate"]

            # Simple action: target position
            actions = []
            if len(pose_estimate) >= 2:
                # Create action sequence moving toward target
                target_x, target_y = pose_estimate[0], pose_estimate[1]
                # Generate intermediate waypoints
                for i in range(self.action_horizon):
                    t = (i + 1) / self.action_horizon
                    # Interpolate toward target (simplified)
                    action = np.array([target_x * t, target_y * t], dtype=np.float32)
                    actions.append(action)

            if len(frames) >= self.obs_horizon and len(actions) >= self.action_horizon:
                trajectories.append({
                    "observations": frames,
                    "actions": actions,
                    "demo_id": demo["demo_id"]
                })

        print(f"Prepared {len(trajectories)} valid trajectories")
        return trajectories

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        try:
            from PIL import Image
            img = Image.fromarray(image.astype(np.uint8))
            img = img.resize(self.image_size, Image.BILINEAR)
            return np.array(img)
        except ImportError:
            # Fallback: simple downsampling (not recommended)
            import warnings
            warnings.warn("PIL not available. Using simple downsampling.")
            # This is a very crude resize - install PIL for production use
            return image[:self.image_size[0], :self.image_size[1]]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            obs: (obs_horizon, 3, H, W) tensor of observations
            actions: (action_horizon, action_dim) tensor of actions
        """
        traj = self.trajectories[idx]

        # Get observations (take the first obs_horizon frames)
        obs_frames = traj["observations"][:self.obs_horizon]

        # Pad if needed
        while len(obs_frames) < self.obs_horizon:
            obs_frames.append(obs_frames[-1])

        # Stack and convert to torch
        obs = np.stack(obs_frames, axis=0)  # (obs_horizon, H, W, 3)

        # Normalize if requested
        if self.normalize:
            obs = obs.astype(np.float32) / 255.0

        # Transpose to (obs_horizon, 3, H, W)
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.from_numpy(obs).float()

        # Get actions
        actions = traj["actions"][:self.action_horizon]

        # Pad if needed
        while len(actions) < self.action_horizon:
            actions.append(actions[-1])

        actions = np.stack(actions, axis=0)  # (action_horizon, action_dim)
        actions = torch.from_numpy(actions).float()

        return obs, actions


def train_diffusion_policy(
    demonstrations_json: str,
    output_dir: str,
    config: DiffusionPolicyConfig,
    val_split: float = 0.1
):
    """
    Train a Diffusion Policy on expert demonstrations.

    Args:
        demonstrations_json: Path to expert_demonstrations.json
        output_dir: Directory to save model checkpoints
        config: Diffusion policy configuration
        val_split: Fraction of data to use for validation
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Training Diffusion Policy")
    print("="*60)

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = ExpertDemonstrationDataset(
        demonstrations_json=demonstrations_json,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon
    )

    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if config.device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )

    # Create policy
    print("\nInitializing policy...")
    policy = DiffusionPolicy(config)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create trainer
    trainer = DiffusionPolicyTrainer(policy, config)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_loss = validate(policy, val_loader, config)

        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, "best_policy.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"policy_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, checkpoint_path)

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir}")
    print("="*60)


@torch.no_grad()
def validate(
    policy: DiffusionPolicy,
    val_loader: DataLoader,
    config: DiffusionPolicyConfig
) -> float:
    """Validate the policy."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    device = torch.device(config.device)

    for obs, actions in val_loader:
        obs = obs.to(device)
        actions = actions.to(device)

        batch = obs.shape[0]

        # Encode observations
        obs_features = policy.encode_observations(obs)

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            config.num_diffusion_steps,
            (batch,),
            device=device
        )

        # Sample noise
        noise = torch.randn_like(actions)

        # Add noise to actions
        sqrt_alpha_t = policy.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_t = policy.sqrt_one_minus_alphas_cumprod[timesteps]

        sqrt_alpha_t = sqrt_alpha_t[:, None, None]
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t[:, None, None]

        noisy_actions = sqrt_alpha_t * actions + sqrt_one_minus_alpha_t * noise

        # Predict noise
        noise_pred = policy(noisy_actions, timesteps, obs_features)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        num_batches += 1

    policy.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")
    parser.add_argument(
        "--demonstrations",
        required=True,
        help="Path to expert_demonstrations.json"
    )
    parser.add_argument(
        "--output-dir",
        default="results/diffusion_policy",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )

    args = parser.parse_args()

    # Create configuration
    config = DiffusionPolicyConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )

    # Train policy
    train_diffusion_policy(
        demonstrations_json=args.demonstrations,
        output_dir=args.output_dir,
        config=config
    )


if __name__ == "__main__":
    main()
