"""
Diffusion Policy Implementation for Visuomotor Control

This implements a simplified version of the Diffusion Policy architecture,
which uses iterative denoising to generate robot action sequences from visual observations.

Based on: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
Chi et al., RSS 2023
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Diffusion Policy will not work.")


@dataclass
class DiffusionPolicyConfig:
    """Configuration for the Diffusion Policy."""

    # Network architecture
    vision_feature_dim: int = 512
    action_dim: int = 2  # For 2D manipulation (x, y)
    action_horizon: int = 8  # Number of future actions to predict
    obs_horizon: int = 2  # Number of past observations to condition on

    # Diffusion parameters
    num_diffusion_steps: int = 100
    num_inference_steps: int = 10
    beta_schedule: str = "linear"  # or "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,)
        Returns:
            embeddings: (batch_size, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat(
            [torch.sin(embeddings), torch.cos(embeddings)], dim=-1
        )
        return embeddings


class VisionEncoder(nn.Module):
    """Simple CNN encoder for visual observations."""

    def __init__(self, output_dim: int = 512):
        super().__init__()
        # Simple CNN for 64x64 RGB images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),  # -> 15x15
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # -> 6x6
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # -> 4x4
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, obs_horizon, 3, H, W)
        Returns:
            features: (batch, obs_horizon, feature_dim)
        """
        batch, obs_horizon, c, h, w = x.shape
        # Flatten batch and obs_horizon
        x = x.view(batch * obs_horizon, c, h, w)
        x = self.conv(x)
        x = x.view(batch * obs_horizon, -1)
        x = self.fc(x)
        # Reshape back
        x = x.view(batch, obs_horizon, -1)
        return x


class ConditionalUNet1D(nn.Module):
    """
    1D U-Net for denoising action sequences conditioned on observations.

    This is a simplified version of the architecture used in Diffusion Policy.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        cond_dim: int,
        time_embed_dim: int = 128
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, 256)

        # U-Net architecture (simplified)
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(action_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128)
        )

        # Middle
        self.mid = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 256)
        )

        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv1d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128)
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64)
        )

        # Output
        self.out = nn.Conv1d(64, action_dim, kernel_size=3, padding=1)

        # Condition injection layers
        self.cond_to_enc1 = nn.Linear(256, 64)
        self.cond_to_enc2 = nn.Linear(256, 128)
        self.cond_to_mid = nn.Linear(256, 256)
        self.time_to_mid = nn.Linear(time_embed_dim, 256)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy actions (batch, action_dim, action_horizon)
            timestep: Diffusion timestep (batch,)
            cond: Observation conditioning (batch, cond_dim)

        Returns:
            Predicted noise (batch, action_dim, action_horizon)
        """
        # Get embeddings
        t_emb = self.time_embed(timestep)  # (batch, time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        cond_emb = self.cond_proj(cond)  # (batch, 256)

        # Encoder
        h1 = self.enc1(x)  # (batch, 64, action_horizon)
        # Inject condition
        h1 = h1 + self.cond_to_enc1(cond_emb)[:, :, None]

        h2 = self.enc2(h1)  # (batch, 128, action_horizon)
        h2 = h2 + self.cond_to_enc2(cond_emb)[:, :, None]

        # Middle
        h = self.mid(h2)  # (batch, 256, action_horizon)
        h = h + self.cond_to_mid(cond_emb)[:, :, None]
        h = h + self.time_to_mid(t_emb)[:, :, None]

        # Decoder with skip connections
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h)

        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h)

        # Output
        out = self.out(h)
        return out


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for visuomotor control.

    This policy learns to denoise action sequences conditioned on visual observations.
    """

    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(output_dim=config.vision_feature_dim)

        # Flatten observation features
        obs_feature_dim = config.vision_feature_dim * config.obs_horizon

        # Denoising network
        self.denoiser = ConditionalUNet1D(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            cond_dim=obs_feature_dim
        )

        # Setup diffusion schedule
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self):
        """Setup noise schedule for diffusion process."""
        config = self.config

        if config.beta_schedule == "linear":
            betas = torch.linspace(
                config.beta_start,
                config.beta_end,
                config.num_diffusion_steps
            )
        elif config.beta_schedule == "cosine":
            # Cosine schedule (more common in practice)
            steps = config.num_diffusion_steps
            s = 0.008
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )

    def encode_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode visual observations.

        Args:
            obs: (batch, obs_horizon, 3, H, W)

        Returns:
            Flattened observation features (batch, obs_feature_dim)
        """
        features = self.vision_encoder(obs)  # (batch, obs_horizon, feature_dim)
        # Flatten
        batch = features.shape[0]
        features = features.view(batch, -1)
        return features

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in the noisy actions.

        Args:
            noisy_actions: (batch, action_horizon, action_dim)
            timestep: (batch,)
            obs_features: (batch, obs_feature_dim)

        Returns:
            Predicted noise (batch, action_horizon, action_dim)
        """
        # Transpose for conv1d
        noisy_actions = noisy_actions.transpose(1, 2)  # (batch, action_dim, action_horizon)

        # Predict noise
        noise_pred = self.denoiser(noisy_actions, timestep, obs_features)

        # Transpose back
        noise_pred = noise_pred.transpose(1, 2)  # (batch, action_horizon, action_dim)

        return noise_pred

    @torch.no_grad()
    def predict_action(
        self,
        obs: torch.Tensor,
        num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate actions from observations using the diffusion process.

        Args:
            obs: (batch, obs_horizon, 3, H, W)
            num_inference_steps: Number of denoising steps (default: use config)

        Returns:
            Actions (batch, action_horizon, action_dim)
        """
        batch = obs.shape[0]
        device = obs.device

        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        # Encode observations
        obs_features = self.encode_observations(obs)

        # Start from random noise
        actions = torch.randn(
            batch,
            self.config.action_horizon,
            self.config.action_dim,
            device=device
        )

        # Denoise using DDIM (Denoising Diffusion Implicit Models)
        timesteps = torch.linspace(
            self.config.num_diffusion_steps - 1,
            0,
            num_inference_steps,
            device=device
        ).long()

        for t in timesteps:
            t_batch = t.repeat(batch)

            # Predict noise
            noise_pred = self.forward(actions, t_batch, obs_features)

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=device)

            # Predicted x0
            pred_x0 = (actions - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred

            # Update
            actions = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return actions


class DiffusionPolicyTrainer:
    """Trainer for the Diffusion Policy."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        config: DiffusionPolicyConfig
    ):
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        self.policy.to(self.device)

        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate
        )

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        """
        Single training step.

        Args:
            obs: (batch, obs_horizon, 3, H, W)
            actions: (batch, action_horizon, action_dim)

        Returns:
            Loss value
        """
        batch = obs.shape[0]

        # Encode observations
        obs_features = self.policy.encode_observations(obs)

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.config.num_diffusion_steps,
            (batch,),
            device=self.device
        )

        # Sample noise
        noise = torch.randn_like(actions)

        # Add noise to actions (forward diffusion)
        sqrt_alpha_t = self.policy.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_t = self.policy.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        sqrt_alpha_t = sqrt_alpha_t[:, None, None]
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t[:, None, None]

        noisy_actions = sqrt_alpha_t * actions + sqrt_one_minus_alpha_t * noise

        # Predict noise
        noise_pred = self.policy(noisy_actions, timesteps, obs_features)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            obs, actions = batch
            obs = obs.to(self.device)
            actions = actions.to(self.device)

            loss = self.train_step(obs, actions)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0
