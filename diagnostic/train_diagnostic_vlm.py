"""
Train Diagnostic VLM

This script provides a framework for training a Vision-Language Model
on the Robo-Oracle dataset for robotic failure diagnosis.

This implementation is designed to be modular and can work with various
VLM architectures (e.g., LLaVA, BLIP-2, etc.)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required. Install with: pip install torch torchvision")


class DiagnosticVLMDataset(Dataset):
    """Dataset for training the diagnostic VLM."""

    def __init__(
        self,
        dataset_json: str,
        image_size: int = 224,
        max_text_length: int = 512
    ):
        """
        Initialize the dataset.

        Args:
            dataset_json: Path to the VLM dataset JSON
            image_size: Target image size for vision encoder
            max_text_length: Maximum text length
        """
        self.image_size = image_size
        self.max_text_length = max_text_length

        # Load dataset
        with open(dataset_json, "r") as f:
            data = json.load(f)

        self.samples = data["samples"]
        self.metadata = data["metadata"]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]

        # Load video/image
        video_path = sample["video"]

        # For simplicity, we'll use the middle frame or first frame
        # In a full implementation, you'd process the entire video
        if video_path.endswith('.npy'):
            # Single frame
            image = np.load(video_path)
        elif video_path.endswith('.gif'):
            # Load first frame of GIF
            img = Image.open(video_path)
            image = np.array(img)
        else:
            # Try to load as image
            try:
                image = np.array(Image.open(video_path))
            except:
                # Placeholder if can't load
                image = np.zeros((64, 64, 3), dtype=np.uint8)

        # Preprocess image
        image = self._preprocess_image(image)

        return {
            "image": image,
            "instruction": sample["instruction"],
            "response": sample["response"],
            "failure_module": sample["metadata"]["failure_module"],
            "error_code": sample["metadata"]["error_code"]
        }

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for VLM."""
        # Convert to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Resize
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # To tensor and normalize
        image_array = np.array(pil_image).astype(np.float32) / 255.0

        # Normalize with ImageNet stats (standard for vision models)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        # Transpose to CHW
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()

        return image_tensor


class SimpleDiagnosticVLM(nn.Module):
    """
    A simplified diagnostic VLM for demonstration.

    In practice, you would use a pre-trained VLM like LLaVA or BLIP-2.
    This is a minimal implementation showing the key components.
    """

    def __init__(
        self,
        num_classes: int = 4,  # Perception, Geometry, Planning, Control
        vision_embed_dim: int = 512,
        text_embed_dim: int = 512,
        fusion_dim: int = 256
    ):
        super().__init__()

        # Vision encoder (simplified)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, vision_embed_dim)
        )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(vision_embed_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )

        # Module mapping
        self.module_to_idx = {
            "Perception": 0,
            "Geometry": 1,
            "Planning": 2,
            "Control": 3
        }
        self.idx_to_module = {v: k for k, v in self.module_to_idx.items()}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: (batch, 3, H, W)

        Returns:
            Logits (batch, num_classes)
        """
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Classify
        logits = self.fusion(vision_features)

        return logits

    def predict(self, images: torch.Tensor) -> List[str]:
        """Predict failure modules."""
        logits = self.forward(images)
        predictions = torch.argmax(logits, dim=1)

        return [self.idx_to_module[idx.item()] for idx in predictions]


class DiagnosticVLMTrainer:
    """Trainer for the diagnostic VLM."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["image"].to(self.device)

            # Get labels
            failure_modules = batch["failure_module"]
            labels = torch.tensor([
                self.model.module_to_idx[mod] for mod in failure_modules
            ]).to(self.device)

            # Forward
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["image"].to(self.device)

            failure_modules = batch["failure_module"]
            labels = torch.tensor([
                self.model.module_to_idx[mod] for mod in failure_modules
            ]).to(self.device)

            # Forward
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels
        }

    def train(self, num_epochs: int, output_dir: str):
        """Train the model."""
        os.makedirs(output_dir, exist_ok=True)

        best_accuracy = 0.0

        print("\nStarting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']*100:.2f}%")

            # Save best model
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                checkpoint_path = os.path.join(output_dir, "best_diagnostic_vlm.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': best_accuracy
                }, checkpoint_path)
                print(f"Saved best model (accuracy: {best_accuracy*100:.2f}%)")

        print("\nTraining complete!")
        print(f"Best validation accuracy: {best_accuracy*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Diagnostic VLM")
    parser.add_argument(
        "--train-dataset",
        required=True,
        help="Path to train_vlm_dataset.json"
    )
    parser.add_argument(
        "--val-dataset",
        required=True,
        help="Path to val_vlm_dataset.json"
    )
    parser.add_argument(
        "--output-dir",
        default="results/diagnostic_vlm",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    args = parser.parse_args()

    # Create datasets
    train_dataset = DiagnosticVLMDataset(args.train_dataset)
    val_dataset = DiagnosticVLMDataset(args.val_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = SimpleDiagnosticVLM()

    # Create trainer
    trainer = DiagnosticVLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr
    )

    # Train
    trainer.train(num_epochs=args.epochs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
