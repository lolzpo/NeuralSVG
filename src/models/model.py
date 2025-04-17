"""
Base model implementation for NeuralSVG.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
from torch import nn
from torch import Tensor

from src.configs.train_config import TrainConfig
from src.models.painter_nerf import PainterNerf

logger = logging.getLogger(__name__)


class SketchModel(nn.Module):
    """Base model class for NeuralSVG implementations."""

    def __init__(
        self,
        cfg: TrainConfig,
        device: str = "cuda",
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """Initialize the base model."""
        super().__init__()

        self.cfg = cfg

        # Validate and set device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = device

        # Initialize renderer
        try:
            self.renderer = self._initialize_renderer()
        except Exception as e:
            logger.error(f"Failed to initialize renderer: {str(e)}")
            raise RuntimeError(f"Renderer initialization failed: {str(e)}")

        # Load checkpoint if specified
        self._load_checkpoint(checkpoint_path)

    def _initialize_renderer(self) -> PainterNerf:
        """Initialize and configure the PainterNerf renderer."""
        logger.debug("Initializing neural renderer")
        renderer = PainterNerf(cfg=self.cfg, device=self.device)
        return renderer.to(self.device)

    def _load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load model weights from checkpoint."""
        # Determine checkpoint path
        cp_path = checkpoint_path or self.cfg.model.checkpoint_path
        if cp_path is None:
            return

        cp_path = Path(cp_path)
        if not cp_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp_path}")

        try:
            logger.info(f"Loading checkpoint from: {cp_path}")
            checkpoint = torch.load(cp_path, map_location=self.device)

            # Load state dict with error checking
            missing_keys, unexpected_keys = self.renderer.mlp.load_state_dict(
                checkpoint["state_dict"], strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

            logger.info("Checkpoint loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass interface."""
        try:
            return self.renderer(x)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise RuntimeError(f"Model forward pass failed: {str(e)}")

    def save_checkpoint(
        self, path: Union[str, Path], additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model state to checkpoint."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "state_dict": self.renderer.mlp.state_dict(),
                "config": self.cfg,
            }

            if additional_data:
                checkpoint.update(additional_data)

            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to: {path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint saving failed: {str(e)}")
