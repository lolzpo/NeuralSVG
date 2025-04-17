"""
Training logger for the NeuralSVG coach implementation.
"""

import dataclasses
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import pyrallis
from git import GitError
from loguru import logger
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
import git

from src.configs.train_config import TrainConfig
from src.training.utils.coach_utils import create_dir, PROJECT_NAME


class CoachLogger:
    """Training logger for NeuralSVG."""

    def __init__(self, cfg: TrainConfig) -> None:
        """Initialize logger.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.step = 0
        self.wandb_task = None

        # Create output directories
        self.log_dir = create_dir(self.cfg.log.exp_dir / "logs")
        self.svg_dir = create_dir(self.cfg.log.exp_dir / "svg_logs")
        self.jpg_dir = create_dir(self.cfg.log.exp_dir / "jpg_logs")

        self.jpg_dir.mkdir(parents=True, exist_ok=True)
        self.svg_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.configure_loguru()
        if self.cfg.log.log2wandb:
            self.init_wandb()

        # Log initial configuration
        self.log_config()
        self.log_git_info()

    def init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        config = dataclasses.asdict(self.cfg)
        self.wandb_task = wandb.init(
            project=PROJECT_NAME, name=self.cfg.log.exp_name, config=config
        )

    def log_config(self) -> None:
        """Log configuration parameters."""
        with (self.cfg.log.exp_dir / "config.yaml").open("w") as f:
            pyrallis.dump(self.cfg, f)
        self.log_message("\n" + pyrallis.dump(self.cfg))

    def configure_loguru(self) -> None:
        """Configure loguru logger settings."""
        logger.remove()
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(sys.stdout, colorize=True, format=format)
        logger.add(self.log_dir / "log.txt", colorize=False, format=format)

    def log_message(self, message: str) -> None:
        """Log a message to all outputs.

        Args:
            message: Message to log
        """
        logger.info(message)

    def log_metrics(self, metrics_dict: Dict[str, float], step: int) -> None:
        """Log training metrics."""
        self.log_message(f"Metrics for step {step}")
        for key, value in metrics_dict.items():
            self.log_message(f"\t{key} = {value:0.4f}")
        if self.cfg.log.log2wandb:
            wandb.log(metrics_dict, step=step)

    def update_step(self, step: int) -> None:
        """Update current training step."""
        self.step = step

    def log_text_prompt(self, text_prompt: str) -> None:
        """Save text prompt to file."""
        text_prompt_filepath = self.cfg.log.exp_dir / "text_prompt.txt"
        with open(text_prompt_filepath, "w") as file:
            file.write(text_prompt)

    def log_image(
        self, image_name: str, image_tensor: torch.Tensor, step: Optional[int] = None
    ) -> None:
        """Save and log an image."""
        if step is None:
            step = self.step

        # Convert tensor to numpy and transpose to [H, W, C]
        image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)

        # Save locally
        plt.imsave(self.jpg_dir / f"{image_name}_{step}.jpg", image_np)

        # Log to wandb
        if self.cfg.log.log2wandb:
            wandb.log({image_name: wandb.Image(image_np)}, step=step)

    def log_svg(self, svg_name: str, svg_str: str, step: Optional[int] = None) -> None:
        """Save an SVG file."""
        if step is None:
            step = self.step

        svg_path = self.svg_dir / f"{svg_name}_{step}.svg"
        with open(svg_path, "w") as f:
            f.write(svg_str)

    def log_checkpoint(self, checkpoint_dict: Dict, step: Optional[int] = None) -> None:
        """Save a model checkpoint."""
        if step is None:
            step = self.step

        checkpoint_dir = self.cfg.log.exp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint_dict, checkpoint_path)

        if self.cfg.log.log2wandb:
            wandb.save(str(checkpoint_path))

    def log_git_info(self) -> None:
        """Save git repository information."""
        try:
            repo = git.Repo(search_parent_directories=True)
            git_hash = repo.head.object.hexsha
            git_diff = repo.git.diff()

            with open(self.cfg.log.exp_dir / "git_hash.txt", "w") as f:
                f.write(git_hash)

            with open(self.cfg.log.exp_dir / "git_diff.txt", "w") as f:
                f.write(git_diff)

        except GitError as e:
            logger.warning(f"Failed to log git info: {e}")
