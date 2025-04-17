"""Training orchestration for NeuralSVG.

This module provides the Coach class that handles the training process,
including model initialization, optimization, logging, and checkpointing.
"""

import re
import cv2
import math
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import random
import matplotlib
import pyrallis
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from tqdm import tqdm
import wandb
from transformers import get_scheduler
import time

from src.configs.train_config import TrainConfig
from src.models.model import SketchModel
from src.training.logging.coach_logger import CoachLogger
from src.training.losses import (
    SDSLoss,
)
from src.training.utils import coach_utils, vis_utils
from src.training.utils.vis_utils import tensor2im


matplotlib.use("Agg")


class Coach:
    """Training orchestrator for NeuralSVG models.

    Handles:
    - Training loop execution
    - Optimization and scheduling
    - Checkpoint management
    - Inference with conditioning
    """

    def __init__(self, cfg: TrainConfig, model: Optional[SketchModel] = None):
        """Initialize the training coach.

        Args:
            cfg: Configuration object
            model: Optional pre-initialized model
        """
        self.cfg = cfg
        self.train_step = 0
        self.device = "cuda"
        self.best_val_loss = None

        # Set training parameters based on mode
        self._setup_training_params()

        # Initialize directories and logger
        self.create_exp_dir()
        self.logger = CoachLogger(cfg=self.cfg)

        # Initialize model and training components
        self.model = self.init_model() if model is None else model
        self.losses = self.init_losses()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

        self.checkpoint_dir = coach_utils.create_dir(
            self.cfg.log.exp_dir / "checkpoints"
        )

        self.text_prompt = self.cfg.data.text_prompt
        self.logger.log_text_prompt(self.text_prompt)

    def _setup_training_params(self) -> None:
        """
        Initialize training parameters and state.

        Sets up:
        - Initial learning rates
        - Optimizer parameters
        - Training state variables
        - Loss function weights
        - Logging intervals
        """
        if self.cfg.optim.pretrain:
            self.max_steps = self.cfg.steps.max_steps_pretrain
            self.scheduler_type = self.cfg.optim.scheduler_type_pretrain.value
            self.image_interval = self.cfg.steps.image_interval_pretrain
        else:
            self.max_steps = self.cfg.steps.max_steps
            self.scheduler_type = self.cfg.optim.scheduler_type.value
            self.image_interval = self.cfg.steps.image_interval

    @coach_utils.nameit
    def train(self) -> SketchModel:
        """
        Execute the main training loop.

        This method runs the main training loop, handling:
        - Batch processing and forward passes
        - Loss calculation and backpropagation
        - Gradient updates and clipping
        - Learning rate scheduling
        - Metric logging and visualization
        - Checkpoint saving
        - Model evaluation and validation

        The training process follows these steps:
        1. Initialize model and optimizer
        2. Load checkpoint if resuming
        3. Enter main training loop:
           - Process batches
           - Calculate losses
           - Update gradients
           - Log metrics
           - Save checkpoints
        4. Finalize training and save final model

        Returns:
            SketchModel: The trained model
        """
        self.model.renderer.mlp.train()

        # Initialize and save initial sketch
        init_sketch = self._initialize_and_save_sketch()

        # Plot attention map if image path provided
        self._plot_attention_map_if_needed()

        # Main training loop
        self.optimizer.zero_grad()
        for step in tqdm(range(self.max_steps + 1)):
            self.train_step = step

            # Get toggle colors for this step
            if self.cfg.model.toggle_color:
                if self.cfg.optim.pretrain:
                    toggle_colors = self.cfg.model.toggle_color_bg_colors
                else:
                    toggle_colors = [
                        random.choice(self.cfg.model.toggle_color_bg_colors)
                    ]

                if not self.cfg.optim.pretrain:
                    if (
                        torch.rand(1).item()
                        < self.cfg.model.toggle_sample_random_color_prob
                    ):
                        random_values = torch.rand(3, dtype=torch.float32)
                        random_color_tns = torch.cat(
                            (random_values, torch.tensor([1.0], dtype=torch.float32))
                        )
                        toggle_colors = [random_color_tns]
            else:
                toggle_colors = [None]

            # Process each toggle color
            for toggle_color_value in toggle_colors:
                # Accumulate Grads
                if self.cfg.optim.pretrain:
                    points, shape_color_rgb_alpha = self.model.renderer.get_points(
                        truncation_indices=[self.cfg.data.num_strokes],  # Don't drop
                        toggle_color_value=toggle_color_value,
                    )
                    if self.cfg.model.toggle_color:
                        gt_shape_color_rgb_alpha = (
                            self.model.renderer.shapes_init_colors[toggle_color_value]
                        )
                    else:
                        gt_shape_color_rgb_alpha = (
                            self.model.renderer.shapes_init_colors
                        )

                    loss, loss_dict = self.calc_pretrain_loss(
                        pred=points,
                        gt=self.model.renderer.points_init,
                        shape_color_rgb_alpha=shape_color_rgb_alpha,
                        gt_shape_color_rgb_alpha=gt_shape_color_rgb_alpha,
                    )
                else:
                    curr_truncation_idx = None

                    if step > self.max_steps - 200:
                        curr_truncation_idx = [self.cfg.data.num_strokes]

                    if step < self.cfg.steps.start_nested_dropout_from_step:
                        curr_truncation_idx = [self.cfg.data.num_strokes]

                    if self.cfg.model.toggle_aspect_ratio:
                        toggle_aspect_ratio_value = random.choice(
                            self.cfg.model.toggle_aspect_ratio_values
                        )
                    else:
                        toggle_aspect_ratio_value = "1:1"

                    sketch, points, _, _, _ = self.model.renderer.get_image(
                        truncation_indices=curr_truncation_idx,
                        toggle_color_value=toggle_color_value,
                        toggle_aspect_ratio_value=toggle_aspect_ratio_value,
                    )
                    sketch = sketch.to(self.device)
                    loss, loss_dict = self.calc_loss(
                        pred=sketch,
                        toggle_color_value=toggle_color_value,
                    )
                loss.backward()

                self.clip_grads_points()
                self.clip_grads_colors()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.scheduler.step()
            self.logger.update_step(step=self.train_step)

            if self.cfg.log.log2wandb:
                if self.cfg.optim.pretrain:
                    wandb.log(
                        {
                            "learning_rate_pretrain": self.optimizer.param_groups[0][
                                "lr"
                            ]
                        },
                        step=self.train_step,
                    )
                else:
                    wandb.log(
                        {"learning_rate": self.optimizer.param_groups[0]["lr"]},
                        step=self.train_step,
                    )

                wandb.log(loss_dict, step=self.train_step)

            if self.time_to(self.image_interval):
                self.model.renderer.mlp.eval()
                all_sketches = []
                for truncation_idx in self.cfg.log.visualization_truncation_idxs:
                    truncation_indices = [truncation_idx]
                    outputs = []
                    if self.cfg.model.toggle_color:
                        for toggle_color_value in self.cfg.model.toggle_color_bg_colors:
                            output = self.model.renderer.get_image(
                                truncation_indices=truncation_indices,
                                toggle_color_value=toggle_color_value,
                            )
                            outputs.append(output)
                    else:
                        output = self.model.renderer.get_image(
                            truncation_indices=truncation_indices,
                            toggle_color_value=None,
                        )
                        outputs.append(output)
                    all_sketches.append(
                        [output[0].to(self.device) for output in outputs]
                    )
                vis_utils.plot_batch(
                    image=None,
                    outputs=all_sketches,
                    init_sketch=init_sketch,
                    caption=self.text_prompt,
                    output_dir=self.logger.jpg_dir,
                    step=self.train_step,
                    use_wandb=self.cfg.log.log2wandb,
                    title=f"iter_{self.train_step}.jpg",
                    cfg=self.cfg,
                    wandb_title=(
                        "output_pretraining" if self.cfg.optim.pretrain else "output"
                    ),
                )

                # Aspect Ration Visualization
                if self.cfg.model.toggle_aspect_ratio:
                    aspect_ratio_outputs = []
                    toggle_color_value = random.choice(
                        self.cfg.model.toggle_color_bg_colors
                    )
                    for aspect_ratio_value in self.cfg.model.toggle_aspect_ratio_values:
                        output = self.model.renderer.get_image(
                            truncation_indices=[self.cfg.data.num_strokes],
                            toggle_aspect_ratio_value=aspect_ratio_value,
                            toggle_color_value=toggle_color_value,
                        )
                        aspect_ratio_outputs.append(output)

                    debug_aspect_ratio_output_white_rects = (
                        self.model.renderer.get_image(
                            truncation_indices=[self.cfg.data.num_strokes],
                            toggle_aspect_ratio_value=aspect_ratio_value,  # last value
                            toggle_color_value=toggle_color_value,
                            aspect_ratio_white_rects=True,
                        )
                    )
                    aspect_ratio_outputs.append(debug_aspect_ratio_output_white_rects)

                    debug_aspect_ratio_output = self.model.renderer.get_image(
                        truncation_indices=[self.cfg.data.num_strokes],
                        toggle_aspect_ratio_value=aspect_ratio_value,
                        toggle_color_value=toggle_color_value,
                        debug_aspect_ratio=True,
                    )
                    aspect_ratio_outputs.append(debug_aspect_ratio_output)

                    titles = [
                        "1:1",
                        "4:1 - rects bg color",
                        "4:1 - rects white",
                        "4:1 - w/o (toggling & rects)",
                    ]
                    vis_utils.plot_sub_layers(
                        outputs=aspect_ratio_outputs,
                        output_dir=self.logger.jpg_dir,
                        step=self.train_step,
                        use_wandb=self.cfg.log.log2wandb,
                        title=f"Aspect Ratio {self.train_step}.jpg",
                        cfg=self.cfg,
                        wandb_title="aspect_ratio",
                        titles=titles,
                    )
                self.model.renderer.mlp.train()
                self.model.renderer.save_svg(
                    self.logger.svg_dir, f"svg_iter_step_{self.train_step}"
                )

            if (
                self.cfg.optim.pretrain
                and loss_dict["l2_loss"] < 3.0
                and loss_dict["color_pretrain_loss"] < 0.02
            ):
                self.checkpoint_me(is_final=True)
                self.logger.log_message(
                    f"OMG! Finished Pretraining with loss={loss.item()}! "
                    "\n\tNow it's time for some False Analysis!"
                    "\n\tTry out the profiling and analyze scripts!"
                )
                break

            if self.is_final_step():
                self.logger.log_message(
                    f"OMG! Finished Training with {self.train_step=}! "
                    "\n\tNow it's time for some False Analysis!"
                    "\n\tTry out the profiling and analyze scripts!"
                )
                break

        # --- From here we are outside of the train loop ---

        # Log final results
        self.checkpoint_me(is_final=True)
        if not self.cfg.optim.pretrain:
            self.model.renderer.save_svg(self.logger.svg_dir, "final_svg")
        all_sketches = []
        for truncation_idx in self.cfg.log.visualization_truncation_idxs:
            truncation_indices = [truncation_idx]
            outputs = [
                self.model.renderer.get_image(
                    truncation_indices=truncation_indices,
                    render_without_background=True,
                )
            ]
            all_sketches.append([output[0].to(self.device) for output in outputs])
        vis_utils.plot_batch(
            image=self.model.renderer.image,
            outputs=all_sketches,
            init_sketch=init_sketch,
            caption=self.text_prompt,
            output_dir=self.logger.jpg_dir,
            step=self.train_step,
            use_wandb=self.cfg.log.log2wandb,
            title=f"final_sketch_{self.text_prompt}.jpg",
            wandb_title=f"final_sketch",
            cfg=self.cfg,
        )

        if not self.cfg.optim.pretrain and self.cfg.data.is_closed:
            inference_dir = Path(self.cfg.log.exp_dir) / "inference"
            inference_dir.mkdir(exist_ok=True)

            if self.cfg.model.toggle_color:
                bg_colors = self.cfg.model.toggle_color_bg_colors
            else:
                bg_colors = [None]
            for bg_color in bg_colors:
                image_tensor = self.model.renderer.get_image(
                    toggle_color_value=bg_color,
                    truncation_indices=[self.cfg.data.num_strokes],
                )[0]
                image = tensor2im(image_tensor[0])
                if bg_color is None:
                    image_save_path = inference_dir / "no_toggle_bg.jpg"
                    self.model.renderer.save_svg(self.logger.svg_dir, "final_svg")
                else:
                    self.model.renderer.save_svg(
                        self.logger.svg_dir, f"final_svg_{bg_color.replace(' ','-')}"
                    )
                    image_save_path = inference_dir / (
                        bg_color.replace(" ", "-") + ".jpg"
                    )
                image.save(image_save_path)

        return self.model

    def _initialize_and_save_sketch(self) -> Any:
        """Initialize and save initial sketch state."""
        if self.cfg.optim.pretrain:
            self.model.renderer.save_svg(
                self.logger.svg_dir, "init_sketch_before_pretrain"
            )
            init_sketch, _, _, opacities, _ = self.model.renderer.get_image(
                "train", render_without_background=True
            )
            init_sketch = init_sketch.to(self.device)[0]
            init_sketch = vis_utils.tensor2im(init_sketch)
            init_sketch.save(self.logger.jpg_dir / "init_sketch_before_pretrain.jpg")
            if self.cfg.log.log2wandb:
                wandb.log(
                    {"init_sketch_before_pretrain": wandb.Image(init_sketch)},
                    commit=False,
                )
        else:
            self.model.renderer.save_svg(
                self.logger.svg_dir, "init_sketch_before_regular_training"
            )
            init_sketch, _, _, _, _ = self.model.renderer.get_image(
                truncation_indices=[self.cfg.data.num_strokes],
                render_without_background=True,
            )
            init_sketch = init_sketch.to(self.device)[0]
            init_sketch = vis_utils.tensor2im(init_sketch)
            init_sketch.save(
                self.logger.jpg_dir / "init_sketch_before_regular_training.jpg"
            )
            if self.cfg.log.log2wandb:
                wandb.log(
                    {"init_sketch_before_regular_training": wandb.Image(init_sketch)},
                    commit=False,
                )
        return init_sketch

    def _plot_attention_map_if_needed(self) -> None:
        """Plot attention maps if available."""
        if self.cfg.data.image_path is not None:
            vis_utils.plot_attention_map(
                image=self.model.renderer.image,
                attn=self.model.renderer.attn_map,
                threshold_map_list=self.model.renderer.attn_map_soft_list,
                bg_threshold_map_list=self.model.renderer.background_attn_map_soft_list,
                inds=self.model.renderer.inds,
                bg_inds=self.model.renderer.bg_inds,
                use_wandb=self.cfg.log.log2wandb,
                output_path=self.logger.log_dir / "attention_map.png",
            )

    def _get_toggle_colors_for_step(self) -> Optional[List[str]]:
        """Get color toggle values for current step."""
        if not self.cfg.model.toggle_color:
            return [None]

        if self.cfg.optim.pretrain:
            return self.cfg.model.toggle_color_bg_colors

        toggle_colors = [random.choice(self.cfg.model.toggle_color_bg_colors)]

        if torch.rand(1).item() < self.cfg.model.toggle_sample_random_color_prob:
            random_values = torch.rand(3, dtype=torch.float32)
            random_color_tns = torch.cat(
                (random_values, torch.tensor([1.0], dtype=torch.float32))
            )
            toggle_colors = [random_color_tns]

        return toggle_colors

    def _process_single_color(
        self, toggle_color_value: Optional[Union[str, Tensor]]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Process a single color during training.

        Args:
            toggle_color_value: Color value to condition on

        Returns:
            tuple: (total_loss, loss_dict)
        """
        if self.cfg.optim.pretrain:
            return self._process_pretrain_color(toggle_color_value)
        else:
            return self._process_regular_color(toggle_color_value)

    def _process_pretrain_color(
        self, toggle_color_value: Optional[Union[str, Tensor]]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Process a color during the pretraining phase.

        Args:
            toggle_color_value: Color value to process

        Returns:
            Tuple containing:
                - Loss tensor
                - Dictionary of loss components
        """
        points, shape_color_rgb_alpha = self.model.renderer.get_points(
            truncation_indices=[self.cfg.data.num_strokes],  # Don't drop
            toggle_color_value=toggle_color_value,
        )
        if self.cfg.model.toggle_color:
            gt_shape_color_rgb_alpha = self.model.renderer.shapes_init_colors[
                toggle_color_value
            ]
        else:
            gt_shape_color_rgb_alpha = self.model.renderer.shapes_init_colors

        return self.calc_pretrain_loss(
            pred=points,
            gt=self.model.renderer.points_init,
            shape_color_rgb_alpha=shape_color_rgb_alpha,
            gt_shape_color_rgb_alpha=gt_shape_color_rgb_alpha,
        )

    def _process_regular_color(
        self, toggle_color_value: Optional[Union[str, Tensor]]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Process a single color value during regular training.

        Args:
            toggle_color_value: Color value to process

        Returns:
            Tuple containing:
                - Loss tensor
                - Dictionary of loss components
        """
        curr_truncation_idx = self._get_truncation_idx()
        toggle_aspect_ratio_value = self._get_aspect_ratio_value()

        sketch, points, _, _, _ = self.model.renderer.get_image(
            truncation_indices=curr_truncation_idx,
            toggle_color_value=toggle_color_value,
            toggle_aspect_ratio_value=toggle_aspect_ratio_value,
        )
        sketch = sketch.to(self.device)
        return self.calc_loss(
            pred=sketch,
            toggle_color_value=toggle_color_value,
        )

    def _get_truncation_idx(self) -> Optional[List[int]]:
        """Calculate truncation indices for shape introduction."""
        if (
            self.train_step > self.max_steps - 200
            or self.train_step < self.cfg.steps.start_nested_dropout_from_step
        ):
            return [self.cfg.data.num_strokes]
        return None

    def _get_aspect_ratio_value(self) -> str:
        """Get aspect ratio value for current step."""
        if self.cfg.model.toggle_aspect_ratio:
            return random.choice(self.cfg.model.toggle_aspect_ratio_values)
        return "1:1"

    def _update_model_and_logging(self, loss_dict: Dict[str, float]) -> None:
        """Update model parameters and logging.

        Args:
            loss_dict: Dictionary of loss components
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.scheduler.step()
        self.logger.update_step(step=self.train_step)

        if self.cfg.log.log2wandb:
            self._log_to_wandb(loss_dict)

    def _log_to_wandb(self, loss_dict: Dict[str, float]) -> None:
        """Log metrics to wandb.

        Args:
            loss_dict: Dictionary of loss components
        """
        if self.cfg.optim.pretrain:
            wandb.log(
                {"learning_rate_pretrain": self.optimizer.param_groups[0]["lr"]},
                step=self.train_step,
            )
        else:
            wandb.log(
                {"learning_rate": self.optimizer.param_groups[0]["lr"]},
                step=self.train_step,
            )

        wandb.log(loss_dict, step=self.train_step)

    def _save_visualization(self, init_sketch: Any) -> None:
        """Save visualization of current model state.

        Args:
            init_sketch: Initial sketch for comparison
        """
        self.model.renderer.mlp.eval()
        all_sketches = []
        for truncation_idx in self.cfg.log.visualization_truncation_idxs:
            truncation_indices = [truncation_idx]
            outputs = []
            if self.cfg.model.toggle_color:
                for toggle_color_value in self.cfg.model.toggle_color_bg_colors:
                    output = self.model.renderer.get_image(
                        truncation_indices=truncation_indices,
                        toggle_color_value=toggle_color_value,
                    )
                    outputs.append(output)
            else:
                output = self.model.renderer.get_image(
                    truncation_indices=truncation_indices,
                    toggle_color_value=None,
                )
                outputs.append(output)
            all_sketches.append([output[0].to(self.device) for output in outputs])
        vis_utils.plot_batch(
            image=None,
            outputs=all_sketches,
            init_sketch=init_sketch,
            caption=self.text_prompt,
            output_dir=self.logger.jpg_dir,
            step=self.train_step,
            use_wandb=self.cfg.log.log2wandb,
            title=f"iter_{self.train_step}.jpg",
            cfg=self.cfg,
            wandb_title=("output_pretraining" if self.cfg.optim.pretrain else "output"),
        )
        self.model.renderer.mlp.train()

    @coach_utils.nameit
    def infer(
        self,
        save_filename: str,
        output_dir: str,
        interpolate_between: Optional[Tuple[str, str]] = None,
        steps: int = 10,
    ) -> List[torch.Tensor]:
        """Run inference with trained model.

        Args:
            save_filename: Base filename for outputs
            output_dir: Directory to save outputs
            interpolate_between: Optional colors to interpolate
            steps: Number of interpolation steps

        Returns:
            List of output tensors
        """
        self.model.renderer.mlp.eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        if interpolate_between is not None:
            return self._interpolate_colors(
                save_filename, output_dir, interpolate_between, steps
            )

        return self._infer_single(save_filename, output_dir)

    def _interpolate_colors(
        self,
        save_filename: str,
        output_dir: Path,
        interpolate_between: Tuple[str, str],
        steps: int,
    ) -> List[torch.Tensor]:
        """Interpolate between two colors during inference."""
        color_1, color_2 = interpolate_between
        color_1_rgba = self.model.renderer.color_name_to_value_map[color_1]
        color_2_rgba = self.model.renderer.color_name_to_value_map[color_2]

        outputs = []
        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated_color = alpha * color_2_rgba + (1 - alpha) * color_1_rgba
            output = self._infer_with_color(
                save_filename, output_dir, interpolated_color, i
            )
            outputs.extend(output)

        return outputs

    def _infer_single(self, save_filename: str, output_dir: Path) -> List[torch.Tensor]:
        """Run single inference without interpolation."""
        outputs = []
        if self.cfg.model.toggle_color:
            for color in self.cfg.model.toggle_color_bg_colors:
                output = self._infer_with_color(save_filename, output_dir, color)
                outputs.extend(output)
        else:
            output = self._infer_with_color(save_filename, output_dir, None)
            outputs.extend(output)
        return outputs

    def _infer_with_color(
        self,
        save_filename: str,
        output_dir: Path,
        color_value: Optional[Union[str, torch.Tensor]],
        step_idx: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Run inference with a specific color.

        Args:
            save_filename (str): Base filename for saving outputs
            output_dir (Path): Directory to save outputs
            color_value (Optional[Union[str, torch.Tensor]]): Color to use
            step_idx (Optional[int]): Step index for interpolation. Defaults to None.

        Returns:
            List[torch.Tensor]: List of output tensors
        """
        image_tensor = self.model.renderer.get_image(
            toggle_color_value=color_value,
            truncation_indices=[self.cfg.data.num_strokes],
        )[0]

        image = tensor2im(image_tensor[0])
        suffix = self._get_color_suffix(color_value, step_idx)
        image_save_path = output_dir / f"{save_filename}{suffix}.jpg"
        image.save(image_save_path)

        if isinstance(color_value, str):
            svg_suffix = color_value.replace(" ", "-")
            self.model.renderer.save_svg(output_dir, f"{save_filename}_{svg_suffix}")

        return [image_tensor]

    def _get_color_suffix(
        self,
        color_value: Optional[Union[str, torch.Tensor]],
        step_idx: Optional[int] = None,
    ) -> str:
        """Generate suffix for color-specific output files.

        Args:
            color_value (Optional[Union[str, torch.Tensor]]): Color value
            step_idx (Optional[int]): Step index for interpolation

        Returns:
            str: Generated suffix
        """
        if color_value is None:
            return "_no_toggle_bg"
        if isinstance(color_value, str):
            return f"_{color_value.replace(' ', '-')}"
        if step_idx is not None:
            return f"_step_{step_idx}"
        return "_custom_color"

    @coach_utils.nameit
    def infer_single_rgb(
        self,
        save_filename: str,
        output_dir: str,
        rgb: torch.Tensor,
        remove_fill_color: bool = False,
    ) -> None:
        """Run inference with a single RGB color.

        Args:
            save_filename (str): Base filename for saving outputs
            output_dir (str): Directory to save outputs
            rgb (torch.Tensor): RGB color tensor
            remove_fill_color (bool): Whether to remove fill color. Defaults to False.
        """
        self.model.renderer.mlp.eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        image_tensor = self.model.renderer.get_image(
            toggle_color_value=rgb,
            truncation_indices=[self.cfg.data.num_strokes],
            remove_fill_color=remove_fill_color,
        )[0]

        image = tensor2im(image_tensor[0])
        image_save_path = output_dir / f"{save_filename}.jpg"
        image.save(image_save_path)

    def time_to(self, interval: int) -> bool:
        """Check if it's time to perform an interval-based action.

        Args:
            interval (int): Interval to check

        Returns:
            bool: True if current step matches interval
        """
        return self.train_step > 0 and self.train_step % interval == 0

    def is_final_step(self) -> bool:
        """Check if current step is the final training step.

        Returns:
            bool: True if current step is final
        """
        return self.train_step >= self.max_steps

    @coach_utils.nameit
    def checkpoint_me(self, is_best: bool = False, is_final: bool = False) -> None:
        """Save a model checkpoint.

        Args:
            is_best (bool): Whether this is the best model so far. Defaults to False.
            is_final (bool): Whether this is the final checkpoint. Defaults to False.
        """
        save_name = "best" if is_best else f"step_{self.train_step}"
        if is_final:
            save_name = "final"

        save_dict = self.get_save_dict()
        save_path = self.checkpoint_dir / f"{save_name}.pt"
        torch.save(save_dict, save_path)

    def init_model(self) -> SketchModel:
        """Initialize the sketch model.

        Returns:
            SketchModel: Initialized model
        """
        return SketchModel(cfg=self.cfg, device=self.device)

    @coach_utils.nameit
    def init_losses(self) -> Dict[str, nn.Module]:
        """Initialize loss functions.

        Returns:
            Dict[str, nn.Module]: Dictionary of loss modules
        """
        losses = {}
        if not self.cfg.optim.pretrain:
            losses["sds_loss"] = SDSLoss(cfg=self.cfg, device=self.device)

        return losses

    @coach_utils.nameit
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer.

        Returns:
            torch.optim.Optimizer: Initialized optimizer

        Raises:
            ValueError: If optimizer type is invalid
        """
        params = list(self.model.renderer.get_parameters())

        if self.cfg.optim.pretrain:
            lr = self.cfg.optim.learning_rate_pretrain
        else:
            lr = self.cfg.optim.learning_rate

        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.9),
            eps=1e-6,
            weight_decay=self.cfg.optim.weight_decay,
        )

        return optimizer

    def init_scheduler(self) -> LRScheduler:
        """Initialize the learning rate scheduler.

        Returns:
            LRScheduler: Initialized scheduler

        Raises:
            ValueError: If scheduler type is invalid
        """
        if self.scheduler_type in [
            "sinusoidal_ramp_exponential_decay",
            "custom",
        ]:
            lr_lambda = SketchLRLambda(
                func_type="sinusoidal_ramp_exponential_decay",
                max_steps=self.max_steps,
                lr_init=self.cfg.optim.learning_rate,
                lr_final=self.cfg.optim.target_lr,
                lr_delay_steps=self.cfg.optim.warmup_steps,
            )
            scheduler = LambdaLR(
                optimizer=self.optimizer, lr_lambda=lr_lambda, last_epoch=-1
            )

        elif self.scheduler_type in [
            "linear_ramp_cosine_decay",
            "exp_ramp_cosine_decay",
        ]:
            lr_lambda = SketchLRLambda(
                func_type=self.scheduler_type,
                max_steps=self.max_steps,
                lr_init=self.cfg.optim.learning_rate,
                lr_final=self.cfg.optim.target_lr,
                lr_delay_steps=self.cfg.optim.warmup_steps,
            )
            scheduler = LambdaLR(
                optimizer=self.optimizer, lr_lambda=lr_lambda, last_epoch=-1
            )

        else:
            scheduler = get_scheduler(
                self.scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.cfg.optim.warmup_steps,
                num_training_steps=self.max_steps,
            )
        return scheduler

    def clip_grads_colors(self) -> None:
        """Apply gradient clipping to color parameters."""
        if self.cfg.optim.use_clip_grad and not self.cfg.optim.pretrain:
            params = []
            if hasattr(self.model.renderer.mlp, "mlp_color"):
                params += list(
                    filter(
                        lambda p: p.requires_grad and p.grad is not None,
                        self.model.renderer.mlp.mlp_color.parameters(),
                    )
                )
            if len(params) > 0:
                return torch.nn.utils.clip_grad_norm_(
                    parameters=params,
                    max_norm=self.cfg.optim.clip_grad_max_norm_colors,
                )

    def clip_grads_points(self) -> None:
        """Apply gradient clipping to point parameters."""
        if self.cfg.optim.use_clip_grad and not self.cfg.optim.pretrain:
            params = []
            params += list(
                filter(
                    lambda p: p.requires_grad and p.grad is not None,
                    self.model.renderer.mlp.mlp_points.parameters(),
                )
            )

            if len(params) > 0:
                return torch.nn.utils.clip_grad_norm_(
                    parameters=params,
                    max_norm=self.cfg.optim.clip_grad_max_norm_points,
                )

    @coach_utils.nameit
    def calc_loss(
        self,
        pred: Tensor,
        toggle_color_value: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Calculate the loss for regular training.

        Args:
            pred (Tensor): Predicted tensor
            toggle_color_value (Optional[str]): Color value. Defaults to None.

        Returns:
            Tuple containing:
                - Total loss tensor
                - Dictionary of loss components
        """
        loss_dict = {}
        loss = 0

        # Compute SDS loss
        loss_sds = self.losses["sds_loss"](
            pred,
            toggle_color_value=toggle_color_value,
        )  # control step value with param
        loss_dict["sds_loss"] = loss_sds.item()
        loss += loss_sds

        return loss, loss_dict

    @coach_utils.nameit
    def calc_pretrain_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        shape_color_rgb_alpha: Tensor,
        gt_shape_color_rgb_alpha: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Calculate the loss for pretraining.

        Args:
            pred (Tensor): Predicted tensor
            gt (Tensor): Ground truth tensor
            shape_color_rgb_alpha (Tensor): Predicted colors
            gt_shape_color_rgb_alpha (Tensor): Ground truth colors

        Returns:
            Tuple containing:
                - Total loss tensor
                - Dictionary of loss components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        l2_loss = torch.nn.functional.mse_loss(pred, gt)
        loss_dict["l2_loss"] = l2_loss.item()
        total_loss += l2_loss

        if self.cfg.model.toggle_color:
            color_loss = torch.nn.functional.mse_loss(
                shape_color_rgb_alpha, gt_shape_color_rgb_alpha
            )
            loss_dict["color_pretrain_loss"] = color_loss.item()
            total_loss += color_loss

        return total_loss, loss_dict

    def create_exp_dir(self) -> None:
        """Create experiment directory structure."""
        coach_utils.create_dir(self.cfg.log.exp_dir)

    def get_save_dict(self) -> Dict[str, Any]:
        """Get dictionary of data to save in checkpoint.

        Returns:
            Dict[str, Any]: Checkpoint data
        """
        state_dict = self.model.renderer.mlp.state_dict()
        save_dict = {
            "state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "cfg": pyrallis.encode(self.cfg),
        }
        return save_dict


class SketchLRLambda:
    """Learning rate scheduler for sketch training.

    This class provides various learning rate scheduling functions:
    - Sinusoidal ramp with exponential decay
    - Linear ramp with cosine decay
    - Exponential ramp with cosine decay
    """

    def __init__(
        self,
        *,
        func_type: str,
        max_steps: int,
        lr_init: float,
        lr_final: float,
        lr_delay_steps: int = 100,
        lr_delay_mult: float = 0.1,
    ):
        """Initialize the learning rate scheduler.

        Args:
            func_type (str): Type of scheduling function
            max_steps (int): Maximum number of steps
            lr_init (float): Initial learning rate
            lr_final (float): Final learning rate
            lr_delay_steps (int): Steps before starting decay. Defaults to 100.
            lr_delay_mult (float): Multiplier during delay. Defaults to 0.1.
        """
        self.func_type = func_type
        self.max_steps = max_steps
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult

    def _sinusoidal_ramp_exponential_decay(self, step: int) -> float:
        """Calculate learning rate using sinusoidal ramp and exponential decay.

        Args:
            step (int): Current step

        Returns:
            float: Learning rate
        """
        if step < self.lr_delay_steps:
            mult = self.lr_delay_mult + (1 - self.lr_delay_mult) * (
                np.sin(0.5 * np.pi * step / self.lr_delay_steps)
            )
        else:
            mult = np.exp(
                (step - self.lr_delay_steps)
                * np.log(self.lr_final)
                / (self.max_steps - self.lr_delay_steps)
            )
        return self.lr_init * mult

    def _linear_ramp_cosine_decay(self, step: int) -> float:
        """Calculate learning rate using linear ramp and cosine decay.

        Args:
            step (int): Current step

        Returns:
            float: Learning rate
        """
        if step < self.lr_delay_steps:
            mult = (
                self.lr_delay_mult
                + (1 - self.lr_delay_mult) * step / self.lr_delay_steps
            )
        else:
            t = (step - self.lr_delay_steps) / (self.max_steps - self.lr_delay_steps)
            mult = np.cos(t * np.pi / 2) ** 2

        return self.lr_init * mult

    def _exp_ramp_cosine_decay(self, step: int) -> float:
        """Calculate learning rate using exponential ramp and cosine decay.

        Args:
            step (int): Current step

        Returns:
            float: Learning rate
        """
        if step < self.lr_delay_steps:
            mult = self.lr_delay_mult + (1 - self.lr_delay_mult) * (
                np.sin(0.5 * np.pi * step / self.lr_delay_steps)
            )
        else:
            t = (step - self.lr_delay_steps) / (self.max_steps - self.lr_delay_steps)
            mult = np.cos(t * np.pi / 2) ** 2

        return self.lr_init * mult

    def __call__(self, step: int) -> float:
        """Calculate learning rate for current step.

        Args:
            step (int): Current step

        Returns:
            float: Learning rate

        Raises:
            ValueError: If function type is invalid
        """
        if self.func_type == "sinusoidal_ramp_exponential_decay":
            return self._sinusoidal_ramp_exponential_decay(step)
        elif self.func_type == "linear_ramp_cosine_decay":
            return self._linear_ramp_cosine_decay(step)
        elif self.func_type == "exp_ramp_cosine_decay":
            return self._exp_ramp_cosine_decay(step)
        else:
            raise ValueError(f"Invalid function type: {self.func_type}")
