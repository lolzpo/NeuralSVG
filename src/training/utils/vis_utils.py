"""Visualization utilities for neural vector graphics.
"""

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import wandb
from torch import Tensor
from PIL import Image
import pydiffvg
import torch
from pathlib import Path
from typing import Optional, List, Union

from src.training.utils import coach_utils
from src.configs.train_config import TrainConfig


def tensor2im(var: Tensor) -> Image.Image:
    """Convert tensor to PIL image."""
    var = var.clone()
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var[var < 0] = 0
    var[var > 1] = 1
    var *= 255
    return Image.fromarray(var.astype("uint8"))


@coach_utils.nameit
def plot_attention_map(
    *,
    image: np.ndarray,
    attn: np.ndarray,
    threshold_map_list: List[np.ndarray],
    bg_threshold_map_list: List[np.ndarray],
    inds: np.ndarray,
    bg_inds: np.ndarray,
    use_wandb: bool,
    output_path: Union[str, Path],
) -> None:
    """Plot attention maps with thresholds."""
    num_rows_for_tau = 3
    fig, axes = plt.subplots(
        nrows=(1 + num_rows_for_tau), ncols=2, figsize=(6, 3 * (1 + num_rows_for_tau))
    )

    # Plot the images in the grid
    for i, ax in enumerate(axes.flat):

        if i == 0:
            # Plot image
            ax.imshow(image, interpolation="nearest")  # Display image
            ax.axis("off")  # Hide axes
            ax.set_title(f"Target Image")

        elif i == 1:
            # Plot Attnention map
            ax.imshow(attn, interpolation="nearest")  # Display image
            ax.axis("off")  # Hide axes
            ax.set_title(f"Attention Map")

        else:
            if i in [2, 3]:
                # Take first map
                map_idx = 0
            elif i in [4, 5]:
                # Take middle map
                map_idx = len(threshold_map_list) // 2
            elif i in [6, 7]:
                # Take last map
                map_idx = -1
            else:
                ValueError(f"Doesn't support {num_rows_for_tau=} > 4")

            # Plot TAU attantions
            if i % 2 == 0:
                threshold_map = threshold_map_list[map_idx]
                # Plot Object attention
                threshold_map_ = (threshold_map - threshold_map.min()) / (
                    threshold_map.max() - threshold_map.min()
                )

                ax.imshow(threshold_map_, interpolation="nearest")  # Display image
                ax.axis("off")  # Hide axes
                ax.set_title(f"Prob Softmax Object tau=")
                ax.scatter(
                    inds[:, 1], inds[:, 0], s=10, c="red", marker="o"
                )  # Add scatter points
            else:
                bg_threshold_map = bg_threshold_map_list[map_idx]
                # Plot Background attention
                bg_threshold_map_ = (bg_threshold_map - bg_threshold_map.min()) / (
                    bg_threshold_map.max() - bg_threshold_map.min()
                )
                ax.imshow(bg_threshold_map_, interpolation="nearest")  # Display image
                ax.axis("off")  # Hide axes
                ax.set_title(f"Prob Softmax Object tau=")
                ax.scatter(
                    bg_inds[:, 1], bg_inds[:, 0], s=10, c="red", marker="o"
                )  # Add scatter points

    fig.suptitle("Attention Map Visualizations!", fontsize=16)
    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)}, commit=False)
    plt.savefig(output_path)
    plt.close()


@coach_utils.nameit
def plot_batch(
    image: Optional[Tensor],
    outputs: List[Union[Tensor, List[Tensor]]],
    caption: str,
    output_dir: Union[str, Path],
    step: int,
    use_wandb: bool,
    title: str,
    cfg: TrainConfig,
    init_sketch: Optional[Tensor] = None,
    layer_svgs: Optional[List[str]] = None,
    wandb_title: str = "output",
) -> None:
    """Plot batch of images with optional layers."""
    if layer_svgs is not None:
        plot_batch_layers(
            image, outputs, layer_svgs, caption, output_dir, step, use_wandb, title, cfg
        )

    if isinstance(
        outputs[0], list
    ):  # In the current code in `coach.py` -> this is always True !
        if len(outputs) == 1:
            # Annoying WA but will do it for now.
            # In case we don't use dropouts and we just want to plot a single output image, just plot it. :)
            simply_plot_image(
                image=outputs[0][0],
                step=step,
                wandb_title=wandb_title,
                output_dir=output_dir,
                title=title,
                caption=caption,
                use_wandb=use_wandb,
            )
        else:
            plot_batch_with_internal_dropout(
                image,
                outputs,
                caption,
                output_dir,
                step,
                use_wandb,
                title,
                cfg,
                layer_svgs=layer_svgs,
                wandb_title=wandb_title,
            )
    else:
        print("Warning: Unexpected output format")  # Better error message
        start_idx = 1
        if image is not None:
            plt.figure(figsize=(8 + 4 * len(outputs), 4))
            plt.subplot(1, 2 + len(outputs), start_idx)
            plt.imshow(image, interpolation="nearest")
            plt.axis("off")
            plt.title("Input")
            start_idx += 1

        if init_sketch is not None:
            # Show the init sketch
            plt.subplot(1, 2 + len(outputs), start_idx)
            plt.imshow(init_sketch, interpolation="nearest")
            plt.axis("off")
            plt.title("Init Sketch")
            start_idx += 1

            output_images = [tensor2im(output[0]) for output in outputs]
            for idx, (output, truncation_idx) in enumerate(
                zip(output_images, cfg.log.visualization_truncation_idxs)
            ):
                plt.subplot(1, 2 + len(outputs), idx + start_idx)
                plt.imshow(output, interpolation="nearest")
                plt.axis("off")
                plt.title(f"Dropout Index: {truncation_idx}")

            plt.suptitle(caption)

            plt.tight_layout()
            if use_wandb:
                wandb.log({wandb_title: wandb.Image(plt)}, step=step)
            plt.savefig(f"{output_dir}/{title}")
            plt.close()


@coach_utils.nameit
def plot_sub_layers(
    outputs: List[List[Tensor]],
    output_dir: Union[str, Path],
    step: int,
    use_wandb: bool,
    title: str,
    cfg: TrainConfig,
    wandb_title: str = "output",
    titles: Optional[List[str]] = None,
) -> None:
    """Plot sub-layer outputs."""
    num_outputs = len(outputs)

    # Create subplots dynamically based on number of outputs
    fig, axes = plt.subplots(1, num_outputs, figsize=(15, 5))

    # Convert tensors to images
    output_images = [tensor2im(output[0][0]) for output in outputs]

    if not titles:
        titles = [f"Layer {idx + 1}" for idx in range(len(output_images))]

    for idx, output in enumerate(output_images):
        axes[idx].imshow(output, interpolation="nearest")
        axes[idx].axis("off")
        axes[idx].set_title(titles[idx])

    fig.suptitle("Layer-wise Visualization", fontsize=16)

    plt.tight_layout()

    # Optionally log to WandB
    if use_wandb:
        wandb.log({wandb_title: wandb.Image(plt)}, step=step)

    # Save the figure
    plt.savefig(f"{output_dir}/{title}.png", bbox_inches="tight")
    plt.close(fig)


@coach_utils.nameit
def simply_plot_image(
    image: Tensor,
    step: int,
    wandb_title: str,
    output_dir: Union[str, Path],
    title: str,
    caption: str,
    use_wandb: bool,
) -> None:
    """Plot single image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(tensor2im(image), interpolation="nearest")
    plt.axis("off")
    plt.title(caption)

    if use_wandb:
        wandb.log({wandb_title: wandb.Image(plt)}, step=step)

    plt.savefig(f"{output_dir}/{title}")
    plt.close()


@coach_utils.nameit
def plot_batch_with_internal_dropout(
    image,
    outputs,
    caption,
    output_dir,
    step,
    use_wandb,
    title,
    cfg: TrainConfig,
    layer_svgs=None,
    wandb_title: str = "output",
):
    """Plot batch with dropout outputs."""
    image = None  # Tmp for now. We don't need to vis image!

    nrows = 1 + len(outputs[0]) if image else len(outputs[0])
    ncols = len(outputs)

    # print(f"plot_batch_with_internal_dropout, {ncols=}")
    # print(f"plot_batch_with_internal_dropout, {nrows=}")

    start_idx = 1
    if image is not None:
        plt.figure(figsize=(8 * ncols, 8 * nrows))
        plt.subplot(ncols, nrows, start_idx)
        if type(image) == Tensor:
            image = tensor2im(image)
        plt.imshow(image, interpolation="nearest")
        plt.axis("off")
        plt.title("Input")
        start_idx += 1

    else:
        plt.figure(figsize=(8 * nrows, 8 * ncols))

    count = 0
    for output, truncation_idx in zip(outputs, cfg.log.visualization_truncation_idxs):
        output = [
            tensor2im(o[0]) if len(o.shape) == 4 else tensor2im(o) for o in output
        ]
        for out in output:
            plt.subplot(ncols, nrows, count + start_idx)
            plt.imshow(out, interpolation="nearest")
            plt.axis("off")
            plt.title(f"Dropout: {truncation_idx}")
            count += 1
        # count += 1

    plt.suptitle(caption)
    plt.tight_layout()

    # If using Weights & Biases for logging
    if use_wandb:
        wandb.log({wandb_title: wandb.Image(plt)}, step=step)

    fig_id = f"{output_dir}/{title}"
    print(f"Save fig @ {fig_id}")
    plt.savefig(fig_id)
    plt.close()


@coach_utils.nameit
def plot_batch_layers(
    image,
    outputs,
    layer_svgs,
    caption,
    output_dir,
    step,
    use_wandb,
    title,
    cfg: TrainConfig,
):
    """Plot batch with layer SVGs."""
    plt.figure(figsize=(8 + 4 * len(layer_svgs), 4))

    start_idx = 1
    n_rows = len(layer_svgs) + 1
    if image is not None:
        n_rows += 1
        plt.subplot(1, n_rows, start_idx)
        plt.imshow(tensor2im(image), interpolation="nearest")
        plt.axis("off")
        plt.title("Target Image")
        start_idx += 1

    full_output = tensor2im(outputs[0][0])
    plt.subplot(1, n_rows, start_idx)
    plt.imshow(full_output, interpolation="nearest")
    plt.axis("off")
    plt.title("Output")
    start_idx += 1

    layer_images = [tensor2im(svg) for svg in layer_svgs]
    for idx, layer in enumerate(layer_images):
        plt.subplot(1, n_rows, start_idx + idx)
        plt.imshow(layer, interpolation="nearest")
        plt.axis("off")
        plt.title(f"Layer: {idx}")

    plt.suptitle(caption)

    plt.tight_layout()
    if use_wandb:
        wandb.log({"per_layer_output": wandb.Image(plt)}, step=step)
    plt.savefig("{}/layers_{}".format(output_dir, title))
    plt.close()
