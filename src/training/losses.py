"""
Stable Diffusion Score (SDS) Loss Implementation for Vector Graphics Generation.
"""

from typing import Optional, Dict, List, Union, Literal
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy.stats import truncnorm
from torchvision import transforms
from src.training.utils import coach_utils

from src.configs.train_config import TrainConfig
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
)

# Constants for timestep scheduling
MIN_TIMESTEP = 50
MAX_TIMESTEP = 950
TOTAL_TIMESTEPS = 1000

# Constants for SDS weights
SDS_WEIGHT_SCHEMES = Literal[
    "default", "constant", "noisematch", "sdslike", "strongerlower"
]
DEFAULT_SDS_WEIGHT = 0.05

# Augmentation parameters
PERSPECTIVE_DISTORTION_SCALE = 0.5
PERSPECTIVE_PROBABILITY = 0.7


class SDSLoss(nn.Module):
    """Score Distillation Sampling loss."""

    def __init__(self, cfg: TrainConfig, device: str):
        """Initialize SDS loss.

        Args:
            cfg: Configuration object
            device: Device to run on
        """
        super(SDSLoss, self).__init__()

        if device not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got {device}")

        self.pipe: StableDiffusionPipeline
        self.device: str = device
        self.cfg: TrainConfig = cfg
        self.timestep_scheduling: str = self.cfg.optim.timestep_scheduling
        self.sd_num_inference_steps: int = self.cfg.optim.sd_num_inference_steps

        self.text: str = self.cfg.data.text_prompt

        if not "stable-diffusion" in cfg.model.text2img_model:
            raise ValueError(
                f"Only stable-diffusion models are supported, got {cfg.model.text2img_model}"
            )

        ddim_scheduler = DDIMScheduler.from_pretrained(
            cfg.model.text2img_model, subfolder="scheduler"
        )
        ddim_scheduler.set_timesteps(self.sd_num_inference_steps)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model.text2img_model,
            torch_dtype=torch.float16,
            scheduler=ddim_scheduler,
        )

        if cfg.model.lora_weights is not None:
            lora_weights_path = cfg.model.lora_weights
            print(f"Loading LoRA weights from: {lora_weights_path}")
            self.pipe.load_lora_weights(lora_weights_path)

        self.pipe = self.pipe.to(self.device)

        self.alphas: Tensor = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas: Tensor = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text = self.cfg.data.text_prompt
        self.negative_prompt: str = self.cfg.data.negative_prompt

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        self.text_embeddings: Tensor = self.embed_text(text=self.text)

        self.text_embedding_bg_colors: Dict[str, Tensor] = {}
        for color_name in cfg.model.toggle_color_bg_colors + ["white"]:
            self.text_embedding_bg_colors[color_name] = self.embed_text(
                text=self.text + f" isolated on {color_name} background"
            )

        del self.tokenizer
        del self.text_encoder

    def sample_timestep(self, train_step: int, batch_size: int) -> Tensor:
        """Sample diffusion timesteps.

        Args:
            train_step: Current training step
            batch_size: Number of samples

        Returns:
            Tensor: Sampled timesteps
        """
        if train_step < 0:
            raise ValueError(f"train_step must be non-negative, got {train_step}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if self.timestep_scheduling == "normal_dist":
            steps_left = self.cfg.steps.max_steps - train_step
            normalized_t = (steps_left / self.cfg.steps.max_steps) * TOTAL_TIMESTEPS
            normalized_t = int(normalized_t)

            dist = get_truncated_normal(
                normalized_t, sd=self.cfg.optim.sds_sample_timestep_sd
            )
            t = torch.from_numpy(
                dist.rvs(batch_size).clip(MIN_TIMESTEP, MAX_TIMESTEP).astype(int)
            ).to(device=self.device)

        elif self.timestep_scheduling == "sqrt":
            percentage = np.sqrt(float(train_step) / self.cfg.steps.max_steps)
            t = (TOTAL_TIMESTEPS - 1) * (1 - percentage)
            t = torch.tensor([t], device=self.device).to(int)

        elif self.timestep_scheduling == "cosine":
            percentage = (
                np.cos(np.pi - float(train_step) / self.cfg.steps.max_steps * np.pi) / 2
                + 0.5
            )
            t = (TOTAL_TIMESTEPS - 1) * (1 - percentage)
            t = torch.tensor([t], device=self.device).to(int)

        else:
            raise ValueError(
                f"Unsupported timestep scheduling: {self.timestep_scheduling}. "
                "Must be one of ['normal_dist', 'sqrt', 'cosine']"
            )

        print(f"Train-step {train_step}, Timestep {t}")
        return t

    @coach_utils.nameit_torch
    def embed_text(self, text) -> Tensor:
        """Generate text embeddings."""
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # Another reference for uncond_input: https://github.com/ximinng/SVGDreamer/blob/c64754d79f655c4f9028206eae56dc63e0839510/svgdreamer/painter/diffusion_pipeline.py#L84
        uncond_input = self.tokenizer(
            [self.negative_prompt],
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def x_augment(self, x: torch.Tensor, img_size: int = 512) -> torch.Tensor:
        """Apply augmentations to input."""
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be a torch.Tensor, got {type(x)}")

        if x.dim() != 4:
            raise ValueError(
                f"Input tensor must have 4 dimensions (B,C,H,W), got shape {x.shape}"
            )

        augment_compose = transforms.Compose(
            [
                transforms.RandomPerspective(
                    distortion_scale=PERSPECTIVE_DISTORTION_SCALE,
                    p=PERSPECTIVE_PROBABILITY,
                ),
                transforms.RandomCrop(
                    size=(img_size, img_size),
                    pad_if_needed=True,
                    padding_mode="reflect",
                ),
            ]
        )
        return augment_compose(x)

    @coach_utils.nameit_torch
    def forward(
        self,
        x_aug: Tensor,
        train_step: Optional[int] = None,
        augment: bool = True,
        toggle_color_value: Optional[str] = None,
    ):
        """Compute SDS loss."""
        sds_loss = 0

        if (
            toggle_color_value is not None
            and isinstance(toggle_color_value, str)
            and self.cfg.optim.sds_use_bg_color_suffix
        ):
            text_embeddings = self.text_embedding_bg_colors[toggle_color_value]
        else:
            text_embeddings = self.text_embeddings

        if augment:
            x_aug = self.x_augment(x=x_aug)
        x = x_aug * 2.0 - 1.0
        with torch.amp.autocast("cuda"):
            init_latent_z = self.pipe.vae.encode(x).latent_dist.sample()
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z

        with torch.inference_mode():
            bsz = latent_z.shape[0]

            if train_step is not None:
                timestep = self.sample_timestep(train_step=train_step, batch_size=bsz)
            else:
                timestep = torch.randint(
                    low=50,
                    high=950,
                    size=(bsz,),
                    device=self.device,
                    dtype=torch.long,
                )

            eps = torch.randn_like(latent_z)

            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            z_in = torch.cat([noised_latent_zt] * 2)
            timestep_in = torch.cat([timestep] * 2)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = (
                    self.pipe.unet(
                        z_in, timestep_in, encoder_hidden_states=text_embeddings
                    )
                    .sample.float()
                    .chunk(2)
                )

            eps_t = eps_t_uncond + self.cfg.optim.sds_guidance_scale * (
                eps_t - eps_t_uncond
            )

            w = self.get_w(timestep=timestep)

            grad_z = w * (eps_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z
        sds_loss = sds_loss.sum(1).mean()
        return sds_loss

    def get_w(self, timestep: int) -> float:
        """Get timestep weight.

        Args:
            timestep: Current timestep

        Returns:
            float: Weight value
        """
        if not isinstance(timestep, (int, torch.Tensor)):
            raise ValueError(f"Timestep must be int or Tensor, got {type(timestep)}")

        weight_scheme: SDS_WEIGHT_SCHEMES = self.cfg.optim.sds_weight

        if weight_scheme == "default":
            w = 1 - self.alphas[timestep]
        elif weight_scheme == "constant":
            w = DEFAULT_SDS_WEIGHT
        elif weight_scheme == "noisematch":
            w = (
                DEFAULT_SDS_WEIGHT
                * self.alphas[timestep] ** 0.5
                / (1 - self.alphas[timestep]) ** 0.5
            )
        elif weight_scheme == "sdslike":
            w = (
                DEFAULT_SDS_WEIGHT
                * self.alphas[timestep] ** 0.5
                * (1 - self.alphas[timestep]) ** 0.5
            )
        elif weight_scheme == "strongerlower":
            w = (
                DEFAULT_SDS_WEIGHT
                * self.alphas[timestep] ** 1.0
                / (1 - self.alphas[timestep]) ** 0.5
            )
        else:
            raise ValueError(
                f"Unsupported SDS weight scheme: {weight_scheme}. "
                f"Must be one of {list(SDS_WEIGHT_SCHEMES.__args__)}"
            )

        return w


def get_truncated_normal(mean: float, sd: float, low: int = 0, upp: int = 1000):
    """
    Create a truncated normal distribution.

    This utility function creates a scipy truncated normal distribution
    with the specified parameters, useful for timestep sampling.

    Args:
        mean (float): Mean of the distribution
        sd (float): Standard deviation of the distribution
        low (int): Lower bound for truncation. Defaults to 0.
        upp (int): Upper bound for truncation. Defaults to 1000.

    Returns:
        scipy.stats.truncnorm: Truncated normal distribution object

    Note:
        This is primarily used for the 'normal_dist' timestep scheduling
        strategy in the SDSLoss class.
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
