import os
import sys
import socket
import random
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import pyrallis
import logging

from diffusers import StableDiffusionPipeline, DDIMScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup
cv2.setNumThreads(0)
torch.autograd.set_detect_anomaly(True)
sys.path.extend([".", ".."])

from src.configs.train_config import TrainConfig
from src.training.coach import Coach


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setRNGSeed(seed)
    logger.info(f"Set random seed to {seed}")


def check_cuda() -> None:
    """Verify CUDA is available for training."""
    if not torch.cuda.is_available():
        hostname = socket.gethostname()
        raise RuntimeError(
            f"CUDA not available on host {hostname}. GPU is required for training."
        )
    logger.info(f"CUDA is available with device: {torch.cuda.get_device_name()}")


def generate_image_if_needed(cfg: TrainConfig) -> str:
    """Generate an image from text prompt if it doesn't already exist."""
    prompt_tag = cfg.data.text_prompt.replace(" ", "-")
    output_path = f"./generated_images/{prompt_tag}_seed{cfg.seed}.png"

    if os.path.exists(output_path):
        logger.info(f"Using existing image: {output_path}")
        return output_path

    logger.info(f"Generating image for prompt: {cfg.data.text_prompt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model.text2img_model, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_pretrained(
            cfg.model.text2img_model, subfolder="scheduler"
        )
        if cfg.model.lora_weights:
            pipe.load_lora_weights(cfg.model.lora_weights)

        pipe = pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
        image = pipe(
            prompt=cfg.data.text_prompt,
            negative_prompt=cfg.data.negative_prompt,
            num_inference_steps=50,
            generator=generator,
        ).images[0]
        image.save(output_path)
        del pipe
        torch.cuda.empty_cache()

        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to generate image: {str(e)}")


def run_pretraining(cfg: TrainConfig) -> Tuple[Optional[torch.nn.Module], Path]:
    """Run the pretraining phase of the model."""
    pretrain_cfg = deepcopy(cfg)
    pretrain_cfg.optim.pretrain = True
    logger.info("Starting pretraining phase")
    coach = Coach(pretrain_cfg)
    model = coach.train()
    final_ckpt = pretrain_cfg.log.exp_dir / "checkpoints" / "final_model.pt"
    logger.info(f"Pretraining completed. Checkpoint saved at: {final_ckpt}")
    return model, final_ckpt


@pyrallis.wrap()
def main(cfg: TrainConfig) -> None:
    """Main training function."""
    logger.info(f"Starting experiment on host: {socket.gethostname()}")
    seed_everything(cfg.seed)

    if cfg.data.generate_target_image:
        cfg.data.image_path = generate_image_if_needed(cfg)

    if cfg.model.checkpoint_path is None:
        logger.info("No checkpoint provided. Starting pretraining phase.")
        model, ckpt = run_pretraining(cfg)
        cfg.model.checkpoint_path = ckpt
    else:
        logger.info(f"Using provided checkpoint: {cfg.model.checkpoint_path}")
        model = None

    logger.info("Starting standard sketching phase")
    cfg.optim.pretrain = False
    Coach(cfg, model=model).train()


if __name__ == "__main__":
    try:
        check_cuda()
        main()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
