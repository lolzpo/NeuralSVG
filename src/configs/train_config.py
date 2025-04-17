from enum import auto, Enum
from pathlib import Path
from typing import Optional, List, Union

from dataclasses import dataclass, field

from .data_config import DataConfig
from .model_config import ModelConfig
from .optim_config import OptimConfig
from .log_config import LogConfig
from .train_steps_config import TrainStepsConfig


class SchedulerType(Enum):
    """Learning rate scheduler types."""

    COSINE = "cosine"
    STEP = auto()
    CONSTANT = "constant"
    SIN_RAMP_EXP_DECAY = "sinusoidal_ramp_exponential_decay"
    LIN_RAMP_COS_DECAY = "linear_ramp_cosine_decay"
    EXP_RAMP_COS_DECAY = "exp_ramp_cosine_decay"
    WARMUP = auto()
    CUSTOM = "custom"  # Same as 'SIN_RAMP_EXP_DECAY' for backward comp (doc)


@dataclass
class OptimConfig:
    """Optimization configuration."""

    learning_rate: float = field(default=5e-3)
    learning_rate_pretrain: float = field(default=1e-2)
    scheduler_type: Union[SchedulerType, str] = field(default=SchedulerType.CONSTANT)
    scheduler_type_pretrain: Union[SchedulerType, str] = field(
        default=SchedulerType.CONSTANT
    )
    target_lr: float = field(default=4e-3)
    warmup_steps: int = field(default=50)
    use_clip_grad: bool = field(default=True)
    clip_grad_max_norm_colors: float = field(default=0.1)
    clip_grad_max_norm_points: float = field(default=0.1)
    weight_decay: float = field(default=0)
    sds_guidance_scale: float = field(default=100)
    sd_num_inference_steps: int = field(default=50)
    pretrain: bool = field(default=False)
    timestep_scheduling: Optional[str] = field(default=None)
    sds_sample_timestep_sd: float = field(default=100)
    lambda_l2_pretraining: float = field(default=1)
    sds_weight: Optional[str] = field(default=None)
    sds_use_bg_color_suffix: bool = field(default=True)


@dataclass
class ModelConfig:
    """Model configuration."""

    mode: str = field(default="train")
    checkpoint_path: Optional[Path] = field(default=None)
    text2img_model: str = field(default="runwayml/stable-diffusion-v1-5")
    lora_weights: Optional[str] = field(default=None)

    mlp_dim: int = field(default=128)
    mlp_num_layers: int = field(default=2)
    use_color: bool = field(default=False)
    use_nested_dropout: bool = field(default=True)
    input_dim: int = field(default=128)
    truncation_start_idx: int = field(default=4)
    nested_dropout_sampling_method: str = field(default="uniform")

    points_prediction_scale: float = field(default=0.1)

    use_dropout_value: bool = field(default=False)
    dropout_emb_dim: int = field(default=16)
    dropout_last_item_prob: float = field(default=0.5)
    dropout_temperature: float = field(default=1.5)

    toggle_color: bool = field(default=False)
    toggle_color_method: str = field(default="rgb")
    toggle_color_input_dim: int = field(default=8)
    toggle_color_bg_colors: list[str] = field(default_factory=lambda: ["blue", "red"])
    toggle_color_init_eps: float = field(default=0.1)
    toggle_sample_random_color_prob: float = field(default=0.5)

    toggle_aspect_ratio: bool = field(default=False)
    toggle_aspect_ratio_values: list[str] = field(default_factory=lambda: ["1:1"])
    aspect_ratio_emb_dim: int = field(default=16)

    def __post_init__(self) -> None:
        if not self.toggle_color:
            self.toggle_color_bg_colors = [None]

        for aspect_ration_value in self.toggle_aspect_ratio_values:
            if aspect_ration_value not in ["1:1", "4:1", None]:
                raise ValueError(
                    f"unsupported value {self.toggle_aspect_ratio_values=}"
                )

        if self.nested_dropout_sampling_method not in ["uniform", "exp_decay"]:
            raise ValueError(
                f"unsupported value {self.nested_dropout_sampling_method=}"
            )


@dataclass
class LogConfig:
    """Logging configuration."""

    exp_root: Path = field(default=Path("./experiments"))
    exp_name: str = field(default="experiment_42")
    allow_overwrite: bool = field(default=False)
    log2wandb: bool = field(default=False)
    visualization_truncation_idxs: list[int] = field(
        default_factory=lambda: [64, 32, 16, 8, 4, 2]
    )

    def __post_init__(self):
        self.exp_dir_base = self.exp_root / self.exp_name
        self.exp_dir = self.exp_root / self.exp_name


@dataclass
class TrainStepsConfig:
    """Training steps configuration."""

    max_steps: int = field(default=2_000)
    max_steps_pretrain: int = field(default=400)
    image_interval: int = field(default=50)
    image_interval_pretrain: int = field(default=5)
    start_nested_dropout_from_step: int = field(default=200)


@dataclass
class TrainConfig:
    """Main training configuration."""

    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log: LogConfig = field(default_factory=LogConfig)
    steps: TrainStepsConfig = field(default_factory=TrainStepsConfig)
    seed: int = field(default=0)

    def __post_init__(self):
        """Initialize and validate relationships between configurations."""
        # Update experiment directory with seed
        self.log.exp_dir = self.log.exp_dir / f"seed_{self.seed}"

        # Filter visualization indices based on data and model constraints
        self.log.visualization_truncation_idxs = [
            idx
            for idx in self.log.visualization_truncation_idxs
            if idx <= self.data.num_strokes
        ]
        self.log.visualization_truncation_idxs = [
            idx
            for idx in self.log.visualization_truncation_idxs
            if idx >= self.model.truncation_start_idx
        ]

        # Clean up color background names
        self.model.toggle_color_bg_colors = list(
            map(
                lambda x: x.replace("-", " ") if x is not None else x,
                self.model.toggle_color_bg_colors,
            )
        )

        # Validate seed
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
