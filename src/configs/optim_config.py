"""Optimizer and learning rate scheduler configuration."""

from enum import auto, Enum
from typing import Optional, Union
from dataclasses import dataclass, field


class SchedulerType(Enum):
    """Learning rate scheduler types."""

    COSINE = "cosine"
    STEP = auto()
    CONSTANT = "constant"
    SIN_RAMP_EXP_DECAY = "sinusoidal_ramp_exponential_decay"
    LIN_RAMP_COS_DECAY = "linear_ramp_cosine_decay"
    EXP_RAMP_COS_DECAY = "exp_ramp_cosine_decay"
    WARMUP = auto()
    CUSTOM = "custom"  # Same as 'SIN_RAMP_EXP_DECAY' for backward compatibility


@dataclass
class OptimConfig:
    """Optimization configuration for training."""

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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.scheduler_type, str):
            try:
                self.scheduler_type = SchedulerType(self.scheduler_type)
            except ValueError:
                raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")

        if isinstance(self.scheduler_type_pretrain, str):
            try:
                self.scheduler_type_pretrain = SchedulerType(
                    self.scheduler_type_pretrain
                )
            except ValueError:
                raise ValueError(
                    f"Invalid scheduler_type_pretrain: {self.scheduler_type_pretrain}"
                )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.learning_rate_pretrain <= 0:
            raise ValueError(
                f"learning_rate_pretrain must be positive, got {self.learning_rate_pretrain}"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )

        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
