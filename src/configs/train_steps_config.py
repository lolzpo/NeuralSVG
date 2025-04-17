"""Training steps and intervals configuration."""

from dataclasses import dataclass, field


@dataclass
class TrainStepsConfig:
    """Configuration for training steps and intervals."""

    max_steps: int = field(default=2_000)
    max_steps_pretrain: int = field(default=400)
    image_interval: int = field(default=50)
    image_interval_pretrain: int = field(default=5)
    start_nested_dropout_from_step: int = field(default=200)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        if self.max_steps_pretrain <= 0:
            raise ValueError(
                f"max_steps_pretrain must be positive, got {self.max_steps_pretrain}"
            )

        if self.image_interval <= 0:
            raise ValueError(
                f"image_interval must be positive, got {self.image_interval}"
            )

        if self.image_interval_pretrain <= 0:
            raise ValueError(
                f"image_interval_pretrain must be positive, got {self.image_interval_pretrain}"
            )

        if self.start_nested_dropout_from_step < 0:
            raise ValueError(
                f"start_nested_dropout_from_step must be non-negative, got {self.start_nested_dropout_from_step}"
            )

        if self.start_nested_dropout_from_step > self.max_steps:
            raise ValueError(
                f"start_nested_dropout_from_step ({self.start_nested_dropout_from_step}) "
                f"cannot be greater than max_steps ({self.max_steps})"
            )
