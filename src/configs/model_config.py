"""Model architecture and training configuration."""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""

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
    toggle_color_bg_colors: List[str] = field(default_factory=lambda: ["blue", "red"])
    toggle_color_init_eps: float = field(default=0.1)
    toggle_sample_random_color_prob: float = field(default=0.5)

    toggle_aspect_ratio: bool = field(default=False)
    toggle_aspect_ratio_values: List[str] = field(default_factory=lambda: ["1:1"])
    aspect_ratio_emb_dim: int = field(default=16)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
            if not self.checkpoint_path.exists():
                raise ValueError(
                    f"Checkpoint path does not exist: {self.checkpoint_path}"
                )

        if self.mode not in ["train", "eval", "inference"]:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {self.mlp_dim}")

        if self.mlp_num_layers <= 0:
            raise ValueError(
                f"mlp_num_layers must be positive, got {self.mlp_num_layers}"
            )

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")

        if self.truncation_start_idx < 0:
            raise ValueError(
                f"truncation_start_idx must be non-negative, got {self.truncation_start_idx}"
            )

        if self.nested_dropout_sampling_method not in ["uniform", "exp_decay"]:
            raise ValueError(
                f"Invalid nested_dropout_sampling_method: {self.nested_dropout_sampling_method}"
            )

        if not (0 <= self.dropout_last_item_prob <= 1):
            raise ValueError(
                f"dropout_last_item_prob must be between 0 and 1, got {self.dropout_last_item_prob}"
            )

        if self.dropout_temperature <= 0:
            raise ValueError(
                f"dropout_temperature must be positive, got {self.dropout_temperature}"
            )

        if self.toggle_color_method not in ["rgb", "hsv"]:
            raise ValueError(f"Invalid toggle_color_method: {self.toggle_color_method}")

        if not (0 <= self.toggle_sample_random_color_prob <= 1):
            raise ValueError(
                f"toggle_sample_random_color_prob must be between 0 and 1, got {self.toggle_sample_random_color_prob}"
            )

        for aspect_ratio_value in self.toggle_aspect_ratio_values:
            if aspect_ratio_value not in ["1:1", "4:1", None]:
                raise ValueError(f"Invalid aspect ratio value: {aspect_ratio_value}")

        if not self.toggle_color:
            self.toggle_color_bg_colors = [None]
