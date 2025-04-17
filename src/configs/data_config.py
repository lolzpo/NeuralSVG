"""Data configuration for training and evaluation."""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for training and evaluation data."""

    text_prompt: str = field(default="A sketch")
    text_prompt_suffix: str = field(
        default="minimal 2d line drawing. on a white background."
    )
    negative_prompt: str = field(default="")
    image_path: Optional[Path] = field(default=None)
    num_strokes: int = field(default=16)
    num_control_points: int = field(default=4)  # Obsolete
    num_segments: int = field(default=1)
    segment_object: bool = field(default=True)
    render_size: int = field(default=512)
    width: float = field(default=1.5)
    radius: float = field(default=0.05)
    is_closed: bool = field(default=True)
    control_points_per_seg: int = field(default=2)
    regular_polygon_closed_shape_init: bool = field(default=True)
    generate_target_image: bool = field(default=True)
    attn_init_tau_max_min: List[float] = field(default_factory=lambda: [0.3, 0.3])
    attn_init_xdog_intersec: bool = field(default=True)
    use_background: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.image_path is not None:
            self.image_path = Path(self.image_path)
            if not self.image_path.exists():
                raise ValueError(f"Image path does not exist: {self.image_path}")

        if not isinstance(self.text_prompt, str) or not self.text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")

        if self.num_strokes <= 0:
            raise ValueError(f"num_strokes must be positive, got {self.num_strokes}")

        if self.num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {self.num_segments}")

        if self.render_size <= 0:
            raise ValueError(f"render_size must be positive, got {self.render_size}")

        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")

        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

        if self.control_points_per_seg <= 0:
            raise ValueError(
                f"control_points_per_seg must be positive, got {self.control_points_per_seg}"
            )

        if len(self.attn_init_tau_max_min) != 2:
            raise ValueError(
                f"attn_init_tau_max_min must have length 2, got {len(self.attn_init_tau_max_min)}"
            )

        if any(tau <= 0 for tau in self.attn_init_tau_max_min):
            raise ValueError(
                f"attn_init_tau_max_min values must be positive, got {self.attn_init_tau_max_min}"
            )
