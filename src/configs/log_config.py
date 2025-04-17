"""Logging and experiment tracking configuration."""

from pathlib import Path
from typing import List
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """Configuration for logging and experiment tracking."""

    exp_root: Path = field(default=Path("./experiments"))
    exp_name: str = field(default="experiment_42")
    allow_overwrite: bool = field(default=False)
    log2wandb: bool = field(default=False)
    visualization_truncation_idxs: List[int] = field(
        default_factory=lambda: [64, 32, 16, 8, 4, 2]
    )

    def __post_init__(self):
        """Initialize and validate experiment directories."""
        # Convert string to Path if needed
        if isinstance(self.exp_root, str):
            self.exp_root = Path(self.exp_root)

        # Validate experiment name
        if not self.exp_name or not isinstance(self.exp_name, str):
            raise ValueError(f"Invalid experiment name: {self.exp_name}")

        # Create experiment directories
        self.exp_dir_base = self.exp_root / self.exp_name
        self.exp_dir = self.exp_root / self.exp_name

        # Validate visualization indices
        if not all(
            isinstance(idx, int) and idx > 0
            for idx in self.visualization_truncation_idxs
        ):
            raise ValueError(
                "All visualization truncation indices must be positive integers"
            )

        # Sort indices in descending order for consistency
        self.visualization_truncation_idxs = sorted(
            self.visualization_truncation_idxs, reverse=True
        )

        # Create experiment directory if it doesn't exist
        if not self.allow_overwrite and self.exp_dir.exists():
            raise ValueError(f"Experiment directory already exists: {self.exp_dir}")

        self.exp_dir.parent.mkdir(parents=True, exist_ok=True)
