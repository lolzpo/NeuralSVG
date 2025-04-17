"""Multi-output NeRF network for vector graphics.

Core functionality:
- Point and color generation
- Positional encodings
- Nested dropout
- Dynamic conditioning
"""

from typing import Optional, Union, Dict, List, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.training.utils import coach_utils


class NerfMLPMulti(nn.Module):
    """Multi-output MLP for vector graphics generation."""

    @coach_utils.nameit_torch
    def __init__(
        self,
        num_strokes: int,
        total_num_points: int,
        input_dim: int = 128,
        intermediate_dim: int = 128,
        num_layers: int = 2,
        use_nested_dropout: bool = False,
        use_color: bool = True,
        nested_dropout_probability: float = 0.5,
        truncation_start_idx: int = 4,
        points_prediction_scale: float = 0.1,
        device: str = "cuda",
        use_dropout_value: bool = False,
        dropout_emb_dim: int = 16,
        toggle_color: bool = False,
        toggle_color_input_dim: int = 12,
        toggle_color_bg_colors: Optional[List[str]] = None,
        color_name_to_value_map: Optional[Dict[str, Tensor]] = None,
        toggle_color_method: str = "rgb",
        toggle_aspect_ratio: bool = False,
        toggle_aspect_ratio_values: Optional[List[str]] = None,
        aspect_ratio_emb_dim: Optional[int] = None,
    ):
        """Initialize multi-output NeRF MLP."""
        super().__init__()

        # Store configuration
        self._store_config(
            num_strokes,
            total_num_points,
            intermediate_dim,
            num_layers,
            use_nested_dropout,
            nested_dropout_probability,
            use_color,
            device,
            truncation_start_idx,
            points_prediction_scale,
            use_dropout_value,
            toggle_color,
            toggle_color_bg_colors,
            color_name_to_value_map,
            toggle_color_method,
            toggle_aspect_ratio,
            toggle_aspect_ratio_values,
        )

        # Initialize dimensions
        self._initialize_dimensions(
            input_dim, dropout_emb_dim, toggle_color_input_dim, aspect_ratio_emb_dim
        )

        # Create embeddings
        self._create_embeddings(aspect_ratio_emb_dim)

        # Create MLPs
        self._create_mlps()

        # Initialize additional parameters
        self._initialize_parameters()

    def _store_config(
        self,
        num_strokes: int,
        total_num_points: int,
        intermediate_dim: int,
        num_layers: int,
        use_nested_dropout: bool,
        nested_dropout_probability: float,
        use_color: bool,
        device: str,
        truncation_start_idx: int,
        points_prediction_scale: float,
        use_dropout_value: bool,
        toggle_color: bool,
        toggle_color_bg_colors: Optional[List[str]],
        color_name_to_value_map: Optional[Dict[str, Tensor]],
        toggle_color_method: str,
        toggle_aspect_ratio: bool,
        toggle_aspect_ratio_values: Optional[List[str]],
    ) -> None:
        """Store configuration parameters."""
        self.num_strokes = num_strokes
        self.total_num_points = total_num_points
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_probability = nested_dropout_probability
        self.use_color = use_color
        self.device = device
        self.truncation_start_idx = truncation_start_idx
        self.points_prediction_scale = points_prediction_scale
        self.use_dropout_value = use_dropout_value
        self.toggle_color = toggle_color
        self.toggle_color_bg_colors = toggle_color_bg_colors
        self.color_name_to_value_map = color_name_to_value_map
        self.toggle_color_method = toggle_color_method
        self.toggle_aspect_ratio = toggle_aspect_ratio
        self.toggle_aspect_ratio_values = toggle_aspect_ratio_values

        if self.num_layers < 1:
            raise Exception(f"Invalid value {self.num_layers=}")

    def _initialize_dimensions(
        self,
        input_dim: int,
        dropout_emb_dim: int,
        toggle_color_input_dim: int,
        aspect_ratio_emb_dim: Optional[int],
    ) -> None:
        """Initialize network dimensions."""
        # Set embedding dimensions
        self.dropout_emb_dim = dropout_emb_dim if self.use_dropout_value else 0
        self.aspect_ratio_dim = aspect_ratio_emb_dim if self.toggle_aspect_ratio else 0

        # Set output dimensions
        self.output_dim_points = self.total_num_points * 2
        self.output_dim_color = 3
        self.output_dim_opacity = 1

        # Handle color toggle dimensions
        if self.toggle_color:
            self.toggle_color_input_dim = toggle_color_input_dim
            if (
                self.toggle_color_method == "rgb"
                and self.toggle_color_input_dim % 3 != 0
            ):
                raise ValueError(f"{self.toggle_color_input_dim=} must be divided by 3")
        else:
            self.toggle_color_input_dim = 0

        # Set MLP input dimensions
        self.shape_input_dim = input_dim
        self.input_dim_mlp_points = (
            self.shape_input_dim + self.dropout_emb_dim + self.aspect_ratio_dim
        )
        self.input_dim_mlp_color = (
            self.shape_input_dim + self.dropout_emb_dim + self.toggle_color_input_dim
        )

    def _create_embeddings(self, aspect_ratio_emb_dim: Optional[int]) -> None:
        """Create embedding layers."""
        # Create main positional embedding
        self.pe = GaussianEmbedding(
            in_channels=1,
            num_shapes=self.num_strokes,
            emb_size=self.shape_input_dim,
        )

        # Create dropout embedding if needed
        if self.use_dropout_value:
            self.dropout_pe = GaussianEmbedding(
                in_channels=1,
                num_shapes=self.num_strokes,
                emb_size=self.dropout_emb_dim,
            )

        # Create aspect ratio embedding if needed
        if self.toggle_aspect_ratio:
            self.aspect_ratio_pe = GaussianEmbedding(
                in_channels=1,
                num_shapes=0,  # irrelevant cause we skip the normalization
                emb_size=self.aspect_ratio_dim,
                skip_normalization=True,
            )

        # Create color toggle embedding if needed
        if self.toggle_color:
            self._create_color_embedding()

    def _create_color_embedding(self) -> None:
        """Create color toggle embedding."""
        if self.toggle_color_method == "discrete":
            self.toggle_color_pe = GaussianEmbedding(
                in_channels=1,
                num_shapes=len(self.toggle_color_bg_colors),
                emb_size=self.toggle_color_input_dim,
            )
        elif self.toggle_color_method == "rgb":
            self.toggle_color_pe = GaussianEmbeddingRGB(
                in_channels=1,
                emb_size=self.toggle_color_input_dim,
            )
        else:
            raise ValueError(f"{self.toggle_color_method=}")

        self.toggle_color_mlp = CustomMLP(
            input_dim=self.toggle_color_input_dim,
            intermediate_dim=self.intermediate_dim,
            output_dim=self.toggle_color_input_dim,
            num_layers=1,
        )

    def _create_mlps(self) -> None:
        """Create main MLPs for points and colors."""
        # Create points MLP
        self.mlp_points = CustomMLP(
            input_dim=self.input_dim_mlp_points,
            intermediate_dim=self.intermediate_dim,
            output_dim=self.output_dim_points,
            num_layers=self.num_layers,
        )

        # Create color MLP if needed
        if self.use_color:
            self.mlp_color = CustomMLP(
                input_dim=self.input_dim_mlp_color,
                intermediate_dim=self.intermediate_dim,
                output_dim=self.output_dim_color,
                num_layers=self.num_layers,
                color_embedding_dim=self.toggle_color_input_dim,
                use_residual_concat=self.toggle_color,
            )

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        self.width_inputs = torch.ones((1, self.num_strokes)).to(self.device) * 1.5
        self.opacities_defaults = torch.ones((1, self.num_strokes)).to(self.device)
        self.widths_output_layer = None
        self.gate = None
        self.background_color_params = nn.Parameter(torch.rand(4))

    @coach_utils.nameit_torch
    def forward(
        self,
        truncation_idx: Optional[int] = None,
        toggle_color_value: Optional[str] = None,
        toggle_aspect_ratio_value: Optional[str] = None,
        indices_to_pass: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Forward pass through network."""
        # Determine number of shapes to process
        number_of_shapes = (
            truncation_idx if truncation_idx is not None else self.num_strokes
        )

        # Get indices to process
        indices_to_pass = (
            indices_to_pass
            if indices_to_pass is not None
            else list(range(number_of_shapes))
        )

        # Get outputs for all strokes
        outputs_dict = self.forward_single_stroke(
            indices_to_pass,
            toggle_color_value=toggle_color_value,
            toggle_aspect_ratio_value=toggle_aspect_ratio_value,
        )

        # Process points
        strokes = self._process_strokes(outputs_dict["points"])

        # Process colors if enabled
        colors = self._process_colors(outputs_dict.get("color"))

        # Get opacities and widths
        opacities = self.opacities_defaults
        widths = self.width_inputs

        return strokes, widths, opacities, colors

    def _process_strokes(self, points: Tensor) -> Tensor:
        """Process stroke points tensor."""
        bs, _ = points.shape
        strokes = points.reshape(bs, -1, 2)
        strokes = strokes.unsqueeze(0)
        return self.points_prediction_scale * strokes

    def _process_colors(self, colors: Optional[Tensor]) -> Optional[Tensor]:
        """Process colors tensor."""
        if not self.use_color or colors is None:
            return None
        colors = colors.T
        return nn.Sigmoid()(colors)

    @coach_utils.nameit_torch
    def forward_single_stroke(
        self,
        indices_to_pass: int,
        toggle_color_value: Optional[str] = None,
        toggle_aspect_ratio_value: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for single stroke."""
        # Get base positional encoding
        stroke_encoding = self._get_base_encoding(indices_to_pass)

        # Add aspect ratio encoding if needed
        stroke_encoding_mlp_points = self._add_aspect_ratio_encoding(
            stroke_encoding, toggle_aspect_ratio_value
        )

        # Add color encoding if needed
        stroke_encoding_mlp_color, color_embedding = self._add_color_encoding(
            stroke_encoding, toggle_color_value
        )

        # Generate outputs
        return self._generate_outputs(
            stroke_encoding_mlp_points, stroke_encoding_mlp_color, color_embedding
        )

    def _get_base_encoding(self, indices_to_pass: int) -> torch.Tensor:
        """Get base positional encoding."""
        indices_b = (
            torch.tensor(indices_to_pass, device=self.device).float().unsqueeze(-1)
        )
        encoding = self.pe.forward(indices_b)

        if self.use_dropout_value:
            dropout_encoding = self.dropout_pe.forward(indices_b)
            encoding = torch.cat((encoding, dropout_encoding), dim=-1)

        return encoding

    def _add_aspect_ratio_encoding(
        self, base_encoding: torch.Tensor, toggle_aspect_ratio_value: Optional[str]
    ) -> torch.Tensor:
        """Add aspect ratio encoding."""
        if not self.toggle_aspect_ratio:
            return base_encoding

        if toggle_aspect_ratio_value is None:
            raise ValueError(f"{toggle_aspect_ratio_value=}")

        ratio_float_value = torch.tensor(
            [calculate_ratio(ratio_string=toggle_aspect_ratio_value)],
            device=self.device,
        ).unsqueeze(-1)

        aspect_ratio_encoding = self.aspect_ratio_pe.forward(ratio_float_value)
        aspect_ratio_encoding = aspect_ratio_encoding.expand(base_encoding.shape[0], -1)

        return torch.cat((base_encoding, aspect_ratio_encoding), dim=1)

    def _add_color_encoding(
        self, base_encoding: torch.Tensor, toggle_color_value: Optional[str]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add color encoding."""
        if not self.toggle_color:
            return base_encoding, None

        if toggle_color_value is None:
            raise ValueError(f"{toggle_color_value=}")

        color_encoding = self._get_color_encoding(toggle_color_value)
        color_embedding = self.toggle_color_mlp(color_encoding)
        color_embedding = color_embedding.expand(base_encoding.shape[0], -1)

        return (torch.cat((base_encoding, color_embedding), dim=1), color_embedding)

    def _get_color_encoding(
        self, toggle_color_value: Union[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get color encoding based on method."""
        if self.toggle_color_method == "discrete":
            color_number = self.toggle_color_bg_colors.index(toggle_color_value)
            return self.toggle_color_pe.forward(
                torch.tensor([color_number], device=self.device).float()
            )
        elif self.toggle_color_method == "rgb":
            if isinstance(toggle_color_value, str):
                rgba = self.color_name_to_value_map[toggle_color_value]
            elif isinstance(toggle_color_value, torch.Tensor):
                rgba = toggle_color_value
            else:
                raise ValueError(f"{type(toggle_color_value)=}")

            rgb = rgba[:3].to(device=self.device).unsqueeze(-1)
            return self.toggle_color_pe.forward(rgb)
        else:
            raise ValueError(f"{self.toggle_color_method=}")

    def _generate_outputs(
        self,
        stroke_encoding_points: torch.Tensor,
        stroke_encoding_color: torch.Tensor,
        color_embedding: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Generate network outputs."""
        outputs = {"points": self.mlp_points(stroke_encoding_points)}

        if self.use_color:
            outputs["color"] = self.mlp_color(
                stroke_encoding_color, color_embedding=color_embedding
            )

        return outputs

    def apply_nested_dropout(
        self,
        embedding: torch.Tensor,
        truncation_idx: Optional[int] = None,
        end_truncation_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply nested dropout to embedding."""
        if truncation_idx is None and self.use_nested_dropout:
            raise ValueError("Truncation index required when nested dropout is enabled")

        if truncation_idx is None:
            return embedding

        if truncation_idx is not None and end_truncation_idx is not None:
            if truncation_idx > end_truncation_idx:
                raise ValueError(f"{truncation_idx=}, {end_truncation_idx=}")

        for idx in torch.arange(embedding.shape[0]):
            if end_truncation_idx is None:
                embedding[idx][truncation_idx:] = 0
            else:
                embedding[idx][truncation_idx:end_truncation_idx] = 0

        return embedding

    @staticmethod
    def sample_truncation_idx(
        vector_size: int = 16, start_idx: int = 1, sampling_method: str = "uniform"
    ) -> int:
        """Sample truncation index."""
        if sampling_method == "uniform":
            return random.randint(start_idx, vector_size)
        elif sampling_method == "exp_decay":
            dist = exponential_decay_distribution(
                size=vector_size - start_idx + 1, temperature=1.5, last_item_prob=0.5
            )
            return np.random.choice(
                np.arange(start_idx, vector_size + 1), size=1, p=dist
            ).item()
        else:
            raise ValueError(f"{sampling_method=}")


class GaussianEmbedding(nn.Module):
    """Gaussian positional embedding for strokes."""

    def __init__(
        self,
        in_channels: int,
        num_shapes: int,
        emb_size: int,
        scale: int = 10,
        eps: float = 1e-4,
        skip_normalization: bool = False,
    ):
        """Initialize Gaussian embedding."""
        super().__init__()
        self.in_channels = in_channels
        self.num_shapes = num_shapes
        self.funcs = [torch.sin, torch.cos]
        self.bvals = nn.Parameter(
            torch.normal(0, 1, (emb_size // 2, in_channels)) * scale
        )
        self.eps = eps
        self.skip_normalization = skip_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding."""
        if not self.skip_normalization:
            x = self.normalize(x)
        x += self.eps

        out = []
        for func in self.funcs:
            out += [func((2.0 * torch.pi * x) @ self.bvals.T)]

        return torch.cat(out, -1)

    def normalize(self, shape_id: torch.Tensor) -> torch.Tensor:
        """Normalize input to [-1, 1] range."""
        return 2 * (shape_id / (self.num_shapes - 1)) - 1


class GaussianEmbeddingRGB(nn.Module):
    """Gaussian embedding for RGB colors."""

    def __init__(
        self,
        in_channels: int,
        emb_size: int,
        scale: int = 10,
    ):
        """Initialize RGB Gaussian embedding."""
        super().__init__()

        if emb_size % 3 != 0:
            raise ValueError(f"{emb_size=}")

        color_emb_size = emb_size // 3

        # Create separate embeddings for each channel
        # RGB inputs are in range [0,1], so num_shapes=2 effectively
        # achieves the normalization
        self.pe_red = GaussianEmbedding(
            in_channels=in_channels,
            emb_size=color_emb_size,
            num_shapes=2,
            scale=scale,
            eps=1e-6,
        )
        self.pe_green = GaussianEmbedding(
            in_channels=in_channels,
            emb_size=color_emb_size,
            num_shapes=2,
            scale=scale,
            eps=1e-6,
        )
        self.pe_blue = GaussianEmbedding(
            in_channels=in_channels,
            emb_size=color_emb_size,
            num_shapes=2,
            scale=scale,
            eps=1e-6,
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Forward pass through RGB embedding."""
        if rgb.shape[0] != 3:
            raise ValueError(f"{rgb=}")

        # Process each channel
        enc_red = self.pe_red(rgb[0].unsqueeze(0))
        enc_green = self.pe_green(rgb[1].unsqueeze(0))
        enc_blue = self.pe_blue(rgb[2].unsqueeze(0))

        # Concatenate channel embeddings
        return torch.cat((enc_red, enc_green, enc_blue), -1)


def init_weights(m: nn.Module) -> None:
    """Initialize network weights."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def exponential_decay_distribution(
    size: int, temperature: float = 1.5, last_item_prob: float = 0.2
) -> np.ndarray:
    """Create exponential decay distribution."""
    if not (0 <= last_item_prob <= 1):
        raise ValueError(f"{last_item_prob=}")

    # Calculate weights with exponential decay
    weights = np.exp(np.linspace(0, -temperature, size - 1))
    normalized_weights = weights / weights.sum() * (1 - last_item_prob)

    # Create final distribution
    distribution = np.zeros(size)
    distribution[:-1] = normalized_weights
    distribution[-1] = last_item_prob

    return distribution


def calculate_ratio(ratio_string: str) -> float:
    """Calculate aspect ratio from string."""
    try:
        # Split the string by ':'
        x, y = ratio_string.split(":")

        # Convert to float to handle decimal numbers
        x = float(x)
        y = float(y)

        # Check for division by zero
        if y == 0:
            raise ValueError("Cannot divide by zero")

        # Return the ratio
        return y / x

    except ValueError as e:
        # Handle cases like:
        # - Incorrect format (not enough or too many ':')
        # - Non-numeric values
        # - Division by zero
        raise ValueError(f"Invalid ratio format: {e}")


class CustomMLP(nn.Module):
    """MLP with optional residual connections."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        num_layers: int,
        color_embedding_dim: int = 0,
        use_residual_concat: bool = False,
    ):
        """Initialize custom MLP."""
        super().__init__()

        if use_residual_concat and color_embedding_dim == 0:
            raise ValueError(f"{use_residual_concat=}, {color_embedding_dim=}")

        # Handle residual settings
        self.use_residual_concat = use_residual_concat
        self.color_embedding_dim = color_embedding_dim if use_residual_concat else 0

        # Set dimensions
        self.intermediate_dim = intermediate_dim
        self.intermediate_dim_extended = intermediate_dim + self.color_embedding_dim

        # Create layers
        self._create_layers(input_dim, output_dim, num_layers)

    def _create_layers(self, input_dim: int, output_dim: int, num_layers: int) -> None:
        """Create network layers."""
        # First layer
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, self.intermediate_dim),
            nn.LayerNorm(self.intermediate_dim),
            nn.LeakyReLU(),
        )

        # Intermediate layers
        self.intermediate_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.intermediate_dim_extended, self.intermediate_dim),
                    nn.LayerNorm(self.intermediate_dim),
                    nn.LeakyReLU(),
                )
                for _ in range(num_layers - 1)
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(self.intermediate_dim_extended, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        color_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through MLP."""
        # First layer
        x = self.first_layer(x)

        # Intermediate layers with optional residual connections
        for layer in self.intermediate_layers:
            if self.use_residual_concat and color_embedding is not None:
                x = torch.cat((x, color_embedding), dim=1)
            x = layer(x)

        # Output layer with optional residual connection
        if self.use_residual_concat and color_embedding is not None:
            x = torch.cat((x, color_embedding), dim=1)

        return self.output_layer(x)
