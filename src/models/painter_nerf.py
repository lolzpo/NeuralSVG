"""Neural renderer for vector graphics.

Core functionality:
- Shape generation with NeRF-based MLP
- Color and background handling
- SVG output
- Dynamic conditioning
"""

import torch
import pydiffvg
import webcolors
import random
from typing import Optional, List, Dict, Tuple, Union

from src.training.utils import coach_utils
from src.configs.train_config import TrainConfig
from src.models.nerf_mlp_multi import NerfMLPMulti, init_weights
from src.models.painter import Painter

# Type aliases
ColorValue = Union[str, torch.Tensor]
ShapeList = List[Union[pydiffvg.Path, pydiffvg.Rect]]
ShapeGroupList = List[pydiffvg.ShapeGroup]


class PainterNerf(Painter):
    """Neural renderer for vector graphics using NeRF architecture."""

    def __init__(self, cfg: TrainConfig, device: str = "cuda"):
        """Initialize renderer with config and device."""
        super().__init__(cfg, device)
        total_num_points = max(self.shape_to_num_points)
        self.color_name_to_value_map = self.get_color_name_to_value_map()

        self.mlp = self._initialize_mlp(total_num_points)
        if self.cfg.model.checkpoint_path is None:
            self.mlp.apply(init_weights)

        self.use_background = self.cfg.data.use_background

    def _initialize_mlp(self, total_num_points: int) -> NerfMLPMulti:
        """Initialize NeRF MLP model."""
        return NerfMLPMulti(
            num_strokes=self.cfg.data.num_strokes,
            total_num_points=total_num_points,
            intermediate_dim=self.cfg.model.mlp_dim,
            num_layers=self.cfg.model.mlp_num_layers,
            use_nested_dropout=self.cfg.model.use_nested_dropout,
            use_color=self.cfg.model.use_color,
            truncation_start_idx=self.cfg.model.truncation_start_idx,
            input_dim=self.cfg.model.input_dim,
            points_prediction_scale=self.cfg.model.points_prediction_scale,
            use_dropout_value=self.cfg.model.use_dropout_value,
            dropout_emb_dim=self.cfg.model.dropout_emb_dim,
            toggle_color=self.cfg.model.toggle_color,
            toggle_color_method=self.cfg.model.toggle_color_method,
            toggle_color_input_dim=self.cfg.model.toggle_color_input_dim,
            toggle_color_bg_colors=self.cfg.model.toggle_color_bg_colors,
            color_name_to_value_map=self.color_name_to_value_map,
            toggle_aspect_ratio=self.cfg.model.toggle_aspect_ratio,
            toggle_aspect_ratio_values=self.cfg.model.toggle_aspect_ratio_values,
            aspect_ratio_emb_dim=self.cfg.model.aspect_ratio_emb_dim,
        )

    def mlp_pass(
        self,
        mode: str,
        eps: float = 1e-4,
        truncation_indices: list[int] = None,
        sub_layers_sizes: list[int] = None,
        toggle_color_value: Optional[ColorValue] = None,
        render_without_background: bool = False,
        toggle_aspect_ratio_value: Optional[str] = None,
        remove_fill_color: bool = False,
        indices_to_pass: Optional[list[int]] = None,
        debug_aspect_ratio: bool = False,
        aspect_ratio_white_rects: bool = False,
    ) -> Tuple:
        """Generate vector graphics through MLP forward pass."""
        # Initialize toggle values
        toggle_color_value, toggle_aspect_ratio_value = self._initialize_toggle_values(
            toggle_color_value, toggle_aspect_ratio_value
        )

        # Validate and process sub-layers
        sub_layers_sizes = self._validate_sub_layers(sub_layers_sizes)

        # Handle nested dropout and get truncation index
        truncation_idx = self._handle_nested_dropout(truncation_indices)

        # Get MLP predictions
        points, widths, opacities, colors = self._get_mlp_predictions(
            truncation_idx,
            toggle_color_value,
            toggle_aspect_ratio_value,
            debug_aspect_ratio,
            indices_to_pass,
        )

        # Process points and apply aspect ratio
        all_points = self._process_points(points, eps, toggle_aspect_ratio_value)

        # Process widths and opacities
        widths, opacities = self._process_widths_and_opacities(
            widths, opacities, truncation_idx
        )

        # Create shapes and shape groups
        shapes, shape_groups, stroke_colors = self._create_shapes_and_groups(
            all_points,
            widths,
            colors,
            opacities,
            truncation_idx,
            indices_to_pass,
            mode,
            remove_fill_color,
            toggle_color_value,
            render_without_background,
            toggle_aspect_ratio_value,
            aspect_ratio_white_rects,
        )

        # Render the scene and update instance variables
        img = self.render_scene(shapes, shape_groups)
        self.strokes = shapes.copy()
        self.shape_groups = shape_groups.copy()

        return img, all_points, None, opacities, stroke_colors

    def _initialize_toggle_values(
        self,
        toggle_color_value: Optional[ColorValue],
        toggle_aspect_ratio_value: Optional[str],
    ) -> Tuple:
        """Initialize color and aspect ratio toggle values."""
        if toggle_color_value is None and self.cfg.model.toggle_color:
            toggle_color_value = random.choice(self.cfg.model.toggle_color_bg_colors)

        if toggle_aspect_ratio_value is None and self.cfg.model.toggle_aspect_ratio:
            toggle_aspect_ratio_value = "1:1"

        return toggle_color_value, toggle_aspect_ratio_value

    def _validate_sub_layers(self, sub_layers_sizes: Optional[List[int]]) -> List[int]:
        """Validate and process sub-layer sizes."""
        if sub_layers_sizes is None:
            return [self.num_strokes]

        if not isinstance(sub_layers_sizes, list):
            raise ValueError(f"{type(sub_layers_sizes)=}")

        if sum(sub_layers_sizes) != self.num_strokes:
            raise ValueError(
                f"sub_layers_sizes list should sum to {self.num_strokes}, "
                f"but got {sum(sub_layers_sizes)=}"
            )

        if not all(x == sub_layers_sizes[0] for x in sub_layers_sizes):
            raise ValueError(
                f"At the moment supporting only equal layers sizes! "
                f"{sub_layers_sizes=}"
            )

        return sub_layers_sizes

    def _handle_nested_dropout(
        self, truncation_indices: Optional[List[int]]
    ) -> Optional[int]:
        """Handle nested dropout and return truncation index."""
        if self.cfg.model.use_nested_dropout and truncation_indices is None:
            truncation_idx = NerfMLPMulti.sample_truncation_idx(
                self.num_strokes,
                start_idx=self.cfg.model.truncation_start_idx,
                sampling_method=self.cfg.model.nested_dropout_sampling_method,
            )
            truncation_indices = [truncation_idx]

        return truncation_indices[0] if truncation_indices else None

    def _get_mlp_predictions(
        self,
        truncation_idx: Optional[int],
        toggle_color_value: Optional[ColorValue],
        toggle_aspect_ratio_value: Optional[str],
        debug_aspect_ratio: bool,
        indices_to_pass: Optional[List[int]],
    ) -> Tuple:
        """Get predictions from MLP."""
        aspect_ratio_value = "1:1" if debug_aspect_ratio else toggle_aspect_ratio_value
        return self.mlp(
            truncation_idx=truncation_idx,
            toggle_color_value=toggle_color_value,
            toggle_aspect_ratio_value=aspect_ratio_value,
            indices_to_pass=indices_to_pass,
        )

    def _process_points(
        self, points: torch.Tensor, eps: float, toggle_aspect_ratio_value: Optional[str]
    ) -> torch.Tensor:
        """Process points tensor and apply aspect ratio."""
        all_points = 0.5 * (points + 1.0) * self.canvas_width
        all_points = all_points + eps * torch.randn_like(all_points)

        if toggle_aspect_ratio_value == "4:1":
            ratio = 0.25
            all_points[:, :, :, 1] = all_points[:, :, :, 1] * ratio + (1 - ratio) * (
                self.canvas_height / 2
            )
        elif toggle_aspect_ratio_value not in ["1:1", None]:
            raise ValueError(f"{toggle_aspect_ratio_value=}")

        return all_points

    def _process_widths_and_opacities(
        self,
        widths: torch.Tensor,
        opacities: torch.Tensor,
        truncation_idx: Optional[int],
    ) -> Tuple:
        """Process width and opacity tensors."""
        widths = self.mlp.apply_nested_dropout(
            self.init_widths.clone().detach(),
            truncation_idx=truncation_idx,
        )
        opacities = self.mlp.apply_nested_dropout(
            opacities.clone().detach(),
            truncation_idx=truncation_idx,
        )
        return widths[0], opacities

    def _create_shapes_and_groups(
        self,
        all_points: torch.Tensor,
        widths: torch.Tensor,
        colors: Optional[torch.Tensor],
        opacities: Optional[torch.Tensor],
        truncation_idx: Optional[int],
        indices_to_pass: Optional[List[int]],
        mode: str,
        remove_fill_color: bool,
        toggle_color_value: Optional[ColorValue],
        render_without_background: bool,
        toggle_aspect_ratio_value: Optional[str],
        aspect_ratio_white_rects: bool,
    ) -> Tuple:
        """Create shapes and shape groups for rendering."""
        shapes: ShapeList = []
        shape_groups: ShapeGroupList = []
        stroke_color_lst = []

        # Add background if needed
        if self.use_background:
            background_color = self._get_background_color(
                toggle_color_value, render_without_background
            )
            shapes.append(self._create_background_shape())
            shape_groups.append(
                pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=background_color,
                )
            )

        # Create strokes
        number_of_shapes = self._get_number_of_shapes(truncation_idx, indices_to_pass)

        for p in range(number_of_shapes):
            width = (
                torch.tensor(widths[p])
                if mode != "init"
                else torch.tensor(self.cfg.data.width)
            )

            path = pydiffvg.Path(
                num_control_points=self.num_control_points_per_shape,
                points=all_points[:, p, :].reshape((-1, 2)),
                stroke_width=width,
                is_closed=self.is_closed,
            )

            stroke_color = self._create_stroke_color(colors, opacities, p)
            stroke_color_lst.append(stroke_color)

            stroke_color_to_render = self._get_stroke_color_to_render(
                stroke_color, remove_fill_color
            )

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=(
                    stroke_color if (self.is_closed and not remove_fill_color) else None
                ),
                stroke_color=stroke_color_to_render,
            )
            shape_groups.append(path_group)

        # Add aspect ratio rectangles if needed
        if self.cfg.model.toggle_aspect_ratio and toggle_aspect_ratio_value == "4:1":
            self._add_aspect_ratio_rects(
                shapes, shape_groups, toggle_color_value, aspect_ratio_white_rects
            )

        return shapes, shape_groups, torch.stack(stroke_color_lst)

    def _get_background_color(
        self, toggle_color_value: Optional[ColorValue], render_without_background: bool
    ) -> torch.Tensor:
        """Get background color based on settings."""
        if render_without_background:
            return self.color_name_to_value_map["white"]
        elif toggle_color_value is None:
            return self.mlp.background_color_params
        elif isinstance(toggle_color_value, str):
            return self.color_name_to_value_map[toggle_color_value]
        elif isinstance(toggle_color_value, torch.Tensor):
            return toggle_color_value
        else:
            raise ValueError(f"{type(toggle_color_value)=}")

    def _create_background_shape(self) -> pydiffvg.Rect:
        """Create background rectangle shape."""
        return pydiffvg.Rect(
            p_min=torch.tensor([0.0, 0.0]),
            p_max=torch.tensor([self.cfg.data.render_size, self.cfg.data.render_size]),
        )

    def _get_number_of_shapes(
        self, truncation_idx: Optional[int], indices_to_pass: Optional[List[int]]
    ) -> int:
        """Get number of shapes to process."""
        if truncation_idx is not None:
            return truncation_idx
        elif indices_to_pass is not None:
            return len(indices_to_pass)
        else:
            return self.num_strokes

    def _create_stroke_color(
        self, colors: Optional[torch.Tensor], opacities: Optional[torch.Tensor], p: int
    ) -> torch.Tensor:
        """Create stroke color with opacity."""
        stroke_color_rgb = (
            colors[:, p]
            if colors is not None
            else torch.tensor([0, 0, 0], device=self.device)
        )

        stroke_color_alpha = (
            opacities[:, p]
            if opacities is not None
            else torch.tensor([1], device=self.device)
        )

        return torch.cat((stroke_color_rgb, stroke_color_alpha))

    def _get_stroke_color_to_render(
        self, stroke_color: torch.Tensor, remove_fill_color: bool
    ) -> Optional[torch.Tensor]:
        """Get stroke color for rendering."""
        if self.is_closed:
            return None
        if remove_fill_color:
            return torch.tensor([0, 0, 0, 1], device=self.device)
        return stroke_color

    def _add_aspect_ratio_rects(
        self,
        shapes: ShapeList,
        shape_groups: ShapeGroupList,
        toggle_color_value: Optional[ColorValue],
        aspect_ratio_white_rects: bool,
    ) -> None:
        """Add aspect ratio rectangles to shapes and groups."""
        rects_fill_color = (
            self._get_background_color(toggle_color_value, False)
            if not aspect_ratio_white_rects
            else self.color_name_to_value_map["white"]
        )

        # Add top rectangle
        shapes.append(
            pydiffvg.Rect(
                p_min=torch.tensor([0.0, 0.0]),
                p_max=torch.tensor(
                    [self.cfg.data.render_size, (self.cfg.data.render_size * 3) / 8]
                ),
            )
        )
        shape_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=rects_fill_color,
            )
        )

        # Add bottom rectangle
        shapes.append(
            pydiffvg.Rect(
                p_min=torch.tensor([0, (self.cfg.data.render_size * 5) / 8]),
                p_max=torch.tensor(
                    [self.cfg.data.render_size, self.cfg.data.render_size]
                ),
            )
        )
        shape_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=rects_fill_color,
            )
        )

    def get_color_name_to_value_map(self) -> Dict[str, torch.Tensor]:
        """Create mapping from color names to normalized RGB values."""
        color_name_to_value = {}

        # Custom colors
        custom_colors = {
            "light red": [238, 75, 43],
            "light green": [144, 238, 144],
            "light blue": [167, 199, 231],
            "red pastel": [238, 75, 43],
            "green pastel": [144, 238, 144],
            "blue pastel": [167, 199, 231],
            "sky blue": [135, 206, 235],
            "turquoise": [64, 224, 208],
            "brown": [165, 42, 42],
            "light gray": [144, 238, 144],
            "gold": [255, 215, 0],
            "turquoise green": [69, 196, 176],
            "tea green": [218, 253, 186],
            "charcoal blue": [37, 54, 89],
            "light salmon": [242, 116, 87],
            "dark magenta": [115, 18, 81],
            "lavender blue": [160, 163, 217],
            "light sky blue": [167, 208, 217],
        }

        # Add custom colors
        for name, rgb in custom_colors.items():
            color_name_to_value[name] = torch.tensor(rgb + [255], dtype=torch.float32)

        # Add web colors
        web_colors = [
            "aqua",
            "black",
            "blue",
            "fuchsia",
            "green",
            "gray",
            "lime",
            "maroon",
            "navy",
            "olive",
            "purple",
            "red",
            "silver",
            "teal",
            "white",
            "yellow",
        ]

        for color_name in web_colors:
            rgb = webcolors.name_to_rgb(color_name)
            color_name_to_value[color_name] = torch.tensor(
                [rgb.red, rgb.green, rgb.blue, 255], dtype=torch.float32
            )

        # Normalize all colors to [0,1] range
        for key, value in color_name_to_value.items():
            color_name_to_value[key] = value / 255

        return color_name_to_value
