from pathlib import Path
import random
from torch import nn
import torch
import numpy as np
import pydiffvg
from src.configs.train_config import TrainConfig
from src.data import data_setups
from src.models.attention_utils import (
    get_clip_attention_map,
    set_init_strokes_with_attention_map,
)
from typing import Optional

"""
Neural painter model implementation for NeuralSVG.
"""


class Painter(nn.Module):
    """
    Neural painter model for generating vector graphics from text.
    """

    def __init__(self, cfg: TrainConfig, device: str = "cuda"):
        """
        Initialize the Painter model.
        """
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.canvas_width, self.canvas_height = (
            cfg.data.render_size,
            cfg.data.render_size,
        )
        self.num_strokes = cfg.data.num_strokes
        self.num_segments = cfg.data.num_segments
        self.is_closed = cfg.data.is_closed
        self.radius = cfg.data.radius
        self.control_points_per_seg = cfg.data.control_points_per_seg
        self.regular_polygon_closed_shape_init = (
            cfg.data.regular_polygon_closed_shape_init
        )

        self.num_control_points_per_shape = (
            torch.zeros(self.num_segments, dtype=torch.long)
            + self.control_points_per_seg
        )

        self.shape_to_num_points = []
        for _ in range(self.num_strokes):
            num_points = Painter._get_total_num_points(
                control_points_per_seg=self.control_points_per_seg,
                num_segments=self.num_segments,
                is_closed=self.is_closed,
            )
            self.shape_to_num_points.append(num_points)

        self.num_control_points = (
            torch.zeros(self.num_segments, dtype=torch.long)
            + self.control_points_per_seg
        )

        if self.control_points_per_seg not in [0, 1, 2]:
            raise ValueError(f"{self.control_points_per_seg=}")

        # Useful for debug
        self.strokes_constant_colors = [
            torch.FloatTensor(np.random.uniform(size=[4]))
            for _ in range(self.num_strokes)
        ]
        self.total_num_points = self._get_total_num_points(
            control_points_per_seg=self.control_points_per_seg,
            num_segments=self.num_segments,
            is_closed=self.is_closed,
        )

        self.mlp = None
        self.radiuses = self._init_radiuses()

        # Initialize strokes either via attention saliency or randomly
        (
            self.image,
            self.image_tensor,
            self.attn_map,
            self.attn_map_soft_list,
            self.inds,
            self.stroke_idxs,
        ) = (None, None, None, None, None, None)
        if self.cfg.data.image_path is not None:
            (
                image,
                image_tensor,
                attn_map,
                attn_map_soft_list,
                background_attn_map_soft_list,
                inds,
                inds_normalised,
                bg_inds,
                bg_inds_normalised,
            ) = self._init_strokes_with_attention_map()
            self.image = image
            self.image_tensor = image_tensor
            self.attn_map = attn_map
            self.attn_map_soft_list = attn_map_soft_list
            self.background_attn_map_soft_list = background_attn_map_soft_list
            self.inds = inds
            self.bg_inds = bg_inds
            self.stroke_idxs = inds_normalised

        if self.cfg.model.toggle_color:
            self.shapes_init_colors = self._init_colors_toggled(
                eps=self.cfg.model.toggle_color_init_eps
            )
        else:
            self.shapes_init_colors = self._init_colors()
        self.strokes, self.shape_groups, self.points_init, self.shape_groups_colored = (
            self._init_strokes()
        )
        self.points_init = torch.stack(self.points_init)
        self.init_widths = torch.ones((1, self.num_strokes)).to(device) * 1.5

    def mlp_pass(self, mode: str, eps: float = 1e-4, truncation_idx: int = None):
        raise NotImplementedError()

    def render_scene(self, shapes, shape_groups):
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, shapes, shape_groups
        )
        img = _render(
            self.canvas_width,  # width
            self.canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args,
        )
        return img

    def get_points(
        self,
        mode="train",
        truncation_indices: list[int] = None,
        toggle_color_value: Optional[str] = None,
    ):
        _, points, _, _, shape_color_rgb = self.mlp_pass(
            mode,
            truncation_indices=truncation_indices,
            toggle_color_value=toggle_color_value,
        )
        return points, shape_color_rgb

    def get_image(
        self,
        mode="train",
        truncation_indices: list[int] = None,
        sub_layers_sizes: list[int] = None,
        toggle_color_value: Optional[str] = None,
        render_without_background: bool = False,
        toggle_aspect_ratio_value: Optional[str] = None,
        remove_fill_color: bool = False,
        debug_aspect_ratio: bool = False,
        aspect_ratio_white_rects: bool = False,
    ):
        if not self.mlp:
            raise ValueError(f"{self.mlp=}")

        if mode != "init":
            # MLP pred
            img, points, _, opacities, shape_color_rgb = self.mlp_pass(
                mode,
                truncation_indices=truncation_indices,
                sub_layers_sizes=sub_layers_sizes,
                toggle_color_value=toggle_color_value,
                render_without_background=render_without_background,
                toggle_aspect_ratio_value=toggle_aspect_ratio_value,
                remove_fill_color=remove_fill_color,
                debug_aspect_ratio=debug_aspect_ratio,
                aspect_ratio_white_rects=aspect_ratio_white_rects,
            )
        else:
            # Construct scene using the strict rules of circular initialization
            raise Exception("'init' mode deprecated")

        img = self.handle_raw_image(img)

        return img, points, None, opacities, shape_color_rgb

    def handle_raw_image(self, img):
        """
        Handle raw image
        https://github.com/BachiLi/diffvg/blob/85802a71fbcc72d79cb75716eb4da4392fd09532/apps/refine_svg.py#L64
        """
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device
        ) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def save_svg(self, output_dir: Path, name: str) -> None:
        print(f"Save SVG @ {output_dir}/{name}.svg")
        pydiffvg.save_svg(
            "{}/{}.svg".format(output_dir, name),
            self.canvas_width,
            self.canvas_height,
            self.strokes,
            self.shape_groups,
        )
        return

    def _init_strokes(self):
        """ """
        strokes, shape_groups, points_init, shape_groups_colored = [], [], [], []
        for idx in range(self.cfg.data.num_strokes):
            if self.cfg.model.toggle_color:
                stroke_color_init = self.shapes_init_colors[
                    self.cfg.model.toggle_color_bg_colors[0]
                ][
                    idx
                ]  # uses first bg color
            else:
                stroke_color_init = self.shapes_init_colors[idx]
            stroke_color_black = torch.tensor([0.0, 0.0, 0.0, 1.0])
            stroke_color_colored = self.strokes_constant_colors[idx]

            path, points = (
                self._get_shape(idx) if self.is_closed else self._get_path(idx)
            )
            # path, points = self._get_shape(idx)
            strokes.append(path)
            points_init.append(points)
            # print(f"{points.shape=}")
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(strokes) - 1]),
                fill_color=stroke_color_init if self.is_closed else None,
                stroke_color=(
                    stroke_color_black if not self.is_closed else None
                ),  # Don't use stroke color if is_closed
            )
            shape_groups.append(shape_group)
            shape_group_colored = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(strokes) - 1]),
                fill_color=stroke_color_colored if self.is_closed else None,
                stroke_color=stroke_color_colored,
            )
            shape_groups_colored.append(shape_group_colored)

        return strokes, shape_groups, points_init, shape_groups_colored

    def _get_path(self, idx: int) -> tuple[pydiffvg.Path, list]:
        points = []
        total_num_points = self.shape_to_num_points[idx]
        num_control_points = self.num_control_points_per_shape

        p0 = (
            self.stroke_idxs[idx]
            if self.stroke_idxs
            else (random.random(), random.random())
        )
        points.append(p0)
        radius = self.radiuses[idx]
        for _ in range(total_num_points - 1):
            p1 = (
                p0[0] + radius * (random.random() - 0.5),
                p0[1] + radius * (random.random() - 0.5),
            )
            points.append(p1)
            p0 = p1
        points = torch.tensor(points).to(self.device, dtype=torch.float32)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(self.cfg.data.width).to(self.device),
            is_closed=self.is_closed,
        )
        return path, points

    def _get_shape(self, idx: int) -> tuple[pydiffvg.Path, list]:
        points = []
        total_num_points = self.shape_to_num_points[idx]
        num_control_points = self.num_control_points_per_shape

        width = self.cfg.data.width

        p0 = (
            self.stroke_idxs[idx]
            if self.stroke_idxs
            else (random.random(), random.random())
        )

        radius = self.radiuses[idx]
        if self.cfg.data.regular_polygon_closed_shape_init:
            # Use p0 as the center
            points = self.create_polygon(
                center=p0, radius=radius, num_edges=self.num_segments * 3
            )
        else:
            points.append(p0)
            for _ in range(total_num_points - 1):
                p1 = (
                    p0[0] + radius * (random.random() - 0.5),
                    p0[1] + radius * (random.random() - 0.5),
                )
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device, dtype=torch.float32)
        eps = 0.00001
        noise = eps * torch.randn_like(points)
        # points += noise

        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(width).to(self.device),
            is_closed=self.is_closed,
        )
        return path, points

    def _init_colors(self) -> torch.Tensor:
        init_color_lst = []
        for p in range(self.num_strokes):
            href, wref = self.inds[p]
            init_color = torch.clone(self.image_tensor[0, :, href, wref])
            init_opacity = torch.tensor([1])
            init_color_lst.append(torch.cat((init_color, init_opacity)))

        init_color_tns = torch.stack(init_color_lst)
        init_color_tns = init_color_tns.to(self.device)
        return init_color_tns

    def _init_colors_toggled(self, eps=1e-1) -> dict[str, torch.Tensor]:
        ans = dict()
        # rand_init_color = [random.uniform(55, 200) / 255 for _ in range(3)]
        # rand_init_color.append(1)  # opacity

        for color_name in self.cfg.model.toggle_color_bg_colors:
            init_color_lst = []
            for p in range(self.num_strokes):
                href, wref = self.inds[p]
                init_color = torch.clone(self.image_tensor[0, :, href, wref])
                init_opacity = torch.tensor([1])
                init_color_lst.append(torch.cat((init_color, init_opacity)))

            init_color_tns = torch.stack(init_color_lst)
            init_color_tns += torch.normal(mean=0.0, std=eps, size=init_color_tns.shape)
            init_color_tns = torch.clamp(init_color_tns, min=0, max=1)
            init_color_tns = init_color_tns.to(self.device)

            ans[color_name] = init_color_tns

        return ans

    def _init_radiuses(self) -> np.ndarray:
        return np.full(self.num_strokes, self.radius)

    @staticmethod
    def _get_total_num_points(
        control_points_per_seg: int, num_segments: int, is_closed: bool
    ) -> int:
        num_points = 0
        num_points += (
            control_points_per_seg + 2
        )  # Control points and two base points for the first segment
        num_points += (control_points_per_seg + 1) * (
            num_segments - 1
        )  # For any additional segment don't need to count the first base point
        num_points -= int(
            is_closed
        )  # If is_closed, the last base point is the first point of the shape
        return num_points

    @staticmethod
    def create_polygon(
        center: tuple[float, float], radius: float, num_edges: int
    ) -> torch.Tensor:
        cx, cy = center
        # Calculate the angle between each edge
        angle = 2 * np.pi / num_edges

        # Adding a random constant for the rotation
        use_rotation = False
        if use_rotation:
            rand_constant = random.uniform(0, 2 * np.pi)
        else:
            rand_constant = 0

        # Generate the vertices of the polygon
        vertices = [
            (
                cx + radius * np.cos(i * angle + rand_constant),
                cy + radius * np.sin(i * angle + rand_constant),
            )
            for i in range(num_edges)
        ]
        vertices_tns = torch.tensor(vertices)
        return vertices_tns

    def _init_strokes_with_attention_map(self):
        image, mask, image_tensor = data_setups.prepare_image(
            self.cfg.data.image_path, cfg=self.cfg.data
        )
        input_image, attn_map = get_clip_attention_map(
            input_image=image, image_size=self.canvas_width, device=self.device
        )
        (
            attn_map_soft_list,
            background_attn_map_soft_list,
            inds,
            inds_normalised,
            bg_inds,
            bg_inds_normalised,
        ) = set_init_strokes_with_attention_map(
            attention_map=attn_map,
            input_image=input_image,
            num_strokes=self.cfg.data.num_strokes,
            xdog_intersec=self.cfg.data.attn_init_xdog_intersec,
            image_size=self.canvas_width,
            mask=mask,
            tau_max_min=self.cfg.data.attn_init_tau_max_min,
        )
        return (
            image,
            image_tensor,
            attn_map,
            attn_map_soft_list,
            background_attn_map_soft_list,
            inds,
            inds_normalised,
            bg_inds,
            bg_inds_normalised,
        )

    def _is_in_canvas(self, canvas_width: int, canvas_height: int, path: pydiffvg.Path):
        shapes, shape_groups = [], []
        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=stroke_color if self.is_closed else None,
            stroke_color=stroke_color if not self.is_closed else None,
        )
        shape_groups.append(path_group)
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            shapes=shapes,
            shape_groups=shape_groups,
        )
        img = _render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args,
        )
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].detach().cpu().numpy()
        return (1 - img).sum()

    def render_warp(self):

        img = self.render_scene(self.strokes, self.shape_groups)
        img_colored = self.render_scene(self.strokes, self.shape_groups_colored)

        all_points = torch.stack([s.points for s in self.strokes]).unsqueeze(0)
        return img, all_points, img_colored

    def get_parameters(self):
        if self.mlp is not None:
            return list(self.mlp.parameters())
        else:
            return [path.points for path in self.strokes]
