# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, minimum_filter

from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    _load_label_font_at_size,
)

if TYPE_CHECKING:
    import trimesh

    from embodichain.lab.sim.scene._support_cpu import DetectedSupportSurface

__all__: list[str] = []

_CANDIDATE_COLORS = ("#D05C26", "#1576B9", "#408043")
_VIEW_SIZE = 512
_BACKGROUND = "#f6f7f9"
_TEXT = "#17232e"


def render_table_support_candidates(
    *,
    table_mesh: trimesh.Trimesh,
    candidates: Sequence[DetectedSupportSurface],
    output_path: Path,
) -> tuple[int, ...]:
    """Draw numbered support patches on one front-elevated view of a Z-up table.

    Return one-based IDs whose badge fits entirely inside the visible patch.
    The sidebar also lists candidates whose patch was occluded or too small.
    """
    if not candidates or len(candidates) > len(_CANDIDATE_COLORS):
        raise ValueError("Expected between one and three tabletop candidates.")

    bounds = table_mesh.bounds
    center = bounds.mean(axis=0)
    span = np.linalg.norm(bounds[1] - bounds[0])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Table preview requires a finite nonzero mesh extent.")

    # Match the reference's Y-up +Z view after conversion to our Z-up frame.
    elevation = np.deg2rad(25.0)
    outward = np.array([0.0, -np.cos(elevation), np.sin(elevation)])
    right = np.array([1.0, 0.0, 0.0])
    view_basis = np.stack((right, np.cross(outward, right), outward), axis=1)
    host = (table_mesh.triangles - center) / span
    projected_host = host @ view_basis
    light = outward + [0.0, 0.0, 0.7]
    light /= np.linalg.norm(light)
    shading = 0.68 + 0.32 * np.abs(table_mesh.face_normals @ light)
    layers = [projected_host]
    colors = [_host_face_colors(table_mesh) * shading[:, None]]
    surface_ids = [np.zeros(len(host), dtype=int)]
    for number, (candidate, color) in enumerate(
        zip(candidates, _CANDIDATE_COLORS), start=1
    ):
        patch = ((candidate.vertices[candidate.faces] - center) / span) @ view_basis
        # Draw coplanar colored polygons just in front of their host faces.
        patch[:, :, 2] += 2e-6
        layers.append(patch)
        colors.append(np.tile(ImageColor.getrgb(color), (len(patch), 1)))
        surface_ids.append(np.full(len(patch), number, dtype=int))

    visible_faces = _rasterize_visible_faces(np.concatenate(layers))
    palette = np.concatenate([*colors, [ImageColor.getrgb(_BACKGROUND)]]).astype(
        np.uint8
    )
    visible_ids = np.concatenate([*surface_ids, [-1]])[visible_faces]
    main_view = Image.fromarray(palette[visible_faces])
    draw = ImageDraw.Draw(main_view)
    badge_font = _load_label_font_at_size(max(18, round(_VIEW_SIZE / 28)))
    labeled = [
        number
        for number, color in enumerate(_CANDIDATE_COLORS[: len(candidates)], start=1)
        if _draw_visible_badge(
            draw, visible_ids == number, str(number), color, badge_font
        )
    ]
    # Keep the overview compact without changing or stretching the camera view.
    visible_rows = np.flatnonzero(np.any(visible_faces >= 0, axis=1))
    if visible_rows.size:
        top = max(int(visible_rows[0]) - 18, 0)
        bottom = min(int(visible_rows[-1]) + 19, _VIEW_SIZE)
        main_view = main_view.crop((0, top, _VIEW_SIZE, bottom))
    image = _compose_candidate_overview(main_view, len(candidates), labeled)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return tuple(labeled)


def _host_face_colors(table_mesh: trimesh.Trimesh) -> np.ndarray:
    """Approximate authored materials with opaque face colors for context."""
    visual = table_mesh.visual
    if visual.kind == "texture":
        try:
            visual = visual.to_color()
        except (AttributeError, TypeError, ValueError):
            pass
    if visual.kind == "vertex" and len(visual.vertex_colors) == len(
        table_mesh.vertices
    ):
        return visual.vertex_colors[table_mesh.faces, :3].mean(axis=1)
    if visual.kind == "face" and len(visual.face_colors) == len(table_mesh.faces):
        return np.asarray(visual.face_colors[:, :3], dtype=float)
    return np.full((len(table_mesh.faces), 3), 197.0)


def _draw_visible_badge(
    draw: ImageDraw.ImageDraw,
    visible: np.ndarray,
    text: str,
    color: str,
    font: ImageFont.ImageFont,
) -> bool:
    """Draw a white badge only if its entire outline fits on the patch."""
    if not visible.any():
        return False
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    half_width = int(np.ceil((right - left) / 2)) + 5
    half_height = int(np.ceil((bottom - top) / 2)) + 5
    radius = max(half_width, half_height)
    distance = distance_transform_edt(np.pad(visible, 1))[1:-1, 1:-1]
    circular = distance.max() > radius + 1
    if circular:
        half_width = half_height = radius
    else:
        fits = minimum_filter(
            visible,
            size=(2 * half_height + 3, 2 * half_width + 3),
            mode="constant",
            cval=0,
        )
        if not fits.any():
            return False
        distance[~fits] = -np.inf

    positions = np.argwhere(distance == distance.max())
    center = np.argwhere(visible).mean(axis=0)
    y, x = positions[np.sum((positions - center) ** 2, axis=1).argmin()]
    bounds = (x - half_width, y - half_height, x + half_width, y + half_height)
    if circular:
        draw.ellipse(bounds, fill="white", outline=color, width=2)
    else:
        draw.rounded_rectangle(bounds, radius=4, fill="white", outline=color, width=2)
    draw.text(
        (x - (left + right) / 2, y - (top + bottom) / 2), text, font=font, fill=_TEXT
    )
    return True


def _compose_candidate_overview(
    main_view: Image.Image, candidate_count: int, labeled: list[int]
) -> Image.Image:
    """Keep exact response IDs in a sidebar next to the projected view."""
    font_size = max(14, round(main_view.width / 35))
    font = _load_label_font_at_size(font_size)
    heading_font = _load_label_font_at_size(font_size + 3)
    small_font = _load_label_font_at_size(max(12, font_size - 2))
    header_height = 50
    row_height = 55
    statuses = ("number on visible surface", "not visible / too small")
    measure = ImageDraw.Draw(main_view)
    text_width = max(measure.textlength(status, font=small_font) for status in statuses)
    image = Image.new(
        "RGB",
        (
            main_view.width + max(195, int(np.ceil(text_width)) + 38),
            header_height + max(main_view.height, 38 + candidate_count * row_height),
        ),
        _BACKGROUND,
    )
    image.paste(
        main_view,
        (0, header_height + (image.height - header_height - main_view.height) // 2),
    )
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), "NUMBERED SURFACES", font=heading_font, fill=_TEXT)
    draw.text(
        (16, 33),
        "Choose the number on a visible surface",
        font=small_font,
        fill="#455563",
    )
    panel_left = main_view.width + 5
    draw.text((panel_left + 6, header_height + 8), "Surface ID", font=font, fill=_TEXT)
    for number, color in enumerate(_CANDIDATE_COLORS[:candidate_count], start=1):
        row_top = header_height + 32 + (number - 1) * row_height
        draw.rounded_rectangle(
            (panel_left, row_top, image.width - 12, row_top + row_height - 7),
            radius=6,
            fill="white",
            outline=color,
            width=2,
        )
        draw.text(
            (panel_left + 9, row_top + 6),
            f"{number} (surface_{number})",
            font=font,
            fill=_TEXT,
        )
        status = statuses[0] if number in labeled else statuses[1]
        draw.text((panel_left + 9, row_top + 26), status, font=small_font, fill=color)
    return image


def _rasterize_visible_faces(triangles: np.ndarray) -> np.ndarray:
    """Return the closest triangle at each pixel of an orthographic view."""
    triangles[:, :, :2] *= [_VIEW_SIZE / 1.08, -_VIEW_SIZE / 1.08]
    triangles[:, :, :2] += (_VIEW_SIZE - 1) / 2
    lower = np.ceil(triangles[:, :, :2].min(axis=1)).astype(int)
    upper = np.floor(triangles[:, :, :2].max(axis=1)).astype(int) + 1
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, _VIEW_SIZE)
    starts = triangles[:, 0]
    edges_1 = triangles[:, 1] - starts
    edges_2 = triangles[:, 2] - starts
    determinants = edges_1[:, 0] * edges_2[:, 1] - edges_1[:, 1] * edges_2[:, 0]
    valid = np.all(lower < upper, axis=1) & (np.abs(determinants) > 1e-12)

    pixel_coords = np.arange(_VIEW_SIZE, dtype=float)
    depths = np.full((_VIEW_SIZE, _VIEW_SIZE), -np.inf)
    visible_faces = np.full(depths.shape, -1, dtype=int)
    for index in np.flatnonzero(valid):
        x_min, y_min = lower[index]
        x_max, y_max = upper[index]
        x, y, z = starts[index]
        edge_1, edge_2 = edges_1[index], edges_2[index]
        dx = pixel_coords[x_min:x_max] - x
        dy = pixel_coords[y_min:y_max, None] - y
        u = (dx * edge_2[1] - dy * edge_2[0]) / determinants[index]
        v = (dy * edge_1[0] - dx * edge_1[1]) / determinants[index]
        depth = z + u * edge_1[2] + v * edge_2[2]
        region = np.s_[y_min:y_max, x_min:x_max]
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        front = inside & (depth > depths[region])
        depths[region][front] = depth[front]
        visible_faces[region][front] = index
    return visible_faces
