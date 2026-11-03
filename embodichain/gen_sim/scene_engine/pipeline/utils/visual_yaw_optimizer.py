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
"""VLM-guided visual yaw selection for canonical SimReady assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim._blender_runtime import _run_blender_job

_RENDER_RESOLUTION = 512
_ALLOWED_CLOCKWISE_YAWS_DEGREES = frozenset(range(0, 360, 15))
_VLM_SYSTEM_PROMPT = """You align one isolated canonical 3D asset to an image observation.
The image has two panels. The LEFT panel is a fixed, slightly distant
orthographic oblique view from above of the canonical SimReady asset in a
right-handed z-up world, resting on a small neutral support patch. It has no
coarse-layout position or rotation applied. The blue arrow marks world +z and
the red arrow marks world +x. The RIGHT panel is the asset's visible RGBA crop
from the source image, which is also a robot-manipulation view.

Choose the absolute clockwise yaw about world z that makes the LEFT asset best
match the visible direction of the RIGHT observation. The output yaw is
measured from the current canonical LEFT view, not from any coarse-layout
rotation. The two views can have different camera poses, perspective,
occlusion, and segmentation error, so do not attempt pixel-level alignment.
Keep 0 unless there is a severe, clearly supported directional mismatch, such
as a handle, spout, blade, long axis, label, or asymmetric silhouette facing
the wrong direction. Clockwise means a physical rotation when looking down
from world +z toward the table, not a screen-plane rotation in either panel.
Use this yaw compass: 0 keeps the canonical asset unchanged; 90 clockwise
turns its world +x direction toward world -y; 180 reverses it; 270 clockwise
turns world +x toward world +y. Choose the nearest 15-degree value only after
first deciding whether a nonzero correction is clearly necessary.

Return JSON only with exactly this schema:
{
  "clockwise_yaw_degrees": 0,
  "reason": "brief visual justification"
}

clockwise_yaw_degrees must be one of 0, 15, 30, ..., 345. Do not return any
other keys or prose outside the JSON object."""


class VisualYawOptimizer:
    """Ask a VLM for one absolute simulator-world z-up yaw for an asset.

    The SimReady GLB has already resolved standing, lying, or stable semantic
    pose and baked the geometry-server scale. Rendering applies that scale's
    inverse only to a temporary Blender root, so every VLM query sees the
    same normalized canonical asset without modifying the saved GLB.
    """

    def __init__(
        self,
        *,
        scene_object: SceneObject,
        baked_scale_y_up: list[float],
        vlm_client: OpenAICompatibleVLM,
        debug_output_root: str | Path,
        json_max_attempts: int = 3,
    ) -> None:
        self._scene_object = scene_object
        self._baked_scale_y_up = baked_scale_y_up
        self._vlm_client = vlm_client
        self._debug_output_root = Path(debug_output_root).expanduser().resolve()
        self._json_max_attempts = json_max_attempts

    def optimize_z_up_yaw_degrees(self) -> float:
        """Return the absolute canonical yaw in z-up world degrees.

        Positive returned angles are counterclockwise in the z-up world. The
        VLM reports clockwise image angles, so this method converts signs at
        the boundary before scene generation applies the result to a layout.
        """
        self._validate_inputs()
        if self._scene_object.visible_rgba_path is None:
            return 0.0

        # Only the two composed comparisons are debug artifacts; raw renders are temporary.
        before_path = (
            self._debug_output_root / f".{self._scene_object.id}_canonical.png"
        )
        after_path = self._debug_output_root / f".{self._scene_object.id}_yawed.png"
        source_path = Path(self._scene_object.visible_rgba_path).expanduser().resolve()
        vlm_input_path = (
            self._debug_output_root / f"{self._scene_object.id}_vlm_input.png"
        )
        try:
            _render_canonical_oblique_view(
                glb_path=self._scene_object.simready_glb_path,
                baked_scale_y_up=self._baked_scale_y_up,
                z_up_yaw_degrees=0.0,
                output_path=before_path,
                resolution=_RENDER_RESOLUTION,
            )
            _compose_render_and_source(
                rendered_path=before_path,
                source_rgba_path=source_path,
                rendered_label="CANONICAL OBLIQUE VIEW",
                source_label="SOURCE RGBA",
                output_path=vlm_input_path,
            )
            clockwise_yaw_degrees, reason = self._query_clockwise_yaw(vlm_input_path)
            z_up_yaw_degrees = -float(clockwise_yaw_degrees)
            _render_canonical_oblique_view(
                glb_path=self._scene_object.simready_glb_path,
                baked_scale_y_up=self._baked_scale_y_up,
                z_up_yaw_degrees=z_up_yaw_degrees,
                output_path=after_path,
                resolution=_RENDER_RESOLUTION,
            )
            _compose_render_and_source(
                rendered_path=after_path,
                source_rgba_path=source_path,
                rendered_label="YAWED OBLIQUE VIEW",
                source_label="SOURCE RGBA",
                output_path=self._debug_output_root
                / f"{self._scene_object.id}_yaw_result.png",
            )
        finally:
            before_path.unlink(missing_ok=True)
            after_path.unlink(missing_ok=True)
        (self._debug_output_root / f"{self._scene_object.id}.json").write_text(
            json.dumps(
                {
                    "object_id": self._scene_object.id,
                    "clockwise_yaw_degrees": clockwise_yaw_degrees,
                    "z_up_yaw_degrees": z_up_yaw_degrees,
                    "reason": reason,
                    "vlm_input_path": str(vlm_input_path),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return z_up_yaw_degrees

    def _query_clockwise_yaw(self, vlm_input_path: Path) -> tuple[int, str]:
        """Query and validate the VLM's discrete clockwise yaw selection."""
        last_error: ValueError | None = None
        for _ in range(self._json_max_attempts):
            response_text = self._vlm_client.complete(
                system_prompt=_VLM_SYSTEM_PROMPT,
                user_prompt=(
                    f"Object description: {self._scene_object.description}\n"
                    "Select the canonical asset's yaw from the paired image."
                ),
                image_path=vlm_input_path,
            )
            try:
                return _parse_clockwise_yaw_response(response_text)
            except ValueError as exc:
                last_error = exc
        assert last_error is not None
        raise ValueError(
            "VLM visual yaw response is invalid after "
            f"{self._json_max_attempts} attempts: {last_error}"
        ) from last_error

    def _validate_inputs(self) -> None:
        if self._json_max_attempts < 1:
            raise ValueError("json_max_attempts must be at least 1.")
        if self._scene_object.simready_glb_path is None:
            raise ValueError(
                "Visual yaw optimization requires a SimReady GLB for "
                f"{self._scene_object.id!r}."
            )
        if not Path(self._scene_object.simready_glb_path).is_file():
            raise FileNotFoundError(
                "Visual yaw optimization SimReady GLB not found for "
                f"{self._scene_object.id!r}: {self._scene_object.simready_glb_path}"
            )
        if len(self._baked_scale_y_up) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in self._baked_scale_y_up
        ):
            raise ValueError("Visual yaw optimization requires three positive scales.")
        if (
            self._scene_object.visible_rgba_path is not None
            and not Path(self._scene_object.visible_rgba_path).is_file()
        ):
            raise FileNotFoundError(
                "Visual yaw optimization RGBA observation not found for "
                f"{self._scene_object.id!r}: "
                f"{self._scene_object.visible_rgba_path}"
            )


def _render_canonical_oblique_view(
    *,
    glb_path: str | Path,
    baked_scale_y_up: list[float],
    z_up_yaw_degrees: float,
    output_path: str | Path,
    resolution: int,
) -> Path:
    """Render an unscaled GLB using the isolated Blender runtime."""
    destination = Path(
        _run_blender_job(
            "visual_yaw",
            {
                "glb_path": str(Path(glb_path).expanduser().resolve()),
                "baked_scale_y_up": baked_scale_y_up,
                "z_up_yaw_degrees": z_up_yaw_degrees,
                "output_path": str(Path(output_path).expanduser().resolve()),
                "resolution": resolution,
            },
        )
    )
    _draw_oblique_view_axes(destination)
    return destination


def _compose_render_and_source(
    *,
    rendered_path: str | Path,
    source_rgba_path: str | Path,
    rendered_label: str,
    source_label: str,
    output_path: str | Path,
) -> Path:
    """Compose a rendered oblique view and an alpha-aware source observation."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(rendered_path) as rendered, Image.open(source_rgba_path) as source:
        panel_size = rendered.size
        source = source.convert("RGBA").resize(panel_size, Image.Resampling.LANCZOS)
        source_panel = Image.new("RGB", panel_size, (36, 36, 36))
        source_panel.paste(source, mask=source.getchannel("A"))
        composed = Image.new("RGB", (panel_size[0] * 2, panel_size[1]), "white")
        composed.paste(rendered.convert("RGB"), (0, 0))
        composed.paste(source_panel, (panel_size[0], 0))
        draw = ImageDraw.Draw(composed)
        font = _label_font(panel_size[1])
        _draw_panel_label(draw, rendered_label, (0, 0), font)
        _draw_panel_label(draw, source_label, (panel_size[0], 0), font)
        composed.save(output_path)
    return output_path


def _draw_oblique_view_axes(rendered_path: Path) -> None:
    """Mark the fixed z-up axes used by the VLM yaw convention."""
    with Image.open(rendered_path) as rendered:
        image = rendered.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _label_font(image.height)
    origin = (image.width - 84, 84)
    _draw_arrow(draw, origin, (image.width - 28, 84), "+X", font, "red")
    _draw_arrow(draw, origin, (image.width - 84, 28), "+Z", font, "blue")
    draw.text((18, image.height - 42), "FIXED OBLIQUE VIEW", fill="white", font=font)
    image.save(rendered_path)


def _parse_clockwise_yaw_response(response_text: str) -> tuple[int, str]:
    """Parse the exact discrete-yaw response contract from the VLM."""
    try:
        value = json.loads(_strip_json_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM visual yaw response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"clockwise_yaw_degrees", "reason"}:
        raise ValueError(
            "VLM visual yaw response must contain exactly clockwise_yaw_degrees and reason."
        )
    yaw = value["clockwise_yaw_degrees"]
    reason = value["reason"]
    if (
        isinstance(yaw, bool)
        or not isinstance(yaw, int)
        or yaw not in _ALLOWED_CLOCKWISE_YAWS_DEGREES
    ):
        raise ValueError(
            "VLM clockwise_yaw_degrees must be a 15-degree value in [0, 345]."
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("VLM visual yaw reason must be a non-empty string.")
    return yaw, reason.strip()


def _strip_json_code_fence(response_text: str) -> str:
    """Remove one optional Markdown JSON fence from a VLM response."""
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _label_font(image_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a readable panel-label font without requiring a system font."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            max(20, image_height // 20),
        )
    except OSError:
        return ImageFont.load_default()


def _draw_panel_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    origin: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw one panel label on an opaque backing rectangle."""
    x, y = origin
    text_box = draw.textbbox((x + 16, y + 16), label, font=font)
    draw.rectangle(
        (text_box[0] - 8, text_box[1] - 6, text_box[2] + 8, text_box[3] + 6),
        fill="white",
    )
    draw.text((x + 16, y + 16), label, fill="black", font=font)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: str,
) -> None:
    """Draw one labeled axis arrow."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(abs(dx), abs(dy))
    if length == 0:
        raise ValueError("Axis arrow start and end must differ.")
    unit_x, unit_y = dx / length, dy / length
    perpendicular_x, perpendicular_y = -unit_y, unit_x
    head_length = 14.0
    head_width = 8.0
    tip_x, tip_y = end
    base_x = tip_x - unit_x * head_length
    base_y = tip_y - unit_y * head_length
    draw.line((*start, *end), fill=color, width=4)
    draw.polygon(
        (
            (tip_x, tip_y),
            (
                base_x + perpendicular_x * head_width,
                base_y + perpendicular_y * head_width,
            ),
            (
                base_x - perpendicular_x * head_width,
                base_y - perpendicular_y * head_width,
            ),
        ),
        fill=color,
    )
    draw.text((int(tip_x + 8), int(tip_y - 8)), label, fill=color, font=font)
