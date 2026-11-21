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

__all__: list[str] = []

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim._blender_runtime import _run_blender_job

from embodichain.gen_sim.services.compat import VisionClient, vision_for_role

_VLM_SYSTEM_PROMPT = """You inspect one isolated 3D object from elevated oblique and top views.
The object is temporarily visual-normalized and placed on a small neutral
support patch. The patch conveys only local contact and up/down context; do
not infer real-world scale from it. Use the object description, needed layout,
and rendered views together.

Choose whether the current asset pose already satisfies the needed layout or
whether it needs one pose switch. Do not reason about coordinate-axis rotations.
The z-up world is right-handed: x is left-right, y is front-back, and z is up.
The OBLIQUE VIEW has x horizontal with visible z height and floor contact; TOP
VIEW is a projection along z onto the x-y plane. For a lying object, its long
dimension is visible in TOP VIEW while the OBLIQUE VIEW shows only a small
vertical thickness. For a standing object, the OBLIQUE VIEW shows its main
height along z while TOP VIEW has a compact footprint. Do not request a pose
switch merely because an object is rotated within the x-y plane; visual yaw is
resolved later.

Treat TOP VIEW as decisive for a clearly visible long, thin silhouette. Its
screen vertical direction is world +y, not world +z: a knife or whisk that
spans the TOP VIEW from end to end is lying in the x-y table plane even if it
looks vertical on screen or foreshortened in the OBLIQUE VIEW. Return
rotate_to_required_pose only when both views clearly support a required pose
switch. When the needed layout asks for a natural stable pose rather than an
explicit standing or lying state, default to keep_current unless an unstable
tip-, edge-, or tiny-contact placement is unambiguous in both views.

After deciding, estimate the object's desired AABB footprint on the x-y plane
in real-world centimetres. The first value is the x size and the second value
is the y size.

Return JSON only with exactly this schema:
{
  "pose_action": "keep_current",
  "reason": "brief visual justification",
  "target_xy_size_cm": [12.0, 5.0]
}

pose_action must be exactly one of: keep_current, rotate_to_required_pose.
Use keep_current when the current views already satisfy the needed pose. Use
rotate_to_required_pose only when the front/top evidence clearly shows that
the current asset is standing but must lie, or lying but must stand.

Examples:
- Fork requested to lie flat: if its length is already visible in TOP VIEW and
  the OBLIQUE VIEW is thin, use keep_current; otherwise use rotate_to_required_pose.
  Use target_xy_size_cm=[15.0, 3.0].
- Fork requested upright in a holder: if the OBLIQUE VIEW already shows the fork
  height and TOP VIEW is compact, use keep_current; otherwise use
  rotate_to_required_pose. Use target_xy_size_cm=[3.0, 3.0].
- Bottle requested standing on its base: use keep_current only when it already
  appears upright in the OBLIQUE VIEW; otherwise use rotate_to_required_pose and use
  target_xy_size_cm=[8.0, 8.0].
"""

_VLM_POSE_CANDIDATE_SYSTEM_PROMPT = """You select the physically and semantically
correct pose for one isolated 3D object. The image shows two temporary
rendered candidates of the same asset, each visual-normalized and placed on a
small neutral support patch in the same fixed, slightly elevated
robot-manipulation view. The LEFT panel is CANDIDATE A and the RIGHT panel is
CANDIDATE B. The candidates differ only in which of two opposite 90-degree
pose switches was used.

Use the object description and needed layout to choose the candidate that
satisfies the requested lying, standing, or natural stable resting pose. Pay
attention to obvious up/down semantics: a pan or bowl opening should face up,
a bottle should rest on its base when standing, and a long tool should not be
upside down when the distinction is visible. Do not infer an exact coordinate
axis direction; simply choose the visually correct candidate.

Return JSON only with exactly this schema:
{
  "selected_candidate": "a",
  "reason": "brief visual justification"
}

selected_candidate must be exactly one of: a, b. Do not return any other keys
or prose outside the JSON object.
"""

_POSE_CANDIDATE_X_ROTATIONS_DEGREES = {"a": 90.0, "b": -90.0}

DEFAULT_NEEDED_LAYOUT = (
    "Rest this asset naturally, physically stably, and functionally correctly "
    "on its direct support. Preserve obvious up/down semantics: a pan, bowl, "
    "plate, or cup opening faces upward; a bottle rests on its base; and a "
    "long tool lies flat unless its requested pose says otherwise."
)


def render_object_front_top_views(
    *,
    glb_path: str | Path,
    output_path: str | Path,
    resolution: int = 512,
) -> Path:
    """Render grounded fixed z-up views in the isolated Blender runtime."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    source = Path(glb_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GLB for VLM rendering not found: {source}")
    return Path(
        _run_blender_job(
            "front_top",
            {
                "glb_path": str(source),
                "output_path": str(Path(output_path).expanduser().resolve()),
                "resolution": resolution,
            },
        )
    )


def render_object_pose_switch_candidates(
    *,
    glb_path: str | Path,
    output_path: str | Path,
    resolution: int = 512,
) -> Path:
    """Render opposite temporary x-rotation candidates without writing GLBs."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    source_path = Path(glb_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"GLB for candidate rendering not found: {source_path}")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_paths = {
        candidate_id: output_path.with_name(
            f".{output_path.stem}_candidate_{candidate_id}.png"
        )
        for candidate_id in _POSE_CANDIDATE_X_ROTATIONS_DEGREES
    }
    try:
        for (
            candidate_id,
            rotation_degrees,
        ) in _POSE_CANDIDATE_X_ROTATIONS_DEGREES.items():
            _render_grounded_pose_candidate(
                glb_path=source_path,
                x_rotation_degrees=rotation_degrees,
                output_path=candidate_paths[candidate_id],
                resolution=resolution,
            )
        with (
            Image.open(candidate_paths["a"]) as candidate_a,
            Image.open(candidate_paths["b"]) as candidate_b,
        ):
            panel_size = candidate_a.size
            composed = Image.new("RGB", (panel_size[0] * 2, panel_size[1]), "white")
            composed.paste(candidate_a.convert("RGB"), (0, 0))
            composed.paste(candidate_b.convert("RGB"), (panel_size[0], 0))
            draw = ImageDraw.Draw(composed)
            font = _label_font(panel_size[1])
            _draw_panel_label(draw, "CANDIDATE A", (0, 0), font)
            _draw_panel_label(draw, "CANDIDATE B", (panel_size[0], 0), font)
            composed.save(output_path)
    finally:
        for candidate_path in candidate_paths.values():
            candidate_path.unlink(missing_ok=True)
    return output_path


def _render_grounded_pose_candidate(
    *,
    glb_path: Path,
    x_rotation_degrees: float,
    output_path: Path,
    resolution: int,
) -> None:
    """Render one temporary candidate in the isolated Blender runtime."""
    _run_blender_job(
        "pose_candidate",
        {
            "glb_path": str(glb_path.resolve()),
            "x_rotation_degrees": x_rotation_degrees,
            "output_path": str(output_path.resolve()),
            "resolution": resolution,
        },
    )


def _label_font(image_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a readable label font without requiring a system font."""
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


def query_vlm_object_pose_and_target_size(
    *,
    scene_object_description: str,
    needed_layout: str,
    rendered_views_path: str | Path,
    vlm_client: VisionClient,
    debug_output_path: str | Path | None = None,
    json_max_attempts: int = 3,
) -> dict[str, object]:
    """Ask the VLM whether to preserve or switch the semantic pose."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vision_for_role(vlm_client, "scene_asset_dimensions").complete(
            system_prompt=_VLM_SYSTEM_PROMPT,
            user_prompt=(
                f"Object description:\n{scene_object_description}\n\n"
                f"Needed layout:\n{needed_layout}\n\n"
                "The image contains an OBLIQUE VIEW on the left and TOP VIEW on the right."
            ),
            image_path=rendered_views_path,
        )
        try:
            value = _parse_vlm_pose_and_target_size_response(response_text)
            break
        except ValueError as exc:
            last_validation_error = exc
    else:
        assert last_validation_error is not None
        raise ValueError(
            "VLM transform response is invalid after "
            f"{json_max_attempts} attempts: {last_validation_error}"
        ) from last_validation_error

    if debug_output_path is not None:
        output_path = Path(debug_output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "description": scene_object_description,
                    "needed_layout": needed_layout,
                    "rendered_views_path": str(
                        Path(rendered_views_path).expanduser().resolve()
                    ),
                    "vlm_output": value,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return value


def query_vlm_pose_switch_candidate(
    *,
    scene_object_description: str,
    needed_layout: str,
    rendered_candidates_path: str | Path,
    vlm_client: VisionClient,
    debug_output_path: str | Path | None = None,
    json_max_attempts: int = 3,
) -> tuple[float, str]:
    """Ask the VLM to choose one opposite temporary semantic-pose candidate."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vision_for_role(vlm_client, "scene_asset_orientation").complete(
            system_prompt=_VLM_POSE_CANDIDATE_SYSTEM_PROMPT,
            user_prompt=(
                f"Object description:\n{scene_object_description}\n\n"
                f"Needed layout:\n{needed_layout}\n\n"
                "Choose the correct grounded candidate from the paired image."
            ),
            image_path=rendered_candidates_path,
        )
        try:
            selected_candidate, reason = _parse_vlm_pose_candidate_response(
                response_text
            )
            break
        except ValueError as exc:
            last_validation_error = exc
    else:
        assert last_validation_error is not None
        raise ValueError(
            "VLM pose-candidate response is invalid after "
            f"{json_max_attempts} attempts: {last_validation_error}"
        ) from last_validation_error

    selected_rotation_degrees = _POSE_CANDIDATE_X_ROTATIONS_DEGREES[selected_candidate]
    if debug_output_path is not None:
        output_path = Path(debug_output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "description": scene_object_description,
                    "needed_layout": needed_layout,
                    "rendered_candidates_path": str(
                        Path(rendered_candidates_path).expanduser().resolve()
                    ),
                    "selected_candidate": selected_candidate,
                    "x_rotation_degrees": selected_rotation_degrees,
                    "reason": reason,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return selected_rotation_degrees, reason


def _parse_vlm_pose_and_target_size_response(
    response_text: str,
) -> dict[str, object]:
    """Validate one VLM semantic-pose and scale JSON response."""
    try:
        value = json.loads(_strip_json_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM transform response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "pose_action",
        "reason",
        "target_xy_size_cm",
    }:
        raise ValueError(
            "VLM transform response must contain exactly pose_action, reason, and "
            "target_xy_size_cm."
        )
    pose_action = value["pose_action"]
    if not isinstance(pose_action, str) or pose_action not in {
        "keep_current",
        "rotate_to_required_pose",
    }:
        raise ValueError(
            "VLM pose_action must be keep_current or rotate_to_required_pose."
        )
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("VLM pose reason must be a non-empty string.")
    target_size = value["target_xy_size_cm"]
    if (
        not isinstance(target_size, list)
        or len(target_size) != 2
        or not all(isinstance(item, (int, float)) for item in target_size)
        or not all(np.isfinite(item) and item > 0 for item in target_size)
    ):
        raise ValueError("VLM target_xy_size_cm must contain two positive numbers.")
    return value


def _parse_vlm_pose_candidate_response(response_text: str) -> tuple[str, str]:
    """Validate one VLM binary semantic-pose candidate selection."""
    try:
        value = json.loads(_strip_json_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"VLM pose-candidate response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or set(value) != {"selected_candidate", "reason"}:
        raise ValueError(
            "VLM pose-candidate response must contain exactly selected_candidate and "
            "reason."
        )
    selected_candidate = value["selected_candidate"]
    reason = value["reason"]
    if selected_candidate not in _POSE_CANDIDATE_X_ROTATIONS_DEGREES:
        raise ValueError("VLM selected_candidate must be a or b.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("VLM pose-candidate reason must be a non-empty string.")
    return selected_candidate, reason.strip()


def compute_uniform_xy_scale_for_target(
    *,
    glb_path: str | Path,
    target_xy_size_cm: list[float],
    rotate_about_x: bool,
) -> float:
    """Compute an isotropic scale from the rotated mesh XY AABB and target size."""
    loaded = trimesh.load(Path(glb_path).expanduser().resolve(), process=False)
    mesh = (
        loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"GLB is not a mesh: {glb_path}")
    if len(target_xy_size_cm) != 2 or any(value <= 0 for value in target_xy_size_cm):
        raise ValueError("target_xy_size_cm must contain two positive values.")
    # GLB geometry is y-up, while the target footprint is defined on z-up table XY.
    y_up_to_z_up = np.eye(4)
    y_up_to_z_up[:3, :3] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    mesh.apply_transform(y_up_to_z_up)
    if rotate_about_x:
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
        mesh.apply_transform(transform)
        mesh.apply_translation(center)
    actual_xy_size = mesh.bounds[1, :2] - mesh.bounds[0, :2]
    if np.any(actual_xy_size <= 0):
        raise ValueError("Rotated mesh must have a positive XY AABB.")
    # Convert the VLM's centimetres to metres before comparing with the GLB AABB.
    target_xy_size_m = np.asarray(target_xy_size_cm, dtype=float) / 100.0
    axis_scales = target_xy_size_m / actual_xy_size
    # Use sqrt(target XY area / actual XY area) as one uniform scale on all axes.
    return float(np.sqrt(axis_scales[0] * axis_scales[1]))


def rotate_glb_about_x_axis(
    *,
    input_path: str | Path,
    output_path: str | Path,
    rotation_degrees: float,
) -> Path:
    """Bake one x-axis rotation around the mesh AABB centre."""
    source_path = Path(input_path).expanduser().resolve()
    destination_path = Path(output_path).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(source_path, process=False)
    mesh = (
        loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"GLB is not a mesh: {source_path}")
    if not np.isfinite(rotation_degrees):
        raise ValueError("rotation_degrees must be finite.")
    if rotation_degrees != 0.0:
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "x", rotation_degrees, degrees=True
        ).as_matrix()
        mesh.apply_transform(transform)
        mesh.apply_translation(center)
    mesh.export(destination_path, file_type="glb")
    return destination_path


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


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: str,
) -> None:
    """Draw one labeled positive-axis arrow on a rendered view."""
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
    arrowhead = (
        (tip_x, tip_y),
        (
            base_x + perpendicular_x * head_width,
            base_y + perpendicular_y * head_width,
        ),
        (
            base_x - perpendicular_x * head_width,
            base_y - perpendicular_y * head_width,
        ),
    )
    draw.line((*start, *end), fill=color, width=4)
    draw.polygon(arrowhead, fill=color)
    # Put each axis label beside its arrowhead so it does not cover the arrow.
    draw.text((int(tip_x + 8), int(tip_y - 8)), label, fill=color, font=font)
