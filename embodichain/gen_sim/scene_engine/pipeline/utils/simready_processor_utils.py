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

import json
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)

_VLM_SYSTEM_PROMPT = """You inspect one isolated 3D object from front and top views.
Use the object description and the rendered views together.

Use the rendered views and the needed layout to decide whether the object should
be rotated around its own center by +90 degrees around the z-up world's x axis.
The z-up world is right-handed: x is left-right, y is front-back, and z is up.
In the composed image, FRONT VIEW is the left panel: x is horizontal and z is
vertical; the upper-right marker shows the positive z direction. TOP VIEW is
the right panel: x is horizontal and y is vertical; the upper-right markers
show the positive x and y directions.
Do not confuse the top view with looking at the object from above in the image
description: it is a projection along the z axis onto the x-y plane.
After deciding and applying that rotation, estimate the object's desired AABB
footprint on the x-y plane in real-world centimetres. The first value is the x
size and the second value is the y size.

Return JSON only with exactly this schema:
{
  "rotate_about_x": false,
  "target_xy_size_cm": [12.0, 5.0]
}

Examples:
- Fork lying flat on a table: in FRONT VIEW the fork is mostly a thin
  horizontal line; in TOP VIEW its length is visible. Keep it flat with
  rotate_about_x=false, and use the tabletop footprint, for example
  target_xy_size_cm=[15.0, 3.0].
- Fork placed in a pen holder: the desired fork is upright, so its long axis is
  approximately z. If the input coarse fork is lying in the x-y plane, set
  rotate_about_x=true; if the input coarse fork is already upright, set it to
  false. The target is the footprint inside the holder, not the fork's full
  length, for example target_xy_size_cm=[3.0, 3.0].
- Fork requested to lie flat on a table even when the input coarse fork is
  upright: set rotate_about_x=true and estimate the final flat footprint, for
  example target_xy_size_cm=[15.0, 3.0].
- Bottle already standing on its flat base: keep it upright with
  rotate_about_x=false and use target_xy_size_cm=[8.0, 8.0].
"""

DEFAULT_NEEDED_LAYOUT = (
    "Place this asset on the table in its natural, physically stable resting "
    "orientation. For example, a fork should lie flat on the table rather "
    "than stand on an edge."
)
STANDING_NEEDED_LAYOUT = (
    "The scene graph requires this asset to stand vertically on the table, "
    "even when its natural stable pose would be lying down. For example, a "
    "bottle should stand on its base and a fork should stand upright. If the "
    "coarse GLB is lying flat, set rotate_about_x=true so its semantic vertical "
    "axis aligns with the z-up world's z axis; if it is already upright, set "
    "it to false."
)
LYING_NEEDED_LAYOUT = (
    "The scene graph requires this asset to lie flat on the table, even when "
    "its natural stable pose would be standing. For example, a bottle should "
    "lie on its side and a fork should lie flat. Choose rotate_about_x so the "
    "asset's semantic long axis remains in the tabletop x-y plane rather than "
    "along the z-up world's z axis."
)


def render_object_front_top_views(
    *,
    glb_path: str | Path,
    output_path: str | Path,
    resolution: int = 512,
) -> Path:
    """Render fixed z-up front/top views and compose them horizontally."""
    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    source_path = Path(glb_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"GLB for VLM rendering not found: {source_path}")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    front_path = output_path.with_name(f"{output_path.stem}_front.png")
    top_path = output_path.with_name(f"{output_path.stem}_top.png")
    try:
        import bpy
        from mathutils import Vector
    except ImportError as exc:
        raise RuntimeError(
            "Blender's bpy is required for SimReady VLM view rendering."
        ) from exc

    _run_blender_operation_silently(
        lambda: bpy.ops.wm.read_factory_settings(use_empty=True)
    )
    _run_blender_operation_silently(
        lambda: bpy.ops.import_scene.gltf(filepath=str(source_path))
    )
    if not any(obj.type == "MESH" for obj in bpy.context.scene.objects):
        raise ValueError(f"GLB contains no mesh objects: {source_path}")
    scene = bpy.context.scene
    # Eevee renders imported GLB materials and textures instead of Workbench previews.
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("VLM_World")
    scene.world.color = (0.08, 0.08, 0.08)
    for name, location, energy in (
        ("VLM_Key", (2.0, -2.0, 3.0), 700.0),
        ("VLM_Fill", (-2.0, 1.0, 2.0), 400.0),
    ):
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = 4.0
        light = bpy.data.objects.new(name, light_data)
        light.location = location
        light.rotation_euler = (
            (Vector((0.0, 0.0, 0.0)) - light.location)
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
        scene.collection.objects.link(light)
    camera_data = bpy.data.cameras.new("VLM_Camera")
    camera = bpy.data.objects.new("VLM_Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.25

    def render_view(path: Path, location: tuple[float, float, float]) -> None:
        camera.location = location
        camera.rotation_euler = (
            (Vector((0.0, 0.0, 0.0)) - camera.location)
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
        scene.render.filepath = str(path)
        _run_blender_operation_silently(lambda: bpy.ops.render.render(write_still=True))

    # Blender uses a right-handed z-up world; front is viewed along +y.
    render_view(front_path, (0.0, -3.0, 0.0))
    render_view(top_path, (0.0, 0.0, 3.0))
    with Image.open(front_path) as front, Image.open(top_path) as top:
        composed = Image.new("RGB", (resolution * 2, resolution), "white")
        composed.paste(front.convert("RGB"), (0, 0))
        composed.paste(top.convert("RGB"), (resolution, 0))
        draw = ImageDraw.Draw(composed)
        # Use a readable scaled font for the panel labels when available.
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
                max(24, resolution // 16),
            )
        except OSError:
            font = ImageFont.load_default()
        # Label each panel so the VLM and manual debugging can distinguish views.
        for label, origin in (("FRONT VIEW", (0, 0)), ("TOP VIEW", (resolution, 0))):
            x, y = origin
            text_box = draw.textbbox((x + 16, y + 16), label, font=font)
            draw.rectangle(
                (text_box[0] - 8, text_box[1] - 6, text_box[2] + 8, text_box[3] + 6),
                fill="white",
            )
            draw.text((x + 16, y + 16), label, fill="black", font=font)
        # Mark the positive axes used by each projection for VLM interpretation.
        _draw_arrow(
            draw,
            (resolution - 62, 62),
            (resolution - 62, 20),
            "+Z",
            font,
            color="blue",
        )
        _draw_arrow(
            draw,
            (2 * resolution - 92, 62),
            (2 * resolution - 42, 62),
            "+X",
            font,
            color="red",
        )
        _draw_arrow(
            draw,
            (2 * resolution - 92, 62),
            (2 * resolution - 92, 20),
            "+Y",
            font,
            color="green",
        )
        composed.save(output_path)
    return output_path


def _run_blender_operation_silently(operation: Callable[[], object]) -> object:
    """Run one bpy operation without forwarding Blender-native console output."""
    # bpy writes render progress directly to process file descriptors, not Python streams.
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_output:
            os.dup2(null_output.fileno(), 1)
            os.dup2(null_output.fileno(), 2)
            return operation()
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


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


def query_vlm_object_rotation_and_target_size(
    *,
    scene_object_description: str,
    needed_layout: str,
    rendered_views_path: str | Path,
    vlm_client: OpenAICompatibleVLM,
    debug_output_path: str | Path | None = None,
    json_max_attempts: int = 3,
) -> dict[str, object]:
    """Ask the VLM for a valid rotation and post-rotation tabletop footprint."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
            system_prompt=_VLM_SYSTEM_PROMPT,
            user_prompt=(
                f"Object description:\n{scene_object_description}\n\n"
                f"Needed layout:\n{needed_layout}\n\n"
                "The image contains front view on the left and top view on the right."
            ),
            image_path=rendered_views_path,
        )
        try:
            value = _parse_vlm_rotation_and_target_size_response(response_text)
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


def _parse_vlm_rotation_and_target_size_response(
    response_text: str,
) -> dict[str, object]:
    """Validate one VLM rotation-and-scale JSON response."""
    try:
        value = json.loads(_strip_json_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM transform response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "rotate_about_x",
        "target_xy_size_cm",
    }:
        raise ValueError(
            "VLM transform response must contain exactly rotate_about_x and "
            "target_xy_size_cm."
        )
    if not isinstance(value["rotate_about_x"], bool):
        raise ValueError("VLM rotate_about_x must be boolean.")
    target_size = value["target_xy_size_cm"]
    if (
        not isinstance(target_size, list)
        or len(target_size) != 2
        or not all(isinstance(item, (int, float)) for item in target_size)
        or not all(np.isfinite(item) and item > 0 for item in target_size)
    ):
        raise ValueError("VLM target_xy_size_cm must contain two positive numbers.")
    return value


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
    rotate: bool,
) -> Path:
    """Bake an optional +90-degree x-axis rotation around the mesh centre."""
    # Current coarse layouts are either flat on xy with possible random z rotation,
    # or upright with almost no random y rotation, so this x-axis toggle is enough.
    source_path = Path(input_path).expanduser().resolve()
    destination_path = Path(output_path).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(source_path, process=False)
    mesh = (
        loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"GLB is not a mesh: {source_path}")
    if rotate:
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
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
