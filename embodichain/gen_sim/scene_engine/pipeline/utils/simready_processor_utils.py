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
_VLM_ORTHOGRAPHIC_SCALE = 1.75  # Leave floor context around normalized assets.
_VISUAL_MAX_OBJECT_EXTENT = 0.75  # Shared temporary extent for VLM pose views.
_VISUAL_FLOOR_SCALE = 1.3  # Preserve a small border around the asset footprint.
_VISUAL_MIN_FLOOR_SIZE = 0.45  # Keep a contact cue for compact upright objects.

DEFAULT_NEEDED_LAYOUT = (
    "Place this asset on the table in its natural, physically stable resting "
    "orientation. For example, a fork should lie flat on the table rather "
    "than stand on an edge."
)
STANDING_NEEDED_LAYOUT = (
    "The scene graph requires this asset to stand vertically on the table, "
    "even when its natural stable pose would be lying down. For example, a "
    "bottle should stand on its base and a fork should stand upright."
)
LYING_NEEDED_LAYOUT = (
    "The scene graph requires this asset to lie flat on the table, even when "
    "its natural stable pose would be standing. For example, a bottle should "
    "lie on its side and a fork should lie flat."
)


def render_object_front_top_views(
    *,
    glb_path: str | Path,
    output_path: str | Path,
    resolution: int = 512,
) -> Path:
    """Render grounded fixed z-up oblique/top views and compose them horizontally."""
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
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
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
    # A wider framing preserves floor and orientation context for VLM inspection.
    camera.data.ortho_scale = _VLM_ORTHOGRAPHIC_SCALE

    _place_meshes_on_visual_floor(
        bpy=bpy,
        mesh_objects=mesh_objects,
    )

    def render_view(path: Path, location: tuple[float, float, float]) -> None:
        camera.location = location
        camera.rotation_euler = (
            (Vector((0.0, 0.0, 0.0)) - camera.location)
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
        scene.render.filepath = str(path)
        _run_blender_operation_silently(lambda: bpy.ops.render.render(write_still=True))

    # A diagonal oblique view avoids looking straight down a long planar tool.
    render_view(front_path, (3.0, -4.0, 3.0))
    render_view(top_path, (0.0, 0.0, 4.0))
    try:
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
            for label, origin in (
                ("OBLIQUE VIEW", (0, 0)),
                ("TOP VIEW", (resolution, 0)),
            ):
                x, y = origin
                text_box = draw.textbbox((x + 16, y + 16), label, font=font)
                draw.rectangle(
                    (
                        text_box[0] - 8,
                        text_box[1] - 6,
                        text_box[2] + 8,
                        text_box[3] + 6,
                    ),
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
    finally:
        front_path.unlink(missing_ok=True)
        top_path.unlink(missing_ok=True)
    return output_path


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
    """Render one temporary x-rotated candidate in the shared visual frame."""
    try:
        import bpy
        from mathutils import Matrix, Vector
    except ImportError as exc:
        raise RuntimeError(
            "Blender's bpy is required for SimReady candidate rendering."
        ) from exc
    _run_blender_operation_silently(
        lambda: bpy.ops.wm.read_factory_settings(use_empty=True)
    )
    _run_blender_operation_silently(
        lambda: bpy.ops.import_scene.gltf(filepath=str(glb_path))
    )
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise ValueError(f"GLB contains no mesh objects: {glb_path}")
    scene = bpy.context.scene
    _configure_vlm_render_scene(
        scene=scene, resolution=resolution, bpy=bpy, Vector=Vector
    )
    root = bpy.data.objects.new("PoseCandidateRoot", None)
    scene.collection.objects.link(root)
    for mesh_object in mesh_objects:
        original_world_matrix = mesh_object.matrix_world.copy()
        mesh_object.parent = root
        mesh_object.matrix_parent_inverse = root.matrix_world.inverted()
        mesh_object.matrix_world = original_world_matrix
    center = _mesh_world_bounds(mesh_objects)[0]
    x_rotation = Rotation.from_euler("x", x_rotation_degrees, degrees=True).as_matrix()
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = x_rotation
    root.matrix_world = Matrix(
        _translation_matrix(center) @ rotation_transform @ _translation_matrix(-center)
    )
    _place_meshes_on_visual_floor(bpy=bpy, mesh_objects=mesh_objects)
    _add_oblique_camera(scene=scene, bpy=bpy, Vector=Vector)
    scene.render.filepath = str(output_path)
    _run_blender_operation_silently(lambda: bpy.ops.render.render(write_still=True))


def _configure_vlm_render_scene(
    *, scene: object, resolution: int, bpy: object, Vector: object
) -> None:
    """Configure the shared illuminated, opaque Blender render scene."""
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


def _add_oblique_camera(*, scene: object, bpy: object, Vector: object) -> None:
    """Add the fixed, wider z-up oblique camera used for pose candidates."""
    camera_data = bpy.data.cameras.new("VLM_CandidateCamera")
    camera = bpy.data.objects.new("VLM_CandidateCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = _VLM_ORTHOGRAPHIC_SCALE
    camera.location = (0.0, -4.0, 5.0)
    camera.rotation_euler = (
        (Vector((0.0, 0.0, 0.3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )


def _place_meshes_on_visual_floor(*, bpy: object, mesh_objects: list[object]) -> None:
    """Normalize one temporary object and place it on a local support patch."""
    from mathutils import Matrix, Vector

    _run_blender_operation_silently(lambda: bpy.context.view_layer.update())
    minimum, maximum = _mesh_world_aabb(mesh_objects)
    extent = maximum - minimum
    largest_extent = float(np.max(extent))
    if not np.isfinite(largest_extent) or largest_extent <= 0.0:
        raise ValueError("VLM pose rendering requires a mesh with positive extent.")
    center = (minimum + maximum) * 0.5
    visual_scale = _VISUAL_MAX_OBJECT_EXTENT / largest_extent
    normalization_transform = np.eye(4)
    normalization_transform[:3, :3] *= visual_scale
    normalization_transform[:3, 3] = center - visual_scale * center
    for mesh_object in mesh_objects:
        mesh_object.matrix_world = (
            Matrix(normalization_transform) @ mesh_object.matrix_world
        )
    _run_blender_operation_silently(lambda: bpy.context.view_layer.update())

    minimum, maximum = _mesh_world_aabb(mesh_objects)
    center = (minimum + maximum) * 0.5
    for mesh_object in mesh_objects:
        # Assign in world space so this remains correct under a rotated parent root.
        mesh_object.matrix_world.translation -= Vector(
            (center[0], center[1], minimum[2])
        )
    _run_blender_operation_silently(lambda: bpy.context.view_layer.update())
    floor_size = max(
        _VISUAL_MIN_FLOOR_SIZE,
        _VISUAL_FLOOR_SCALE * float(np.max(maximum[:2] - minimum[:2])),
    )
    bpy.ops.mesh.primitive_plane_add(size=floor_size, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "VLM_VisualFloor"
    material = bpy.data.materials.new("VLM_VisualFloorMaterial")
    material.diffuse_color = (0.22, 0.22, 0.22, 1.0)
    floor.data.materials.append(material)


def _mesh_world_bounds(mesh_objects: list[object]) -> tuple[np.ndarray, float]:
    """Return a world-space AABB centre and its minimum z coordinate."""
    minimum, maximum = _mesh_world_aabb(mesh_objects)
    return (minimum + maximum) * 0.5, float(minimum[2])


def _mesh_world_aabb(mesh_objects: list[object]) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space AABB minimum and maximum for one temporary object."""
    from mathutils import Vector

    points = np.asarray(
        [
            tuple(mesh_object.matrix_world @ Vector(corner))
            for mesh_object in mesh_objects
            for corner in mesh_object.bound_box
        ],
        dtype=float,
    )
    return points.min(axis=0), points.max(axis=0)


def _translation_matrix(translation: np.ndarray) -> np.ndarray:
    """Return one homogeneous translation transform."""
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform


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
    vlm_client: OpenAICompatibleVLM,
    debug_output_path: str | Path | None = None,
    json_max_attempts: int = 3,
) -> dict[str, object]:
    """Ask the VLM whether to preserve or switch the semantic pose."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
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
    vlm_client: OpenAICompatibleVLM,
    debug_output_path: str | Path | None = None,
    json_max_attempts: int = 3,
) -> tuple[float, str]:
    """Ask the VLM to choose one opposite temporary semantic-pose candidate."""
    if json_max_attempts < 1:
        raise ValueError("json_max_attempts must be at least 1.")
    last_validation_error: ValueError | None = None
    for _ in range(json_max_attempts):
        response_text = vlm_client.complete(
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
