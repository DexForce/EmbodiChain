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

"""Render Scene Engine review images inside the isolated Blender worker."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Callable

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

__all__ = []

_VLM_ORTHOGRAPHIC_SCALE = 1.75
_VISUAL_MAX_OBJECT_EXTENT = 0.75
_VISUAL_FLOOR_SCALE = 1.3
_VISUAL_MIN_FLOOR_SIZE = 0.45


def _render_object_front_top_views(
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


def _render_grounded_pose_candidate(
    *,
    glb_path: str | Path,
    x_rotation_degrees: float,
    output_path: str | Path,
    resolution: int,
) -> None:
    """Render one temporary x-rotated candidate in the shared visual frame."""
    _render_grounded_pose_view(
        glb_path=Path(glb_path),
        rotation_z_up=Rotation.from_euler(
            "x", x_rotation_degrees, degrees=True
        ).as_matrix(),
        output_path=Path(output_path),
        resolution=resolution,
    )


def _render_asset_posture(
    *,
    glb_path: str | Path,
    rotation_y_up: list[list[float]],
    output_path: str | Path,
    resolution: int,
) -> None:
    """Render an isolated candidate at an absolute, proper Y-up rotation."""
    rotation = np.asarray(rotation_y_up, dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Asset posture preview requires a proper Y-up rotation matrix."
        )
    frame = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=float)
    _render_grounded_pose_view(
        glb_path=Path(glb_path),
        rotation_z_up=frame @ rotation @ frame.T,
        output_path=Path(output_path),
        resolution=resolution,
        y_up_axes_z_up=frame,
    )


def _render_grounded_pose_view(
    *,
    glb_path: Path,
    rotation_z_up: np.ndarray,
    output_path: Path,
    resolution: int,
    y_up_axes_z_up: np.ndarray | None = None,
) -> None:
    """Ground and normalize one rotated mesh for an isolated oblique render."""
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
    render_posture = y_up_axes_z_up is not None
    if render_posture:
        for obj in scene.objects:
            if obj.type == "LIGHT":
                obj.hide_render = True
    _configure_vlm_render_scene(
        scene=scene,
        resolution=resolution,
        bpy=bpy,
        Vector=Vector,
        use_cycles=render_posture,
    )
    root = bpy.data.objects.new("PoseCandidateRoot", None)
    scene.collection.objects.link(root)
    for mesh_object in mesh_objects:
        original_world_matrix = mesh_object.matrix_world.copy()
        mesh_object.parent = root
        mesh_object.matrix_parent_inverse = root.matrix_world.inverted()
        mesh_object.matrix_world = original_world_matrix
    center = _mesh_world_bounds(mesh_objects)[0]
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = rotation_z_up
    root.matrix_world = Matrix(
        _translation_matrix(center) @ rotation_transform @ _translation_matrix(-center)
    )
    _place_meshes_on_visual_floor(
        bpy=bpy, mesh_objects=mesh_objects, posture_preview=render_posture
    )
    _add_oblique_camera(
        scene=scene,
        bpy=bpy,
        Vector=Vector,
        location=(3.0, -4.0, 2.9) if render_posture else (0.0, -4.0, 5.0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    if y_up_axes_z_up is None:
        _run_blender_operation_silently(lambda: bpy.ops.render.render(write_still=True))
    else:
        bpy.ops.render.render(write_still=True)
        world_to_camera = scene.camera.rotation_euler.to_quaternion().inverted()
        camera_axes = [world_to_camera @ Vector(axis) for axis in y_up_axes_z_up.T]
        _annotate_posture_axes(output_path, [(axis.x, -axis.y) for axis in camera_axes])


def _configure_vlm_render_scene(
    *,
    scene: object,
    resolution: int,
    bpy: object,
    Vector: object,
    use_cycles: bool = False,
) -> None:
    """Configure the shared illuminated, opaque Blender render scene."""
    if use_cycles:
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 4
    else:
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
    if use_cycles:
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes["Background"]
        background.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1.0)
        background.inputs["Strength"].default_value = 0.8
    else:
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


def _add_oblique_camera(
    *, scene: object, bpy: object, Vector: object, location: tuple[float, float, float]
) -> None:
    """Add the fixed, wider z-up oblique camera used for pose candidates."""
    camera_data = bpy.data.cameras.new("VLM_CandidateCamera")
    camera = bpy.data.objects.new("VLM_CandidateCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = _VLM_ORTHOGRAPHIC_SCALE
    camera.location = location
    camera.rotation_euler = (
        (Vector((0.0, 0.0, 0.3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )


def _place_meshes_on_visual_floor(
    *, bpy: object, mesh_objects: list[object], posture_preview: bool = False
) -> None:
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
    if posture_preview:
        material.use_nodes = True
        shader = material.node_tree.nodes["Principled BSDF"]
        shader.inputs["Base Color"].default_value = (0.52, 0.47, 0.33, 1.0)
        shader.inputs["Roughness"].default_value = 0.9
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
    label_offset: tuple[int, int] = (8, -8),
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
    draw.text(
        (tip_x + label_offset[0], tip_y + label_offset[1]),
        label,
        fill=color,
        font=font,
    )


def _annotate_posture_axes(path: Path, axes: list[tuple[float, float]]) -> None:
    """Add the fixed world axes in a small card away from the rendered object."""
    with Image.open(path) as image:
        view = image.convert("RGB")
    width, height = view.size
    origin = (round(width * 0.82), round(height * 0.86))
    scale = width * 0.11
    font = ImageFont.load_default(size=max(12, width // 28))
    draw = ImageDraw.Draw(view)
    draw.rounded_rectangle(
        (origin[0] - scale - 10, origin[1] - scale - 10, width - 8, origin[1] + scale),
        radius=5,
        fill="white",
        outline="#d0d6dc",
    )
    for (x, y), label, color in zip(
        axes, ("+X", "+Y", "+Z"), ("#b52b2b", "#277143", "#2656a7"), strict=True
    ):
        end = (round(origin[0] + scale * x), round(origin[1] + scale * y))
        label_offset = (-18, 3) if x < -0.1 else (8, -8)
        _draw_arrow(draw, origin, end, label, font, color, label_offset)
    view.save(path)


def _render_canonical_oblique_view(
    *,
    glb_path: str | Path,
    baked_scale_y_up: list[float],
    z_up_yaw_degrees: float,
    output_path: str | Path,
    resolution: int,
) -> Path:
    """Render a temporary unscaled GLB from a fixed z-up oblique view."""
    source_path = Path(glb_path).expanduser().resolve()
    destination_path = Path(output_path).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import bpy
        from mathutils import Matrix, Vector
    except ImportError as exc:
        raise RuntimeError(
            "Blender's bpy is required for visual-yaw rendering."
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
        scene.world = bpy.data.worlds.new("VisualYawWorld")
    scene.world.color = (0.08, 0.08, 0.08)

    # Blender imports the y-up GLB into z-up. Undo only baked y-up scale here.
    y_up_to_z_up = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    inverse_scale_z_up = (
        y_up_to_z_up
        @ np.diag(1.0 / np.asarray(baked_scale_y_up, dtype=float))
        @ y_up_to_z_up.T
    )
    yaw_z_up = Rotation.from_euler("z", z_up_yaw_degrees, degrees=True).as_matrix()
    root = bpy.data.objects.new("VisualYawRoot", None)
    scene.collection.objects.link(root)
    for mesh_object in mesh_objects:
        original_world_matrix = mesh_object.matrix_world.copy()
        mesh_object.parent = root
        mesh_object.matrix_parent_inverse = root.matrix_world.inverted()
        mesh_object.matrix_world = original_world_matrix
    transform = np.eye(4)
    transform[:3, :3] = yaw_z_up @ inverse_scale_z_up
    root.matrix_world = Matrix(transform)
    visual_floor_size = _center_root_on_visual_floor(
        bpy=bpy,
        mesh_objects=mesh_objects,
        root=root,
    )
    _add_visual_floor(bpy=bpy, size=visual_floor_size)

    light_data = bpy.data.lights.new("VisualYawKey", type="AREA")
    light_data.energy = 900.0
    light_data.shape = "DISK"
    light_data.size = 4.0
    light = bpy.data.objects.new("VisualYawKey", light_data)
    light.location = (2.0, -2.0, 4.0)
    light.rotation_euler = (
        (Vector((0.0, 0.0, 0.0)) - light.location).to_track_quat("-Z", "Y").to_euler()
    )
    scene.collection.objects.link(light)
    camera_data = bpy.data.cameras.new("VisualYawCamera")
    camera = bpy.data.objects.new("VisualYawCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    # Keep a wider fixed robot view with floor context, without coarse-layout scale.
    camera.data.ortho_scale = 1.75
    camera.location = (0.0, -4.0, 5.0)
    camera.rotation_euler = (
        (Vector((0.0, 0.0, 0.3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    scene.render.filepath = str(destination_path)
    _run_blender_operation_silently(lambda: bpy.ops.render.render(write_still=True))
    return destination_path


def _center_root_on_visual_floor(
    *, bpy: object, mesh_objects: list[object], root: object
) -> float:
    """Translate a temporary transformed root so its mesh rests at the origin."""
    from mathutils import Matrix, Vector

    _run_blender_operation_silently(lambda: bpy.context.view_layer.update())
    points = np.asarray(
        [
            tuple(mesh_object.matrix_world @ Vector(corner))
            for mesh_object in mesh_objects
            for corner in mesh_object.bound_box
        ],
        dtype=float,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    translation = np.eye(4)
    translation[:3, 3] = [
        -(minimum[0] + maximum[0]) * 0.5,
        -(minimum[1] + maximum[1]) * 0.5,
        -minimum[2],
    ]
    root.matrix_world = Matrix(translation) @ root.matrix_world
    _run_blender_operation_silently(lambda: bpy.context.view_layer.update())
    return max(
        _VISUAL_MIN_FLOOR_SIZE,
        _VISUAL_FLOOR_SCALE * float(np.max(maximum[:2] - minimum[:2])),
    )


def _add_visual_floor(*, bpy: object, size: float) -> None:
    """Add a footprint-scaled support patch for local contact context."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "VisualYawFloor"
    material = bpy.data.materials.new("VisualYawFloorMaterial")
    material.diffuse_color = (0.22, 0.22, 0.22, 1.0)
    floor.data.materials.append(material)


def _render_scene_heading_views(
    *,
    glb_paths: list[str],
    output_paths: list[str],
    resolution: int,
    baseline_transforms: list[list[list[float]]] | None = None,
    flipped_transforms: list[list[list[float]]] | None = None,
    candidate_transforms: list[list[list[list[float]]]] | None = None,
) -> None:
    """Render world layouts with one shared, front-elevated Blender camera."""
    from mathutils import Matrix, Vector

    if candidate_transforms is None:
        if baseline_transforms is None or flipped_transforms is None:
            raise ValueError("Heading preview requires candidate transforms.")
        candidates = [baseline_transforms, flipped_transforms]
    else:
        candidates = candidate_transforms
    if (
        resolution <= 0
        or not glb_paths
        or not candidates
        or len(output_paths) != len(candidates)
        or len(set(output_paths)) != len(output_paths)
        or any(len(candidate) != len(glb_paths) for candidate in candidates)
    ):
        raise ValueError(
            "Heading preview requires complete candidates and distinct outputs."
        )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # The GLB importer maps Y-up assets into Blender's Z-up frame.
    frame = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    inverse_frame = frame.transposed()
    roots = []
    meshes = []
    for path in glb_paths:
        before = set(scene.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        imported = set(scene.objects) - before
        imported_meshes = [obj for obj in imported if obj.type == "MESH"]
        if not imported_meshes:
            raise ValueError(f"Heading preview GLB contains no mesh: {path}")
        meshes.extend(imported_meshes)

        root = bpy.data.objects.new(Path(path).stem + "_HeadingRoot", None)
        scene.collection.objects.link(root)
        roots.append(root)
        for obj in imported:
            if obj.type == "LIGHT":
                obj.hide_render = True
            if obj.parent not in imported:
                source_world = obj.matrix_world.copy()
                obj.parent = root
                obj.matrix_world = source_world

    candidate_matrices = [
        [frame @ Matrix(transform) @ inverse_frame for transform in candidate]
        for candidate in candidates
    ]
    angle = np.deg2rad(25.0)
    # Rows are the fixed camera's right, up and outward directions in Z-up.
    camera_basis = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.sin(angle), np.cos(angle)),
            (0.0, -np.cos(angle), np.sin(angle)),
        )
    )
    fit_four_headings = candidate_transforms is not None and len(candidates) == 4
    bounds = []
    projected_bounds = []
    for transforms in candidate_matrices:
        for root, transform in zip(roots, transforms, strict=True):
            root.matrix_world = transform
        bpy.context.view_layer.update()
        bounds.append(_mesh_world_aabb(meshes))
        if fit_four_headings:
            projected_bounds.append(
                _heading_camera_projected_bounds(meshes, camera_basis)
            )
    lower = np.minimum.reduce([bound[0] for bound in bounds])
    upper = np.maximum.reduce([bound[1] for bound in bounds])
    extent = float(np.linalg.norm(upper - lower))
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("Heading preview requires a finite, nonzero scene extent.")
    span = max(1.8 * extent, 0.01)
    center = Vector((lower + upper) / 2.0)
    camera_span = span
    camera_center = center
    if fit_four_headings:
        projected_lower = np.minimum.reduce([bound[0] for bound in projected_bounds])
        projected_upper = np.maximum.reduce([bound[1] for bound in projected_bounds])
        # Share the union framing across all candidates, including legs and any
        # objects extending beyond the tabletop. Add 22% to the limiting span.
        camera_span = max(
            1.22 * float(np.max((projected_upper - projected_lower)[:2])), 0.01
        )
        camera_center = Vector(
            camera_basis.T @ ((projected_lower + projected_upper) / 2.0)
        )

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Heading Preview World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.62, 0.65, 0.70, 1.0)
    background.inputs[1].default_value = 0.8

    bpy.ops.mesh.primitive_plane_add(
        size=span * 5.0, location=(center.x, center.y, lower[2] - span * 0.002)
    )
    floor = bpy.context.object
    floor.name = "Heading Preview Floor"
    floor_material = bpy.data.materials.new("Heading Preview Checker")
    floor_material.use_nodes = True
    nodes, links = floor_material.node_tree.nodes, floor_material.node_tree.links
    shader = nodes["Principled BSDF"]
    shader.inputs["Roughness"].default_value = 0.9
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.43, 0.46, 0.49, 1.0)
    checker.inputs["Color2"].default_value = (0.57, 0.60, 0.63, 1.0)
    checker.inputs["Scale"].default_value = 6.0 / span
    coordinates = nodes.new("ShaderNodeTexCoord")
    links.new(coordinates.outputs["Object"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], shader.inputs["Base Color"])
    floor.data.materials.append(floor_material)

    for direction in ((3.0, -4.0, 6.0), (-4.0, 2.0, 3.0)):
        bpy.ops.object.light_add(
            type="AREA", location=center + Vector(direction) * span
        )
        light = bpy.context.object
        light.data.energy = 350.0 * span**2
        light.data.shape = "DISK"
        light.data.size = 4.0 * span
        light.rotation_euler = (
            (center - light.location).to_track_quat("-Z", "Y").to_euler()
        )

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = camera_span
    camera.data.clip_start = max(1e-6, span * 1e-4)
    camera.data.clip_end = max(1000.0, span * 100.0)
    camera.location = camera_center + 4.0 * span * Vector(
        (0.0, -np.cos(angle), np.sin(angle))
    )
    camera.rotation_euler = (
        (camera_center - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    scene.camera = camera

    for transforms, output_path in zip(candidate_matrices, output_paths, strict=True):
        for root, transform in zip(roots, transforms, strict=True):
            root.matrix_world = transform
        bpy.context.view_layer.update()
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)


def _heading_camera_projected_bounds(
    meshes: list[object], camera_basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bound all mesh boxes in one fixed camera frame without sampling vertices."""
    projected = []
    for mesh in meshes:
        transform = np.asarray(mesh.matrix_world, dtype=float)
        corners = np.asarray(mesh.bound_box, dtype=float)
        world = corners @ transform[:3, :3].T + transform[:3, 3]
        projected.append(world @ camera_basis.T)
    points = np.concatenate(projected, axis=0)
    return points.min(axis=0), points.max(axis=0)


def _render_articulation_orientation_views(
    *,
    usdc_path: str | Path,
    native_to_upright: list[list[float]],
    proxy_yaw_degrees: float,
    output_paths: list[str],
    resolution: int = 384,
) -> None:
    """Render four yaw offsets of one USD asset with its imported materials.

    The input USD already expresses visible geometry in the native root
    frame at the default joint configuration. The output paths are in order
    of positive offsets 0, 90, 180 and 270 degrees about world Z.
    """
    from mathutils import Matrix, Vector

    if resolution <= 0 or len(output_paths) != 4 or len(set(output_paths)) != 4:
        raise ValueError("Expected a positive resolution and four distinct outputs.")
    source = Path(usdc_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"USD articulation preview source not found: {source}")
    rotation = np.asarray(native_to_upright, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Expected a finite 3×3 native-to-upright rotation.")
    if not np.isfinite(proxy_yaw_degrees):
        raise ValueError("Expected a finite proxy yaw.")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(
        filepath=str(source),
        import_cameras=False,
        import_curves=False,
        import_lights=False,
        import_volumes=False,
        import_points=False,
        import_guide=False,
        import_proxy=False,
        import_render=True,
        import_visible_only=True,
        import_materials=True,
        import_usd_preview=True,
        import_textures_mode="IMPORT_PACK",
        support_scene_instancing=False,
    )
    scene = bpy.context.scene
    objects = list(scene.objects)
    meshes = [obj for obj in objects if obj.type == "MESH" and len(obj.data.vertices)]
    if not meshes:
        raise ValueError(f"USD articulation contains no visible mesh: {source}")

    root = bpy.data.objects.new("ArticulationPreviewRoot", None)
    scene.collection.objects.link(root)
    for obj in objects:
        if obj.parent is None:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    upright = Matrix(rotation.tolist()).to_4x4()
    root.matrix_world = upright
    bpy.context.view_layer.update()
    lower, upper = _articulation_preview_bounds(meshes)
    extent = float((upper - lower).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Articulation preview requires a finite, nonzero mesh extent.")
    ground_center = (lower + upper) * 0.5
    ground_center[2] = lower[2]
    scale = _VISUAL_MAX_OBJECT_EXTENT / extent
    normalized = (
        Matrix.Scale(scale, 4) @ Matrix.Translation(Vector(-ground_center)) @ upright
    )

    _configure_vlm_render_scene(
        scene=scene, resolution=resolution, bpy=bpy, Vector=Vector, use_cycles=True
    )
    camera_data = bpy.data.cameras.new("ArticulationPreviewCamera")
    camera = bpy.data.objects.new("ArticulationPreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.3
    center = Vector((0.0, 0.0, scale * (upper[2] - lower[2]) * 0.5))
    elevation = np.deg2rad(25.0)
    outward = Vector((0.0, -np.cos(elevation), np.sin(elevation)))
    camera.location = center + 4.0 * outward
    camera.rotation_euler = (-outward).to_track_quat("-Z", "Y").to_euler()
    for offset, output_path in zip((0, 90, 180, 270), output_paths, strict=True):
        yaw = Matrix.Rotation(np.deg2rad(proxy_yaw_degrees + offset), 4, "Z")
        root.matrix_world = yaw @ normalized
        bpy.context.view_layer.update()
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(destination)
        _run_blender_operation_silently(lambda: bpy.ops.render.render(write_still=True))


def _articulation_preview_bounds(
    meshes: list[bpy.types.Object],
) -> tuple[np.ndarray, np.ndarray]:
    """Measure transformed visual vertices without allocating a combined mesh."""
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for obj in meshes:
        vertices = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", vertices)
        transform = np.asarray(obj.matrix_world)
        world = vertices.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
        np.minimum(lower, world.min(axis=0), out=lower)
        np.maximum(upper, world.max(axis=0), out=upper)
    return lower, upper
