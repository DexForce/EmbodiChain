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

import os
from pathlib import Path
import sys
from typing import Any, Callable

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


def modify_mtl_file(mtl_path: Path, diffuse_name: str, normal_name: str) -> None:
    """Modify an exported OBJ .mtl to reference baked textures."""
    mtl_path = Path(mtl_path)
    if not mtl_path.exists():
        return

    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    new_lines = []
    for line in lines:
        if line.startswith("Ns "):
            new_lines.append("Ns 500.000000\n")
        elif line.startswith("Ka "):
            new_lines.append("Ka 1.000000 1.000000 1.000000\n")
        elif line.startswith("Ks "):
            new_lines.append("Ks 0.500000 0.500000 0.500000\n")
        else:
            new_lines.append(line)

    new_lines.append(f"map_Kd {diffuse_name}\n")
    new_lines.append(f"map_Bump {normal_name}\n")
    new_lines.append(f"bump {normal_name} -bm 1.0\n")

    mtl_path.write_text("".join(new_lines), encoding="utf-8")


def _blender_remesh_bake(
    source_file: Path,
    asset_source: Path,
    texture_size: int | None = None,
    png_name: str | None = None,
    voxel_size: float | None = None,
    decimate_ratio: float | None = None,
    obj_name: str = "asset.obj",
    config: Dict[str, Any] | None = None,
):
    """Remesh a high-poly mesh into a low-poly one and bake textures via Blender."""
    import bpy

    config = config or {}
    remesh_config = config.get("remesh", {})
    decimate_config = config.get("decimate", {})
    uv_config = config.get("uv", {})
    bake_config = config.get("bake", {})
    material_config = config.get("material", {})

    texture_size = int(
        texture_size
        or bake_config.get("texture_size", config.get("texture_size", 2048))
    )
    diffuse_texture_name = png_name or bake_config.get(
        "diffuse_texture_name", config.get("texture_name", "diffuse.png")
    )
    normal_texture_name = bake_config.get(
        "normal_texture_name", config.get("normal_texture_name", "normal.png")
    )
    voxel_size = float(
        voxel_size
        if voxel_size is not None
        else remesh_config.get("voxel_size", config.get("voxel_size", 0.01))
    )
    min_voxel_size_ratio = float(remesh_config.get("min_voxel_size_ratio", 0.005))
    use_smooth_shade = bool(remesh_config.get("use_smooth_shade", True))
    decimate_ratio = float(
        decimate_ratio
        if decimate_ratio is not None
        else decimate_config.get("ratio", config.get("decimate_ratio", 0.5))
    )
    angle_limit = float(uv_config.get("angle_limit", 66.0))
    island_margin = float(uv_config.get("island_margin", 0.02))
    cage_extrusion_ratio = float(bake_config.get("cage_extrusion_ratio", 0.05))
    material_name = material_config.get("name", "BakeMat")

    asset_source = Path(asset_source)
    asset_source.mkdir(parents=True, exist_ok=True)
    source_file = Path(source_file)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = source_file.suffix.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source_file))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source_file))
    elif ext in [".gltf", ".glb"]:
        bpy.ops.import_scene.gltf(filepath=str(source_file))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(source_file))
    else:
        raise RuntimeError(f"Unsupported extension: {ext}")

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    imported_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not imported_meshes:
        raise RuntimeError("No mesh object after import")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_meshes[0]

    if len(imported_meshes) > 1:
        bpy.ops.object.join()
    high_poly = bpy.context.view_layer.objects.active
    if not high_poly or high_poly.type != "MESH":
        raise RuntimeError("No active mesh object after import")
    high_poly.name = "High_Poly"

    auto_extrusion = max(high_poly.dimensions) * cage_extrusion_ratio

    bpy.ops.object.select_all(action="DESELECT")
    high_poly.select_set(True)
    bpy.context.view_layer.objects.active = high_poly
    bpy.ops.object.duplicate()
    low_poly = bpy.context.active_object
    if not low_poly:
        raise RuntimeError("Failed to duplicate object")
    low_poly.name = "Low_Poly_Target"
    try:
        low_poly.data.materials.clear()
    except Exception:
        pass

    rem = low_poly.modifiers.new(name="Remesh", type="REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = max(
        float(voxel_size), max(high_poly.dimensions) * min_voxel_size_ratio
    )
    rem.use_smooth_shade = use_smooth_shade
    bpy.ops.object.modifier_apply(modifier="Remesh")

    dec = low_poly.modifiers.new(name="Decimate", type="DECIMATE")
    dec.ratio = float(decimate_ratio)
    bpy.ops.object.modifier_apply(modifier="Decimate")

    bpy.context.view_layer.objects.active = low_poly
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    low_poly.data.materials.append(mat)
    nodes = mat.node_tree.nodes
    nodes.clear()

    def setup_node(name: str, is_color: bool):
        img = bpy.data.images.new(
            name, width=int(texture_size), height=int(texture_size)
        )
        node = nodes.new("ShaderNodeTexImage")
        node.image = img
        if not is_color:
            img.colorspace_settings.name = "Non-Color"
        return node, img

    diff_node, diff_img = setup_node(diffuse_texture_name, True)
    norm_node, norm_img = setup_node(normal_texture_name, False)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = auto_extrusion

    bpy.ops.object.select_all(action="DESELECT")
    high_poly.select_set(True)
    low_poly.select_set(True)
    bpy.context.view_layer.objects.active = low_poly

    nodes.active = diff_node
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    diff_img.filepath_raw = str(asset_source / diffuse_texture_name)
    diff_img.save()

    nodes.active = norm_node
    bpy.ops.object.bake(type="NORMAL")
    norm_img.filepath_raw = str(asset_source / normal_texture_name)
    norm_img.save()

    export_path = asset_source / obj_name
    bpy.ops.object.select_all(action="DESELECT")
    low_poly.select_set(True)
    bpy.ops.wm.obj_export(filepath=str(export_path), export_selected_objects=True)

    mtl_path = asset_source / Path(obj_name).with_suffix(".mtl").name
    modify_mtl_file(mtl_path, diffuse_texture_name, normal_texture_name)

    return {
        "png": str(asset_source / diffuse_texture_name),
        "obj": str(export_path),
        "mtl": str(mtl_path.name),
    }


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False, confirm=False)

    for block in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.collections,
    ):
        for item in list(block):
            try:
                block.remove(item)
            except:
                pass


def load_obj(filepath):
    bpy.ops.wm.obj_import(filepath=str(filepath))
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return objs


def join_meshes(objs):
    if not objs:
        raise RuntimeError("No mesh objects to join.")

    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)

    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def decimate_optimized(
    obj,
    ratio: float = 0.5,
    weld_distance: float = 0.0001,
    collapse_triangulate: bool = True,
):

    bpy.context.view_layer.objects.active = obj

    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # 1) Weld
    # weld_mod = obj.modifiers.new(name="Weld", type="WELD")
    # weld_mod.merge_threshold = weld_distance
    # bpy.ops.object.modifier_apply(modifier=weld_mod.name)
    # bpy.ops.object.mode_set(mode="EDIT")
    # bpy.ops.mesh.select_all(action="SELECT")

    # bpy.ops.mesh.normals_make_consistent(inside=False)
    # bpy.ops.mesh.customdata_custom_splitnormals_clear()

    # bpy.ops.object.mode_set(mode="OBJECT")

    # 2) remove loose
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_loose()
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # 3) decimate
    print(f"Simplifying mesh (Ratio: {ratio})...")
    decimate_mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    decimate_mod.ratio = ratio
    decimate_mod.use_collapse_triangulate = collapse_triangulate
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    # 4) post clean
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=weld_distance)
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode="OBJECT")

    print(
        f"[Info] Optimized state: Vertices {len(obj.data.vertices)}, Faces {len(obj.data.polygons)}"
    )

    return obj


def clean_mesh(obj, merge_dist=1e-5, remove_non_manifold=True, triangulate=False):
    bpy.context.view_layer.objects.active = obj

    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.mesh.remove_doubles(threshold=merge_dist)

    bpy.ops.mesh.delete_loose()

    bpy.ops.mesh.dissolve_degenerate()

    bpy.ops.mesh.normals_make_consistent(inside=False)

    if remove_non_manifold:
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.delete(type="VERT")

    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=merge_dist)
    bpy.ops.mesh.delete_loose()

    if triangulate:
        bpy.ops.mesh.quads_convert_to_tris()

    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def fill_holes(obj, max_sides=8):
    bpy.context.view_layer.objects.active = obj

    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.mesh.fill_holes(sides=max_sides)

    bpy.ops.mesh.beautify_fill()
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.mesh.normals_make_consistent(inside=False)

    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def export_obj(obj, out_path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.wm.obj_export(filepath=str(out_path), export_selected_objects=True)


def process_obj(
    input_path,
    output_path,
    ratio=0.5,
    weld_distance=0.0001,
    merge_dist=1e-5,
    remove_non_manifold=True,
    triangulate=False,
    collapse_triangulate=True,
):
    clear_scene()
    objs = load_obj(input_path)
    if not objs:
        raise RuntimeError("No mesh objects imported.")

    obj = join_meshes(objs)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    clean_mesh(
        obj,
        merge_dist=merge_dist,
        remove_non_manifold=remove_non_manifold,
        triangulate=triangulate,
    )
    decimate_optimized(
        obj,
        ratio=ratio,
        weld_distance=weld_distance,
        collapse_triangulate=collapse_triangulate,
    )

    export_obj(obj, output_path)
    print("Clean mesh saved to:", output_path)
