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
from pathlib import Path
from typing import Any

__all__ = ["scale_mesh_to_real_dimensions"]


def scale_mesh_to_real_dimensions(
    *,
    mesh_path: Path,
    output_path: Path,
    dimensions_m: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    """Scale a canonical GLB mesh for Blender with object up along -Y.

    glTF/GLB stores assets in y-up coordinates. Blender converts glTF y-up
    assets to its z-up scene coordinates during import. The exported vertices
    are arranged so that after Blender import the object's original y-up axis
    becomes Blender -Y, and the bbox bottom-center is at the world origin.
    """
    trimesh = _require_trimesh()
    np = _require_numpy()
    mesh_path = mesh_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    scene = trimesh.load(str(mesh_path), force="scene")
    mesh = _scene_to_world_mesh(scene)
    bounds = _mesh_bounds(mesh)
    extents = bounds[1] - bounds[0]
    axis_map = _axis_mapping(extents)
    target_extents = np.asarray(
        [
            dimensions_m[axis_map["x"]],
            dimensions_m[axis_map["y"]],
            dimensions_m[axis_map["z"]],
        ],
        dtype=np.float64,
    )
    source_max_extent = float(max(extents) or 1.0)
    target_max_extent = float(max(target_extents))
    uniform_scale = target_max_extent / source_max_extent
    scale = np.asarray([uniform_scale, uniform_scale, uniform_scale], dtype=np.float64)
    bottom_center_y_up = _bottom_center_y_up(bounds)
    gltf_to_blender = _gltf_y_up_to_blender_z_up_matrix(np)
    original_to_blender = _original_y_up_to_blender_negative_y_up_matrix(np)
    original_to_export = np.linalg.inv(gltf_to_blender) @ original_to_blender
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = original_to_export @ np.diag(scale)
    transform[:3, 3] = -(original_to_export @ np.diag(scale) @ bottom_center_y_up)
    mesh.apply_transform(transform)
    exported_bounds = _mesh_bounds(mesh)
    blender_bounds = _bounds(_transform_vertices(mesh.vertices, gltf_to_blender))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    report = {
        "input_mesh_path": str(mesh_path),
        "scaled_mesh_path": str(output_path),
        "axis_convention": (
            "Input GLB is treated as y-up. After Blender's glTF import, the "
            "object's original y-up axis is aligned to Blender -Y. length_m "
            "maps to the larger generated horizontal axis among input x/z; "
            "width_m maps to the other."
        ),
        "scaling_policy": (
            "The mesh is scaled uniformly to preserve generated geometry "
            "proportions. The source mesh is first considered normalized by "
            "its maximum bbox extent; the uniform scale is computed as "
            "estimated_max_real_extent / mesh_max_extent."
        ),
        "origin_policy": (
            "The input y-up bbox bottom-center is subtracted before GLB export. "
            "After Blender import, the -Y-up bbox bottom-center is at "
            "(0, 0, 0), so its XZ-plane location is (0, 0)."
        ),
        "axis_map": axis_map,
        "estimated_dimensions_m": dimensions_m,
        "estimated_target_extents_by_mesh_axes": target_extents.tolist(),
        "source_max_extent": source_max_extent,
        "estimated_max_real_extent": target_max_extent,
        "original_bounds": bounds.tolist(),
        "original_extents": extents.tolist(),
        "bottom_center_y_up_subtracted": bottom_center_y_up.tolist(),
        "gltf_to_blender_matrix": gltf_to_blender.tolist(),
        "original_to_blender_matrix": original_to_blender.tolist(),
        "original_to_export_matrix": original_to_export.tolist(),
        "uniform_scale": uniform_scale,
        "applied_transform": transform.tolist(),
        "exported_gltf_bounds": exported_bounds.tolist(),
        "exported_gltf_extents": (exported_bounds[1] - exported_bounds[0]).tolist(),
        "blender_import_bounds": blender_bounds.tolist(),
        "blender_import_extents": (blender_bounds[1] - blender_bounds[0]).tolist(),
        "blender_import_bottom_center_negative_y_up": _bottom_center_negative_y_up(
            blender_bounds
        ).tolist(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _axis_mapping(extents: Any) -> dict[str, str]:
    if float(extents[0]) >= float(extents[2]):
        return {"x": "length_m", "y": "height_m", "z": "width_m"}
    return {"x": "width_m", "y": "height_m", "z": "length_m"}


def _bottom_center_negative_y_up(bounds: Any) -> Any:
    np = _require_numpy()
    return np.asarray(
        [
            0.5 * (bounds[0][0] + bounds[1][0]),
            bounds[1][1],
            0.5 * (bounds[0][2] + bounds[1][2]),
        ],
        dtype=np.float64,
    )


def _bottom_center_y_up(bounds: Any) -> Any:
    np = _require_numpy()
    return np.asarray(
        [
            0.5 * (bounds[0][0] + bounds[1][0]),
            bounds[0][1],
            0.5 * (bounds[0][2] + bounds[1][2]),
        ],
        dtype=np.float64,
    )


def _bounds(vertices: Any) -> Any:
    np = _require_numpy()
    return np.vstack([vertices.min(axis=0), vertices.max(axis=0)])


def _transform_vertices(vertices: Any, matrix: Any) -> Any:
    np = _require_numpy()
    vertices_array = np.asarray(vertices, dtype=np.float64)
    matrix_array = np.asarray(matrix, dtype=np.float64)
    return vertices_array @ matrix_array.T


def _gltf_y_up_to_blender_z_up_matrix(np: Any) -> Any:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _original_y_up_to_blender_negative_y_up_matrix(np: Any) -> Any:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _mesh_bounds(mesh: Any) -> Any:
    np = _require_numpy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError("Mesh contains no vertices.")
    return _bounds(vertices)


def _scene_to_world_mesh(scene: Any) -> Any:
    """Convert a loaded GLB scene to one world-space mesh.

    This intentionally bakes scene graph transforms into vertex coordinates so
    later z-up conversion and origin anchoring are visible to downstream tools
    that only inspect mesh vertices.
    """
    try:
        mesh = scene.dump(concatenate=True)
    except Exception as exc:
        raise ValueError("Failed to concatenate GLB scene into a mesh.") from exc
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError("GLB scene contains no mesh vertices.")
    return mesh


def _require_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required to scale GLB meshes.") from exc
    return trimesh


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required to scale GLB meshes.") from exc
    return np
