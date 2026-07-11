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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import json
import math
import struct

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_io import read_glb

__all__ = [
    "_GLTF_TO_SIM_FRAME_KEY",
    "_apply_tabletop_z_placement",
    "_clean_vector3",
    "_dual_ur5_init_z_from_table_top",
    "_iter_generated_scene_object_configs",
    "_mesh_config_has_distinct_xy_axis",
    "_mesh_config_world_xy_bounds",
    "_mesh_config_world_xy_center",
    "_mesh_config_world_xy_extents",
    "_mesh_config_world_z_bounds",
    "_mesh_config_world_zmax",
    "_resolve_table_mesh_world_zmax",
    "_vector3",
]

_GLTF_TO_SIM_FRAME_KEY = "_gltf_to_sim_frame"

_DUAL_UR5_LEGACY_INIT_Z = 0.5
_DUAL_UR5_ARM_COMPONENT_Z = 0.4
_DUAL_UR5_TABLETOP_CLEARANCE = 0.7
_DUAL_FRANKA_TABLETOP_CLEARANCE = 0.7
_TABLETOP_OBJECT_CLEARANCE = 0.003
_GLTF_COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_GLTF_TYPE_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}
_GLTF_NORMALIZED_UNSIGNED_MAX = {
    5121: 255.0,
    5123: 65535.0,
    5125: 4294967295.0,
}
_GLTF_NORMALIZED_SIGNED_MAX = {
    5120: 127.0,
    5122: 32767.0,
}


def _dual_ur5_init_z_from_table_top(table_top_z: float | None) -> float:
    if table_top_z is None:
        return _DUAL_UR5_LEGACY_INIT_Z

    init_z = table_top_z + _DUAL_UR5_TABLETOP_CLEARANCE - _DUAL_UR5_ARM_COMPONENT_Z
    return round(init_z, 6)


def _apply_tabletop_z_placement(
    gym_config: dict[str, Any],
    table_top_z: float | None,
) -> None:
    if table_top_z is None:
        return
    target_bottom_z = float(table_top_z) + _TABLETOP_OBJECT_CLEARANCE
    for obj in _iter_generated_scene_object_configs(gym_config):
        if obj.get("uid") == "table":
            continue
        mesh_min_z = _mesh_config_local_zmin_after_rotation(obj)
        if mesh_min_z is None:
            continue
        init_pos = _clean_vector3(obj.get("init_pos", [0.0, 0.0, 0.0]))
        init_pos[2] = round(target_bottom_z - mesh_min_z, 6)
        obj["init_pos"] = init_pos


def _iter_generated_scene_object_configs(
    gym_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for section in ("background", "rigid_object"):
        value = gym_config.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            continue
        objects.extend(obj for obj in value if isinstance(obj, dict))
    return objects


def _mesh_config_world_zmax(obj_config: Mapping[str, Any]) -> float | None:
    bounds = _mesh_config_world_z_bounds(obj_config)
    if bounds is None:
        return None
    return bounds[1]


def _mesh_config_world_xy_extents(
    obj_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    bounds = _mesh_config_world_xy_bounds(obj_config)
    if bounds is None:
        return None
    mins, maxs = bounds
    return (
        float(maxs[0]) - float(mins[0]),
        float(maxs[1]) - float(mins[1]),
    )


def _mesh_config_has_distinct_xy_axis(
    obj_config: Mapping[str, Any],
    *,
    min_aspect_ratio: float = 1.2,
) -> bool:
    extents = _mesh_config_scaled_xy_extents(obj_config)
    if extents is None:
        return False
    x_extent, y_extent = (abs(float(extents[0])), abs(float(extents[1])))
    long_extent = max(x_extent, y_extent)
    short_extent = min(x_extent, y_extent)
    if long_extent <= 1e-6:
        return False
    if short_extent <= 1e-6:
        return True
    return long_extent / short_extent >= min_aspect_ratio


def _mesh_config_scaled_xy_extents(
    obj_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    shape = obj_config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    mesh_path = shape.get("fpath")
    if not isinstance(mesh_path, str):
        return None
    vertices = _load_mesh_vertices(
        Path(mesh_path).expanduser().resolve(),
        gltf_to_sim_frame=bool(shape.get(_GLTF_TO_SIM_FRAME_KEY, False)),
    )
    if not vertices:
        return None

    scale = _vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    x_values = [float(vertex[0]) * scale[0] for vertex in vertices]
    y_values = [float(vertex[1]) * scale[1] for vertex in vertices]
    return (
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
    )


def _mesh_config_world_xy_center(
    obj_config: Mapping[str, Any],
) -> list[float] | None:
    bounds = _mesh_config_world_xy_bounds(obj_config)
    if bounds is None:
        return None
    mins, maxs = bounds
    return [
        round((float(mins[0]) + float(maxs[0])) / 2.0, 6),
        round((float(mins[1]) + float(maxs[1])) / 2.0, 6),
    ]


def _mesh_config_world_xy_bounds(
    obj_config: Mapping[str, Any],
) -> tuple[list[float], list[float]] | None:
    shape = obj_config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    mesh_path = shape.get("fpath")
    if not isinstance(mesh_path, str):
        return None
    vertices = _load_mesh_vertices(
        Path(mesh_path).expanduser().resolve(),
        gltf_to_sim_frame=bool(shape.get(_GLTF_TO_SIM_FRAME_KEY, False)),
    )
    if not vertices:
        return None

    matrix = _mesh_config_transform_matrix(obj_config)
    transformed_vertices = [_transform_point(matrix, vertex) for vertex in vertices]
    x_values = [vertex[0] for vertex in transformed_vertices]
    y_values = [vertex[1] for vertex in transformed_vertices]
    return (
        [min(x_values), min(y_values)],
        [max(x_values), max(y_values)],
    )


def _mesh_config_local_zmin_after_rotation(
    obj_config: Mapping[str, Any],
) -> float | None:
    shape = obj_config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    mesh_path = shape.get("fpath")
    if not isinstance(mesh_path, str):
        return None
    vertices = _load_mesh_vertices(
        Path(mesh_path).expanduser().resolve(),
        gltf_to_sim_frame=bool(shape.get(_GLTF_TO_SIM_FRAME_KEY, False)),
    )
    if not vertices:
        return None

    matrix = _mesh_config_transform_matrix(
        obj_config,
        translation=[0.0, 0.0, 0.0],
    )
    return min(_transform_point(matrix, vertex)[2] for vertex in vertices)


def _mesh_config_world_z_bounds(
    obj_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    shape = obj_config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    mesh_path = shape.get("fpath")
    if not isinstance(mesh_path, str):
        return None
    vertices = _load_mesh_vertices(
        Path(mesh_path).expanduser().resolve(),
        gltf_to_sim_frame=bool(shape.get(_GLTF_TO_SIM_FRAME_KEY, False)),
    )
    if not vertices:
        return None

    matrix = _mesh_config_transform_matrix(obj_config)
    z_values = [_transform_point(matrix, vertex)[2] for vertex in vertices]
    return (min(z_values), max(z_values))


def _mesh_config_transform_matrix(
    obj_config: Mapping[str, Any],
    *,
    translation: list[float] | None = None,
) -> list[list[float]]:
    scale = _vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    init_local_pose = obj_config.get("init_local_pose")
    if init_local_pose is not None and translation is None:
        root_matrix = _matrix4(init_local_pose)
    else:
        root_matrix = _euler_xyz_degrees_matrix(
            _vector3(obj_config.get("init_rot", [0.0, 0.0, 0.0])),
            (
                _vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
                if translation is None
                else translation
            ),
        )
    return _matrix_multiply(root_matrix, _scale_matrix4(scale))


def _resolve_table_mesh_world_zmax(
    scene_dir: Path,
    table_obj: _SceneObject,
) -> float | None:
    shape = table_obj.config.get("shape", {})
    if not isinstance(shape, Mapping):
        return None
    if shape.get("shape_type") != "Mesh" or not shape.get("fpath"):
        return None

    mesh_path = _source_asset_path(scene_dir, str(shape["fpath"]))
    try:
        vertices = _load_mesh_vertices(
            mesh_path,
            gltf_to_sim_frame=bool(shape.get(_GLTF_TO_SIM_FRAME_KEY, False)),
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        struct.error,
    ):
        return None
    if not vertices:
        return None

    world_matrix = _table_mesh_world_matrix(table_obj.config)
    return max(_transform_point(world_matrix, vertex)[2] for vertex in vertices)


def _source_asset_path(scene_dir: Path, fpath: str) -> Path:
    raw_path = Path(fpath)
    if raw_path.is_absolute():
        return raw_path.resolve()

    scene_candidate = (scene_dir / raw_path).resolve()
    if scene_candidate.exists():
        return scene_candidate

    repo_candidate = (_repo_root() / raw_path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return scene_candidate


def _load_mesh_vertices(
    mesh_path: Path,
    *,
    gltf_to_sim_frame: bool = False,
) -> list[tuple[float, float, float]] | None:
    if mesh_path.suffix.lower() == ".glb":
        try:
            vertices = list(_iter_glb_world_position_vertices(mesh_path))
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            struct.error,
        ):
            vertices = _load_mesh_vertices_with_trimesh(mesh_path)
        return _maybe_convert_gltf_vertices_to_sim_frame(
            vertices,
            enabled=gltf_to_sim_frame,
        )
    if mesh_path.suffix.lower() == ".gltf":
        return _maybe_convert_gltf_vertices_to_sim_frame(
            _load_mesh_vertices_with_trimesh(mesh_path),
            enabled=gltf_to_sim_frame,
        )
    if mesh_path.suffix.lower() == ".obj":
        vertices = _load_obj_position_vertices(mesh_path)
        if vertices is not None:
            return vertices
    return _load_mesh_vertices_with_trimesh(mesh_path)


def _maybe_convert_gltf_vertices_to_sim_frame(
    vertices: list[tuple[float, float, float]] | None,
    *,
    enabled: bool,
) -> list[tuple[float, float, float]] | None:
    if not enabled or vertices is None:
        return vertices
    return [(x, -z, y) for x, y, z in vertices]


def _load_obj_position_vertices(
    mesh_path: Path,
) -> list[tuple[float, float, float]] | None:
    try:
        vertices = []
        for line in mesh_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) < 4:
                continue
            vertices.append((float(values[1]), float(values[2]), float(values[3])))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return vertices or None


def _load_mesh_vertices_with_trimesh(
    mesh_path: Path,
) -> list[tuple[float, float, float]] | None:
    try:
        import trimesh
    except ImportError:
        return None

    try:
        scene_or_mesh = trimesh.load(str(mesh_path), force="scene")
        if hasattr(scene_or_mesh, "to_geometry"):
            mesh = scene_or_mesh.to_geometry()
        elif hasattr(scene_or_mesh, "dump"):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
    except Exception as exc:
        raise ValueError(f"Failed to load mesh vertices from {mesh_path}.") from exc
    vertices = getattr(mesh, "vertices", None)
    if vertices is None or len(vertices) == 0:
        return None
    return [
        (float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in vertices
    ]


def _iter_glb_world_position_vertices(
    mesh_path: Path,
):
    doc, binary_chunk = read_glb(mesh_path)
    nodes = doc.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("GLB nodes must be a list.")

    scenes = doc.get("scenes", [])
    if scenes:
        scene_index = int(doc.get("scene", 0))
        root_node_ids = scenes[scene_index].get("nodes", [])
    else:
        root_node_ids = list(range(len(nodes)))

    stack = [(int(node_id), _identity_matrix4()) for node_id in root_node_ids]
    while stack:
        node_id, parent_matrix = stack.pop()
        node = nodes[node_id]
        node_matrix = _matrix_multiply(parent_matrix, _gltf_node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            for vertex in _iter_gltf_mesh_position_vertices(
                doc,
                binary_chunk,
                int(mesh_index),
            ):
                yield _transform_point(node_matrix, vertex)
        for child_id in node.get("children", []) or []:
            stack.append((int(child_id), node_matrix))


def _iter_gltf_mesh_position_vertices(
    doc: Mapping[str, Any],
    binary_chunk: bytes,
    mesh_index: int,
):
    meshes = doc.get("meshes", [])
    accessors = doc.get("accessors", [])
    mesh = meshes[mesh_index]
    for primitive in mesh.get("primitives", []) or []:
        attributes = primitive.get("attributes", {})
        position_accessor = attributes.get("POSITION")
        if position_accessor is None:
            continue
        if int(position_accessor) >= len(accessors):
            raise ValueError("POSITION accessor index is out of range.")
        yield from _iter_gltf_accessor_vec3(doc, binary_chunk, int(position_accessor))


def _iter_gltf_accessor_vec3(
    doc: Mapping[str, Any],
    binary_chunk: bytes,
    accessor_index: int,
):
    accessor = doc["accessors"][accessor_index]
    if accessor.get("sparse"):
        raise ValueError("Sparse GLB accessors are not supported.")
    if accessor.get("type") != "VEC3":
        raise ValueError("POSITION accessor must be VEC3.")
    if "bufferView" not in accessor:
        raise ValueError("POSITION accessor must reference a bufferView.")

    component_type = int(accessor["componentType"])
    if component_type not in _GLTF_COMPONENT_FORMATS:
        raise ValueError(f"Unsupported GLB component type: {component_type}.")
    component_format, component_size = _GLTF_COMPONENT_FORMATS[component_type]
    component_count = _GLTF_TYPE_COMPONENT_COUNTS[accessor["type"]]
    buffer_view = doc["bufferViews"][int(accessor["bufferView"])]
    if int(buffer_view.get("buffer", 0)) != 0:
        raise ValueError("Only GLB embedded binary buffers are supported.")

    stride = int(buffer_view.get("byteStride", component_size * component_count))
    offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    element_format = "<" + component_format * component_count
    normalized = bool(accessor.get("normalized", False))
    for index in range(int(accessor["count"])):
        values = struct.unpack_from(
            element_format,
            binary_chunk,
            offset + index * stride,
        )
        if normalized:
            values = tuple(
                _normalize_gltf_component(value, component_type) for value in values
            )
        yield (float(values[0]), float(values[1]), float(values[2]))


def _normalize_gltf_component(value: int | float, component_type: int) -> float:
    if component_type in _GLTF_NORMALIZED_UNSIGNED_MAX:
        return float(value) / _GLTF_NORMALIZED_UNSIGNED_MAX[component_type]
    if component_type in _GLTF_NORMALIZED_SIGNED_MAX:
        return max(float(value) / _GLTF_NORMALIZED_SIGNED_MAX[component_type], -1.0)
    return float(value)


def _table_mesh_world_matrix(table_config: Mapping[str, Any]) -> list[list[float]]:
    scale = _vector3(table_config.get("body_scale", [1.0, 1.0, 1.0]))
    init_local_pose = table_config.get("init_local_pose")
    if init_local_pose is not None:
        root_matrix = _matrix4(init_local_pose)
    else:
        root_matrix = _euler_xyz_degrees_matrix(
            _vector3(table_config.get("init_rot", [0.0, 0.0, 0.0])),
            _vector3(table_config.get("init_pos", [0.0, 0.0, 0.0])),
        )
    return _matrix_multiply(root_matrix, _scale_matrix4(scale))


def _gltf_node_matrix(node: Mapping[str, Any]) -> list[list[float]]:
    if "matrix" in node:
        values = [float(value) for value in node["matrix"]]
        if len(values) != 16:
            raise ValueError("GLB node matrix must contain 16 values.")
        return [[values[column * 4 + row] for column in range(4)] for row in range(4)]

    translation = [float(value) for value in node.get("translation", [0.0, 0.0, 0.0])]
    scale = [float(value) for value in node.get("scale", [1.0, 1.0, 1.0])]
    rotation = [float(value) for value in node.get("rotation", [0.0, 0.0, 0.0, 1.0])]
    if len(translation) != 3 or len(scale) != 3 or len(rotation) != 4:
        raise ValueError("Invalid GLB node TRS transform.")

    x, y, z, w = rotation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = [
        [
            (1.0 - 2.0 * (yy + zz)) * scale[0],
            (2.0 * (xy - wz)) * scale[1],
            (2.0 * (xz + wy)) * scale[2],
            translation[0],
        ],
        [
            (2.0 * (xy + wz)) * scale[0],
            (1.0 - 2.0 * (xx + zz)) * scale[1],
            (2.0 * (yz - wx)) * scale[2],
            translation[1],
        ],
        [
            (2.0 * (xz - wy)) * scale[0],
            (2.0 * (yz + wx)) * scale[1],
            (1.0 - 2.0 * (xx + yy)) * scale[2],
            translation[2],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matrix


def _euler_xyz_degrees_matrix(
    rotation_deg: Sequence[float],
    translation: Sequence[float],
) -> list[list[float]]:
    rx, ry, rz = (math.radians(float(value)) for value in rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_y = [
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_z = [
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # Match RigidObject.reset and Prompt2Scene's intrinsic XYZ convention.
    matrix = _matrix_multiply(_matrix_multiply(rot_x, rot_y), rot_z)
    matrix[0][3] = float(translation[0])
    matrix[1][3] = float(translation[1])
    matrix[2][3] = float(translation[2])
    return matrix


def _identity_matrix4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _scale_matrix4(scale: Sequence[float]) -> list[list[float]]:
    return [
        [float(scale[0]), 0.0, 0.0, 0.0],
        [0.0, float(scale[1]), 0.0, 0.0],
        [0.0, 0.0, float(scale[2]), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix4(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected a 4x4 matrix, got {value!r}.")
    matrix = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f"Expected a 4x4 matrix, got {value!r}.")
        matrix.append([float(item) for item in row])
    return matrix


def _matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            sum(
                float(left[row][inner]) * float(right[inner][column])
                for inner in range(4)
            )
            for column in range(4)
        ]
        for row in range(4)
    ]


def _transform_point(
    matrix: Sequence[Sequence[float]],
    point: Sequence[float],
) -> tuple[float, float, float]:
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return (
        float(matrix[0][0]) * x
        + float(matrix[0][1]) * y
        + float(matrix[0][2]) * z
        + float(matrix[0][3]),
        float(matrix[1][0]) * x
        + float(matrix[1][1]) * y
        + float(matrix[1][2]) * z
        + float(matrix[1][3]),
        float(matrix[2][0]) * x
        + float(matrix[2][1]) * y
        + float(matrix[2][2]) * z
        + float(matrix[2][3]),
    )


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() and (parent / "embodichain").exists():
            return parent
    return Path.cwd().resolve()


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}.")
    return [float(item) for item in value]


def _clean_vector3(value: Any) -> list[float]:
    cleaned = []
    for item in _vector3(value):
        if abs(item - 1.0) < 1e-9:
            cleaned.append(1.0)
        elif abs(item) < 1e-12:
            cleaned.append(0.0)
        else:
            cleaned.append(item)
    return cleaned
