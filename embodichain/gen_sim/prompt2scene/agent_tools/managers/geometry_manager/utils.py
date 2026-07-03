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

from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    DetectTabletopRequest,
    GeometryManager,
)

__all__ = [
    "_compose_json_matrices",
    "_compose_simready_to_aligned_matrix",
    "_decompose_transform_matrix",
    "_aabb_bottom_to_xy_plane_transform",
    "_aabb_center",
    "_compose_sam3d_multi_object_transform",
    "_copy_scene_with_transform",
    "_estimate_support_normal",
    "_glb_to_sam3d_local_matrix",
    "_load_scene_with_transform",
    "_matrix_from_json",
    "_quaternion_wxyz_to_matrix",
    "_rotation_between_vectors",
    "_row_linear_to_trimesh_matrix",
    "_scale_transform",
    "_scene_to_mesh",
    "_support_normal_flip_transform",
    "_transform_point",
    "_validate_vector",
    "_xy_aabb_center",
    "_xy_aabb_size",
    "_z_up_to_glb_y_up_transform",
    "_z_yaw_transform",
]


def _compose_json_matrices(*values: Any) -> list[list[float]]:
    matrices = [np.asarray(value, dtype=np.float64) for value in values]
    if any(matrix.shape != (4, 4) for matrix in matrices):
        return []
    result = np.eye(4, dtype=np.float64)
    for matrix in matrices:
        result = result @ matrix
    return result.tolist()


def _compose_simready_to_aligned_matrix(
    *, raw_to_aligned_matrix: Any, raw_to_simready_matrix: Any
) -> list[list[float]]:
    raw_to_aligned = np.asarray(raw_to_aligned_matrix, dtype=np.float64)
    raw_to_simready = np.asarray(raw_to_simready_matrix, dtype=np.float64)
    if raw_to_aligned.shape != (4, 4) or raw_to_simready.shape != (4, 4):
        return []
    try:
        return (raw_to_aligned @ np.linalg.inv(raw_to_simready)).tolist()
    except np.linalg.LinAlgError:
        return []


def _decompose_transform_matrix(matrix_value: Any) -> dict[str, Any]:
    matrix = np.asarray(matrix_value, dtype=np.float64)
    if matrix.shape != (4, 4):
        return {"translation": [], "rotation_matrix": [], "scale": []}
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    rotation = np.eye(3, dtype=np.float64)
    for index in range(3):
        if scale[index] > 1.0e-12:
            rotation[:, index] = linear[:, index] / scale[index]
    return {
        "translation": matrix[:3, 3].tolist(),
        "rotation_matrix": rotation.tolist(),
        "scale": scale.tolist(),
    }


def _support_normal_flip_transform(
    *,
    support_normal: np.ndarray,
    normal_alignment: np.ndarray,
) -> np.ndarray:
    flipped_normal_alignment = _rotation_between_vectors(
        -support_normal,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    return flipped_normal_alignment @ np.linalg.inv(normal_alignment)


def _z_yaw_transform(yaw_degrees: float) -> np.ndarray:
    angle = np.deg2rad(yaw_degrees)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform


def _z_up_to_glb_y_up_transform() -> np.ndarray:
    return _rotation_between_vectors(
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )


def _copy_scene_with_transform(scene: Any, transform: np.ndarray) -> Any:
    copied = scene.copy()
    copied.apply_transform(transform)
    return copied


def _matrix_from_json(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix.")
    return matrix


def _load_scene_with_transform(
    *,
    path: Path,
    transform: np.ndarray,
    trimesh: Any,
) -> Any:
    scene = trimesh.load(path, force="scene")
    scene.apply_transform(transform)
    return scene


def _scene_to_mesh(scene: Any, *, trimesh: Any) -> Any:
    if isinstance(scene, trimesh.Trimesh):
        return scene
    dumped = scene.dump(concatenate=True)
    if isinstance(dumped, trimesh.Trimesh):
        return dumped
    meshes = [item for item in dumped if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("Scene contains no mesh geometry.")
    return trimesh.util.concatenate(meshes)


def _estimate_support_normal(mesh: Any) -> np.ndarray:
    geom = GeometryManager()
    try:
        detect_result = geom.detect_tabletop(DetectTabletopRequest(mesh=mesh))
        normal = np.asarray(detect_result.oriented_normal, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm > 0.0:
            return normal / norm
    except Exception:
        pass

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if normals.size == 0 or areas.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = normals[int(np.argmax(areas))]
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.0:
            return np.eye(4, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(source, axis))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cross = np.cross(source, axis)
    axis = cross / np.linalg.norm(cross)
    angle = float(np.arccos(dot))
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point
    return (transform @ homogeneous)[:3]


def _aabb_center(bounds: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.asarray(bounds[0], dtype=np.float64)
        + np.asarray(bounds[1], dtype=np.float64)
    )


def _xy_aabb_center(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    return 0.5 * (bounds[0, :2] + bounds[1, :2])


def _xy_aabb_size(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    return np.maximum(bounds[1, :2] - bounds[0, :2], 1e-6)


def _aabb_bottom_to_xy_plane_transform(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    min_z = float(bounds[0][2])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.0, 0.0, -min_z]
    return transform


def _scale_transform(scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= float(scale)
    return transform


def _compose_sam3d_multi_object_transform(
    *,
    rotation_quaternion_wxyz: list[float],
    translation: list[float],
    scale: list[float],
) -> np.ndarray:
    """Compose the transform equivalent to the old baked multi-object export."""
    rotation = _quaternion_wxyz_to_matrix(rotation_quaternion_wxyz)
    scale_matrix = np.diag(_validate_vector(scale, expected_len=3, name="scale"))
    linear_row = _glb_to_sam3d_local_matrix() @ scale_matrix @ rotation
    return _row_linear_to_trimesh_matrix(
        linear_row=linear_row,
        translation=translation,
    )


def _row_linear_to_trimesh_matrix(
    *,
    linear_row: np.ndarray,
    translation: list[float],
) -> np.ndarray:
    """Convert a row-vector linear transform to trimesh's 4x4 matrix format."""
    translation_vector = _validate_vector(
        translation,
        expected_len=3,
        name="translation",
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear_row.T
    transform[:3, 3] = translation_vector
    return transform


def _validate_vector(
    values: list[float],
    *,
    expected_len: int,
    name: str,
) -> np.ndarray:
    """Validate and convert a numeric vector."""
    if len(values) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values")
    return np.asarray(values, dtype=np.float64)


def _glb_to_sam3d_local_matrix() -> np.ndarray:
    """Return the basis conversion used by the old baked multi-object exporter."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _quaternion_wxyz_to_matrix(quaternion: list[float]) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    if len(quaternion) != 4:
        raise ValueError("rotation_quaternion_wxyz must have 4 values")
    w, x, y, z = [float(v) for v in quaternion]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("rotation quaternion must be non-zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _detect_table_fit_support_quad(
    mesh: Any,
    *,
    target_aspect: float,
) -> dict[str, Any]:
    geom = GeometryManager()
    detect = geom.detect_tabletop(DetectTabletopRequest(mesh=mesh))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    support_vertices = vertices[
        np.unique(faces[np.asarray(detect.selected.face_indices, dtype=np.int64)])
    ]
    hull_xy = _table_fit_convex_hull_2d(support_vertices[:, :2])
    quad = _largest_centered_table_fit_inscribed_rect(
        hull_xy,
        target_aspect=max(float(target_aspect), 1.0e-6),
    )
    center_z = float(np.mean(support_vertices[:, 2]))
    return {
        "method": "sampled_centered_inscribed_rectangle_on_support_convex_hull",
        "normal": detect.oriented_normal,
        "area": float(detect.selected.area),
        "center": [quad["center_xy"][0], quad["center_xy"][1], center_z],
        "center_xy": quad["center_xy"],
        "size_xy": quad["size_xy"],
        "yaw_radians": quad["yaw_radians"],
        "yaw_degrees": float(np.rad2deg(quad["yaw_radians"])),
        "corners_xy": quad["corners_xy"],
        "support_hull_xy": hull_xy.tolist(),
    }


def _largest_centered_table_fit_inscribed_rect(
    hull_xy: np.ndarray,
    *,
    target_aspect: float,
    yaw_samples: int = 180,
) -> dict[str, Any]:
    if hull_xy.shape[0] < 3:
        raise ValueError("Support hull must contain at least 3 points.")
    best: dict[str, Any] | None = None
    centers = [
        np.mean(hull_xy, axis=0),
        0.5 * (np.min(hull_xy, axis=0) + np.max(hull_xy, axis=0)),
    ]
    for yaw in np.linspace(0.0, np.pi, yaw_samples, endpoint=False):
        rot = _table_fit_rot2(-yaw)
        inv_rot = _table_fit_rot2(yaw)
        rotated_hull = hull_xy @ rot.T
        for center_world in centers:
            center = center_world @ rot.T
            lo = 0.0
            bbox_size = np.max(rotated_hull, axis=0) - np.min(rotated_hull, axis=0)
            hi = float(max(bbox_size[0] / target_aspect, bbox_size[1], 1.0e-6))
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                width = target_aspect * mid
                depth = mid
                corners = _table_fit_rect_corners(
                    center=center,
                    width=width,
                    depth=depth,
                )
                corners_world = corners @ inv_rot.T
                if all(
                    _table_fit_point_in_convex_polygon(point, hull_xy)
                    for point in corners_world
                ):
                    lo = mid
                else:
                    hi = mid
            width = target_aspect * lo
            depth = lo
            area = width * depth
            corners_world = (
                _table_fit_rect_corners(center=center, width=width, depth=depth)
                @ inv_rot.T
            )
            candidate = {
                "center_xy": center_world.tolist(),
                "size_xy": [float(width), float(depth)],
                "yaw_radians": float(yaw),
                "corners_xy": corners_world.tolist(),
                "area": float(area),
            }
            if best is None or area > float(best["area"]):
                best = candidate
    if best is None:
        raise ValueError("Failed to estimate an inscribed support rectangle.")
    return best


def _load_table_fit_scene_internal_z(
    path: Path,
    *,
    trimesh: Any,
    y_to_z: np.ndarray,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"GLB not found: {path}")
    scene = trimesh.load(path, force="scene")
    scene.apply_transform(y_to_z)
    return scene


def _table_fit_scene_union_bounds(scenes: list[Any], *, trimesh: Any) -> np.ndarray:
    bounds = [
        np.asarray(_scene_to_mesh(scene, trimesh=trimesh).bounds, dtype=np.float64)
        for scene in scenes
    ]
    return np.vstack(
        [
            np.vstack([item[0] for item in bounds]).min(axis=0),
            np.vstack([item[1] for item in bounds]).max(axis=0),
        ]
    )


def _table_fit_bounds_xy_manifest(
    bounds: np.ndarray,
    *,
    unit_scale: float,
) -> dict[str, Any]:
    min_xy = bounds[0, :2] * unit_scale
    max_xy = bounds[1, :2] * unit_scale
    size_xy = max_xy - min_xy
    center_xy = 0.5 * (min_xy + max_xy)
    return {
        "unit": "cm",
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "center_xy": center_xy.tolist(),
        "size_xy": size_xy.tolist(),
        "area": float(size_xy[0] * size_xy[1]),
    }


def _table_fit_uniform_xy_scale_transform(
    *,
    center_xy: np.ndarray,
    scale: float,
) -> np.ndarray:
    center = np.eye(4, dtype=np.float64)
    center[:3, 3] = [float(center_xy[0]), float(center_xy[1]), 0.0]
    uncenter = np.eye(4, dtype=np.float64)
    uncenter[:3, 3] = [-float(center_xy[0]), -float(center_xy[1]), 0.0]
    scale_mat = np.eye(4, dtype=np.float64)
    scale_mat[0, 0] = float(scale)
    scale_mat[1, 1] = float(scale)
    return center @ scale_mat @ uncenter


def _table_fit_uniform_scale_transform(
    *,
    center_xy: np.ndarray,
    scale: float,
) -> np.ndarray:
    center = np.eye(4, dtype=np.float64)
    center[:3, 3] = [float(center_xy[0]), float(center_xy[1]), 0.0]
    uncenter = np.eye(4, dtype=np.float64)
    uncenter[:3, 3] = [-float(center_xy[0]), -float(center_xy[1]), 0.0]
    scale_mat = np.eye(4, dtype=np.float64)
    scale_mat[:3, :3] *= float(scale)
    return center @ scale_mat @ uncenter


def _table_fit_safe_positive_ratio(numerator: float, denominator: float) -> float:
    return max(float(numerator) / max(float(denominator), 1.0e-6), 1.0e-6)


def _table_fit_convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points)[:, :2]})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(
        o: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _table_fit_point_in_convex_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
) -> bool:
    previous = 0.0
    for index in range(len(polygon)):
        a = polygon[index]
        b = polygon[(index + 1) % len(polygon)]
        cross = float(np.cross(b - a, point - a))
        if abs(cross) < 1.0e-9:
            continue
        if previous == 0.0:
            previous = cross
        elif cross * previous < -1.0e-9:
            return False
    return True


def _table_fit_rect_corners(
    *,
    center: np.ndarray,
    width: float,
    depth: float,
) -> np.ndarray:
    half_w = 0.5 * float(width)
    half_d = 0.5 * float(depth)
    return np.asarray(
        [
            [center[0] - half_w, center[1] - half_d],
            [center[0] + half_w, center[1] - half_d],
            [center[0] + half_w, center[1] + half_d],
            [center[0] - half_w, center[1] + half_d],
        ],
        dtype=np.float64,
    )


def _table_fit_rot2(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)
