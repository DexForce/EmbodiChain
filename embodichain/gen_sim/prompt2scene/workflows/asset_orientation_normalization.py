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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np

__all__ = [
    "DEFAULT_ORIENTATION_NORMALIZATION_KEYWORDS",
    "AssetOrientationNormalizationResult",
    "asset_orientation_is_upper_larger",
    "export_z_axis_normalized_asset",
    "match_asset_orientation_keyword",
]

DEFAULT_ORIENTATION_NORMALIZATION_KEYWORDS: tuple[str, ...] = (
    "bottle",
    "can",
    "canned food",
    "tin can",
    "food can",
    "soda can",
    "paper cup",
    "disposable cup",
    "coffee cup",
    "cup",
    "瓶",
    "瓶子",
    "罐头",
    "易拉罐",
    "纸杯",
    "杯子",
)
_UPPER_LARGER_ORIENTATION_KEYWORDS = frozenset(
    {
        "paper cup",
        "disposable cup",
        "coffee cup",
        "cup",
        "纸杯",
        "杯子",
    }
)
_DEFAULT_SURFACE_SAMPLE_COUNT = 10000


@dataclass(frozen=True)
class AssetOrientationNormalizationResult:
    """Pose needed to restore a normalized asset to its original scene pose."""

    init_pos: list[float]
    init_rot: list[float]
    rotation_matrix: list[list[float]]
    normalization_transform: list[list[float]]


def match_asset_orientation_keyword(
    *,
    object_id: str,
    name: str = "",
    description: str = "",
    keywords: Sequence[str] = DEFAULT_ORIENTATION_NORMALIZATION_KEYWORDS,
) -> str | None:
    """Return the matched hardcoded object keyword, if orientation should normalize."""
    text = " ".join([object_id, name, description]).lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    compact_text = " ".join(re.findall(r"[a-z0-9]+", text))
    for raw_keyword in keywords:
        keyword = str(raw_keyword or "").strip().lower()
        if not keyword:
            continue
        keyword_tokens = re.findall(r"[a-z0-9]+", keyword)
        if not keyword_tokens:
            if keyword in text:
                return keyword
            continue
        if len(keyword_tokens) == 1:
            if keyword_tokens[0] in tokens:
                return keyword
        elif " ".join(keyword_tokens) in compact_text:
            return keyword
    return None


def asset_orientation_is_upper_larger(keyword: str | None) -> bool:
    """Return whether a matched asset has a larger top than bottom."""
    if keyword is None:
        return False
    return str(keyword).strip().lower() in _UPPER_LARGER_ORIENTATION_KEYWORDS


def export_z_axis_normalized_asset(
    source_path: Path,
    output_path: Path,
    *,
    glb_to_sim_rotation: np.ndarray,
    scale_factor: float = 1.0,
    is_upper_larger: bool = False,
    surface_sample_count: int = _DEFAULT_SURFACE_SAMPLE_COUNT,
) -> AssetOrientationNormalizationResult:
    """Export an asset upright along imported Z while preserving its scene pose.

    ``source_path`` is a placed GLB whose scene pose is already baked into the
    vertices. The exported GLB is reoriented so the dominant object axis is
    Z-up after the GLB Y-up importer conversion, and its imported AABB bottom
    center is at the origin. The returned pose is the inverse transform needed
    by the gym config so the loaded scene still appears exactly like the
    original placed mesh.
    """
    mesh = _load_mesh_as_trimesh(source_path)
    verts_glb = np.asarray(mesh.vertices, dtype=np.float64)
    if verts_glb.ndim != 2 or verts_glb.shape[1] != 3 or verts_glb.shape[0] == 0:
        raise ValueError(f"GLB contains no valid vertices: {source_path}")

    basis = _validate_basis(glb_to_sim_rotation)
    scale = _safe_positive_scale(scale_factor)
    verts_sim = (basis @ verts_glb.T).T * scale

    sample_points = _sample_surface_points(
        mesh,
        basis=basis,
        scale=scale,
        sample_count=surface_sample_count,
    )
    normalization_transform = _z_axis_normalization_transform(
        sample_points,
        is_upper_larger=is_upper_larger,
    )

    normalized_sim = _transform_points(verts_sim, normalization_transform)
    bottom_center = _bottom_center(normalized_sim)
    asset_sim = normalized_sim - bottom_center.reshape(1, 3)

    baked = mesh.copy()
    baked.vertices = (basis.T @ asset_sim.T).T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baked.export(output_path)

    asset_to_scene = np.linalg.inv(normalization_transform) @ _translation_matrix(
        bottom_center
    )
    rotation_matrix = asset_to_scene[:3, :3]
    return AssetOrientationNormalizationResult(
        init_pos=_clean_float_list(asset_to_scene[:3, 3]),
        init_rot=_rotation_matrix_to_euler_degrees(rotation_matrix),
        rotation_matrix=_clean_matrix(rotation_matrix),
        normalization_transform=_clean_matrix(normalization_transform),
    )


def _load_mesh_as_trimesh(glb_path: Path) -> Any:
    import trimesh

    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return scene
    dumped = scene.dump(concatenate=True)
    if isinstance(dumped, trimesh.Trimesh):
        return dumped
    meshes = [item for item in dumped if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise ValueError(f"GLB contains no mesh geometry: {glb_path}")
    return trimesh.util.concatenate(meshes)


def _validate_basis(value: np.ndarray) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("glb_to_sim_rotation must be a finite 3x3 matrix")
    return basis


def _safe_positive_scale(scale_factor: float) -> float:
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _sample_surface_points(
    mesh: Any,
    *,
    basis: np.ndarray,
    scale: float,
    sample_count: int,
) -> np.ndarray:
    points = None
    if getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0:
        try:
            import trimesh

            points, _ = trimesh.sample.sample_surface(
                mesh,
                max(int(sample_count), 4),
            )
        except Exception:
            points = None
    if points is None or len(points) < 4:
        points = np.asarray(mesh.vertices, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("at least 4 valid mesh points are required")
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 4:
        raise ValueError("at least 4 finite mesh points are required")
    return (basis @ points.T).T * scale


def _z_axis_normalization_transform(
    points: np.ndarray,
    *,
    is_upper_larger: bool,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center.reshape(1, 3)

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if np.linalg.det(vt) < 0.0:
        vt[2, :] = -vt[2, :]

    align_long_axis_to_z = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotation = align_long_axis_to_z @ vt

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ center)

    standardized = _transform_points(points, transform)
    upper_volume, lower_volume = _end_cap_volumes(standardized)
    if is_upper_larger:
        flip = upper_volume < lower_volume
    else:
        flip = upper_volume > lower_volume
    if flip:
        upside_down = np.eye(4, dtype=np.float64)
        upside_down[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        transform = upside_down @ transform
    return transform


def _end_cap_volumes(points: np.ndarray) -> tuple[float, float]:
    axis_min = float(points[:, 2].min())
    axis_max = float(points[:, 2].max())
    upper_th = axis_min + (axis_max - axis_min) * 0.8
    lower_th = axis_min + (axis_max - axis_min) * 0.2
    upper_part = points[points[:, 2] > upper_th]
    lower_part = points[points[:, 2] < lower_th]
    return _convex_hull_volume(upper_part), _convex_hull_volume(lower_part)


def _convex_hull_volume(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        return 0.0
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 4:
        return 0.0
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return 0.0

    try:
        return float(ConvexHull(points).volume)
    except (QhullError, ValueError):
        return 0.0


def _bottom_center(points: np.ndarray) -> np.ndarray:
    bounds = np.vstack((points.min(axis=0), points.max(axis=0)))
    return np.array(
        [
            0.5 * (bounds[0, 0] + bounds[1, 0]),
            0.5 * (bounds[0, 1] + bounds[1, 1]),
            bounds[0, 2],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3].reshape(1, 3)


def _translation_matrix(offset: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(offset, dtype=np.float64)
    return transform


def _rotation_matrix_to_euler_degrees(rotation_matrix: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation as R

    radians = R.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_euler(
        "XYZ",
        degrees=False,
    )
    return _clean_float_list(np.rad2deg(np.asarray(radians, dtype=np.float64)))


def _clean_matrix(matrix: np.ndarray) -> list[list[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    return [_clean_float_list(row) for row in arr.tolist()]


def _clean_float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        number = float(value)
        if abs(number) < 1.0e-12:
            number = 0.0
        cleaned.append(number)
    return cleaned
