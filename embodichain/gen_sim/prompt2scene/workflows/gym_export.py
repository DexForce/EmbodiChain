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
import math
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.workflows.paths import (
    UNIFIED_SCENE_GEN_STEP,
    PipelinePaths,
)

__all__ = ["export_gym_config"]

_DEFAULT_OBJECT_ATTRS: dict[str, Any] = {
    "mass": 0.01,
    "contact_offset": 0.003,
    "rest_offset": 0.001,
    "restitution": 0.01,
    "max_depenetration_velocity": 10.0,
    "min_position_iters": 32,
    "min_velocity_iters": 8,
}

_DEFAULT_TABLE_ATTRS: dict[str, Any] = {
    "mass": 10.0,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "restitution": 0.01,
}

_DEFAULT_OBJECT_MAX_CONVEX_HULL_NUM = 16
_DEFAULT_TABLE_MAX_CONVEX_HULL_NUM = 16
_DEFAULT_CONVEX_DECOMPOSITION_METHOD = "vhacd"


@dataclass(frozen=True)
class _UprightFrameStandardizationProfile:
    name: str
    is_upper_larger: bool


_BOTTLE_FRAME_STANDARDIZATION = _UprightFrameStandardizationProfile(
    name="bottle",
    is_upper_larger=False,
)
_CUP_FRAME_STANDARDIZATION = _UprightFrameStandardizationProfile(
    name="cup",
    is_upper_larger=True,
)
_CAN_FRAME_STANDARDIZATION = _UprightFrameStandardizationProfile(
    name="can",
    is_upper_larger=False,
)
_FRAME_STANDARDIZATION_KEYWORDS = (
    (_CUP_FRAME_STANDARDIZATION, ("cup", "cups", "paper_cup", "纸杯", "杯子")),
    (
        _BOTTLE_FRAME_STANDARDIZATION,
        ("bottle", "bottles", "water_bottle", "soda_bottle", "瓶子", "瓶"),
    ),
    (
        _CAN_FRAME_STANDARDIZATION,
        ("can", "cans", "soda_can", "tin_can", "罐头", "易拉罐", "罐"),
    ),
)


def _resolve_path(value: str, output_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_dir():
        raise IsADirectoryError(f"Expected JSON file but got directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _resolve_table_fit_manifest_path(
    *,
    manifest_path_value: Any,
    output_root: Path,
    paths: PipelinePaths,
) -> Path:
    if not manifest_path_value:
        raise FileNotFoundError(
            "table_fit_to_clutter manifest_path is missing or empty"
        )

    resolved = _resolve_path(str(manifest_path_value), output_root)
    if resolved.is_file():
        return resolved

    default_manifest = paths.table_fit_manifest
    if default_manifest.is_file():
        return default_manifest

    if resolved.is_dir():
        raise IsADirectoryError(
            "table_fit_to_clutter manifest_path points to a directory, not a JSON "
            f"file: value={manifest_path_value!r} resolved={resolved}"
        )
    raise FileNotFoundError(
        "table_fit_to_clutter manifest_path does not point to a JSON file: "
        f"value={manifest_path_value!r} resolved={resolved}"
    )


def _glb_to_sim_rotation() -> np.ndarray:
    """Return the loader basis conversion from GLB Y-up to sim Z-up."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _glb_scale_to_sim(scale: Sequence[float]) -> list[float]:
    """Convert GLB-axis scale components to sim-axis body_scale components."""
    values = [float(v) for v in scale]
    if len(values) != 3:
        raise ValueError("scale must have three components")
    return [values[0], values[2], values[1]]


def _sim_world_xy_aabb(
    glb_path: Path,
    rotation_matrix: list[list[float]] | None,
    scale: float | Sequence[float],
    init_pos: Sequence[float],
) -> dict[str, Any]:
    """Project a transformed simready GLB onto the simulation XY plane."""
    import trimesh

    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        dumped = scene.dump(concatenate=True)
        mesh = (
            dumped
            if isinstance(dumped, trimesh.Trimesh)
            else trimesh.util.concatenate(
                [m for m in dumped if isinstance(m, trimesh.Trimesh)]
            )
        )
    verts = np.asarray(mesh.vertices.copy(), dtype=np.float64)
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)):
        scale_array = np.asarray(list(scale), dtype=np.float64)
        if scale_array.shape != (3,):
            raise ValueError("scale must be a scalar or a 3-vector")
        verts *= scale_array
    else:
        verts *= float(scale)
    if rotation_matrix is not None:
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        if rot.shape == (4, 4):
            rot = rot[:3, :3]
        verts = (rot @ verts.T).T

    init = np.asarray(list(init_pos), dtype=np.float64)
    if init.shape != (3,):
        raise ValueError("init_pos must have three components")
    sim_xy = np.column_stack((verts[:, 0] + init[0], -verts[:, 2] + init[1]))
    min_xy = sim_xy.min(axis=0)
    max_xy = sim_xy.max(axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    size_xy = np.maximum(max_xy - min_xy, 0.0)
    return {
        "unit": "m",
        "center_xy": center_xy.tolist(),
        "aabb_xy": [min_xy.tolist(), max_xy.tolist()],
        "size_xy": size_xy.tolist(),
    }


def _support_region_2d(table_fit_manifest: dict[str, Any]) -> dict[str, Any]:
    support = table_fit_manifest.get("final_support_quad_centered") or {}
    support_hull = np.asarray(support.get("support_hull_xy", []), dtype=np.float64)
    corners = np.asarray(support.get("corners_xy", []), dtype=np.float64)
    if (
        support_hull.ndim == 2
        and support_hull.shape[0] >= 3
        and support_hull.shape[1] == 2
    ):
        points = support_hull
        source = "support_hull_aabb"
    elif corners.shape == (4, 2):
        points = corners
        source = "support_quad_aabb"
    else:
        return {
            "unit": "m",
            "center_xy": [],
            "aabb_xy": [],
            "size_xy": [],
            "corners_xy": [],
        }
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    size_xy = max_xy - min_xy
    aabb_corners = np.asarray(
        [
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ],
        dtype=np.float64,
    )
    return {
        "unit": "m",
        "source": source,
        "center_xy": center_xy.tolist(),
        "aabb_xy": [min_xy.tolist(), max_xy.tolist()],
        "size_xy": size_xy.tolist(),
        "corners_xy": aabb_corners.tolist(),
        "support_hull_xy": (
            support_hull.tolist()
            if support_hull.ndim == 2 and support_hull.shape[1] == 2
            else []
        ),
    }


def _write_scene_state(
    *,
    export_dir: Path,
    config_path: Path,
    table_desc: str,
    table_support_region_2d: dict[str, Any],
    object_states: list[dict[str, Any]],
    source_snapshots: dict[str, str],
) -> Path:
    scene_state_dir = export_dir / "scene_state"
    scene_state_dir.mkdir(parents=True, exist_ok=True)
    plot_path = scene_state_dir / "topdown_2d.png"
    state_path = scene_state_dir / "result.json"
    state = {
        "version": 1,
        "coordinate_frame": {
            "unit": "m",
            "plane": "simulation_xy",
            "x_axis": "simulation +X",
            "y_axis": "simulation +Y",
            "note": "2D values are top-down projections onto the simulation XY plane.",
        },
        "gym_config_path": str(config_path.relative_to(export_dir)),
        "topdown_2d_plot_path": str(plot_path.relative_to(export_dir)),
        "source_snapshots": source_snapshots,
        "table": {
            "id": "table",
            "role": "background",
            "description": table_desc,
            "support_region_2d": table_support_region_2d,
        },
        "objects": object_states,
    }
    state_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _render_scene_state_topdown(
        support_region=table_support_region_2d,
        objects=object_states,
        output_path=plot_path,
    )
    return state_path


def _copy_scene_source_snapshots(
    *,
    paths: PipelinePaths,
    export_dir: Path,
    scene_state_dir: Path,
) -> dict[str, str]:
    scene_state_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, str] = {}
    sources = {
        "unified_scene": paths.unified_scene_result,
        "unified_scene_gen": paths.step_result(UNIFIED_SCENE_GEN_STEP),
    }
    for name, source in sources.items():
        if not source.is_file():
            continue
        destination = scene_state_dir / f"{name}.json"
        shutil.copy2(source, destination)
        snapshots[name] = str(destination.relative_to(export_dir))
    return snapshots


def _render_scene_state_topdown(
    *,
    support_region: dict[str, Any],
    objects: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))

    data_points: list[np.ndarray] = []
    corners = np.asarray(support_region.get("corners_xy", []), dtype=np.float64)
    if corners.shape == (4, 2):
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=(0.18, 0.62, 0.32, 0.14),
                edgecolor=(0.05, 0.38, 0.16, 1.0),
                linewidth=2.0,
                label="table support region",
            )
        )
        data_points.append(corners)

    for obj in objects:
        footprint = obj.get("footprint_2d") or {}
        aabb = np.asarray(footprint.get("aabb_xy", []), dtype=np.float64)
        center = np.asarray(footprint.get("center_xy", []), dtype=np.float64)
        if aabb.shape != (2, 2) or center.shape != (2,):
            continue
        size = np.maximum(aabb[1] - aabb[0], 0.0)
        ax.add_patch(
            Rectangle(
                aabb[0],
                size[0],
                size[1],
                facecolor=(0.25, 0.48, 0.95, 0.22),
                edgecolor=(0.08, 0.20, 0.65, 1.0),
                linewidth=1.5,
            )
        )
        ax.plot(center[0], center[1], "o", color="#102a7a", markersize=4)
        label = str(obj.get("id", "")).replace("interact_", "")
        ax.text(
            center[0],
            center[1],
            f"{label}\n({center[0]:.3f}, {center[1]:.3f})",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )
        data_points.append(aabb)

    if data_points:
        all_points = np.vstack(data_points)
        data_min = all_points.min(axis=0)
        data_max = all_points.max(axis=0)
    else:
        data_min = np.array([-0.5, -0.5], dtype=np.float64)
        data_max = np.array([0.5, 0.5], dtype=np.float64)
    span = np.maximum(data_max - data_min, 1.0e-3)
    padding = max(float(span.max()) * 0.18, 0.05)
    x_limits = (float(data_min[0] - padding), float(data_max[0] + padding))
    y_limits = (float(data_min[1] - padding), float(data_max[1] + padding))

    ax.axhline(0.0, color="#303030", linewidth=1.2, alpha=0.75)
    ax.axvline(0.0, color="#303030", linewidth=1.2, alpha=0.75)
    ax.annotate(
        "+X",
        xy=(x_limits[1], 0.0),
        xytext=(x_limits[1] - 0.08 * (x_limits[1] - x_limits[0]), 0.02),
        arrowprops={"arrowstyle": "->", "color": "#303030", "lw": 1.4},
        color="#303030",
    )
    ax.annotate(
        "+Y",
        xy=(0.0, y_limits[1]),
        xytext=(0.02, y_limits[1] - 0.08 * (y_limits[1] - y_limits[0])),
        arrowprops={"arrowstyle": "->", "color": "#303030", "lw": 1.4},
        color="#303030",
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Prompt2Scene Gym Export Top-Down 2D State")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


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


def _upright_frame_standardization_for_object(
    *,
    object_id: str,
    description: str,
    mesh_path: Path,
) -> _UprightFrameStandardizationProfile | None:
    text = " ".join(
        [
            object_id,
            description,
            mesh_path.stem,
            *(part for part in mesh_path.parts[-5:-1]),
        ]
    )
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", text).lower()
    tokens = set(normalized.split())
    compact = normalized.replace(" ", "")
    for profile, keywords in _FRAME_STANDARDIZATION_KEYWORDS:
        for keyword in keywords:
            key = keyword.lower()
            if any("\u4e00" <= char <= "\u9fff" for char in key):
                if key in compact:
                    return profile
                continue
            if "_" in key:
                if key.replace("_", "") in compact:
                    return profile
                continue
            if key in tokens:
                return profile
    return None


def _bake_glb_bottom_center_to_origin(
    source_path: Path,
    output_path: Path,
    *,
    scale_factor: float = 1.0,
    upright_frame_standardization: _UprightFrameStandardizationProfile | None = None,
) -> dict[str, Any] | None:
    mesh = _load_mesh_as_trimesh(source_path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    basis = _glb_to_sim_rotation()
    sample_points = _sample_mesh_points(mesh)
    verts_sim = (basis @ verts.T).T
    verts_sim *= scale
    sample_points_sim = (basis @ sample_points.T).T
    sample_points_sim *= scale
    standardization_report = None
    if upright_frame_standardization is not None:
        # Work in sim coordinates so the object mouth/opening maps to runtime +Z.
        try:
            verts_sim, standardization_report = _standardize_upright_sim_vertices(
                vertices=verts_sim,
                sample_points=sample_points_sim,
                profile=upright_frame_standardization,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            standardization_report = {
                "profile": upright_frame_standardization.name,
                "axis": "sim_z",
                "status": "skipped",
                "reason": str(exc),
            }
    bounds = np.asarray(
        np.vstack((verts_sim.min(axis=0), verts_sim.max(axis=0))),
        dtype=np.float64,
    )
    bottom_center = np.array(
        [
            0.5 * (bounds[0, 0] + bounds[1, 0]),
            0.5 * (bounds[0, 1] + bounds[1, 1]),
            bounds[0, 2],
        ],
        dtype=np.float64,
    )
    verts_sim -= bottom_center
    baked = mesh.copy()
    baked.vertices = (basis.T @ verts_sim.T).T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baked.export(output_path)
    return standardization_report


def _sample_mesh_points(mesh: Any) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = getattr(mesh, "faces", None)
    if faces is None or len(faces) == 0:
        return vertices
    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        return vertices
    try:
        triangles = vertices[faces_array]
    except IndexError:
        return vertices
    centers = triangles.mean(axis=1)
    edge_midpoints = np.vstack(
        (
            (triangles[:, 0, :] + triangles[:, 1, :]) * 0.5,
            (triangles[:, 1, :] + triangles[:, 2, :]) * 0.5,
            (triangles[:, 2, :] + triangles[:, 0, :]) * 0.5,
        )
    )
    near_vertices = np.vstack(
        (
            triangles[:, 0, :] * 0.9
            + triangles[:, 1, :] * 0.05
            + triangles[:, 2, :] * 0.05,
            triangles[:, 1, :] * 0.9
            + triangles[:, 0, :] * 0.05
            + triangles[:, 2, :] * 0.05,
            triangles[:, 2, :] * 0.9
            + triangles[:, 0, :] * 0.05
            + triangles[:, 1, :] * 0.05,
        )
    )
    points = np.vstack((vertices, centers, edge_midpoints, near_vertices))
    return points if len(points) >= 4 else vertices


def _standardize_upright_sim_vertices(
    *,
    vertices: np.ndarray,
    sample_points: np.ndarray,
    profile: _UprightFrameStandardizationProfile,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_points = np.asarray(sample_points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    if sample_points.ndim != 2 or sample_points.shape[1] != 3:
        raise ValueError(
            f"sample_points must have shape (N, 3), got {sample_points.shape}"
        )
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if len(sample_points) < 4 or len(vertices) < 4:
        raise ValueError("at least 4 points are required")
    if not np.all(np.isfinite(sample_points)) or not np.all(np.isfinite(vertices)):
        raise ValueError("mesh points must be finite")

    center = sample_points.mean(axis=0)
    centered_sample = sample_points - center
    if np.linalg.matrix_rank(centered_sample) < 3:
        raise ValueError("mesh points do not span a 3D volume")
    _, _, vt = np.linalg.svd(centered_sample, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2, :] = -vt[2, :]

    rotation = _rotation_y_matrix3(90.0) @ vt
    standardized_sample = centered_sample @ rotation.T
    standardized_vertices = (vertices - center) @ rotation.T

    upper_volume, lower_volume = _upper_lower_convex_hull_volumes(standardized_sample)
    flip = (
        upper_volume < lower_volume
        if profile.is_upper_larger
        else upper_volume > lower_volume
    )
    if flip:
        flip_rotation = _rotation_x_matrix3(180.0)
        standardized_sample = standardized_sample @ flip_rotation.T
        standardized_vertices = standardized_vertices @ flip_rotation.T
        upper_volume, lower_volume = lower_volume, upper_volume

    return standardized_vertices, {
        "profile": profile.name,
        "axis": "sim_z",
        "status": "applied",
        "is_upper_larger": profile.is_upper_larger,
        "flip_applied": bool(flip),
        "upper_volume": float(upper_volume),
        "lower_volume": float(lower_volume),
    }


def _upper_lower_convex_hull_volumes(points: np.ndarray) -> tuple[float, float]:
    axis_values = points[:, 2]
    axis_min = float(axis_values.min())
    axis_max = float(axis_values.max())
    axis_span = axis_max - axis_min
    if axis_span <= 1e-9:
        raise ValueError("standardized mesh has no height along sim Z")
    upper_th = axis_min + axis_span * 0.8
    lower_th = axis_min + axis_span * 0.2
    upper = points[axis_values > upper_th]
    lower = points[axis_values < lower_th]
    return _convex_hull_volume(upper), _convex_hull_volume(lower)


def _convex_hull_volume(points: np.ndarray) -> float:
    if len(points) < 4:
        raise ValueError("at least 4 points are required for convex hull volume")
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return _aabb_volume(points)
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return _aabb_volume(points)


def _aabb_volume(points: np.ndarray) -> float:
    extents = np.ptp(points, axis=0)
    return float(max(extents[0], 0.0) * max(extents[1], 0.0) * max(extents[2], 0.0))


def _rotation_x_matrix3(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_value, -sin_value],
            [0.0, sin_value, cos_value],
        ],
        dtype=np.float64,
    )


def _rotation_y_matrix3(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    return np.array(
        [
            [cos_value, 0.0, sin_value],
            [0.0, 1.0, 0.0],
            [-sin_value, 0.0, cos_value],
        ],
        dtype=np.float64,
    )


def _ensure_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not written: {path}")


def _resolve_existing_table_fit_path(
    value: Any,
    *,
    output_root: Path,
    paths: PipelinePaths,
) -> Path:
    resolved = _resolve_path(str(value or ""), output_root)
    if resolved.is_file():
        return resolved

    fallback_root = paths.table_fit_dir
    fallback = fallback_root / resolved.name
    if fallback.is_file():
        return fallback

    if resolved.parent.name == "table_fit_to_clutter":
        alt = (
            output_root
            / "unified_scene_gen"
            / "glb_gen"
            / "table_fit_to_clutter"
            / resolved.name
        )
        if alt.is_file():
            return alt

    return resolved


def _build_object_manifest(
    output_root: Path,
    step_result: dict[str, Any],
    table_fit_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Merge table-fit object paths with their descriptions."""
    objects_info = step_result.get("objects") or []

    # index table-fit object records by object id
    table_fit_by_id: dict[str, dict[str, Any]] = {}
    for e in table_fit_manifest.get("objects") or []:
        eid = str(e.get("id", "")) if isinstance(e, dict) else ""
        if eid and isinstance(e, dict):
            table_fit_by_id[eid] = e

    # index world_aabb_bottom_center from table-fit manifest
    world_bc_by_id: dict[str, list[float]] = {}
    for eid, e in table_fit_by_id.items():
        wbc = e.get("world_aabb_bottom_center") if isinstance(e, dict) else None
        if eid and isinstance(wbc, list) and len(wbc) == 3:
            world_bc_by_id[eid] = [float(v) for v in wbc]

    consolidated: dict[str, Any] = {}
    skipped_no_glb: list[str] = []
    for obj in objects_info:
        oid = str(obj.get("id", ""))
        if not oid:
            continue

        table_fit_item = table_fit_by_id.get(oid)
        if table_fit_item is None:
            skipped_no_glb.append(oid)
            continue

        description = str(obj.get("description") or obj.get("name") or "").strip()
        source = table_fit_item.get("path") or ""
        table_fit_path = _resolve_existing_table_fit_path(
            source,
            output_root=output_root,
            paths=PipelinePaths(output_root),
        )
        if not table_fit_path.is_file():
            skipped_no_glb.append(oid)
            continue

        wbc = world_bc_by_id.get(oid)

        consolidated[oid] = {
            "id": oid,
            "description": description,
            "table_fit_path": table_fit_path,
            "world_aabb_bottom_center": wbc,
        }

    if skipped_no_glb:
        print(
            "  [WARN] object(s) skipped (table-fit GLB not found): "
            + ", ".join(skipped_no_glb)
        )
    extra_in_manifest = set(world_bc_by_id) - set(consolidated)
    if extra_in_manifest:
        print(
            "  [WARN] object(s) in table-fit manifest but not in step_result: "
            + ", ".join(sorted(extra_in_manifest))
        )

    return consolidated


def export_gym_config(
    output_root: Path,
    *,
    export_dir: Path | None = None,
) -> Path:
    """Export the unified-scene-gen result as a gym_config.json bundle.

    Uses table-fit GLBs baked so their sim-space AABB bottom-centre sits at
    the local origin. Object pose is then restored with ``init_pos`` only.
    """
    output_root = output_root.expanduser().resolve()
    if export_dir is None:
        export_dir = output_root / "gym_export"
    else:
        export_dir = export_dir.expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    paths = PipelinePaths(output_root)

    step_result = _read_json(paths.step_result(UNIFIED_SCENE_GEN_STEP))
    table_fit = step_result.get("table_fit_to_clutter") or {}
    if table_fit.get("status") != "ok":
        raise RuntimeError(
            "Cannot export gym_config because table_fit_to_clutter did not "
            f"succeed: status={table_fit.get('status')!r} "
            f"reason={table_fit.get('reason', '')}"
        )
    manifest_path_value = table_fit.get("manifest_path") or ""
    table_fit_manifest = _read_json(
        _resolve_table_fit_manifest_path(
            manifest_path_value=manifest_path_value,
            output_root=output_root,
            paths=paths,
        )
    )

    object_manifest = _build_object_manifest(
        output_root, step_result, table_fit_manifest
    )

    table_info = step_result.get("table") or {}
    table_desc = str(
        table_info.get("complete_table_description")
        or table_info.get("description", "")
    ).strip()

    mesh_assets_dir = export_dir / "mesh_assets"
    mesh_assets_dir.mkdir(parents=True, exist_ok=True)

    table_dst = mesh_assets_dir / "table" / "table_0.glb"
    table_dst.parent.mkdir(parents=True, exist_ok=True)
    table_fit_output = _resolve_existing_table_fit_path(
        table_fit_manifest.get("table_output_path", ""),
        output_root=output_root,
        paths=paths,
    )
    if not table_fit_output.is_file():
        raise FileNotFoundError(f"Table fit GLB not found: {table_fit_output}")
    _bake_glb_bottom_center_to_origin(table_fit_output, table_dst)
    table_init_pos = [0.0, 0.0, 0.0]
    table_init_rot = [0.0, 0.0, 0.0]
    table_body_scale = [1.0, 1.0, 1.0]

    rigid_objects: list[dict[str, Any]] = []
    object_states: list[dict[str, Any]] = []

    total = len(object_manifest)
    for idx, (oid, om) in enumerate(object_manifest.items()):
        safe_name = oid.replace("interact_", "").strip("_") or "object"
        obj_dir = mesh_assets_dir / safe_name / oid
        obj_dir.mkdir(parents=True, exist_ok=True)
        object_dst = obj_dir / f"{oid}.glb"
        frame_standardization = _upright_frame_standardization_for_object(
            object_id=oid,
            description=om["description"],
            mesh_path=om["table_fit_path"],
        )
        frame_standardization_report = _bake_glb_bottom_center_to_origin(
            om["table_fit_path"],
            object_dst,
            upright_frame_standardization=frame_standardization,
        )
        wbc = om["world_aabb_bottom_center"]
        if wbc is not None:
            init_pos = [float(wbc[0]), float(wbc[1]), float(wbc[2])]
        else:
            raise ValueError(f"Missing table-fit world_aabb_bottom_center for {oid}")
        init_rot = [0.0, 0.0, 0.0]
        body_scale = [1.0, 1.0, 1.0]

        rigid_objects.append(
            {
                "uid": oid,
                "description": om["description"],
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": str(object_dst.relative_to(export_dir)),
                    "compute_uv": False,
                },
                "attrs": dict(_DEFAULT_OBJECT_ATTRS),
                "body_type": "dynamic",
                "init_pos": init_pos,
                "init_rot": init_rot,
                "body_scale": body_scale,
                "max_convex_hull_num": _DEFAULT_OBJECT_MAX_CONVEX_HULL_NUM,
                "convex_decomposition_method": _DEFAULT_CONVEX_DECOMPOSITION_METHOD,
            }
        )
        footprint_2d = _sim_world_xy_aabb(
            object_dst,
            None,
            1.0,
            init_pos,
        )
        object_state = {
            "id": oid,
            "name": safe_name,
            "role": "interact",
            "description": om["description"],
            "init_pos": init_pos,
            "init_rot": init_rot,
            "body_scale": body_scale,
            "footprint_2d": footprint_2d,
        }
        if frame_standardization_report is not None:
            object_state["mesh_frame_standardization"] = frame_standardization_report
        object_states.append(object_state)
        wbc_flag = "wbc" if wbc is not None else "missing_wbc"
        frame_flag = (
            f"  frame={frame_standardization_report['profile']}"
            if frame_standardization_report is not None
            and frame_standardization_report.get("status") == "applied"
            else ""
        )
        print(
            f"  [{idx+1}/{total}] [{oid}] {om['description']}"
            f"  pos={init_pos}  rot={init_rot}  scale={body_scale}"
            f"  src={wbc_flag}{frame_flag}"
        )

    config = {
        "id": f"Prompt2Scene-{int(time.time() * 1000)}-v0",
        "max_episodes": 10,
        "max_episode_steps": 300,
        "env": {"events": {}, "observations": {}, "dataset": {}},
        "robot": {},
        "sensor": [],
        "light": {},
        "background": [
            {
                "uid": "table",
                "description": table_desc,
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": str(table_dst.relative_to(export_dir)),
                    "compute_uv": False,
                },
                "attrs": dict(_DEFAULT_TABLE_ATTRS),
                "body_scale": table_body_scale,
                "body_type": "kinematic",
                "init_pos": table_init_pos,
                "init_rot": table_init_rot,
                "max_convex_hull_num": _DEFAULT_TABLE_MAX_CONVEX_HULL_NUM,
            }
        ],
        "rigid_object": rigid_objects,
    }

    config_path = export_dir / "gym_config.json"
    config_path.write_text(
        json.dumps(config, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _ensure_file(config_path, label="gym_config.json")
    scene_state_dir = export_dir / "scene_state"
    source_snapshots = _copy_scene_source_snapshots(
        paths=paths,
        export_dir=export_dir,
        scene_state_dir=scene_state_dir,
    )
    scene_state_path = _write_scene_state(
        export_dir=export_dir,
        config_path=config_path,
        table_desc=table_desc,
        table_support_region_2d=_support_region_2d(table_fit_manifest),
        object_states=object_states,
        source_snapshots=source_snapshots,
    )
    _ensure_file(scene_state_path, label="scene_state/result.json")
    _ensure_file(scene_state_dir / "topdown_2d.png", label="scene_state/topdown_2d.png")
    print(f"  scene_state={scene_state_path.relative_to(export_dir)}")

    return config_path
