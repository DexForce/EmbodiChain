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
import shutil
import time
from collections.abc import Sequence
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

_DEFAULT_MAX_CONVEX_HULL_NUM = 32


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
        raise FileNotFoundError("table_fit_to_clutter manifest_path is missing or empty")

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


def _matrix_to_euler_xyz_deg(matrix: list[list[float]]) -> list[float]:
    """Decompose a 3×3 or 4×4 rotation matrix into XYZ Euler angles (degrees)."""
    m = np.asarray(matrix, dtype=np.float64)
    r = m[:3, :3]
    sy = math.sqrt(float(r[0, 0]) ** 2 + float(r[1, 0]) ** 2)
    if sy > 1e-6:
        x = math.atan2(float(r[2, 1]), float(r[2, 2]))
        y = math.atan2(-float(r[2, 0]), sy)
        z = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        x = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        y = math.atan2(-float(r[2, 0]), sy)
        z = 0.0
    return [math.degrees(x), math.degrees(y), math.degrees(z)]


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


def _glb_rotation_to_sim(rotation_matrix: list[list[float]]) -> list[list[float]]:
    """Convert a GLB-space local rotation into simulation-space rotation."""
    rot = np.asarray(rotation_matrix, dtype=np.float64)
    if rot.shape == (4, 4):
        rot = rot[:3, :3]
    basis = _glb_to_sim_rotation()
    return (basis @ rot @ basis.T).tolist()


def _glb_scale_to_sim(scale: Sequence[float]) -> list[float]:
    """Convert GLB-axis scale components to sim-axis body_scale components."""
    values = [float(v) for v in scale]
    if len(values) != 3:
        raise ValueError("scale must have three components")
    return [values[0], values[2], values[1]]


def _glb_max_z(glb_path: Path) -> float:
    """Maximum height (Y in GLB, Z in simulation) of a mesh."""
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
    return float(np.asarray(mesh.bounds, dtype=np.float64)[1, 1])  # max Y


def _rotated_aabb_offsets(
    glb_path: Path,
    rotation_matrix: list[list[float]] | None,
    scale: float | Sequence[float] = 1.0,
) -> tuple[float, float, float]:
    """Compute the AABB shift caused by rotation + scale alone.

    Loads the simready GLB, applies *rotation_matrix* and *scale_factor*
    around the local origin (the AABB bottom-centre), and returns the XY
    centre and minimum Z of the resulting AABB.  These offsets are
    subtracted from the fitted AABB bottom-centre to recover the true
    world-space position of the simready local origin (the ``init_pos``
    that the simulation expects).
    """
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
    verts = mesh.vertices.copy()
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
    b = np.zeros((2, 3), dtype=np.float64)
    b[0] = verts.min(axis=0)
    b[1] = verts.max(axis=0)
    return (
        float(0.5 * (b[0, 0] + b[1, 0])),   # AABB centre X → sim X
        float(-0.5 * (b[0, 2] + b[1, 2])),  # -centre Z → sim Y
        float(b[0, 1]),                       # min Y → sim Z
    )


def _build_object_manifest(
    output_root: Path,
    step_result: dict[str, Any],
    table_fit_manifest: dict[str, Any],
    aligned_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge world_bc, rotation, scale into one per-object record.

    Returns a dict keyed by object id, each value containing everything
    needed to compute ``init_pos`` / ``init_rot`` / ``body_scale``.
    """
    objects_info = step_result.get("objects") or []

    # index metric_scale by object id
    metric_by_id: dict[str, float] = {}
    for obj in objects_info:
        oid = str(obj.get("id", ""))
        if not oid:
            continue
        ms = obj.get("metric_scale")
        sf = float(ms.get("scale_factor", 1.0)) if isinstance(ms, dict) else 1.0
        metric_by_id[oid] = sf

    # index world_aabb_bottom_center from table-fit manifest
    world_bc_by_id: dict[str, list[float]] = {}
    for e in table_fit_manifest.get("objects") or []:
        eid = str(e.get("id", "")) if isinstance(e, dict) else ""
        wbc = e.get("world_aabb_bottom_center") if isinstance(e, dict) else None
        if eid and isinstance(wbc, list) and len(wbc) == 3:
            world_bc_by_id[eid] = [float(v) for v in wbc]

    consolidated: dict[str, Any] = {}
    skipped_no_glb: list[str] = []
    for obj in objects_info:
        oid = str(obj.get("id", ""))
        if not oid:
            continue

        source = obj.get("simready_geometry_path") or obj.get("mesh_path")
        simready_path = _resolve_path(source or "", output_root)
        if not simready_path.is_file():
            skipped_no_glb.append(oid)
            continue

        description = str(obj.get("description") or obj.get("name") or "").strip()
        scale_factor = metric_by_id.get(oid, 1.0)

        aligned = aligned_by_id.get(oid)
        rot_matrix: list[list[float]] | None = None
        transform_scale: list[float] | None = None
        if aligned:
            raw = aligned.get("rotation_matrix")
            if raw and isinstance(raw, list):
                rot_matrix = raw
            raw_scale = aligned.get("scale")
            if isinstance(raw_scale, list) and len(raw_scale) == 3:
                transform_scale = [float(v) for v in raw_scale]

        wbc = world_bc_by_id.get(oid)

        consolidated[oid] = {
            "id": oid,
            "description": description,
            "simready_path": simready_path,
            "scale_factor": scale_factor,
            "transform_scale": transform_scale,
            "rotation_matrix": rot_matrix,
            "world_aabb_bottom_center": wbc,
        }

    if skipped_no_glb:
        print(
            "  [WARN] object(s) skipped (simready GLB not found): "
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

    Uses **simready** GLBs — transforms are written explicitly as
    ``body_scale``, ``init_pos``, and ``init_rot``.
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

    table_info = step_result.get("table") or {}
    table_desc = str(
        table_info.get("complete_table_description")
        or table_info.get("description", "")
    ).strip()
    object_desc_by_id = {
        str(item.get("id", "")): str(
            item.get("description") or item.get("name") or ""
        ).strip()
        for item in step_result.get("objects") or []
        if isinstance(item, dict) and item.get("id")
    }

    mesh_assets_dir = export_dir / "mesh_assets"
    mesh_assets_dir.mkdir(parents=True, exist_ok=True)

    table_fit_output = _resolve_path(
        table_fit_manifest.get("table_output_path", ""),
        output_root,
    )
    if not table_fit_output.is_file():
        raise FileNotFoundError(f"Table-fit GLB not found: {table_fit_output}")
    table_dst = mesh_assets_dir / "table" / "table_0.glb"
    table_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(table_fit_output, table_dst)

    rigid_objects: list[dict[str, Any]] = []

    fitted_objects = [
        item
        for item in table_fit_manifest.get("objects", []) or []
        if isinstance(item, dict) and item.get("id") and item.get("path")
    ]
    total = len(fitted_objects)
    for idx, item in enumerate(fitted_objects):
        oid = str(item["id"])
        safe_name = oid.replace("interact_", "").strip("_") or "object"
        obj_dir = mesh_assets_dir / safe_name / oid
        obj_dir.mkdir(parents=True, exist_ok=True)
        object_dst = obj_dir / f"{oid}.glb"
        object_fit_path = _resolve_path(str(item["path"]), output_root)
        if not object_fit_path.is_file():
            raise FileNotFoundError(f"Table-fit object GLB not found: {object_fit_path}")
        shutil.copy2(object_fit_path, object_dst)

        # Table-fit GLBs already have the relative layout baked into vertices.
        # Preview/export should not rebuild placement from simready transforms.
        init_pos = [0.0, 0.0, 0.0]
        init_rot = [0.0, 0.0, 0.0]
        body_scale = [1.0, 1.0, 1.0]
        description = object_desc_by_id.get(oid, oid)

        rigid_objects.append(
            {
                "uid": oid,
                "description": description,
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
                "max_convex_hull_num": _DEFAULT_MAX_CONVEX_HULL_NUM,
            }
        )
        print(
            f"  [{idx+1}/{total}] [{oid}] {description}"
            f"  pos={init_pos}  rot={init_rot}  scale={body_scale}  src=table_fit_glb"
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
                "body_scale": [1.0, 1.0, 1.0],
                "body_type": "kinematic",
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
            }
        ],
        "rigid_object": rigid_objects,
    }

    config_path = export_dir / "gym_config.json"
    config_path.write_text(
        json.dumps(config, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return config_path
