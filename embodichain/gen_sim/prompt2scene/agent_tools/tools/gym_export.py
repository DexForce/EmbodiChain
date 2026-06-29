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
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    STEP_RESULT_FILENAME,
    UNIFIED_SCENE_GEN_STEP,
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

_DEFAULT_MAX_CONVEX_HULL_NUM = 8


def _resolve_path(value: str, output_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


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


def _glb_aabb_bottom_center(glb_path: Path) -> list[float]:
    """``[x, y, z]`` bottom-centre position in **simulation Z-up** space.

    The GLB is stored in Y-up convention (X=width, Y=up, Z=depth).
    EmbodiChain simulation converts to Z-up on load, so we return the
    position in Z-up space:  ``center_X``, ``-center_Z``, ``min_Y``.
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
    b = np.asarray(mesh.bounds, dtype=np.float64)
    return [
        float(0.5 * (b[0, 0] + b[1, 0])),   # centre X
        float(-0.5 * (b[0, 2] + b[1, 2])),  # -centre Z (GLB Z → internal -Y)
        float(b[0, 1]),                       # min Y (GLB up → internal Z)
    ]


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

    # ── step result & table-fit manifest ──────────────────────────────
    step_result = _read_json(
        output_root / UNIFIED_SCENE_GEN_STEP / STEP_RESULT_FILENAME
    )
    table_fit = step_result.get("table_fit_to_clutter") or {}
    manifest = _read_json(
        _resolve_path(table_fit.get("manifest_path", ""), output_root)
    )

    # ── per-object metadata from simready→aligned manifest ────────────
    aligned_by_id: dict[str, dict[str, Any]] = {}
    aligned_manifest_path = (
        output_root / UNIFIED_SCENE_GEN_STEP / "glb_gen" / "simready_to_aligned_manifest.json"
    )
    if aligned_manifest_path.is_file():
        aligned_manifest = _read_json(aligned_manifest_path)
        for item in aligned_manifest.get("items", []) or []:
            if isinstance(item, dict):
                aligned_by_id[str(item.get("id", ""))] = item

    # ── table surface Z (from fitted table GLB) ───────────────────────
    fitted_table_path = _resolve_path(
        manifest.get("table_output_path", ""), output_root
    )
    table_surface_z = (
        _glb_max_z(fitted_table_path) if fitted_table_path.is_file() else 0.0
    )

    # ── description lookup ────────────────────────────────────────────
    object_meta_by_id: dict[str, dict[str, str]] = {}
    for obj in step_result.get("objects", []) or []:
        if isinstance(obj, dict):
            oid = str(obj.get("id", ""))
            if oid:
                object_meta_by_id[oid] = {
                    "description": str(obj.get("description", "")).strip(),
                    "name": str(obj.get("name", "")).strip(),
                }

    table_info = step_result.get("table") or {}
    table_desc = str(
        table_info.get("complete_table_description")
        or table_info.get("description", "")
    ).strip()

    mesh_assets_dir = export_dir / "mesh_assets"
    mesh_assets_dir.mkdir(parents=True, exist_ok=True)

    # ── table ─────────────────────────────────────────────────────────
    table_simready = _resolve_path(
        table_info.get("simready_geometry_path")
        or table_info.get("mesh_path", ""),
        output_root,
    )
    if not table_simready.is_file():
        raise FileNotFoundError(f"Table simready GLB not found: {table_simready}")
    table_dst = mesh_assets_dir / "table" / "table_0.glb"
    table_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(table_simready, table_dst)

    uniform_scale = 1.0
    ts = manifest.get("table_xy_scale")
    if isinstance(ts, dict):
        uniform_scale = float(ts.get("uniform_scale", 1.0))

    # ── objects ───────────────────────────────────────────────────────
    table_fit_objects = {
        str(e["id"]): _resolve_path(e["path"], output_root)
        for e in (manifest.get("objects") or [])
        if isinstance(e, dict)
    }
    objects_info = step_result.get("objects") or []
    rigid_objects: list[dict[str, Any]] = []

    def _obj_desc(obj: dict[str, Any]) -> str:
        meta = object_meta_by_id.get(str(obj.get("id", "")))
        return (meta["description"] or meta["name"]) if meta else ""

    for obj in objects_info:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("id", ""))
        if not object_id:
            continue

        # ── GLB: simready (normalised, no baked transforms) ──────────
        source = obj.get("simready_geometry_path") or obj.get("mesh_path")
        object_src = _resolve_path(source, output_root)
        if not object_src.is_file():
            continue

        safe_name = object_id.replace("interact_", "").strip("_") or "object"
        obj_dir = mesh_assets_dir / safe_name / object_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        object_dst = obj_dir / f"{object_id}.glb"
        shutil.copy2(object_src, object_dst)

        # ── body_scale ────────────────────────────────────────────────
        ms = obj.get("metric_scale")
        scale_factor = float(ms.get("scale_factor", 1.0)) if isinstance(ms, dict) else 1.0
        body_scale = [scale_factor, scale_factor, scale_factor]

        # ── init_pos: read from fitted on-table GLB ───────────────────
        fitted_path = table_fit_objects.get(object_id)
        if fitted_path and fitted_path.is_file():
            init_pos = _glb_aabb_bottom_center(fitted_path)
        else:
            init_pos = [0.0, 0.0, table_surface_z]

        # ── init_rot: decompose from simready→aligned rotation ────────
        init_rot: list[float] = [0.0, 0.0, 0.0]
        aligned = aligned_by_id.get(object_id)
        if aligned:
            rot = aligned.get("rotation_matrix")
            if rot and isinstance(rot, list):
                init_rot = _matrix_to_euler_xyz_deg(rot)

        rigid_objects.append(
            {
                "uid": object_id,
                "description": _obj_desc(obj),
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

    # ── write config ──────────────────────────────────────────────────
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
                "body_scale": [uniform_scale, uniform_scale, 1.0],
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
