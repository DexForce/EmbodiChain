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

_DEFAULT_MAX_CONVEX_HULL_NUM = 32


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# consolidated object manifest
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# main export
# ---------------------------------------------------------------------------


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

    # ── data sources ────────────────────────────────────────────────────
    step_result = _read_json(
        output_root / UNIFIED_SCENE_GEN_STEP / STEP_RESULT_FILENAME
    )
    table_fit = step_result.get("table_fit_to_clutter") or {}
    table_fit_manifest = _read_json(
        _resolve_path(table_fit.get("manifest_path", ""), output_root)
    )

    aligned_by_id: dict[str, dict[str, Any]] = {}
    aligned_manifest_path = (
        output_root
        / UNIFIED_SCENE_GEN_STEP
        / "glb_gen"
        / "simready_to_aligned_manifest.json"
    )
    if aligned_manifest_path.is_file():
        for item in _read_json(aligned_manifest_path).get("items", []) or []:
            if isinstance(item, dict) and item.get("id"):
                aligned_by_id[str(item["id"])] = item

    # ── consolidated per-object manifest ─────────────────────────────────
    object_manifest = _build_object_manifest(
        output_root, step_result, table_fit_manifest, aligned_by_id
    )

    # ── table ────────────────────────────────────────────────────────────
    table_info = step_result.get("table") or {}
    table_desc = str(
        table_info.get("complete_table_description")
        or table_info.get("description", "")
    ).strip()

    mesh_assets_dir = export_dir / "mesh_assets"
    mesh_assets_dir.mkdir(parents=True, exist_ok=True)

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

    table_surface_z = _glb_max_z(table_simready)

    uniform_scale = 1.0
    ts = table_fit_manifest.get("table_xy_scale")
    if isinstance(ts, dict):
        uniform_scale = float(ts.get("uniform_scale", 1.0))

    # ── objects ──────────────────────────────────────────────────────────
    rigid_objects: list[dict[str, Any]] = []

    total = len(object_manifest)
    for idx, (oid, om) in enumerate(object_manifest.items()):
        # Copy simready GLB
        safe_name = oid.replace("interact_", "").strip("_") or "object"
        obj_dir = mesh_assets_dir / safe_name / oid
        obj_dir.mkdir(parents=True, exist_ok=True)
        object_dst = obj_dir / f"{oid}.glb"
        shutil.copy2(om["simready_path"], object_dst)

        # body_scale.  Image-scene alignment may contain a full simready→aligned
        # scale; text-scene layout only has the per-object metric scale.
        sf = om["scale_factor"]
        scale_glb = om.get("transform_scale") or [sf, sf, sf]
        body_scale = _glb_scale_to_sim(scale_glb)

        # init_rot
        init_rot: list[float] = [0.0, 0.0, 0.0]
        if om["rotation_matrix"] is not None:
            init_rot = _matrix_to_euler_xyz_deg(
                _glb_rotation_to_sim(om["rotation_matrix"])
            )

        # init_pos = world_bc - rotated_aabb_offset
        ro = _rotated_aabb_offsets(
            om["simready_path"], om["rotation_matrix"], scale_glb
        )
        wbc = om["world_aabb_bottom_center"]
        if wbc is not None:
            init_pos = [wbc[0] - ro[0], wbc[1] - ro[1], wbc[2] - ro[2]]
        else:
            init_pos = [-ro[0], -ro[1], table_surface_z - ro[2]]

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
                "max_convex_hull_num": _DEFAULT_MAX_CONVEX_HULL_NUM,
            }
        )
        wbc = om["world_aabb_bottom_center"]
        wbc_flag = "wbc" if wbc is not None else "fallback"
        print(
            f"  [{idx+1}/{total}] [{oid}] {om['description']}"
            f"  pos={init_pos}  rot={init_rot}  scale={body_scale}  src={wbc_flag}"
        )

    # ── write gym config ─────────────────────────────────────────────────
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
