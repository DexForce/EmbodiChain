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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.utils.io import relative_path
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager.scene_geometry import (
    _copy_scene_with_transform,
    _scene_to_mesh,
    _z_up_to_glb_y_up_transform,
    _detect_table_fit_support_quad,
    _load_table_fit_scene_internal_z,
    _table_fit_bounds_xy_manifest,
    _table_fit_safe_positive_ratio,
    _table_fit_scene_union_bounds,
    _table_fit_uniform_xy_scale_transform,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager import (
    SimulationManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.schemas import (
    GravityDropRequest,
)

__all__ = ["fit_table_to_clutter"]


def _resolve_generated_path(value: Any, output_root: Path) -> Path:
    if not value:
        return Path()
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root.expanduser().resolve() / path).resolve()


def _gravity_settle_table_fit_internal_z_scene(
    scene: Any,
    *,
    z_to_y: np.ndarray,
    sim_device: str,
) -> Any:
    sim = SimulationManager(headless=True, sim_device=sim_device)
    with tempfile.TemporaryDirectory(prefix="p2s_table_fit_gravity_") as tmp:
        tmp_path = Path(tmp)
        pre_gravity = tmp_path / "table_pre_gravity.glb"
        _copy_scene_with_transform(scene, z_to_y).export(pre_gravity)
        result = sim.run_gravity_simulation(
            GravityDropRequest(
                glb_path=pre_gravity,
                max_convex_hull_num=16,
                initial_height=0.05,
            )
        )
    settled = scene.copy()
    settled.apply_transform(np.asarray(result.final_pose, dtype=np.float64))
    return settled


def _write_table_fit_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fit_table_to_clutter(
    *,
    table_result: dict[str, Any],
    clutter_result: dict[str, Any],
    output_root: Path,
    output_dir: Path,
    margin_cm: float = 10.0,
    support_occupancy_ratio: float = 0.80,
    object_coverage_percent: int | None = None,
    gravity_settle_table: bool = True,
    sim_device: str = "cpu",
) -> dict[str, Any]:
    """Fit a table mesh to an already laid-out clutter result.

    Args:
        object_coverage_percent: If set (1-100), overrides
            ``support_occupancy_ratio`` by converting the percentage to a ratio
            (e.g. 30 → 0.30). The required table size is computed as
            clutter_size / ratio. When None, the default
            ``support_occupancy_ratio`` is used.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Table fitting requires trimesh.") from exc

    output_root = output_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the table geometry.
    table_simready_path = _resolve_generated_path(
        table_result.get("simready_geometry_path") or table_result.get("mesh_path"),
        output_root,
    )
    if not table_simready_path.is_file():
        raise FileNotFoundError(f"Table simready GLB not found: {table_simready_path}")

    # Resolve the clutter object geometries.
    settled_objects = [
        item
        for item in clutter_result.get("objects", [])
        if isinstance(item, dict) and item.get("status") == "ok"
    ]
    if not settled_objects:
        raise ValueError("No successfully settled objects for table fitting.")

    object_glb_paths: list[tuple[str, Path]] = []
    for item in settled_objects:
        glb_path = _resolve_generated_path(
            item.get("laid_out_glb_path") or item.get("settled_glb_path"),
            output_root,
        )
        if glb_path.is_file():
            object_glb_paths.append((str(item["id"]), glb_path))

    if not object_glb_paths:
        raise ValueError("No valid settled object GLBs for table fitting.")

    z_to_y = _z_up_to_glb_y_up_transform()
    y_to_z = np.linalg.inv(z_to_y)

    # Load the table and detect its support surface.
    table_scene = _load_table_fit_scene_internal_z(
        table_simready_path,
        trimesh=trimesh,
        y_to_z=y_to_z,
    )
    table_mesh = _scene_to_mesh(table_scene, trimesh=trimesh)
    clutter_aabb = clutter_result.get("clutter_2d_aabb_cm") or {}
    clutter_size = clutter_aabb.get("size_xy", [1.0, 1.0])
    target_aspect = float(clutter_size[0]) / max(float(clutter_size[1]), 1.0e-6)
    initial_support = _detect_table_fit_support_quad(
        table_mesh,
        target_aspect=target_aspect,
    )

    # Load the clutter scenes.
    clutter_scenes = [
        (oid, _load_table_fit_scene_internal_z(path, trimesh=trimesh, y_to_z=y_to_z))
        for oid, path in object_glb_paths
    ]
    clutter_bounds = _table_fit_scene_union_bounds(
        [scene for _, scene in clutter_scenes],
        trimesh=trimesh,
    )

    # Compute the required table size and uniform scale.
    clutter_size_cm = (clutter_bounds[1, :2] - clutter_bounds[0, :2]) * 100.0
    if object_coverage_percent is not None:
        support_occupancy_ratio = float(
            np.clip(object_coverage_percent / 100.0, 0.1, 1.0)
        )
    occupancy = float(np.clip(support_occupancy_ratio, 0.1, 1.0))
    required_size_cm = clutter_size_cm / occupancy + 2.0 * float(margin_cm)
    support_size_cm = np.asarray(initial_support["size_xy"], dtype=np.float64) * 100.0
    scale_x = _table_fit_safe_positive_ratio(required_size_cm[0], support_size_cm[0])
    scale_y = _table_fit_safe_positive_ratio(required_size_cm[1], support_size_cm[1])
    uniform_scale = max(scale_x, scale_y)
    table_scale_transform = _table_fit_uniform_xy_scale_transform(
        center_xy=np.asarray(initial_support["center_xy"], dtype=np.float64),
        scale=uniform_scale,
    )
    table_scene.apply_transform(table_scale_transform)

    # Settle the table under gravity.
    if gravity_settle_table:
        table_scene = _gravity_settle_table_fit_internal_z_scene(
            table_scene,
            z_to_y=z_to_y,
            sim_device=sim_device,
        )

    # Reposition the table at the origin.
    final_table_mesh = _scene_to_mesh(table_scene, trimesh=trimesh)
    final_support = _detect_table_fit_support_quad(
        final_table_mesh,
        target_aspect=float(required_size_cm[0] / max(required_size_cm[1], 1.0e-6)),
    )
    support_center = np.asarray(final_support["center"], dtype=np.float64)
    table_bounds = np.asarray(final_table_mesh.bounds, dtype=np.float64)
    table_bottom_z = float(table_bounds[0, 2])

    table_shift = np.eye(4, dtype=np.float64)
    table_shift[:3, 3] = [-support_center[0], -support_center[1], -table_bottom_z]
    table_scene.apply_transform(table_shift)
    support_z_after = float((support_center + table_shift[:3, 3])[2])

    # Measure the table surface height.
    # Use the highest point of the table mesh (after scaling + gravity + shift)
    # rather than the support-plane mean Z, so that thin objects sit above the
    # actual geometry even when the tabletop has slight unevenness.
    _table_mesh_after_shift = _scene_to_mesh(table_scene, trimesh=trimesh)
    _table_max_z = float(
        np.asarray(_table_mesh_after_shift.bounds, dtype=np.float64)[1, 2]
    )
    _surface_z_margin = 0.01  # 1 cm above the highest table point

    # Place the objects on the table.
    placed_objects: list[dict[str, Any]] = []
    shifted_clutter: list[tuple[str, Any]] = []
    clutter_after = _table_fit_scene_union_bounds(
        [scene for _, scene in clutter_scenes],
        trimesh=trimesh,
    )
    clutter_center_xy = 0.5 * (clutter_after[0, :2] + clutter_after[1, :2])
    for oid, scene in clutter_scenes:
        obj_mesh = _scene_to_mesh(scene, trimesh=trimesh)
        obj_bounds = np.asarray(obj_mesh.bounds, dtype=np.float64)
        obj_bottom_z = float(obj_bounds[0, 2])
        obj_shift = np.eye(4, dtype=np.float64)
        obj_shift[:3, 3] = [
            -float(clutter_center_xy[0]),
            -float(clutter_center_xy[1]),
            _table_max_z - obj_bottom_z + _surface_z_margin,
        ]
        scene.apply_transform(obj_shift)
        shifted_clutter.append((oid, scene))

    # Export the fitted table and placed objects.
    final_table_path = output_dir / "table_fit_to_clutter.glb"
    _copy_scene_with_transform(table_scene, z_to_y).export(final_table_path)

    for oid, scene in shifted_clutter:
        object_path = output_dir / f"{oid}_on_table.glb"
        _copy_scene_with_transform(scene, z_to_y).export(object_path)
        placed_objects.append({"id": oid, "path": str(object_path)})

    # Write the fit manifest.
    final_clutter_bounds = _table_fit_scene_union_bounds(
        [scene for _, scene in shifted_clutter],
        trimesh=trimesh,
    )
    final_clutter_aabb_cm = _table_fit_bounds_xy_manifest(
        final_clutter_bounds,
        unit_scale=100.0,
    )
    final_support_centered = {
        **final_support,
        "center": (support_center + table_shift[:3, 3]).tolist(),
        "center_xy": (
            np.asarray(final_support["center_xy"], dtype=np.float64)
            - support_center[:2]
        ).tolist(),
        "corners_xy": (
            np.asarray(final_support["corners_xy"], dtype=np.float64)
            - support_center[:2]
        ).tolist(),
    }
    manifest = {
        "status": "ok",
        "output_dir": str(output_dir),
        "table_simready_path": str(table_simready_path),
        "table_output_path": str(final_table_path),
        "objects": placed_objects,
        "margin_cm": margin_cm,
        "support_occupancy_ratio": occupancy,
        "gravity_settle_table": gravity_settle_table,
        "table_bottom_z_after_shift": 0.0,
        "support_z_after_shift": support_z_after,
        "initial_support_quad": initial_support,
        "final_support_quad_centered": final_support_centered,
        "clutter_2d_aabb_cm": final_clutter_aabb_cm,
        "required_support_size_cm": required_size_cm.tolist(),
        "table_xy_scale": {
            "uniform_scale": uniform_scale,
            "scale_x_raw": scale_x,
            "scale_y_raw": scale_y,
            "support_size_before_scale_cm": support_size_cm.tolist(),
        },
        "fit_check": {
            "fits_width": float(final_clutter_aabb_cm["size_xy"][0])
            <= float(np.asarray(final_support_centered["size_xy"])[0] * 100.0),
            "fits_depth": float(final_clutter_aabb_cm["size_xy"][1])
            <= float(np.asarray(final_support_centered["size_xy"])[1] * 100.0),
        },
    }
    manifest_path = output_dir / "table_fit_to_clutter_manifest.json"
    _write_table_fit_json(manifest_path, manifest)
    return {
        "status": "ok",
        "manifest_path": relative_path(manifest_path, output_root),
    }
