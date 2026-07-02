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

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    GeometryManager,
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
) -> tuple[Any, np.ndarray]:
    sim = SimulationManager(headless=True, sim_device=sim_device)
    with tempfile.TemporaryDirectory(prefix="p2s_table_fit_gravity_") as tmp:
        tmp_path = Path(tmp)
        pre_gravity = tmp_path / "table_pre_gravity.glb"
        GeometryManager.copy_scene_with_transform(scene, z_to_y).export(pre_gravity)
        result = sim.run_gravity_simulation(
            GravityDropRequest(
                glb_path=pre_gravity,
                max_convex_hull_num=8,
                initial_height=0.05,
            )
        )
    gravity_transform = np.asarray(result.final_pose, dtype=np.float64)
    settled = scene.copy()
    settled.apply_transform(gravity_transform)
    return settled, gravity_transform


def fit_table_to_clutter(
    *,
    table_result: dict[str, Any],
    clutter_result: dict[str, Any],
    output_root: Path,
    output_dir: Path,
    table_output_path: Path | None = None,
    object_output_paths: dict[str, Path] | None = None,
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
    if table_output_path is None:
        table_output_path = output_dir / "table_fit_to_clutter.glb"
    table_output_path = table_output_path.expanduser().resolve()
    table_output_path.parent.mkdir(parents=True, exist_ok=True)
    object_output_paths = {
        str(key): path.expanduser().resolve()
        for key, path in (object_output_paths or {}).items()
    }

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

    z_to_y = GeometryManager.z_up_to_glb_y_up_transform()
    y_to_z = np.linalg.inv(z_to_y)

    # Load the table and detect its support surface.
    table_scene = GeometryManager.load_table_fit_scene_internal_z(
        table_simready_path,
        trimesh=trimesh,
        y_to_z=y_to_z,
    )
    table_fit_transform = np.eye(4, dtype=np.float64)

    table_mesh = GeometryManager.scene_to_mesh(table_scene, trimesh=trimesh)
    clutter_aabb = clutter_result.get("clutter_2d_aabb_cm") or {}
    clutter_size = clutter_aabb.get("size_xy", [1.0, 1.0])
    target_aspect = float(clutter_size[0]) / max(float(clutter_size[1]), 1.0e-6)
    initial_support = GeometryManager.detect_table_fit_support_quad(
        table_mesh,
        target_aspect=target_aspect,
    )

    # Load the clutter scenes.
    clutter_scenes = [
        (
            oid,
            GeometryManager.load_table_fit_scene_internal_z(
                path,
                trimesh=trimesh,
                y_to_z=y_to_z,
            ),
        )
        for oid, path in object_glb_paths
    ]
    clutter_bounds = GeometryManager.table_fit_scene_union_bounds(
        [scene for _, scene in clutter_scenes],
        trimesh=trimesh,
    )

    # Compute the required table size and optional uniform scale.
    clutter_size_cm = (clutter_bounds[1, :2] - clutter_bounds[0, :2]) * 100.0
    if object_coverage_percent is not None:
        support_occupancy_ratio = float(
            np.clip(object_coverage_percent / 100.0, 0.1, 1.0)
        )
    occupancy = float(np.clip(support_occupancy_ratio, 0.1, 1.0))
    required_size_cm = clutter_size_cm / occupancy + 2.0 * float(margin_cm)
    scale_method = "fit_to_clutter_occupancy_margin"
    relative_scale_hint = None
    if table_result.get("is_complete_visible_table"):
        hint = table_result.get("complete_table_relative_scale_hint")
        if isinstance(hint, dict) and hint.get("status") == "ok":
            ratio_xy = np.asarray(
                hint.get("support_to_clutter_size_ratio_xy", []),
                dtype=np.float64,
            )
            if ratio_xy.shape == (2,) and np.all(np.isfinite(ratio_xy)):
                ratio_xy = np.maximum(ratio_xy, 1.0)
                required_size_cm = clutter_size_cm * ratio_xy
                scale_method = "complete_table_sam3d_raw_relative_uniform_xyz"
                relative_scale_hint = hint
    support_size_cm = np.asarray(initial_support["size_xy"], dtype=np.float64) * 100.0
    scale_x = GeometryManager.table_fit_safe_positive_ratio(
        required_size_cm[0],
        support_size_cm[0],
    )
    scale_y = GeometryManager.table_fit_safe_positive_ratio(
        required_size_cm[1],
        support_size_cm[1],
    )
    uniform_scale = max(scale_x, scale_y)
    if scale_method == "complete_table_sam3d_raw_relative_uniform_xyz":
        table_scale_transform = GeometryManager.table_fit_uniform_scale_transform(
            center_xy=np.asarray(initial_support["center_xy"], dtype=np.float64),
            scale=uniform_scale,
        )
    else:
        table_scale_transform = GeometryManager.table_fit_uniform_xy_scale_transform(
            center_xy=np.asarray(initial_support["center_xy"], dtype=np.float64),
            scale=uniform_scale,
        )
    table_scene.apply_transform(table_scale_transform)
    table_fit_transform = table_scale_transform @ table_fit_transform

    # Settle the table under gravity.
    if gravity_settle_table:
        table_scene, gravity_transform = _gravity_settle_table_fit_internal_z_scene(
            table_scene,
            z_to_y=z_to_y,
            sim_device=sim_device,
        )
        table_fit_transform = gravity_transform @ table_fit_transform

    # Reposition the table at the origin.
    final_table_mesh = GeometryManager.scene_to_mesh(table_scene, trimesh=trimesh)
    final_support = GeometryManager.detect_table_fit_support_quad(
        final_table_mesh,
        target_aspect=float(required_size_cm[0] / max(required_size_cm[1], 1.0e-6)),
    )
    support_center = np.asarray(final_support["center"], dtype=np.float64)
    table_bounds = np.asarray(final_table_mesh.bounds, dtype=np.float64)
    table_bottom_z = float(table_bounds[0, 2])

    table_shift = np.eye(4, dtype=np.float64)
    table_shift[:3, 3] = [-support_center[0], -support_center[1], -table_bottom_z]
    table_scene.apply_transform(table_shift)
    table_fit_transform = table_shift @ table_fit_transform
    support_z_after = float((support_center + table_shift[:3, 3])[2])

    # Measure the table surface height.
    # Use the highest point of the table mesh (after scaling + gravity + shift)
    # rather than the support-plane mean Z, so that thin objects sit above the
    # actual geometry even when the tabletop has slight unevenness.
    _table_mesh_after_shift = GeometryManager.scene_to_mesh(
        table_scene,
        trimesh=trimesh,
    )
    _table_max_z = float(
        np.asarray(_table_mesh_after_shift.bounds, dtype=np.float64)[1, 2]
    )
    _surface_z_margin = 0.01  # 1 cm above the highest table point

    # Place the objects on the table.
    placed_objects: list[dict[str, Any]] = []
    shifted_clutter: list[tuple[str, Any]] = []
    clutter_after = GeometryManager.table_fit_scene_union_bounds(
        [scene for _, scene in clutter_scenes],
        trimesh=trimesh,
    )
    clutter_center_xy = 0.5 * (clutter_after[0, :2] + clutter_after[1, :2])
    for oid, scene in clutter_scenes:
        obj_mesh = GeometryManager.scene_to_mesh(scene, trimesh=trimesh)
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
    GeometryManager.copy_scene_with_transform(table_scene, z_to_y).export(
        table_output_path
    )

    for oid, scene in shifted_clutter:
        object_path = object_output_paths.get(oid, output_dir / f"{oid}_on_table.glb")
        object_path.parent.mkdir(parents=True, exist_ok=True)
        GeometryManager.copy_scene_with_transform(scene, z_to_y).export(object_path)
        # Compute world-space AABB bottom-centre (sim Z-up coords) before
        # the scene is converted to GLB Y-up for export.  This is the
        # reference position that gym_export uses to derive ``init_pos``.
        _placed_mesh = GeometryManager.scene_to_mesh(scene, trimesh=trimesh)
        _placed_b = np.asarray(_placed_mesh.bounds, dtype=np.float64)
        world_aabb_bottom_center = [
            float(0.5 * (_placed_b[0, 0] + _placed_b[1, 0])),
            float(0.5 * (_placed_b[0, 1] + _placed_b[1, 1])),
            float(_placed_b[0, 2]),
        ]
        placed_objects.append(
            {
                "id": oid,
                "path": str(object_path),
                "world_aabb_bottom_center": world_aabb_bottom_center,
            }
        )

    final_clutter_bounds = GeometryManager.table_fit_scene_union_bounds(
        [scene for _, scene in shifted_clutter],
        trimesh=trimesh,
    )
    final_clutter_aabb_cm = GeometryManager.table_fit_bounds_xy_manifest(
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
        "table_output_path": str(table_output_path),
        "objects": placed_objects,
        "margin_cm": margin_cm,
        "support_occupancy_ratio": occupancy,
        "gravity_settle_table": gravity_settle_table,
        "table_bottom_z_after_shift": 0.0,
        "support_z_after_shift": support_z_after,
        "table_fit_transform": table_fit_transform.tolist(),
        "initial_support_quad": initial_support,
        "final_support_quad_centered": final_support_centered,
        "clutter_2d_aabb_cm": final_clutter_aabb_cm,
        "required_support_size_cm": required_size_cm.tolist(),
        "table_xy_scale": {
            "method": scale_method,
            "uniform_scale": uniform_scale,
            "scale_x_raw": scale_x,
            "scale_y_raw": scale_y,
            "support_size_before_scale_cm": support_size_cm.tolist(),
            "complete_table_relative_scale_hint": relative_scale_hint,
        },
        "fit_check": {
            "fits_width": float(final_clutter_aabb_cm["size_xy"][0])
            <= float(np.asarray(final_support_centered["size_xy"])[0] * 100.0),
            "fits_depth": float(final_clutter_aabb_cm["size_xy"][1])
            <= float(np.asarray(final_support_centered["size_xy"])[1] * 100.0),
        },
    }
    return manifest
