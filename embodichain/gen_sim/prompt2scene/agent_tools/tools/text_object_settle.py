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
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager import (
    SimulationManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.schemas import (
    GravityDropRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager import (
    LayoutManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    GeometryManager,
)
from embodichain.gen_sim.prompt2scene.utils.io import (
    relative_path,
    write_json,
)
from embodichain.gen_sim.prompt2scene.utils.log import log_warning
from embodichain.gen_sim.prompt2scene.agent_tools.managers.matplotlib_manager import (
    MatplotlibManager,
    RenderFootprintLayoutRequest,
)

__all__ = ["settle_text_objects_to_ground"]


def settle_text_objects_to_ground(
    *,
    objects: list[dict[str, Any]],
    spatial_relations: list[dict[str, Any]] | None = None,
    table_constraints: list[dict[str, Any]] | None = None,
    output_dir: Path,
    output_root: Path,
    sim_device: str = "cpu",
) -> dict[str, Any]:
    """Scale simready objects to real-world size, gravity-settle, layout on table.

    For each text-input object:
    1. Load simready GLB (GLB Y-up) → convert to internal Z-up
    2. Apply scene-level metric scale_factor → real-world size
    3. Gravity simulation to settle on ground plane
    4. Move AABB bottom centre to XY origin at Z=0
    5. Build grid/rank initialization from left_of/front_of and table constraints
    6. Run SA-based 2D point optimization and mesh AABB collision cleanup
    7. Apply layout positions

    Returns laid-out scenes and per-object metadata.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Text object gravity settling requires trimesh.") from exc

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationManager(headless=True, sim_device=sim_device)
    z_to_y = GeometryManager.z_up_to_glb_y_up_transform()
    y_to_z = np.linalg.inv(z_to_y)

    settled_objects: list[dict[str, Any]] = []
    object_scenes: list[tuple[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="p2s_text_settle_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for obj in objects:
            obj_id = str(obj.get("id", ""))
            obj_name = str(obj.get("name", ""))

            # Validate the metric scale.
            metric_scale = obj.get("metric_scale")
            if not isinstance(metric_scale, dict):
                settled_objects.append(
                    {
                        "id": obj_id,
                        "name": obj_name,
                        "status": "skipped",
                        "reason": "missing_metric_scale",
                    }
                )
                continue
            scale_factor = float(metric_scale.get("scale_factor", 1.0))
            if not np.isfinite(scale_factor) or scale_factor <= 0.0:
                settled_objects.append(
                    {
                        "id": obj_id,
                        "name": obj_name,
                        "status": "skipped",
                        "reason": "invalid_scale_factor",
                    }
                )
                continue

            # Load the simulation-ready GLB.
            simready_path = _resolve_generated_path(
                obj.get("simready_geometry_path") or obj.get("mesh_path"),
                output_root,
            )
            if not simready_path.is_file():
                settled_objects.append(
                    {
                        "id": obj_id,
                        "name": obj_name,
                        "status": "skipped",
                        "reason": "missing_simready_glb",
                    }
                )
                continue

            try:
                # Load simready (GLB Y-up) → convert to internal Z-up
                scene_yup = trimesh.load(simready_path, force="scene")
                scene = GeometryManager.copy_scene_with_transform(scene_yup, y_to_z)

                # Apply real-world scale
                scale_transform = GeometryManager.scale_transform(scale_factor)
                scene.apply_transform(scale_transform)

                # Settle the object under gravity.
                mesh = GeometryManager.scene_to_mesh(scene, trimesh=trimesh)
                mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
                mesh_z_height = max(float(mesh_bounds[1][2] - mesh_bounds[0][2]), 0.0)
                bottom_to_xy = GeometryManager.aabb_bottom_to_xy_plane_transform(
                    mesh_bounds
                )
                normalized_scene = GeometryManager.copy_scene_with_transform(
                    scene,
                    bottom_to_xy,
                )

                # Export to Y-up GLB for gravity
                pre_gravity_scene = GeometryManager.copy_scene_with_transform(
                    normalized_scene,
                    z_to_y,
                )
                pre_gravity_path = tmp_path / f"{obj_id}_pre_gravity.glb"
                pre_gravity_scene.export(pre_gravity_path)
                gravity_initial_height = mesh_z_height * 0.1

                gravity_status = "ok"
                gravity_transform = np.eye(4, dtype=np.float64)
                gravity_reason = ""
                try:
                    gravity_result = sim.run_gravity_simulation(
                        GravityDropRequest(
                            glb_path=pre_gravity_path,
                            max_convex_hull_num=32,
                            initial_height=gravity_initial_height,
                        )
                    )
                    gravity_transform = GeometryManager.matrix_from_json(
                        gravity_result.final_pose,
                        name=f"{obj_id}.gravity_final_pose",
                    )
                except Exception:
                    gravity_status = "failed"
                    gravity_reason = traceback.format_exc()

                # Apply gravity result (in internal Z-up space)
                settled_scene = GeometryManager.copy_scene_with_transform(
                    normalized_scene,
                    gravity_transform,
                )

                # Center the bottom of the AABB at the XY origin.
                settled_mesh = GeometryManager.scene_to_mesh(
                    settled_scene,
                    trimesh=trimesh,
                )
                settled_bounds = np.asarray(settled_mesh.bounds, dtype=np.float64)
                settled_xy_center = GeometryManager.xy_aabb_center(settled_bounds)
                settled_xy_size = GeometryManager.xy_aabb_size(settled_bounds)
                settled_bottom_z = float(settled_bounds[0, 2])

                centre_transform = np.eye(4, dtype=np.float64)
                centre_transform[:3, 3] = [
                    -float(settled_xy_center[0]),
                    -float(settled_xy_center[1]),
                    -settled_bottom_z,
                ]
                centred_scene = GeometryManager.copy_scene_with_transform(
                    settled_scene,
                    centre_transform,
                )

                # Verify final bounds
                centred_mesh = GeometryManager.scene_to_mesh(
                    centred_scene,
                    trimesh=trimesh,
                )
                centred_bounds = np.asarray(centred_mesh.bounds, dtype=np.float64)
                centred_xy_size = GeometryManager.xy_aabb_size(centred_bounds)

                # Export settled GLB (Z-up → Y-up for GLB output)
                settled_glb_path = output_dir / f"{obj_id}_settled.glb"
                GeometryManager.copy_scene_with_transform(centred_scene, z_to_y).export(
                    settled_glb_path
                )

                item = {
                    "id": obj_id,
                    "name": obj_name,
                    "status": "ok",
                    "gravity_status": gravity_status,
                    "gravity_reason": gravity_reason,
                    "scale_factor": scale_factor,
                    "settled_glb_path": relative_path(
                        str(settled_glb_path),
                        output_root,
                    ),
                    "settled_xy_size_m": centred_xy_size.tolist(),
                    "settled_xy_size_cm": (centred_xy_size * 100.0).tolist(),
                    "settled_bounds_m": centred_bounds.tolist(),
                    "mesh_z_height_m": mesh_z_height,
                    "bottom_to_xy_transform": bottom_to_xy.tolist(),
                    "gravity_transform": gravity_transform.tolist(),
                    "centre_transform": centre_transform.tolist(),
                    "composed_settle_transform": (
                        centre_transform
                        @ gravity_transform
                        @ bottom_to_xy
                        @ scale_transform
                        @ y_to_z
                    ).tolist(),
                }
                settled_objects.append(item)
                object_scenes.append((obj_id, centred_scene))

            except Exception:
                settled_objects.append(
                    {
                        "id": obj_id,
                        "name": obj_name,
                        "status": "failed",
                        "reason": traceback.format_exc(),
                    }
                )

    # Optimize the spatial layout.
    layout_result = None
    if object_scenes:
        xy_sizes = {
            oid: np.asarray(
                GeometryManager.xy_aabb_size(
                    GeometryManager.scene_to_mesh(scene, trimesh=trimesh).bounds
                ),
                dtype=np.float64,
            )
            for oid, scene in object_scenes
        }
        relations = list(spatial_relations or [])
        layout_result = LayoutManager.layout_text_objects_grid(
            object_ids=[oid for oid, _ in object_scenes],
            xy_sizes=xy_sizes,
            spatial_relations=relations,
            table_constraints=list(table_constraints or []),
        )
        target_centers = layout_result["centers"]
        initial_centers = layout_result.get("initial_centers", {})

        # Render footprint layout diagnostics.
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_object_ids = [oid for oid, _ in object_scenes]
        debug_before_centers = {
            oid: np.zeros(2, dtype=np.float64) for oid in debug_object_ids
        }
        debug_renders = (
            (
                "footprint_layout_xy_before.png",
                "Before Layout (all at origin)",
                debug_before_centers,
            ),
            (
                "footprint_layout_xy_grid_init.png",
                "After Grid Initialisation",
                initial_centers,
            ),
            (
                "footprint_layout_xy_after.png",
                "After SA Optimisation",
                target_centers,
            ),
        )
        for filename, title, debug_centers in debug_renders:
            try:
                MatplotlibManager(figsize=(8, 8), dpi=180).render_footprint_layout(
                    RenderFootprintLayoutRequest(
                        object_ids=debug_object_ids,
                        centers=debug_centers,
                        xy_sizes=xy_sizes,
                        output_path=debug_dir / filename,
                        title=title,
                    )
                )
            except Exception as exc:
                log_warning(
                    f"text clutter debug render failed file={filename} error={exc}"
                )

        # Apply layout positions to centred scenes
        laid_out_scenes: list[tuple[str, Any]] = []
        for oid, scene in object_scenes:
            target_xy = target_centers[oid]
            settled_mesh = GeometryManager.scene_to_mesh(scene, trimesh=trimesh)
            settled_bounds = np.asarray(settled_mesh.bounds, dtype=np.float64)
            current_xy = GeometryManager.xy_aabb_center(settled_bounds)
            placement = np.eye(4, dtype=np.float64)
            placement[:3, 3] = [
                float(target_xy[0] - current_xy[0]),
                float(target_xy[1] - current_xy[1]),
                0.0,
            ]
            laid_out_scene = GeometryManager.copy_scene_with_transform(
                scene,
                placement,
            )
            laid_out_scenes.append((oid, laid_out_scene))

            # Export laid-out GLB (replaces the origin-centred one)
            laid_out_glb_path = output_dir / f"{oid}_laid_out.glb"
            GeometryManager.copy_scene_with_transform(
                laid_out_scene,
                z_to_y,
            ).export(laid_out_glb_path)

            # Update per-object metadata with layout position
            for item in settled_objects:
                if item.get("id") == oid:
                    item["layout_target_xy"] = target_xy.tolist()
                    item["layout_placement_transform"] = placement.tolist()
                    item["laid_out_glb_path"] = relative_path(
                        str(laid_out_glb_path), output_root
                    )
                    laid_out_bounds = np.asarray(
                        GeometryManager.scene_to_mesh(
                            laid_out_scene,
                            trimesh=trimesh,
                        ).bounds,
                        dtype=np.float64,
                    )
                    item["laid_out_xy_size_cm"] = (
                        GeometryManager.xy_aabb_size(laid_out_bounds) * 100.0
                    ).tolist()
                    break

        object_scenes = laid_out_scenes

    clutter_2d_aabb_cm = LayoutManager.object_scenes_xy_aabb_manifest(
        object_scenes=object_scenes,
        trimesh=trimesh,
        unit_scale=100.0,
        unit="cm",
    )

    debug_manifest = {
        "status": "ok",
        "output_dir": relative_path(str(output_dir), output_root),
        "object_count": len(objects),
        "settled_count": len(object_scenes),
        "clutter_2d_aabb_cm": clutter_2d_aabb_cm,
        "debug_image_before_path": (
            relative_path(
                str(debug_dir / "footprint_layout_xy_before.png"),
                output_root,
            )
            if object_scenes
            else ""
        ),
        "debug_image_grid_init_path": (
            relative_path(
                str(debug_dir / "footprint_layout_xy_grid_init.png"),
                output_root,
            )
            if object_scenes
            else ""
        ),
        "debug_image_after_path": (
            relative_path(
                str(debug_dir / "footprint_layout_xy_after.png"),
                output_root,
            )
            if object_scenes
            else ""
        ),
        "layout_optimization": layout_result["metadata"] if layout_result else None,
        "objects": settled_objects,
    }
    debug_manifest_path = output_dir / "debug" / "settle_diagnostics.json"
    write_json(debug_manifest_path, debug_manifest)

    # Keep workflow state limited to the contract consumed by table fitting.
    workflow_objects = [
        {
            key: item[key]
            for key in (
                "id",
                "name",
                "status",
                "reason",
                "settled_glb_path",
                "laid_out_glb_path",
            )
            if key in item
        }
        for item in settled_objects
    ]
    return {
        "status": "ok",
        "clutter_2d_aabb_cm": clutter_2d_aabb_cm,
        "objects": workflow_objects,
        "debug_manifest_path": relative_path(str(debug_manifest_path), output_root),
    }


def _resolve_generated_path(value: Any, output_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()
