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
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager.scene_geometry import (
    _aabb_bottom_to_xy_plane_transform,
    _copy_scene_with_transform,
    _matrix_from_json,
    _scene_to_mesh,
    _xy_aabb_center,
    _xy_aabb_size,
    _z_up_to_glb_y_up_transform,
)
from embodichain.gen_sim.prompt2scene.utils.io import (
    relative_path,
)

__all__ = [
    "_center_xy_aabb_layout",
    "_object_scenes_xy_aabb_manifest",
    "_settle_and_pack_object_footprints",
    "_xy_aabb_overlap",
    "_xy_union_area",
    "_xy_union_bounds",
]

def _object_scenes_xy_aabb_manifest(
    *,
    object_scenes: list[tuple[str, Any]],
    trimesh: Any,
    unit_scale: float,
    unit: str,
) -> dict[str, Any]:
    if not object_scenes:
        return {
            "status": "empty",
            "unit": unit,
            "object_count": 0,
        }
    bounds = [
        np.asarray(_scene_to_mesh(scene, trimesh=trimesh).bounds, dtype=np.float64)
        for _, scene in object_scenes
    ]
    union_bounds = np.vstack(
        [
            np.vstack([item[0] for item in bounds]).min(axis=0),
            np.vstack([item[1] for item in bounds]).max(axis=0),
        ]
    )
    min_xy = union_bounds[0, :2] * unit_scale
    max_xy = union_bounds[1, :2] * unit_scale
    size_xy = max_xy - min_xy
    center_xy = 0.5 * (min_xy + max_xy)
    return {
        "status": "ok",
        "unit": unit,
        "object_count": len(object_scenes),
        "min_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "center_xy": center_xy.tolist(),
        "size_xy": size_xy.tolist(),
        "area": float(size_xy[0] * size_xy[1]),
    }



def _settle_and_pack_object_footprints(
    *,
    object_scenes: list[tuple[str, Any]],
    output_dir: Path,
    output_root: Path,
    trimesh: Any,
) -> dict[str, Any]:
    sim = SimulationManager(headless=True, sim_device="cpu")
    footprint_items: list[dict[str, Any]] = []
    settled_entries: list[dict[str, Any]] = []
    output_axis_transform = _z_up_to_glb_y_up_transform()
    output_to_internal_transform = np.linalg.inv(output_axis_transform)

    with tempfile.TemporaryDirectory(prefix="p2s_footprint_drop_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for object_id, scene in object_scenes:
            mesh = _scene_to_mesh(scene, trimesh=trimesh)
            mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
            mesh_z_height = max(float(mesh_bounds[1][2] - mesh_bounds[0][2]), 0.0)
            bottom_to_xy_plane_transform = _aabb_bottom_to_xy_plane_transform(
                mesh_bounds
            )
            normalized_scene = _copy_scene_with_transform(
                scene,
                bottom_to_xy_plane_transform,
            )
            normalized_output_scene = _copy_scene_with_transform(
                normalized_scene,
                output_axis_transform,
            )
            pre_gravity_path = tmp_path / f"{object_id}_pre_gravity.glb"
            normalized_output_scene.export(pre_gravity_path)
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
                gravity_transform = _matrix_from_json(
                    gravity_result.final_pose,
                    name=f"{object_id}.gravity_final_pose",
                )
            except Exception:
                gravity_status = "failed"
                gravity_reason = traceback.format_exc()

            settled_origin_scene = _copy_scene_with_transform(
                normalized_scene,
                gravity_transform,
            )
            settled_mesh = _scene_to_mesh(settled_origin_scene, trimesh=trimesh)
            settled_bounds = np.asarray(settled_mesh.bounds, dtype=np.float64)
            settled_xy_center = _xy_aabb_center(settled_bounds)
            settled_xy_size = _xy_aabb_size(settled_bounds)
            settled_entries.append(
                {
                    "id": object_id,
                    "scene": scene,
                    "bottom_to_xy_plane_transform": bottom_to_xy_plane_transform,
                    "mesh_z_height": mesh_z_height,
                    "gravity_initial_height": gravity_initial_height,
                    "gravity_transform": gravity_transform,
                    "settled_bounds": settled_bounds,
                    "settled_xy_center": settled_xy_center,
                    "settled_xy_size": settled_xy_size,
                    "gravity_status": gravity_status,
                    "gravity_reason": gravity_reason,
                }
            )

    layout_result = _optimize_xy_aabb_footprint_layout(
        object_ids=[str(entry["id"]) for entry in settled_entries],
        xy_sizes={
            str(entry["id"]): np.asarray(entry["settled_xy_size"], dtype=np.float64)
            for entry in settled_entries
        },
        current_centers={
            str(entry["id"]): _xy_aabb_center(
                _scene_to_mesh(entry["scene"], trimesh=trimesh).bounds
            )
            for entry in settled_entries
        },
    )
    target_centers = layout_result["centers"]

    packed_object_scenes: list[tuple[str, Any]] = []
    object_layout_transforms: dict[str, np.ndarray] = {}
    for entry in settled_entries:
        object_id = str(entry["id"])
        settled_bounds = np.asarray(entry["settled_bounds"], dtype=np.float64)
        target_xy = target_centers[object_id]
        placement_transform = np.eye(4, dtype=np.float64)
        placement_transform[:3, 3] = [
            float(target_xy[0] - entry["settled_xy_center"][0]),
            float(target_xy[1] - entry["settled_xy_center"][1]),
            -float(settled_bounds[0][2]),
        ]
        object_transform = (
            placement_transform
            @ entry["gravity_transform"]
            @ entry["bottom_to_xy_plane_transform"]
        )
        packed_scene = _copy_scene_with_transform(entry["scene"], object_transform)
        packed_object_scenes.append((object_id, packed_scene))
        object_layout_transforms[object_id] = object_transform

        packed_bounds = np.asarray(
            _scene_to_mesh(packed_scene, trimesh=trimesh).bounds,
            dtype=np.float64,
        )
        footprint_items.append(
            {
                "id": object_id,
                "gravity_status": entry["gravity_status"],
                "gravity_reason": entry["gravity_reason"],
                "bottom_to_xy_plane_transform": entry[
                    "bottom_to_xy_plane_transform"
                ].tolist(),
                "mesh_z_height": entry["mesh_z_height"],
                "gravity_initial_height": entry["gravity_initial_height"],
                "gravity_transform": entry["gravity_transform"].tolist(),
                "placement_transform": placement_transform.tolist(),
                "object_layout_transform": object_transform.tolist(),
                "settled_xy_size": entry["settled_xy_size"].tolist(),
                "target_xy_center": target_xy.tolist(),
                "packed_bounds": packed_bounds.tolist(),
            }
        )

    manifest = {
        "status": "ok",
        "method": "per_object_gravity_then_geometry_knn_2d_aabb_relaxation",
        "output_dir": relative_path(str(output_dir), output_root),
        "internal_up_axis": [0.0, 0.0, 1.0],
        "gravity_glb_up_axis": [0.0, 1.0, 0.0],
        "internal_to_gravity_glb_transform": output_axis_transform.tolist(),
        "gravity_glb_to_internal_transform": output_to_internal_transform.tolist(),
        "layout_optimization": layout_result["metadata"],
        "items": footprint_items,
    }
    return {
        "object_scenes": packed_object_scenes,
        "object_layout_transforms": object_layout_transforms,
        "manifest": manifest,
    }



def _optimize_xy_aabb_footprint_layout(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    current_centers: dict[str, np.ndarray],
    padding_ratio: float = 0.08,
) -> dict[str, Any]:
    if not object_ids:
        return {
            "centers": {},
            "metadata": {
                "method": "geometry_knn_2d_aabb_relaxation",
                "iterations": 0,
                "confidence_score": 1.0,
            },
        }

    max_extent = max(
        float(max(xy_sizes[object_id][0], xy_sizes[object_id][1]))
        for object_id in object_ids
    )
    padding = max(max_extent * padding_ratio, 1e-3)
    max_iterations = 300
    overlap_strength = 1.0
    neighbor_strength = 0.04
    compactness_strength = 0.01
    target_expansion_ratio = 1.2
    knn_k = min(3, max(len(object_ids) - 1, 0))
    centers = {
        object_id: np.asarray(
            current_centers.get(object_id, np.zeros(2, dtype=np.float64)),
            dtype=np.float64,
        ).copy()
        for object_id in object_ids
    }
    centers = _center_xy_aabb_layout(
        centers=centers,
        xy_sizes=xy_sizes,
    )
    initial_centers = {
        object_id: center.copy()
        for object_id, center in centers.items()
    }
    initial_union_bounds = _xy_union_bounds(
        centers=initial_centers,
        xy_sizes=xy_sizes,
    )
    neighbor_edges = _knn_neighbor_edges(
        centers=initial_centers,
        k=knn_k,
    )

    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        max_delta = 0.0

        for i, object_id in enumerate(object_ids):
            for other_id in object_ids[i + 1 :]:
                overlap = _xy_aabb_overlap(
                    center_a=centers[object_id],
                    size_a=xy_sizes[object_id],
                    center_b=centers[other_id],
                    size_b=xy_sizes[other_id],
                    padding=padding,
                )
                if overlap is None:
                    continue
                overlap_x, overlap_y = overlap
                if overlap_x <= overlap_y:
                    axis = 0
                    sign = (
                        -1.0
                        if centers[object_id][0] <= centers[other_id][0]
                        else 1.0
                    )
                    amount = overlap_x
                else:
                    axis = 1
                    sign = (
                        -1.0
                        if centers[object_id][1] <= centers[other_id][1]
                        else 1.0
                    )
                    amount = overlap_y
                shift = 0.5 * (amount + 1e-6) * overlap_strength
                centers[object_id][axis] += sign * shift
                centers[other_id][axis] -= sign * shift
                max_delta = max(max_delta, shift)

        for edge in neighbor_edges:
            object_id = edge["object"]
            neighbor_id = edge["neighbor"]
            initial_delta = np.asarray(edge["initial_delta"], dtype=np.float64)
            error = (centers[object_id] - centers[neighbor_id]) - initial_delta
            correction = 0.5 * neighbor_strength * error
            centers[object_id] -= correction
            centers[neighbor_id] += correction
            max_delta = max(max_delta, float(np.linalg.norm(correction)))

        max_delta = max(
            max_delta,
            _apply_compactness_pull(
                centers=centers,
                xy_sizes=xy_sizes,
                initial_union_bounds=initial_union_bounds,
                target_expansion_ratio=target_expansion_ratio,
                strength=compactness_strength,
            ),
        )

        centers = _center_xy_aabb_layout(
            centers=centers,
            xy_sizes=xy_sizes,
        )
        if iteration >= 20 and max_delta < 1e-5:
            break

    diagnostics = _footprint_layout_diagnostics(
        object_ids=object_ids,
        centers=centers,
        initial_centers=initial_centers,
        xy_sizes=xy_sizes,
        padding=padding,
        initial_union_bounds=initial_union_bounds,
    )
    metadata = {
        "method": "geometry_knn_2d_aabb_relaxation",
        "relation_usage": "disabled",
        "iterations": iterations,
        "padding": padding,
        "padding_ratio": padding_ratio,
        "max_iterations": max_iterations,
        "overlap_strength": overlap_strength,
        "neighbor_strength": neighbor_strength,
        "compactness_strength": compactness_strength,
        "target_expansion_ratio": target_expansion_ratio,
        "knn_k": knn_k,
        "neighbor_edges": neighbor_edges,
        "final_centers": {
            object_id: centers[object_id].tolist()
            for object_id in object_ids
        },
        **diagnostics,
    }
    return {"centers": centers, "metadata": metadata}



def _knn_neighbor_edges(
    *,
    centers: dict[str, np.ndarray],
    k: int,
) -> list[dict[str, Any]]:
    if k <= 0 or len(centers) < 2:
        return []
    object_ids = sorted(centers)
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for object_id in object_ids:
        distances = []
        for other_id in object_ids:
            if other_id == object_id:
                continue
            distance = float(np.linalg.norm(centers[object_id] - centers[other_id]))
            distances.append((distance, other_id))
        for _, neighbor_id in sorted(distances)[:k]:
            edge_key = tuple(sorted((object_id, neighbor_id)))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edges.append(
                {
                    "object": object_id,
                    "neighbor": neighbor_id,
                    "initial_delta": (
                        centers[object_id] - centers[neighbor_id]
                    ).tolist(),
                }
            )
    return edges



def _apply_compactness_pull(
    *,
    centers: dict[str, np.ndarray],
    xy_sizes: dict[str, np.ndarray],
    initial_union_bounds: np.ndarray,
    target_expansion_ratio: float,
    strength: float,
) -> float:
    current_bounds = _xy_union_bounds(centers=centers, xy_sizes=xy_sizes)
    expansion_ratio = _xy_union_area(current_bounds) / max(
        _xy_union_area(initial_union_bounds),
        1.0e-12,
    )
    if expansion_ratio <= target_expansion_ratio:
        return 0.0
    excess = min(expansion_ratio / target_expansion_ratio - 1.0, 1.0)
    union_center = 0.5 * (current_bounds[0] + current_bounds[1])
    factor = strength * excess
    max_delta = 0.0
    for object_id, center in centers.items():
        delta = factor * (union_center - center)
        centers[object_id] = center + delta
        max_delta = max(max_delta, float(np.linalg.norm(delta)))
    return max_delta



def _footprint_layout_diagnostics(
    *,
    object_ids: list[str],
    centers: dict[str, np.ndarray],
    initial_centers: dict[str, np.ndarray],
    xy_sizes: dict[str, np.ndarray],
    padding: float,
    initial_union_bounds: np.ndarray,
) -> dict[str, Any]:
    remaining_overlaps = _remaining_xy_overlaps(
        object_ids=object_ids,
        centers=centers,
        xy_sizes=xy_sizes,
        padding=padding,
    )
    displacements = [
        float(np.linalg.norm(centers[object_id] - initial_centers[object_id]))
        for object_id in object_ids
    ]
    current_union_bounds = _xy_union_bounds(centers=centers, xy_sizes=xy_sizes)
    expansion_ratio = _xy_union_area(current_union_bounds) / max(
        _xy_union_area(initial_union_bounds),
        1.0e-12,
    )
    average_displacement = float(np.mean(displacements)) if displacements else 0.0
    max_displacement = float(np.max(displacements)) if displacements else 0.0
    confidence_score = _footprint_confidence_score(
        remaining_overlap_count=len(remaining_overlaps),
        average_displacement=average_displacement,
        max_extent=max(
            float(max(xy_sizes[object_id][0], xy_sizes[object_id][1]))
            for object_id in object_ids
        )
        if object_ids
        else 1.0,
        expansion_ratio=expansion_ratio,
    )
    return {
        "remaining_overlaps": remaining_overlaps,
        "average_displacement": average_displacement,
        "max_displacement": max_displacement,
        "union_aabb_expansion_ratio": expansion_ratio,
        "confidence_score": confidence_score,
    }



def _remaining_xy_overlaps(
    *,
    object_ids: list[str],
    centers: dict[str, np.ndarray],
    xy_sizes: dict[str, np.ndarray],
    padding: float,
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for index, object_id in enumerate(object_ids):
        for other_id in object_ids[index + 1 :]:
            overlap = _xy_aabb_overlap(
                center_a=centers[object_id],
                size_a=xy_sizes[object_id],
                center_b=centers[other_id],
                size_b=xy_sizes[other_id],
                padding=padding,
            )
            if overlap is None:
                continue
            overlaps.append(
                {
                    "object": object_id,
                    "other": other_id,
                    "overlap_x": overlap[0],
                    "overlap_y": overlap[1],
                }
            )
    return overlaps



def _footprint_confidence_score(
    *,
    remaining_overlap_count: int,
    average_displacement: float,
    max_extent: float,
    expansion_ratio: float,
) -> float:
    displacement_scale = max(max_extent, 1.0e-6)
    overlap_penalty = min(0.35 * remaining_overlap_count, 0.7)
    displacement_penalty = min(0.1 * average_displacement / displacement_scale, 0.2)
    expansion_penalty = min(max(expansion_ratio - 1.2, 0.0) * 0.25, 0.2)
    return float(
        np.clip(
            1.0
            - overlap_penalty
            - displacement_penalty
            - expansion_penalty,
            0.0,
            1.0,
        )
    )



def _center_xy_aabb_layout(
    *,
    centers: dict[str, np.ndarray],
    xy_sizes: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not centers:
        return centers
    bounds_min = []
    bounds_max = []
    for object_id, center in centers.items():
        half_size = 0.5 * np.asarray(xy_sizes[object_id], dtype=np.float64)
        bounds_min.append(center - half_size)
        bounds_max.append(center + half_size)
    clutter_center = 0.5 * (
        np.vstack(bounds_min).min(axis=0)
        + np.vstack(bounds_max).max(axis=0)
    )
    return {
        object_id: np.asarray(center, dtype=np.float64) - clutter_center
        for object_id, center in centers.items()
    }



def _xy_union_bounds(
    *,
    centers: dict[str, np.ndarray],
    xy_sizes: dict[str, np.ndarray],
) -> np.ndarray:
    if not centers:
        return np.zeros((2, 2), dtype=np.float64)
    bounds_min = []
    bounds_max = []
    for object_id, center in centers.items():
        half_size = 0.5 * np.asarray(xy_sizes[object_id], dtype=np.float64)
        bounds_min.append(np.asarray(center, dtype=np.float64) - half_size)
        bounds_max.append(np.asarray(center, dtype=np.float64) + half_size)
    return np.vstack(
        [
            np.vstack(bounds_min).min(axis=0),
            np.vstack(bounds_max).max(axis=0),
        ]
    )



def _xy_union_area(bounds: np.ndarray) -> float:
    bounds = np.asarray(bounds, dtype=np.float64)
    size = np.maximum(bounds[1] - bounds[0], 1.0e-9)
    return float(size[0] * size[1])



def _xy_aabb_overlap(
    *,
    center_a: np.ndarray,
    size_a: np.ndarray,
    center_b: np.ndarray,
    size_b: np.ndarray,
    padding: float,
) -> tuple[float, float] | None:
    half_a = 0.5 * np.asarray(size_a, dtype=np.float64)
    half_b = 0.5 * np.asarray(size_b, dtype=np.float64)
    delta = np.abs(
        np.asarray(center_b, dtype=np.float64)
        - np.asarray(center_a, dtype=np.float64)
    )
    overlap = half_a + half_b + padding - delta
    if float(overlap[0]) <= 0.0 or float(overlap[1]) <= 0.0:
        return None
    return float(overlap[0]), float(overlap[1])
