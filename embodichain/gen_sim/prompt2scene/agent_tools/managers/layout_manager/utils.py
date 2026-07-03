# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.


from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager import (
    SimulationManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.schemas import (
    GravityDropRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    GeometryManager,
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

_WEIGHTS: dict[str, float] = {
    "seed": 1.0,
    "overlap": 200.0,
    "grid": 3.0,
}

_SLSQP_OPTIONS: dict[str, Any] = {
    "maxiter": 300,
    "ftol": 1.0e-6,
    "disp": False,
}

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
        np.asarray(
            GeometryManager.scene_to_mesh(scene, trimesh=trimesh).bounds,
            dtype=np.float64,
        )
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
    gravity_settle_mode: str = "physics",
) -> dict[str, Any]:
    sim = SimulationManager(headless=True, sim_device="cpu")
    footprint_items: list[dict[str, Any]] = []
    settled_entries: list[dict[str, Any]] = []
    output_axis_transform = GeometryManager.z_up_to_glb_y_up_transform()
    output_to_internal_transform = np.linalg.inv(output_axis_transform)

    with tempfile.TemporaryDirectory(prefix="p2s_footprint_drop_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for object_id, scene in object_scenes:
            mesh = GeometryManager.scene_to_mesh(scene, trimesh=trimesh)
            mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
            mesh_z_height = max(float(mesh_bounds[1][2] - mesh_bounds[0][2]), 0.0)
            bottom_to_xy_plane_transform = GeometryManager.aabb_bottom_to_xy_plane_transform(
                mesh_bounds
            )
            normalized_scene = GeometryManager.copy_scene_with_transform(
                scene,
                bottom_to_xy_plane_transform,
            )
            normalized_output_scene = GeometryManager.copy_scene_with_transform(
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
                        max_convex_hull_num=16,
                        initial_height=gravity_initial_height,
                        gravity_settle_mode=gravity_settle_mode,
                    )
                )
                gravity_transform = GeometryManager.matrix_from_json(
                    gravity_result.final_pose,
                    name=f"{object_id}.gravity_final_pose",
                )
            except Exception:
                gravity_status = "failed"
                gravity_reason = traceback.format_exc()

            settled_origin_scene = GeometryManager.copy_scene_with_transform(
                normalized_scene,
                gravity_transform,
            )
            settled_mesh = GeometryManager.scene_to_mesh(
                settled_origin_scene,
                trimesh=trimesh,
            )
            settled_bounds = np.asarray(settled_mesh.bounds, dtype=np.float64)
            settled_xy_center = GeometryManager.xy_aabb_center(settled_bounds)
            settled_xy_size = GeometryManager.xy_aabb_size(settled_bounds)
            settled_entries.append(
                {
                    "id": object_id,
                    "scene": scene,
                    "bottom_to_xy_plane_transform": bottom_to_xy_plane_transform,
                    "mesh_z_height": mesh_z_height,
                    "gravity_initial_height": gravity_initial_height,
                    "gravity_settle_mode": gravity_settle_mode,
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
            str(entry["id"]): GeometryManager.xy_aabb_center(
                GeometryManager.scene_to_mesh(
                    entry["scene"],
                    trimesh=trimesh,
                ).bounds
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
        packed_scene = GeometryManager.copy_scene_with_transform(
            entry["scene"],
            object_transform,
        )
        packed_object_scenes.append((object_id, packed_scene))
        object_layout_transforms[object_id] = object_transform

        packed_bounds = np.asarray(
            GeometryManager.scene_to_mesh(packed_scene, trimesh=trimesh).bounds,
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
                "gravity_settle_mode": entry["gravity_settle_mode"],
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
        "gravity_settle_mode": gravity_settle_mode,
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
#     http://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,



from typing import Any

import numpy as np

__all__: list[str] = []

def _transitive_closure(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Floyd–Warshall transitive closure over a small set of nodes."""
    if not nodes or not edges:
        return list(edges)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = [[False] * n for _ in range(n)]
    for src, dst in edges:
        if src in idx and dst in idx:
            adj[idx[src]][idx[dst]] = True
    for k in range(n):
        for i in range(n):
            if adj[i][k]:
                row_k = adj[k]
                row_i = adj[i]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True
    closed: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                closed.append((nodes[i], nodes[j]))
    return closed



def _longest_path_ranks(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, int]:
    """Assign integer ranks satisfying ``(A,B)`` → rank[A] < rank[B].

    Uses topological sort + longest-path DP.  Returns a rank dict for every
    node in *nodes* (default 0 for isolated nodes).
    """
    ranks: dict[str, int] = {n: 0 for n in nodes}
    if not edges:
        return ranks
    # Build adjacency and in-degree
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    in_deg: dict[str, int] = {n: 0 for n in nodes}
    present = set(nodes)
    for src, dst in edges:
        if src not in present or dst not in present:
            continue
        adj[src].append(dst)
        in_deg[dst] += 1
    # Kahn topological sort
    queue = [n for n in nodes if in_deg[n] == 0]
    order: list[str] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    # Longest path
    for u in order:
        for v in adj[u]:
            if ranks[v] < ranks[u] + 1:
                ranks[v] = ranks[u] + 1
    # Remaining nodes (cycles / isolated) keep rank 0
    return ranks



def _layout_text_objects_grid(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    spatial_relations: list[dict[str, Any]],
    table_constraints: list[dict[str, Any]] | None = None,
    grid_spacing: float = 0.02,
    padding_ratio: float = 0.08,
) -> dict[str, Any]:
    """Lay out text-scene objects — transitive closure + longest-path ranks.

    1. Transitive closure of left_of / front_of.
    2. Pick centre: explicit 9‑grid ʻcenterʼ, else highest-degree node.
    3. Longest-path rank assignment (left_of→X, front_of→Y).
    4. Shift 9‑grid anchors to their grid positions.
    5. Free objects auto‑wrap below.
    6. Convert ranks→XY using per‑column/row max sizes + gaps.
    7. SA point optimisation + mesh AABB collision cleanup.
    """
    if not object_ids:
        return {
            "centers": {},
            "initial_centers": {},
            "metadata": {
                "method": "transitive_closure_longest_path_with_9grid",
                "iterations": 0,
            },
        }

    # Parse spatial relations.
    left_of_edges: list[tuple[str, str]] = []
    front_of_edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in spatial_relations:
        subject = str(rel.get("subject") or "")
        obj = str(rel.get("object") or "")
        relation = str(rel.get("relation") or "")
        if not subject or not obj or subject == obj:
            continue
        key = (subject, relation, obj)
        if key in seen:
            continue
        seen.add(key)
        if relation == "left_of":
            left_of_edges.append((subject, obj))
        elif relation == "front_of":
            front_of_edges.append((subject, obj))

    # Compute transitive closures.
    left_of_closed = _transitive_closure(object_ids, left_of_edges)
    front_of_closed = _transitive_closure(object_ids, front_of_edges)

    # Parse nine-grid constraints.
    # −Y = front, so front row = 0, back row = 2
    _GRID_TO_RC: dict[str, tuple[int, int]] = {
        "left_front": (0, 0), "center_front": (1, 0), "right_front": (2, 0),
        "left_center": (0, 1), "center": (1, 1), "right_center": (2, 1),
        "left_back": (0, 2), "center_back": (1, 2), "right_back": (2, 2),
        "front": (1, 0), "back": (1, 2),
        "left": (0, 1), "right": (2, 1),
    }
    grid_targets: dict[str, tuple[int, int]] = {}
    for tc in (table_constraints or []):
        asset = str(tc.get("asset") or "")
        grid_name = str(tc.get("grid") or "").strip()
        if asset in object_ids and grid_name in _GRID_TO_RC:
            grid_targets[asset] = _GRID_TO_RC[grid_name]

    # Select a center object when none is explicit.
    auto_center_oid: str | None = None
    has_explicit_center = any(
        tc.get("grid") == "center" for tc in (table_constraints or [])
    )
    if not has_explicit_center:
        # Degree = appearances in left_of + front_of (subject or object)
        degree: dict[str, int] = {oid: 0 for oid in object_ids}
        for src, dst in left_of_closed + front_of_closed:
            if src in degree:
                degree[src] += 1
            if dst in degree:
                degree[dst] += 1
        max_deg = max(degree.values()) if degree else 0
        if max_deg > 0:
            candidates = [oid for oid, d in degree.items() if d == max_deg]
            # Tie-breaker: largest AABB area
            centre_oid = max(
                candidates,
                key=lambda oid: float(xy_sizes[oid][0]) * float(xy_sizes[oid][1]),
            )
            grid_targets[centre_oid] = (1, 1)  # 9‑grid centre
            auto_center_oid = centre_oid

    # Derive ranks from the transitive closures.
    x_rank = _longest_path_ranks(object_ids, left_of_closed)
    # −Y = front:  A front_of B  →  A.y < B.y  →  row[A] < row[B].
    # _longest_path_ranks gives rank[src] < rank[dst]; edges are
    # already (A,B) for "A front_of B", so NO reversal needed.
    y_rank = _longest_path_ranks(object_ids, front_of_closed)

    # Apply nine-grid shifts.
    # Pin 9‑grid objects to their target ranks; shift all connected
    # objects (both upstream and downstream) to preserve topology.
    if grid_targets:
        # Build undirected connected-components via relation edges
        all_edges = left_of_closed + front_of_closed
        neighbours: dict[str, set[str]] = {oid: set() for oid in object_ids}
        for src, dst in all_edges:
            if src in neighbours and dst in neighbours:
                neighbours[src].add(dst)
                neighbours[dst].add(src)
        for oid in grid_targets:
            neighbours.setdefault(oid, set())

        # For each 9‑grid object, BFS the component and shift uniformly
        shifted: set[str] = set()
        for oid, (target_col, target_row) in grid_targets.items():
            if oid in shifted:
                continue
            dx = target_col - x_rank.get(oid, 0)
            dy = target_row - y_rank.get(oid, 0)

            # BFS to collect the full connected component
            component: set[str] = {oid}
            queue = [oid]
            while queue:
                u = queue.pop(0)
                for v in neighbours.get(u, set()):
                    if v not in component:
                        component.add(v)
                        queue.append(v)

            for oid2 in component:
                if oid2 not in grid_targets:  # only shift non‑anchored objects
                    x_rank[oid2] = x_rank.get(oid2, 0) + dx
                    y_rank[oid2] = y_rank.get(oid2, 0) + dy
            shifted.update(component)

    # Propagate row and column alignment.
    # left_of A B  →  same row  (y_rank[A] = y_rank[B])
    # front_of A B →  same col  (x_rank[A] = x_rank[B])
    # Priority (higher wins): 9‑grid > higher degree > larger area.
    _prio = {
        oid: (
            oid in grid_targets,
            sum(1 for e in left_of_closed + front_of_closed if oid in e),
            float(xy_sizes[oid][0]) * float(xy_sizes[oid][1]),
        )
        for oid in object_ids
    }
    for src, dst in left_of_closed:
        if _prio[src] >= _prio[dst]:
            y_rank[dst] = y_rank.get(src, 0)
        else:
            y_rank[src] = y_rank.get(dst, 0)
    for src, dst in front_of_closed:
        if _prio[src] >= _prio[dst]:
            x_rank[dst] = x_rank.get(src, 0)
        else:
            x_rank[src] = x_rank.get(dst, 0)

    # Normalise to >= 0
    min_x = min(x_rank.values()) if x_rank else 0
    min_y = min(y_rank.values()) if y_rank else 0
    for oid in object_ids:
        x_rank[oid] = x_rank.get(oid, 0) - min_x
        y_rank[oid] = y_rank.get(oid, 0) - min_y

    # Resolve cell collisions: spread objects sharing the same (col, row)
    cell_occupants: dict[tuple[int, int], list[str]] = {}
    for oid in object_ids:
        cell = (x_rank[oid], y_rank[oid])
        cell_occupants.setdefault(cell, []).append(oid)
    for (col, row), occupants in cell_occupants.items():
        if len(occupants) > 1:
            for offset, oid in enumerate(occupants[1:], start=1):
                x_rank[oid] = col + offset

    # Place unconstrained objects in wrapped rows.
    constrained = set()
    for src, dst in left_of_closed + front_of_closed:
        constrained.update([src, dst])
    constrained.update(grid_targets)
    free_objects = [oid for oid in object_ids if oid not in constrained]

    if free_objects:
        free_row = max(y_rank.values()) + 1 if y_rank else 0
        # Max row width ≈ existing union width × 1.5 (at least 3 cols)
        col_keys = list(x_rank.values())
        existing_cols = max(col_keys) - min(col_keys) + 1 if col_keys else 1
        max_cols_per_row = max(existing_cols, 3)
        free_sorted = sorted(
            free_objects,
            key=lambda oid: float(xy_sizes[oid][0]),
            reverse=True,
        )
        col = 0
        row_offset = 0
        for oid in free_sorted:
            x_rank[oid] = col
            y_rank[oid] = free_row + row_offset
            col += 1
            if col >= max_cols_per_row:
                col = 0
                row_offset += 1

    # Convert ranks to XY positions.
    col_widths: dict[int, float] = {}
    row_heights: dict[int, float] = {}
    for oid in object_ids:
        c = x_rank[oid]
        r = y_rank[oid]
        col_widths[c] = max(col_widths.get(c, 0.0), float(xy_sizes[oid][0]))
        row_heights[r] = max(row_heights.get(r, 0.0), float(xy_sizes[oid][1]))

    x_cumsum: dict[int, float] = {}
    cumulative = 0.0
    for c in sorted(col_widths):
        x_cumsum[c] = cumulative
        cumulative += col_widths[c] + grid_spacing

    y_cumsum: dict[int, float] = {}
    cumulative = 0.0
    for r in sorted(row_heights):
        y_cumsum[r] = cumulative
        cumulative += row_heights[r] + grid_spacing

    centers: dict[str, np.ndarray] = {}
    for oid in object_ids:
        c = x_rank[oid]
        r = y_rank[oid]
        cx = x_cumsum[c] + 0.5 * float(xy_sizes[oid][0])
        cy = y_cumsum[r] + 0.5 * float(xy_sizes[oid][1])
        centers[oid] = np.array([cx, cy], dtype=np.float64)

    centers = _center_xy_aabb_layout(centers=centers, xy_sizes=xy_sizes)

    initial_centers = {oid: c.copy() for oid, c in centers.items()}

    # Snap initial grid positions as 9‑grid spring targets
    grid_spring_targets: dict[str, np.ndarray] = {
        oid: initial_centers[oid].copy()
        for oid in grid_targets
        if oid in initial_centers
    }

    # Optimize positions and remove mesh AABB collisions.
    optimized = _optimize_text_layout_slp(
        object_ids=object_ids,
        xy_sizes=xy_sizes,
        initial_centers=initial_centers,
        left_of_edges=left_of_closed,
        front_of_edges=front_of_closed,
        grid_spring_targets=grid_spring_targets,
        padding_ratio=padding_ratio,
    )
    centers = optimized["centers"]
    optimization_metadata = optimized["metadata"]

    # Collect layout metadata.
    metadata = {
        "method": "transitive_closure_longest_path_with_9grid_and_sa",
        "grid_spacing": grid_spacing,
        "auto_center_oid": auto_center_oid,
        "has_explicit_center": has_explicit_center,
        "table_constraint_count": len(grid_targets),
        "left_of_count": len(left_of_edges),
        "left_of_closed_count": len(left_of_closed),
        "front_of_count": len(front_of_edges),
        "front_of_closed_count": len(front_of_closed),
        "free_object_count": len(free_objects),
        "x_ranks": {oid: x_rank.get(oid, 0) for oid in object_ids},
        "y_ranks": {oid: y_rank.get(oid, 0) for oid in object_ids},
        "optimization": optimization_metadata,
    }
    return {
        "centers": centers,
        "initial_centers": initial_centers,
        "metadata": metadata,
    }
#     http://www.apache.org/licenses/LICENSE-2.0

def _optimize_text_layout_slp(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    initial_centers: dict[str, np.ndarray],
    left_of_edges: list[tuple[str, str]],
    front_of_edges: list[tuple[str, str]],
    grid_spring_targets: dict[str, np.ndarray],
    padding_ratio: float,
    fixed_object_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Optimize 2D centres with scipy SLSQP, then remove mesh AABB overlap.

    Mirroring the original example_optimization/SA pipeline:
    - left_of / front_of → linear inequality constraints
    - bounding box → variable bounds (2× initial union)
    - seed / overlap / grid → soft penalties in the objective
    - post‑solve collision cleanup on actual footprint AABBs
    """
    if not object_ids:
        return {
            "centers": {},
            "metadata": {
                "method": "text_slsqp_then_mesh_aabb_collision_removal",
                "slsqp_iterations": 0,
                "collision_iterations": 0,
            },
        }

    max_extent = max(
        float(max(xy_sizes[oid][0], xy_sizes[oid][1])) for oid in object_ids
    )
    padding = max(max_extent * padding_ratio, 1e-3)

    initial_centers = {
        oid: np.asarray(initial_centers[oid], dtype=np.float64).copy()
        for oid in object_ids
    }
    fixed_ids = {
        oid for oid in (fixed_object_ids or []) if oid in initial_centers
    }
    initial_union_bounds = _xy_union_bounds(
        centers=initial_centers,
        xy_sizes=xy_sizes,
    )

    index_by_id = {oid: i for i, oid in enumerate(object_ids)}
    x0 = _pack_centers(object_ids, initial_centers)

    # Build linear inequality constraints for left_of and front_of.
    constraints: list[dict[str, Any]] = []
    _build_relation_constraints(
        constraints=constraints,
        object_ids=object_ids,
        index_by_id=index_by_id,
        xy_sizes=xy_sizes,
        left_of_edges=left_of_edges,
        front_of_edges=front_of_edges,
        padding=padding,
    )

    # Bound variables to twice the initial union size.
    init_size = initial_union_bounds[1] - initial_union_bounds[0]
    margin = init_size * 0.5  # 50 % each side → 2× total
    bounds = []
    for oid in object_ids:
        if oid in fixed_ids:
            bounds.append(
                (
                    float(initial_centers[oid][0]),
                    float(initial_centers[oid][0]),
                )
            )
            bounds.append(
                (
                    float(initial_centers[oid][1]),
                    float(initial_centers[oid][1]),
                )
            )
            continue
        bounds.append(
            (
                float(initial_union_bounds[0, 0] - margin[0]),
                float(initial_union_bounds[1, 0] + margin[0]),
            )
        )  # x
        bounds.append(
            (
                float(initial_union_bounds[0, 1] - margin[1]),
                float(initial_union_bounds[1, 1] + margin[1]),
            )
        )  # y

    # Define the optimization objective.
    def _objective(xvec: np.ndarray) -> float:
        centers = _unpack_centers(object_ids, xvec)
        loss = 0.0

        # seed: stay close to initial positions
        for oid in object_ids:
            delta = centers[oid] - initial_centers[oid]
            loss += _WEIGHTS["seed"] * float(np.dot(delta, delta))

        # overlap: AABB overlap area penalty
        for i, oid in enumerate(object_ids):
            for other_id in object_ids[i + 1 :]:
                ov = _xy_aabb_overlap(
                    center_a=centers[oid],
                    size_a=xy_sizes[oid],
                    center_b=centers[other_id],
                    size_b=xy_sizes[other_id],
                    padding=padding,
                )
                if ov is not None:
                    loss += _WEIGHTS["overlap"] * float(ov[0] * ov[1])

        # grid: spring toward 9‑grid targets
        for oid, target in grid_spring_targets.items():
            if oid not in centers:
                continue
            delta = centers[oid] - target
            loss += _WEIGHTS["grid"] * float(np.dot(delta, delta))

        return float(loss)

    # Solve the constrained optimization problem.
    slsqp_result: dict[str, Any] = {"success": False, "nit": 0, "message": ""}
    try:
        result = minimize(
            _objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=_SLSQP_OPTIONS,
        )
        slsqp_result = {
            "success": bool(result.success),
            "nit": int(getattr(result, "nit", 0)),
            "message": str(result.message),
            "fun": float(result.fun) if result.fun is not None else None,
        }
        if result.success:
            x_opt = result.x
        else:
            # SLSQP failed — fall back to seed positions
            x_opt = x0.copy()
    except Exception:
        x_opt = x0.copy()
        slsqp_result["message"] = "SLSQP raised an exception; using seed positions."

    centers = _unpack_centers(object_ids, x_opt)
    centers = _center_xy_aabb_layout(centers=centers, xy_sizes=xy_sizes)

    # Remove residual collisions.
    centers, collision_metadata = _remove_mesh_aabb_collisions(
        object_ids=object_ids,
        xy_sizes=xy_sizes,
        centers=centers,
        initial_centers=initial_centers,
        left_of_edges=left_of_edges,
        front_of_edges=front_of_edges,
        padding=padding,
        fixed_object_ids=fixed_ids,
    )
    centers = _center_xy_aabb_layout(centers=centers, xy_sizes=xy_sizes)

    # Collect optimization metadata.
    diagnostics = _footprint_layout_diagnostics(
        object_ids=object_ids,
        centers=centers,
        initial_centers=initial_centers,
        xy_sizes=xy_sizes,
        padding=padding,
        initial_union_bounds=initial_union_bounds,
    )
    metadata: dict[str, Any] = {
        "method": "text_slsqp_then_mesh_aabb_collision_removal",
        "relation_usage": "left_of_front_of_hard_constraints",
        "padding": float(padding),
        "padding_ratio": float(padding_ratio),
        "weights": dict(_WEIGHTS),
        "fixed_object_ids": sorted(fixed_ids),
        "slsqp": slsqp_result,
        "bounds_expansion": 2.0,
        "initial_union_size": init_size.tolist(),
        **collision_metadata,
        "final_centers": {
            oid: centers[oid].tolist() for oid in object_ids
        },
        **diagnostics,
    }
    return {"centers": centers, "metadata": metadata}


# Build relation constraints.


def _build_relation_constraints(
    *,
    constraints: list[dict[str, Any]],
    object_ids: list[str],
    index_by_id: dict[str, int],
    xy_sizes: dict[str, np.ndarray],
    left_of_edges: list[tuple[str, str]],
    front_of_edges: list[tuple[str, str]],
    padding: float,
) -> None:
    """Append SLSQP inequality constraints for left_of / front_of edges."""

    for subject, obj in left_of_edges:
        if subject not in index_by_id or obj not in index_by_id:
            continue
        i_a = index_by_id[subject]
        i_b = index_by_id[obj]
        # A.x + gap ≤ B.x  →  B.x - A.x - gap ≥ 0
        gap = (
            0.5 * float(xy_sizes[subject][0])
            + 0.5 * float(xy_sizes[obj][0])
            + padding
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, ia=i_a, ib=i_b, g=gap: float(
                    x[2 * ib] - x[2 * ia] - g
                ),
            }
        )

    for subject, obj in front_of_edges:
        if subject not in index_by_id or obj not in index_by_id:
            continue
        i_a = index_by_id[subject]
        i_b = index_by_id[obj]
        # A.y + gap ≤ B.y  →  B.y - A.y - gap ≥ 0
        gap = (
            0.5 * float(xy_sizes[subject][1])
            + 0.5 * float(xy_sizes[obj][1])
            + padding
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, ia=i_a, ib=i_b, g=gap: float(
                    x[2 * ib + 1] - x[2 * ia + 1] - g
                ),
            }
        )


# Remove AABB collisions.


def _remove_mesh_aabb_collisions(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    initial_centers: dict[str, np.ndarray],
    left_of_edges: list[tuple[str, str]],
    front_of_edges: list[tuple[str, str]],
    padding: float,
    fixed_object_ids: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    relation_pairs = set(left_of_edges + front_of_edges)
    relation_pairs.update((b, a) for a, b in left_of_edges + front_of_edges)
    fixed_ids = set(fixed_object_ids or set())
    current = {
        oid: np.asarray(center, dtype=np.float64).copy()
        for oid, center in centers.items()
    }
    max_rounds = 80
    total_pushes = 0
    last_overlap_count = 0

    for iteration in range(max_rounds):
        overlaps = _mesh_aabb_collision_pairs(
            object_ids=object_ids,
            xy_sizes=xy_sizes,
            centers=current,
            padding=padding,
        )
        last_overlap_count = len(overlaps)
        if not overlaps:
            return current, {
                "collision_iterations": iteration,
                "collision_pushes": total_pushes,
                "collision_remaining": 0,
                "collision_removal": "iterative_mesh_aabb_push",
            }
        for item in overlaps:
            object_a = item["object"]
            object_b = item["other"]
            axis = int(item["axis"])
            sign = -1.0 if current[object_a][axis] <= current[object_b][axis] else 1.0
            amount = 0.5 * (float(item["overlap"]) + 1.0e-6)
            a_fixed = object_a in fixed_ids
            b_fixed = object_b in fixed_ids
            if a_fixed and b_fixed:
                continue
            if (object_a, object_b) in relation_pairs:
                if a_fixed:
                    current[object_b][axis] -= sign * amount * 2.0
                elif b_fixed:
                    current[object_a][axis] += sign * amount * 2.0
                else:
                    current[object_a][axis] += sign * amount
                    current[object_b][axis] -= sign * amount
            elif a_fixed:
                current[object_b][axis] -= sign * amount * 2.0
            elif b_fixed:
                current[object_a][axis] += sign * amount * 2.0
            else:
                drift_a = np.linalg.norm(
                    current[object_a] - initial_centers[object_a]
                )
                drift_b = np.linalg.norm(
                    current[object_b] - initial_centers[object_b]
                )
                if drift_a <= drift_b:
                    current[object_a][axis] += sign * amount * 1.25
                    current[object_b][axis] -= sign * amount * 0.75
                else:
                    current[object_a][axis] += sign * amount * 0.75
                    current[object_b][axis] -= sign * amount * 1.25
            total_pushes += 1
        current = _center_xy_aabb_layout(centers=current, xy_sizes=xy_sizes)

    return current, {
        "collision_iterations": max_rounds,
        "collision_pushes": total_pushes,
        "collision_remaining": last_overlap_count,
        "collision_removal": "iterative_mesh_aabb_push",
    }


def _mesh_aabb_collision_pairs(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    padding: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for i, oid in enumerate(object_ids):
        for other_id in object_ids[i + 1 :]:
            ov = _xy_aabb_overlap(
                center_a=centers[oid],
                size_a=xy_sizes[oid],
                center_b=centers[other_id],
                size_b=xy_sizes[other_id],
                padding=padding,
            )
            if ov is None:
                continue
            axis = 0 if ov[0] <= ov[1] else 1
            pairs.append(
                {
                    "object": oid,
                    "other": other_id,
                    "axis": axis,
                    "overlap": float(ov[axis]),
                    "overlap_x": float(ov[0]),
                    "overlap_y": float(ov[1]),
                }
            )
    pairs.sort(key=lambda item: item["overlap"], reverse=True)
    return pairs


# Pack and unpack center coordinates.


def _pack_centers(
    object_ids: list[str],
    centers: dict[str, np.ndarray],
) -> np.ndarray:
    values: list[float] = []
    for oid in object_ids:
        c = np.asarray(centers[oid], dtype=np.float64)
        values.extend([float(c[0]), float(c[1])])
    return np.asarray(values, dtype=np.float64)


def _unpack_centers(
    object_ids: list[str],
    xvec: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        oid: np.asarray(
            [xvec[2 * i], xvec[2 * i + 1]],
            dtype=np.float64,
        )
        for i, oid in enumerate(object_ids)
    }
