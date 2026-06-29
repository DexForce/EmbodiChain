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

from typing import Any

import numpy as np
from scipy.optimize import minimize

from embodichain.gen_sim.prompt2scene.agent_tools.managers.optimization_manager import (
    _center_xy_aabb_layout,
    _footprint_layout_diagnostics,
    _xy_aabb_overlap,
    _xy_union_bounds,
)

__all__ = ["_optimize_text_layout_slp"]

# SLSQP solve options — matching the original example_optimization SA pipeline.
_SLSQP_OPTIONS: dict[str, Any] = {"maxiter": 500, "ftol": 1e-6, "disp": False}

# Objective weights (relations are hard constraints, not in the objective).
_WEIGHTS: dict[str, float] = {
    "seed": 1.0,
    "overlap": 200.0,
    "grid": 100.0,
}


def _optimize_text_layout_slp(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    initial_centers: dict[str, np.ndarray],
    left_of_edges: list[tuple[str, str]],
    front_of_edges: list[tuple[str, str]],
    grid_spring_targets: dict[str, np.ndarray],
    padding_ratio: float,
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
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    relation_pairs = set(left_of_edges + front_of_edges)
    relation_pairs.update((b, a) for a, b in left_of_edges + front_of_edges)
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
            if (object_a, object_b) in relation_pairs:
                current[object_a][axis] += sign * amount
                current[object_b][axis] -= sign * amount
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
