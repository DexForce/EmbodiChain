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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraphRelation
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    load_scene_object_z_up_mesh,
)

if TYPE_CHECKING:
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_constructor import (
        SceneLayoutGroup,
        SceneLayoutProblem,
    )


@dataclass
class TableSurfaceLayoutProblem:
    """All scene-graph and geometry inputs for one table-surface solve."""

    assets_by_id: dict[str, SceneObject]
    root_ids: list[str]
    root_seed_xy_by_id: dict[str, list[float]]
    imported_root_ids: set[str]
    fixed_root_xy_by_id: dict[str, list[float] | None]
    root_table_regions_by_id: dict[str, str | None]
    table_optimization_rect_xy: list[list[float]]
    root_relations: list[SceneGraphRelation]

    @classmethod
    def from_layout_problem(
        cls,
        *,
        layout_problem: SceneLayoutProblem,
        group: SceneLayoutGroup,
        current_xy_by_id: dict[str, list[float] | None],
    ) -> TableSurfaceLayoutProblem:
        """Build one table-surface problem without mutating layout state."""
        table = layout_problem.post_edit_scene.table
        if table is None:
            raise ValueError("Table group optimization requires a table.")
        if table.support_optimization_rect_xy is None:
            raise ValueError(
                "Table group optimization requires a table support optimization rectangle."
            )
        root_ids = set(group.child_ids)
        nodes_by_id = layout_problem.goal_scene_graph.node_by_id()
        root_seed_xy_by_id = {}
        for root_id in group.child_ids:
            inherited_xy = current_xy_by_id[root_id]
            # New roots begin from the table origin; imported roots retain their seed.
            root_seed_xy_by_id[root_id] = (
                [0.0, 0.0] if inherited_xy is None else list(inherited_xy)
            )
        return cls(
            assets_by_id={
                asset.id: asset for asset in layout_problem.post_edit_scene.assets
            },
            root_ids=group.child_ids,
            root_seed_xy_by_id=root_seed_xy_by_id,
            imported_root_ids={
                root_id
                for root_id in group.child_ids
                if layout_problem.initial_xy_by_id[root_id] is not None
            },
            fixed_root_xy_by_id={
                root_id: (
                    None
                    if root_id in layout_problem.layout_variable_ids
                    else current_xy_by_id[root_id]
                )
                for root_id in group.child_ids
            },
            root_table_regions_by_id={
                root_id: nodes_by_id[root_id].table_region
                for root_id in group.child_ids
            },
            table_optimization_rect_xy=table.support_optimization_rect_xy,
            root_relations=[
                relation
                for relation in layout_problem.goal_scene_graph.relations
                if relation.source_id in root_ids and relation.target_id in root_ids
            ],
        )


@dataclass(frozen=True)
class TableSurfaceLayoutOptimizerConfig:
    """Numerical controls for one direct-table sibling layout solve."""

    relation_clearance_m: float = 0.03
    collision_margin_m: float = 0.02
    max_slsqp_iterations: int = 500
    slsqp_ftol: float = 1e-6
    max_collision_rounds: int = 8
    max_added_collision_pairs: int = 64
    imported_seed_weight: float = 5.0
    min_center_distance_m: float = 0.01
    min_center_distance_weight: float = 0.05

    def __post_init__(self) -> None:
        """Reject invalid controls before assembling table-surface constraints."""
        if self.relation_clearance_m < 0.0:
            raise ValueError("relation_clearance_m must be non-negative.")
        if self.collision_margin_m < 0.0:
            raise ValueError("collision_margin_m must be non-negative.")
        if self.max_slsqp_iterations <= 0:
            raise ValueError("max_slsqp_iterations must be positive.")
        if self.slsqp_ftol <= 0.0:
            raise ValueError("slsqp_ftol must be positive.")
        if self.max_collision_rounds <= 0:
            raise ValueError("max_collision_rounds must be positive.")
        if self.max_added_collision_pairs <= 0:
            raise ValueError("max_added_collision_pairs must be positive.")


class TableSurfaceLayoutOptimizer:
    """Solve direct table children with table, relation, and collision constraints."""

    def __init__(
        self,
        *,
        config: TableSurfaceLayoutOptimizerConfig | None = None,
    ) -> None:
        self.config = (
            config if config is not None else TableSurfaceLayoutOptimizerConfig()
        )

    def optimize(
        self,
        problem: TableSurfaceLayoutProblem,
    ) -> dict[str, list[float]]:
        """Return the table-frame XY centers satisfying this atomic problem."""
        # Measure only this sibling group from the complete scene-asset index.
        root_half_extents_xy = _asset_half_extents_xy(
            assets_by_id=problem.assets_by_id,
            object_ids=problem.root_ids,
        )
        # Equality constraints for fixed roots, and inequality constraints for table-region and planar-relation bounds.
        inequality_constraints, equality_constraints = _build_constraints(
            problem=problem,
            root_half_extents_xy=root_half_extents_xy,
            config=self.config,
        )
        # Solve with the SLSQP optimizer.
        solved_root_xy_by_id = _solve_root_xy(
            root_ids=problem.root_ids,
            root_seed_xy_by_id=problem.root_seed_xy_by_id,
            imported_root_ids=problem.imported_root_ids,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            config=self.config,
        )
        return _refine_root_collisions(
            root_ids=problem.root_ids,
            root_seed_xy_by_id=problem.root_seed_xy_by_id,
            imported_root_ids=problem.imported_root_ids,
            root_half_extents_xy=root_half_extents_xy,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_root_xy_by_id=problem.fixed_root_xy_by_id,
            solved_root_xy_by_id=solved_root_xy_by_id,
            config=self.config,
        )


class _LayoutInfeasibleError(ValueError):
    """Internal marker for an SLSQP failure while testing one collision direction."""


def _build_constraints(
    *,
    problem: TableSurfaceLayoutProblem,
    root_half_extents_xy: dict[str, np.ndarray],
    config: TableSurfaceLayoutOptimizerConfig,
) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
    """Build variable table-region, planar-relation, and fixed-root constraints."""
    # Objects which need to be optimized.
    root_index = {root_id: index for index, root_id in enumerate(problem.root_ids)}
    table_bounds = _bounds_from_points(problem.table_optimization_rect_xy)
    # Initi constraints.
    inequality_constraints: list[tuple[np.ndarray, float]] = []
    equality_constraints: list[tuple[np.ndarray, float]] = []
    for root_id in problem.root_ids:
        fixed_xy = problem.fixed_root_xy_by_id[root_id]
        if fixed_xy is not None:
            # Imported, unedited roots retain their pose even when partly off-table.
            _append_fixed_root_constraints(
                constraints=equality_constraints,
                root_index=root_index,
                root_id=root_id,
                fixed_xy=fixed_xy,
            )
            continue
        # Only layout variables must keep their complete AABB inside the table region.
        region_bounds = _table_region_bounds(
            table_bounds=table_bounds,
            table_region=problem.root_table_regions_by_id[root_id],
        )
        _append_aabb_center_bounds(
            constraints=inequality_constraints,
            root_index=root_index,
            root_id=root_id,
            bounds=region_bounds,
            half_extents_xy=root_half_extents_xy[root_id],
        )
    for relation in problem.root_relations:
        # Add planar-relation constraints for each sibling relation in this group.
        _append_planar_relation_constraint(
            constraints=inequality_constraints,
            root_index=root_index,
            source_id=relation.source_id,
            relation=relation.relation,
            target_id=relation.target_id,
            source_half_extents_xy=root_half_extents_xy[relation.source_id],
            target_half_extents_xy=root_half_extents_xy[relation.target_id],
            relation_clearance_m=config.relation_clearance_m,
        )
    return inequality_constraints, equality_constraints


def _table_region_bounds(
    *,
    table_bounds: np.ndarray,
    table_region: str | None,
) -> np.ndarray:
    """Return the requested 3x3 table region, with y increasing toward front."""
    if table_region is None:
        return table_bounds.copy()
    column_by_region = {
        "left_back": 0,
        "left_center": 0,
        "left_front": 0,
        "back_center": 1,
        "center": 1,
        "front_center": 1,
        "right_back": 2,
        "right_center": 2,
        "right_front": 2,
    }
    row_by_region = {
        "left_back": 0,
        "back_center": 0,
        "right_back": 0,
        "left_center": 1,
        "center": 1,
        "right_center": 1,
        "left_front": 2,
        "front_center": 2,
        "right_front": 2,
    }
    if table_region not in column_by_region:
        raise ValueError(f"Unsupported table region {table_region!r}.")
    minimum, maximum = table_bounds
    # 9-grid.
    cell_size = (maximum - minimum) / 3.0
    region_minimum = minimum + cell_size * np.array(
        [column_by_region[table_region], row_by_region[table_region]]
    )
    return np.stack([region_minimum, region_minimum + cell_size])


def _asset_half_extents_xy(
    *, assets_by_id: dict[str, SceneObject], object_ids: list[str]
) -> dict[str, np.ndarray]:
    """Measure each optimized asset's oriented z-up XY half-extents."""
    result = {}
    for object_id in object_ids:
        asset = assets_by_id.get(object_id)
        if asset is None:
            raise ValueError(f"Table root {object_id!r} is not an asset.")
        mesh = load_scene_object_z_up_mesh(scene_object=asset)
        result[object_id] = (mesh.bounds[1, :2] - mesh.bounds[0, :2]) / 2.0
    return result


def _bounds_from_points(points: list[list[float]]) -> np.ndarray:
    coordinates = np.asarray(points, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("XY bounds must contain at least two finite points.")
    return np.stack([coordinates.min(axis=0), coordinates.max(axis=0)])


def _append_aabb_center_bounds(
    *,
    constraints: list[tuple[np.ndarray, float]],
    root_index: dict[str, int],
    root_id: str,
    bounds: np.ndarray,
    half_extents_xy: np.ndarray,
) -> None:
    minimum, maximum = bounds[0] + half_extents_xy, bounds[1] - half_extents_xy
    if np.any(minimum > maximum):
        raise ValueError(
            f"Asset {root_id!r} cannot fit inside its assigned table region."
        )
    # root_id is the sibling whose center is constrained in this AABB bound.
    offset, count = 2 * root_index[root_id], 2 * len(root_index)
    # offset selects this root's XY pair; count is the full flattened XY vector size.
    for axis in range(2):
        upper, lower = np.zeros(count), np.zeros(count)
        upper[offset + axis], lower[offset + axis] = 1.0, -1.0
        constraints.extend(
            [(upper, float(maximum[axis])), (lower, -float(minimum[axis]))]
        )


def _append_fixed_root_constraints(
    *,
    constraints: list[tuple[np.ndarray, float]],
    root_index: dict[str, int],
    root_id: str,
    fixed_xy: list[float],
) -> None:
    offset, count = 2 * root_index[root_id], 2 * len(root_index)
    for axis, coordinate in enumerate(fixed_xy):
        row = np.zeros(count)
        row[offset + axis] = 1.0
        constraints.append((row, float(coordinate)))


def _append_planar_relation_constraint(
    *,
    constraints: list[tuple[np.ndarray, float]],
    root_index: dict[str, int],
    source_id: str,
    relation: str,
    target_id: str,
    source_half_extents_xy: np.ndarray,
    target_half_extents_xy: np.ndarray,
    relation_clearance_m: float,
) -> None:
    axis, sign = {
        "left_of": (0, 1.0),
        "right_of": (0, -1.0),
        "behind": (1, 1.0),
        "in_front_of": (1, -1.0),
    }.get(relation, (None, None))
    if axis is None or source_id not in root_index or target_id not in root_index:
        raise ValueError(f"Unsupported table-root planar relation {relation!r}.")
    row = np.zeros(2 * len(root_index))
    row[2 * root_index[source_id] + axis] = sign
    row[2 * root_index[target_id] + axis] = -sign
    constraints.append(
        (
            row,
            -float(
                source_half_extents_xy[axis]
                + target_half_extents_xy[axis]
                + relation_clearance_m
            ),
        )
    )


def _solve_root_xy(
    *,
    root_ids: list[str],
    root_seed_xy_by_id: dict[str, list[float]],
    imported_root_ids: set[str],
    inequality_constraints: list[tuple[np.ndarray, float]],
    equality_constraints: list[tuple[np.ndarray, float]],
    config: TableSurfaceLayoutOptimizerConfig,
) -> dict[str, list[float]]:
    # Init with XY-seeds.
    initial = np.asarray(
        [root_seed_xy_by_id[root_id] for root_id in root_ids], dtype=float
    )

    def objective(values: np.ndarray) -> float:
        xy = values.reshape(-1, 2)
        loss = 0.0
        for index, root_id in enumerate(root_ids):
            if root_id in imported_root_ids:
                delta = xy[index] - initial[index]
                loss += config.imported_seed_weight * float(delta @ delta)
        return loss

    constraints = [
        {
            "type": "ineq",
            "fun": lambda values, row=row, bound=bound: bound - float(row @ values),
        }
        for row, bound in inequality_constraints
    ] + [
        {
            "type": "eq",
            "fun": lambda values, row=row, bound=bound: float(row @ values) - bound,
        }
        for row, bound in equality_constraints
    ]
    result = minimize(
        objective,
        initial.reshape(-1),
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": config.max_slsqp_iterations,
            "ftol": config.slsqp_ftol,
            "disp": False,
        },
    )
    if not result.success:
        raise _LayoutInfeasibleError(
            f"Table layout optimization failed: {result.message}"
        )
    return {
        root_id: [float(result.x[2 * index]), float(result.x[2 * index + 1])]
        for index, root_id in enumerate(root_ids)
    }


def _refine_root_collisions(
    *,
    root_ids: list[str],
    root_seed_xy_by_id: dict[str, list[float]],
    imported_root_ids: set[str],
    root_half_extents_xy: dict[str, np.ndarray],
    inequality_constraints: list[tuple[np.ndarray, float]],
    equality_constraints: list[tuple[np.ndarray, float]],
    fixed_root_xy_by_id: dict[str, list[float] | None],
    solved_root_xy_by_id: dict[str, list[float]],
    config: TableSurfaceLayoutOptimizerConfig,
) -> dict[str, list[float]]:
    # Get current SLSQP solution.
    current = solved_root_xy_by_id
    seen: set[tuple[str, str]] = set()
    for _ in range(config.max_collision_rounds):
        # Fine overlaps.
        overlaps = [
            pair
            for pair in _root_aabb_overlaps(
                root_ids=root_ids, half_extents=root_half_extents_xy, xy_by_id=current
            )
            if fixed_root_xy_by_id[pair[1]] is None
            or fixed_root_xy_by_id[pair[2]] is None
        ]
        if not overlaps:
            return current
        added = 0
        for _, first_id, second_id in overlaps[: config.max_added_collision_pairs]:
            key = tuple(sorted((first_id, second_id)))
            if key in seen:
                continue
            # Earlier pair updates may already have separated this stale overlap.
            if key not in {
                tuple(sorted((first, second)))
                for _, first, second in _root_aabb_overlaps(
                    root_ids=root_ids,
                    half_extents=root_half_extents_xy,
                    xy_by_id=current,
                )
            }:
                continue
            for separation_constraint in _aabb_separation_constraints(
                root_ids=root_ids,
                first_id=first_id,
                second_id=second_id,
                half_extents=root_half_extents_xy,
                xy_by_id=current,
                margin=config.collision_margin_m,
            ):
                # Keep a candidate only when it is compatible with all hard constraints.
                try:
                    solved_xy_by_id = _solve_root_xy(
                        root_ids=root_ids,
                        root_seed_xy_by_id=current,
                        imported_root_ids=imported_root_ids,
                        inequality_constraints=[
                            *inequality_constraints,
                            separation_constraint,
                        ],
                        equality_constraints=equality_constraints,
                        config=config,
                    )
                except _LayoutInfeasibleError:
                    continue
                inequality_constraints.append(separation_constraint)
                current = solved_xy_by_id
                seen.add(key)
                added += 1
                break
            else:
                raise ValueError(
                    "Table-root AABB pair has no feasible separation direction: "
                    f"{first_id!r}, {second_id!r}."
                )
        if not added:
            break
    raise ValueError("Table-root AABB collisions remain after layout refinement.")


def _root_aabb_overlaps(
    *,
    root_ids: list[str],
    half_extents: dict[str, np.ndarray],
    xy_by_id: dict[str, list[float]],
) -> list[tuple[float, str, str]]:
    """Return all overlapping root pairs with their minimum XY overlap distance."""
    result = []
    for index, first_id in enumerate(root_ids):
        for second_id in root_ids[index + 1 :]:
            overlap = np.minimum(
                np.asarray(xy_by_id[first_id]) + half_extents[first_id],
                np.asarray(xy_by_id[second_id]) + half_extents[second_id],
            ) - np.maximum(
                np.asarray(xy_by_id[first_id]) - half_extents[first_id],
                np.asarray(xy_by_id[second_id]) - half_extents[second_id],
            )
            if np.all(overlap > 1e-9):
                result.append((float(np.min(overlap)), first_id, second_id))
    return sorted(result, reverse=True)


def _aabb_separation_constraints(
    *,
    root_ids: list[str],
    first_id: str,
    second_id: str,
    half_extents: dict[str, np.ndarray],
    xy_by_id: dict[str, list[float]],
    margin: float,
) -> list[tuple[np.ndarray, float]]:
    """Return ordered feasible-direction candidates for one overlapping AABB pair."""
    first, second = np.asarray(xy_by_id[first_id]), np.asarray(xy_by_id[second_id])
    # Positive overlap on both axes means these two center-based AABBs intersect.
    overlap = np.minimum(
        first + half_extents[first_id], second + half_extents[second_id]
    ) - np.maximum(first - half_extents[first_id], second - half_extents[second_id])
    # Try the least-penetrating axis first, but permit order reversal if required.
    axes = np.argsort(overlap)
    constraints = []
    for axis in axes:
        current_order = first[axis] < second[axis] or (
            first[axis] == second[axis] and first_id < second_id
        )
        for first_is_lower in (current_order, not current_order):
            constraints.append(
                _aabb_separation_constraint_for_direction(
                    root_ids=root_ids,
                    first_id=first_id,
                    second_id=second_id,
                    half_extents=half_extents,
                    axis=int(axis),
                    first_is_lower=first_is_lower,
                    margin=margin,
                )
            )
    return constraints


def _aabb_separation_constraint_for_direction(
    *,
    root_ids: list[str],
    first_id: str,
    second_id: str,
    half_extents: dict[str, np.ndarray],
    axis: int,
    first_is_lower: bool,
    margin: float,
) -> tuple[np.ndarray, float]:
    """Return one directed AABB separation inequality on a selected axis."""
    index = {root_id: i for i, root_id in enumerate(root_ids)}
    # One row addresses the x/y variable pair of each root in the flattened solver vector.
    row = np.zeros(2 * len(root_ids))
    sign = 1.0 if first_is_lower else -1.0
    row[2 * index[first_id] + axis], row[2 * index[second_id] + axis] = sign, -sign
    # row @ values <= bound keeps the selected AABB faces apart by the requested margin.
    return row, -float(
        half_extents[first_id][axis] + half_extents[second_id][axis] + margin
    )
