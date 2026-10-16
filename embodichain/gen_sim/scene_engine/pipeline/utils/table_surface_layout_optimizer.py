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
from embodichain.lab.sim.scene.layout import (
    LayoutCfg,
    LayoutProblem,
    PlanarRelation,
    solve_layout,
)

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


__all__ = [
    "TableSurfaceLayoutProblem",
    "TableSurfaceLayoutOptimizerConfig",
    "TableSurfaceLayoutOptimizer",
]


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
    support_uid: str = "table"

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
            support_uid=table.id,
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

    def optimize(self, problem: TableSurfaceLayoutProblem) -> dict[str, list[float]]:
        """Adapt GenSim semantics to the shared arena-frame layout service."""
        half_extents = _asset_half_extents_xy(
            assets_by_id=problem.assets_by_id,
            object_ids=problem.root_ids,
        )
        table_bounds = _bounds_from_points(problem.table_optimization_rect_xy)
        bounds = {
            uid: _table_region_bounds(
                table_bounds=table_bounds,
                table_region=problem.root_table_regions_by_id[uid],
            )
            for uid in problem.root_ids
            if problem.fixed_root_xy_by_id[uid] is None
        }
        solution = solve_layout(
            LayoutProblem(
                support_uid=problem.support_uid,
                seed_xy_by_uid=problem.root_seed_xy_by_id,
                half_extents_xy_by_uid=half_extents,
                bounds_xy_by_uid=bounds,
                fixed_xy_by_uid={
                    uid: xy
                    for uid, xy in problem.fixed_root_xy_by_id.items()
                    if xy is not None
                },
                preferred_uids=tuple(problem.imported_root_ids),
                relations=tuple(
                    PlanarRelation(r.source_id, r.relation, r.target_id)
                    for r in problem.root_relations
                ),
                skip_fixed_collisions=True,
            ),
            LayoutCfg(
                relation_clearance_m=self.config.relation_clearance_m,
                collision_margin_m=self.config.collision_margin_m,
                max_slsqp_iterations=self.config.max_slsqp_iterations,
                slsqp_ftol=self.config.slsqp_ftol,
                max_collision_rounds=self.config.max_collision_rounds,
                max_added_collision_pairs=self.config.max_added_collision_pairs,
                imported_seed_weight=self.config.imported_seed_weight,
                enforce_all_clearances=False,
            ),
        )
        if solution.status != "solved":
            raise ValueError(
                f"Table layout optimization failed ({solution.status}): {solution.message}"
            )
        return {uid: list(xy) for uid, xy in solution.xy_by_uid.items()}


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
