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
from embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler import (
    GravitySettleBody,
    GravitySettler,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    load_scene_object_z_up_mesh,
    measure_scene_object_z_up_world_aabb,
    scene_object_y_up_layout,
)

if TYPE_CHECKING:
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_constructor import (
        SceneLayoutGroup,
        SceneLayoutProblem,
    )


__all__ = [
    "ParentSurfaceLayoutProblem",
    "ParentSurfaceLayoutOptimizerConfig",
    "ParentSurfaceLayoutOptimizer",
]


@dataclass
class ParentSurfaceLayoutProblem:
    """All geometry and layout-state inputs for one parent-surface solve."""

    assets_by_id: dict[str, SceneObject]
    child_ids: list[str]
    child_seed_xy_by_id: dict[str, list[float]]
    imported_child_ids: set[str]
    fixed_child_xy_by_id: dict[str, list[float] | None]
    parent_aabb_xy: list[list[float]]
    parent_top_z: float
    child_relations: list[SceneGraphRelation]
    support_uid: str = "parent"

    @classmethod
    def from_layout_problem(
        cls,
        *,
        layout_problem: SceneLayoutProblem,
        group: SceneLayoutGroup,
        current_xy_by_id: dict[str, list[float] | None],
    ) -> ParentSurfaceLayoutProblem:
        """Build one parent-surface problem without mutating layout state."""
        assets_by_id = {
            asset.id: asset for asset in layout_problem.post_edit_scene.assets
        }
        child_ids = set(group.child_ids)
        parent = assets_by_id.get(group.parent_id)
        if parent is None:
            raise ValueError(f"Parent {group.parent_id!r} is not an asset.")
        parent_aabb = measure_scene_object_z_up_world_aabb(scene_object=parent)
        parent_aabb_xy = [
            [parent_aabb[0][0], parent_aabb[0][1]],
            [parent_aabb[1][0], parent_aabb[1][1]],
        ]
        parent_center_xy = [
            (parent_aabb[0][0] + parent_aabb[1][0]) / 2.0,
            (parent_aabb[0][1] + parent_aabb[1][1]) / 2.0,
        ]
        child_seed_xy_by_id = {}
        for child_id in group.child_ids:
            inherited_xy = current_xy_by_id[child_id]
            # New children begin from the solved parent's AABB center.
            child_seed_xy_by_id[child_id] = (
                parent_center_xy if inherited_xy is None else list(inherited_xy)
            )
        return cls(
            support_uid=parent.id,
            assets_by_id=assets_by_id,
            child_ids=group.child_ids,
            child_seed_xy_by_id=child_seed_xy_by_id,
            imported_child_ids={
                child_id
                for child_id in group.child_ids
                if layout_problem.initial_xy_by_id[child_id] is not None
            },
            fixed_child_xy_by_id={
                child_id: (
                    None
                    if child_id in layout_problem.layout_variable_ids
                    else current_xy_by_id[child_id]
                )
                for child_id in group.child_ids
            },
            parent_aabb_xy=parent_aabb_xy,
            parent_top_z=parent_aabb[1][2],
            child_relations=[
                relation
                for relation in layout_problem.goal_scene_graph.relations
                if relation.source_id in child_ids and relation.target_id in child_ids
            ],
        )


@dataclass(frozen=True)
class ParentSurfaceLayoutOptimizerConfig:
    """Numerical controls for one non-table parent-surface sibling solve."""

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
        """Reject invalid controls before assembling parent-surface constraints."""
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


class ParentSurfaceLayoutOptimizer:
    """Solve direct ``on`` children inside one parent's current XY footprint."""

    def __init__(
        self,
        *,
        config: ParentSurfaceLayoutOptimizerConfig | None = None,
    ) -> None:
        self.config = (
            config if config is not None else ParentSurfaceLayoutOptimizerConfig()
        )

    def optimize(self, problem: ParentSurfaceLayoutProblem) -> dict[str, list[float]]:
        """Adapt GenSim semantics to the shared arena-frame layout service."""
        half_extents = _asset_half_extents_xy(
            assets_by_id=problem.assets_by_id,
            object_ids=problem.child_ids,
        )
        parent_bounds = _bounds_from_points(problem.parent_aabb_xy)
        bounds = {uid: parent_bounds for uid in problem.child_ids}
        solution = solve_layout(
            LayoutProblem(
                support_uid=problem.support_uid,
                seed_xy_by_uid=problem.child_seed_xy_by_id,
                half_extents_xy_by_uid=half_extents,
                bounds_xy_by_uid=bounds,
                fixed_xy_by_uid={
                    uid: xy
                    for uid, xy in problem.fixed_child_xy_by_id.items()
                    if xy is not None
                },
                preferred_uids=tuple(problem.imported_child_ids),
                relations=tuple(
                    PlanarRelation(r.source_id, r.relation, r.target_id)
                    for r in problem.child_relations
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
                f"Parent layout optimization failed ({solution.status}): {solution.message}"
            )
        return {uid: list(xy) for uid, xy in solution.xy_by_uid.items()}

    def settle_dynamic_children(
        self,
        *,
        table: SceneObject,
        parent: SceneObject,
        problem: ParentSurfaceLayoutProblem,
        dynamic_child_ids: set[str],
    ) -> dict[str, dict[str, list[float]]]:
        """Settle variable children against their parent and fixed siblings.

        Children must already have been placed 2cm above ``parent`` by the
        caller. The physical table remains mandatory, while the parent and
        unedited siblings become kinematic collision bodies for this pass.
        """
        child_ids = set(problem.child_ids)
        if not dynamic_child_ids.issubset(child_ids):
            raise ValueError("Only parent-surface children may be dynamic.")
        if not dynamic_child_ids:
            return {}
        participant_ids = {parent.id, *child_ids}
        if parent.id in child_ids:
            raise ValueError("A parent-surface group cannot contain its parent.")
        return GravitySettler(
            table_body=GravitySettleBody(
                scene_object=table,
                y_up_layout=scene_object_y_up_layout(table),
            ),
            participant_bodies=[
                GravitySettleBody(
                    scene_object=problem.assets_by_id[object_id],
                    y_up_layout=scene_object_y_up_layout(
                        problem.assets_by_id[object_id]
                    ),
                )
                for object_id in participant_ids
            ],
            dynamic_asset_ids=dynamic_child_ids,
            static_asset_ids=participant_ids - dynamic_child_ids,
        ).settle()


def _asset_half_extents_xy(
    *, assets_by_id: dict[str, SceneObject], object_ids: list[str]
) -> dict[str, np.ndarray]:
    """Measure each optimized child asset's z-up XY half-extents."""
    result = {}
    for object_id in object_ids:
        asset = assets_by_id.get(object_id)
        if asset is None:
            raise ValueError(f"Parent child {object_id!r} is not an asset.")
        mesh = load_scene_object_z_up_mesh(scene_object=asset)
        result[object_id] = (mesh.bounds[1, :2] - mesh.bounds[0, :2]) / 2.0
    return result


def _bounds_from_points(points: list[list[float]]) -> np.ndarray:
    """Return finite XY minimum and maximum bounds from polygon points."""
    coordinates = np.asarray(points, dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("XY bounds must contain at least two finite points.")
    return np.stack([coordinates.min(axis=0), coordinates.max(axis=0)])
