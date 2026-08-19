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
from pathlib import Path

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    TABLE_OBJECT_ID,
    SceneGraph,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer import (
    ParentSurfaceLayoutOptimizer,
    ParentSurfaceLayoutOptimizerConfig,
    ParentSurfaceLayoutProblem,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    translate_scene_object_y_up_by_z_up_delta,
    update_scene_object_y_up_pose_from_z_up_support,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_surface_layout_optimizer import (
    TableSurfaceLayoutOptimizer,
    TableSurfaceLayoutOptimizerConfig,
    TableSurfaceLayoutProblem,
)


@dataclass(frozen=True)
class SceneLayoutGroup:
    """One parent and its direct on-children handled in one layout pass."""

    parent_id: str
    child_ids: list[str]


@dataclass(frozen=True)
class SceneLayoutProblem:
    """Prepared graph-constrained inputs for one scene-layout construction."""

    post_edit_scene: Scene
    goal_scene_graph: SceneGraph
    layout_variable_ids: set[str]
    initial_xy_by_id: dict[str, list[float] | None]
    groups: list[SceneLayoutGroup]


class SceneLayoutConstructor:
    """Construct a scene layout from its goal graph.

    ``formal_scene`` may be empty for text-to-scene. In that case every table
    and asset object must be supplied through ``generated_scene_objects``.
    """

    def __init__(
        self,
        *,
        formal_scene: Scene,
        goal_scene_graph: SceneGraph,
        layout_variable_ids: set[str],
        generated_scene_objects: list[SceneObject],
        output_root: str | Path,
        table_surface_config: TableSurfaceLayoutOptimizerConfig | None = None,
        parent_surface_config: ParentSurfaceLayoutOptimizerConfig | None = None,
    ) -> None:
        self.formal_scene = formal_scene
        self.goal_scene_graph = goal_scene_graph
        self.layout_variable_ids = layout_variable_ids
        self.generated_scene_objects = generated_scene_objects
        self.output_root = Path(output_root).expanduser().resolve()
        # Table surface optimizer.
        self.table_surface_layout_optimizer = TableSurfaceLayoutOptimizer(
            config=table_surface_config
        )
        # Parent surface (on) optimizer.
        self.parent_surface_layout_optimizer = ParentSurfaceLayoutOptimizer(
            config=parent_surface_config
        )
        self._current_xy_by_id: dict[str, list[float] | None] = {}
        self._solved_delta_xy_by_id: dict[str, list[float]] = {}
        self._updated_object_ids: set[str] = set()

    def construct(self) -> Scene:
        """Construct table-root layouts before later stacked-group refinement."""
        # Build layout problem.
        layout_problem = self._build_problem()
        # Get current XY centers.
        self._current_xy_by_id = {
            object_id: list(initial_xy) if initial_xy is not None else None
            for object_id, initial_xy in layout_problem.initial_xy_by_id.items()
        }
        self._solved_delta_xy_by_id = {}
        self._updated_object_ids = set()
        # Check the group.
        if (
            layout_problem.groups
            and layout_problem.groups[0].parent_id != TABLE_OBJECT_ID
        ):
            raise ValueError("The first layout group must be rooted at the table.")

        # Optimize each group in BFS order, propagating solved deltas to descendants.
        for group in layout_problem.groups:
            if group.parent_id == TABLE_OBJECT_ID:
                self._optimize_table_group(
                    layout_problem=layout_problem,
                    group=group,
                )
                continue
            self._optimize_parent_group(
                layout_problem=layout_problem,
                group=group,
            )

        return layout_problem.post_edit_scene

    def _optimize_table_group(
        self,
        *,
        layout_problem: SceneLayoutProblem,
        group: SceneLayoutGroup,
    ) -> None:
        """Optimize all direct on-table children before any stacked child groups."""
        table_surface_problem = TableSurfaceLayoutProblem.from_layout_problem(
            layout_problem=layout_problem,
            group=group,
            current_xy_by_id=self._current_xy_by_id,
        )
        solved_root_xy_by_id = self.table_surface_layout_optimizer.optimize(
            table_surface_problem
        )
        table = layout_problem.post_edit_scene.table
        if table is None:
            raise ValueError("Table group optimization requires a table.")
        # Check the table's z.
        if table.support_surface_z is None and any(
            root_id in layout_problem.layout_variable_ids for root_id in group.child_ids
        ):
            raise ValueError("Table group optimization requires support_surface_z.")
        assets_by_id = {
            asset.id: asset for asset in layout_problem.post_edit_scene.assets
        }
        for root_id, solved_xy in solved_root_xy_by_id.items():
            seed_xy = table_surface_problem.root_seed_xy_by_id[root_id]
            delta_xy = [
                solved_xy[0] - seed_xy[0],
                solved_xy[1] - seed_xy[1],
            ]
            self._current_xy_by_id[root_id] = list(solved_xy)
            self._solved_delta_xy_by_id[root_id] = delta_xy
            if root_id in layout_problem.layout_variable_ids:
                # Direct add/move roots receive a new pose on the table support.
                assert table.support_surface_z is not None
                update_scene_object_y_up_pose_from_z_up_support(
                    scene_object=assets_by_id[root_id],
                    support_region_z=table.support_surface_z,
                    center_xy=solved_xy,
                    clearance_m=0.00, # Directly place on the support surface.
                )
                self._updated_object_ids.add(root_id)
            self._propagate_descendant_delta(
                scene=layout_problem.post_edit_scene,
                root_id=root_id,
                delta_xy=delta_xy,
            )

    def _propagate_descendant_delta(
        self,
        *,
        scene: Scene,
        root_id: str,
        delta_xy: list[float],
    ) -> None:
        """Move every positioned descendant by one solved ancestor XY delta."""
        # A zero root delta cannot change any descendant pose, so skip the subtree walk.
        if delta_xy == [0.0, 0.0]:
            return
        assets_by_id = {asset.id: asset for asset in scene.assets}
        children_by_parent: dict[str, list[str]] = {}
        for node in self.goal_scene_graph.nodes:
            if node.parent_id is not None:
                children_by_parent.setdefault(node.parent_id, []).append(node.object_id)

        pending = list(children_by_parent.get(root_id, []))
        while pending:
            descendant_id = pending.pop(0)
            descendant_xy = self._current_xy_by_id[descendant_id]
            if descendant_xy is not None:
                self._current_xy_by_id[descendant_id] = [
                    descendant_xy[0] + delta_xy[0],
                    descendant_xy[1] + delta_xy[1],
                ]
                translate_scene_object_y_up_by_z_up_delta(
                    scene_object=assets_by_id[descendant_id],
                    delta_xy=delta_xy,
                )
                self._updated_object_ids.add(descendant_id)
            pending.extend(children_by_parent.get(descendant_id, []))

    def _optimize_parent_group(
        self,
        *,
        layout_problem: SceneLayoutProblem,
        group: SceneLayoutGroup,
    ) -> None:
        """Optimize one settled parent's direct on-children in local XY coordinates."""
        parent_surface_problem = ParentSurfaceLayoutProblem.from_layout_problem(
            layout_problem=layout_problem,
            group=group,
            current_xy_by_id=self._current_xy_by_id,
        )
        # Get results.
        solved_child_xy_by_id = self.parent_surface_layout_optimizer.optimize(
            parent_surface_problem
        )
        for child_id, solved_xy in solved_child_xy_by_id.items():
            seed_xy = parent_surface_problem.child_seed_xy_by_id[child_id]
            delta_xy = [
                solved_xy[0] - seed_xy[0],
                solved_xy[1] - seed_xy[1],
            ]
            self._current_xy_by_id[child_id] = list(solved_xy)
            self._solved_delta_xy_by_id[child_id] = delta_xy
            if child_id in layout_problem.layout_variable_ids:
                # Variable children are placed directly above the parent's current top.
                update_scene_object_y_up_pose_from_z_up_support(
                    scene_object=parent_surface_problem.assets_by_id[child_id],
                    support_region_z=parent_surface_problem.parent_top_z,
                    center_xy=solved_xy,
                )
                self._updated_object_ids.add(child_id)
            self._propagate_descendant_delta(
                scene=layout_problem.post_edit_scene,
                root_id=child_id,
                delta_xy=delta_xy,
            )

    def _build_problem(self) -> SceneLayoutProblem:
        """Build post-edit objects and preserve formal-scene centers as seeds."""
        # Validate the graph first.
        self.goal_scene_graph.validate()
        graph_object_ids = set(self.goal_scene_graph.node_by_id())
        generated_objects_by_id = self._generated_scene_objects_by_id()

        # The goal graph removes deleted formal-scene objects from the layout input.
        post_edit_objects = [
            scene_object
            for scene_object in self.formal_scene.objects
            if scene_object.id in graph_object_ids
        ]
        imported_object_ids = {scene_object.id for scene_object in post_edit_objects}
        if imported_object_ids.intersection(generated_objects_by_id):
            raise ValueError(
                "Generated scene objects must not reuse formal scene object ids."
            )
        post_edit_objects.extend(generated_objects_by_id.values())

        post_edit_scene = Scene(objects=post_edit_objects)
        post_edit_object_ids = {
            scene_object.id for scene_object in post_edit_scene.objects
        }
        if post_edit_object_ids != graph_object_ids:
            raise ValueError("Goal scene graph and post-edit scene have different ids.")
        # Get the movable asset ids.
        if not self.layout_variable_ids.issubset(post_edit_object_ids - {"table"}):
            raise ValueError(
                "Only post-edit assets may participate in layout optimization."
            )
        # Get initial XY centers.
        initial_xy_by_id = {
            asset.id: self._initial_xy(  # The assets' center XY should always be updated whenever changes are made.
                asset,
                is_generated=asset.id in generated_objects_by_id,
            )
            for asset in post_edit_scene.assets
        }
        for object_id, initial_xy in initial_xy_by_id.items():
            if initial_xy is None and object_id not in self.layout_variable_ids:
                raise ValueError(
                    f"New asset {object_id!r} must participate in layout optimization."
                )
        # Build the table-rooted BFS groups.
        groups = self._build_groups()

        return SceneLayoutProblem(
            post_edit_scene=post_edit_scene,
            goal_scene_graph=self.goal_scene_graph,
            layout_variable_ids=set(self.layout_variable_ids),
            initial_xy_by_id=initial_xy_by_id,
            groups=groups,
        )

    def _build_groups(self) -> list[SceneLayoutGroup]:
        """Build table-rooted BFS groups of direct on-children."""
        children_by_parent: dict[str, list[str]] = {}
        for node in self.goal_scene_graph.nodes:
            if node.parent_id is None:
                continue
            if node.parent_relation != "on":
                raise ValueError(f"Node {node.object_id!r} must be on its parent.")
            children_by_parent.setdefault(node.parent_id, []).append(node.object_id)

        groups: list[SceneLayoutGroup] = []
        pending = [TABLE_OBJECT_ID]
        while pending:
            parent_id = pending.pop(0)
            child_ids = children_by_parent.get(parent_id, [])
            if not child_ids:
                continue
            groups.append(SceneLayoutGroup(parent_id=parent_id, child_ids=child_ids))
            pending.extend(child_ids)
        return groups

    def _generated_scene_objects_by_id(self) -> dict[str, SceneObject]:
        """Index generated scene objects before merging them into the formal scene."""
        generated_objects_by_id = {
            scene_object.id: scene_object
            for scene_object in self.generated_scene_objects
        }
        if len(generated_objects_by_id) != len(self.generated_scene_objects):
            raise ValueError("Generated scene objects must use unique object ids.")
        return generated_objects_by_id

    @staticmethod
    def _initial_xy(
        asset: SceneObject,
        *,
        is_generated: bool,
    ) -> list[float] | None:
        """Retain formal-scene centers while generated assets await initialization."""
        if is_generated:
            return None
        if asset.center_xy is None or len(asset.center_xy) != 2:
            raise ValueError(
                f"Formal-scene asset {asset.id!r} must have a 2D center_xy."
            )
        return [float(value) for value in asset.center_xy]
