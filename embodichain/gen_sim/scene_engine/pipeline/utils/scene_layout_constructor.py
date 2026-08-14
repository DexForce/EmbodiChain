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
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_optimizer import (
    SceneLayoutOptimizerConfig,
    SceneLayoutOptimizer,
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
        config: SceneLayoutOptimizerConfig | None = None,
    ) -> None:
        self.formal_scene = formal_scene
        self.goal_scene_graph = goal_scene_graph
        self.layout_variable_ids = layout_variable_ids
        self.generated_scene_objects = generated_scene_objects
        self.output_root = Path(output_root).expanduser().resolve()
        self.layout_optimizer = SceneLayoutOptimizer(config=config)
        self._current_xy_by_id: dict[str, list[float] | None] = {}
        self._solved_delta_xy_by_id: dict[str, list[float]] = {}
        self._updated_object_ids: set[str] = set()

    def construct(self) -> Scene:
        """Construct table-root layouts before later stacked-group refinement."""
        layout_problem = self._build_problem()
        self._current_xy_by_id = {
            object_id: list(initial_xy) if initial_xy is not None else None
            for object_id, initial_xy in layout_problem.initial_xy_by_id.items()
        }
        self._solved_delta_xy_by_id = {}
        self._updated_object_ids = set()
        if (
            layout_problem.groups
            and layout_problem.groups[0].parent_id != TABLE_OBJECT_ID
        ):
            raise ValueError("The first layout group must be rooted at the table.")

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
        table = layout_problem.post_edit_scene.table
        if table is None:
            raise ValueError("Table group optimization requires a table.")
        if table.support_optimization_rect_xy is None:
            raise ValueError(
                "Table group optimization requires a table support optimization rectangle."
            )

        root_ids = set(group.child_ids)
        root_relations = [
            relation
            for relation in layout_problem.goal_scene_graph.relations
            if relation.source_id in root_ids and relation.target_id in root_ids
        ]
        root_seed_xy_by_id: dict[str, list[float]] = {}
        for root_id in group.child_ids:
            inherited_xy = self._current_xy_by_id[root_id]
            # New roots start from the table-local origin; imported roots keep their pose.
            root_seed_xy_by_id[root_id] = (
                [0.0, 0.0] if inherited_xy is None else list(inherited_xy)
            )
            self._current_xy_by_id[root_id] = root_seed_xy_by_id[root_id]

        nodes_by_id = layout_problem.goal_scene_graph.node_by_id()
        solved_root_xy_by_id = self.layout_optimizer.optimize_table_root_xy(
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
                    else self._current_xy_by_id[root_id]
                )
                for root_id in group.child_ids
            },
            root_table_regions_by_id={
                root_id: nodes_by_id[root_id].table_region
                for root_id in group.child_ids
            },
            table_optimization_rect_xy=table.support_optimization_rect_xy,
            root_relations=root_relations,
        )
        if table.support_surface_z is None and any(
            root_id in layout_problem.layout_variable_ids for root_id in group.child_ids
        ):
            raise ValueError("Table group optimization requires support_surface_z.")
        assets_by_id = {
            asset.id: asset for asset in layout_problem.post_edit_scene.assets
        }
        for root_id, solved_xy in solved_root_xy_by_id.items():
            seed_xy = root_seed_xy_by_id[root_id]
            delta_xy = [
                solved_xy[0] - seed_xy[0],
                solved_xy[1] - seed_xy[1],
            ]
            self._current_xy_by_id[root_id] = list(solved_xy)
            self._solved_delta_xy_by_id[root_id] = delta_xy
            if root_id in layout_problem.layout_variable_ids:
                # Direct add/move roots receive a new pose on the table support.
                assert table.support_surface_z is not None
                self.layout_optimizer.update_scene_object_y_up_pose_from_z_up_support(
                    scene_object=assets_by_id[root_id],
                    support_region_z=table.support_surface_z,
                    center_xy=solved_xy,
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
                self.layout_optimizer.translate_scene_object_y_up_by_z_up_delta(
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
        assets_by_id = {
            asset.id: asset for asset in layout_problem.post_edit_scene.assets
        }
        parent = assets_by_id.get(group.parent_id)
        if parent is None:
            raise ValueError(f"Parent {group.parent_id!r} is not an asset.")
        parent_aabb = self.layout_optimizer.scene_object_z_up_world_aabb(
            scene_object=parent
        )
        parent_aabb_xy = [
            [parent_aabb[0][0], parent_aabb[0][1]],
            [parent_aabb[1][0], parent_aabb[1][1]],
        ]
        parent_center_xy = [
            (parent_aabb[0][0] + parent_aabb[1][0]) / 2.0,
            (parent_aabb[0][1] + parent_aabb[1][1]) / 2.0,
        ]
        child_seed_xy_by_id: dict[str, list[float]] = {}
        for child_id in group.child_ids:
            inherited_xy = self._current_xy_by_id[child_id]
            # New children start at their parent's current AABB center.
            child_seed_xy_by_id[child_id] = (
                parent_center_xy if inherited_xy is None else list(inherited_xy)
            )
            self._current_xy_by_id[child_id] = child_seed_xy_by_id[child_id]

        solved_child_xy_by_id = self.layout_optimizer.optimize_parent_child_xy(
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
                    else self._current_xy_by_id[child_id]
                )
                for child_id in group.child_ids
            },
            parent_aabb_xy=parent_aabb_xy,
        )
        parent_top_z = parent_aabb[1][2]
        for child_id, solved_xy in solved_child_xy_by_id.items():
            seed_xy = child_seed_xy_by_id[child_id]
            delta_xy = [
                solved_xy[0] - seed_xy[0],
                solved_xy[1] - seed_xy[1],
            ]
            self._current_xy_by_id[child_id] = list(solved_xy)
            self._solved_delta_xy_by_id[child_id] = delta_xy
            if child_id in layout_problem.layout_variable_ids:
                # Variable children are placed directly above the parent's current top.
                self.layout_optimizer.update_scene_object_y_up_pose_from_z_up_support(
                    scene_object=assets_by_id[child_id],
                    support_region_z=parent_top_z,
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
        if not self.layout_variable_ids.issubset(post_edit_object_ids - {"table"}):
            raise ValueError(
                "Only post-edit assets may participate in layout optimization."
            )

        initial_xy_by_id = {
            asset.id: self._initial_xy(
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

        return SceneLayoutProblem(
            post_edit_scene=post_edit_scene,
            goal_scene_graph=self.goal_scene_graph,
            layout_variable_ids=set(self.layout_variable_ids),
            initial_xy_by_id=initial_xy_by_id,
            groups=self._build_groups(),
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
