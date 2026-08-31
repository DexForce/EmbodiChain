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

from dataclasses import dataclass, field
from typing import Literal

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneConstraintType,
    SceneGraph,
    TableRegion,
    TABLE_OBJECT_ID,
)

SceneEditOperationType = Literal["add", "move", "delete"]


@dataclass(frozen=True)
class SceneEditOperation:
    """Describe one normalized add, move, or delete operation.

    Attributes:
        op: Operation kind. Add creates a new object, move repositions an
            existing object, and delete removes an existing object.
        object_id: Existing object ID for move/delete, or the generated ID for
            an added object.
        target_id: Optional existing scene object used as the spatial target.
        relation: Spatial relation between the edited object and ``target_id``.
        table_region: Optional named tabletop region. It is valid only when the
            target is the table and the relation is ``"on"``.
        category: Semantic category required for an added object.
        name: Human-readable name required for an added object.
        description: Generation prompt and semantic description required for
            an added object.
        orientation_state: Optional standing or lying intent for an added
            object. Move operations may only preserve the existing state.
    """

    op: SceneEditOperationType
    object_id: str | None = None
    target_id: str | None = None
    relation: SceneConstraintType | None = None
    table_region: TableRegion | None = None
    category: str | None = None
    name: str | None = None
    description: str | None = None
    pose_description: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize one normalized edit operation."""
        return {
            "op": self.op,
            "object_id": self.object_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "table_region": self.table_region,
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "pose_description": self.pose_description,
        }


@dataclass
class SceneEditPlan:
    """Validate edit operations against one pre-edit scene state.

    Construction validates every object reference and rejects conflicting
    operations without mutating the supplied scene or scene graph.

    Attributes:
        scene: Scene state that exists before the edit is applied.
        scene_graph: Pre-edit support and spatial-relation graph. Its node IDs
            must match the scene object IDs.
        operations: Normalized operations in application order.

    Raises:
        ValueError: If scene IDs are inconsistent, an operation has invalid
            fields or references, edits conflict, or a deletion would orphan a
            support descendant.
    """

    scene: Scene
    scene_graph: SceneGraph
    operations: list[SceneEditOperation] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the plan before later stages prepare assets or edit layouts."""
        self.validate()

    def to_dict(self) -> dict[str, object]:
        """Serialize the input scene state and normalized edit operations."""
        return {
            "scene": self.scene.to_dict(),
            "scene_graph": self.scene_graph.to_dict(),
            "operations": [operation.to_dict() for operation in self.operations],
        }

    def validate(self) -> None:
        """Validate object references and edit conflicts against the input scene."""
        # Edit-plan rules:
        # - move and delete identify one existing non-table object with object_id.
        # - add carries generated object_id plus non-empty category, name, and description.
        # - add and move may carry an explicit self-pose description.
        # - move always supplies target_id and relation; add may omit both.
        # - table_region is only valid with target_id=table and relation=on.
        # - target_id and relation are otherwise supplied together or both absent.
        # - every target is from the pre-edit scene; new and deleted objects are invalid targets.
        # - an existing object has at most one move or delete operation in one plan.
        # - delete carries no placement or new-object metadata and must delete every descendant.
        # - these checks validate intent only; they do not mutate Scene or SceneGraph.
        # Scene object IDs must remain a one-to-one lookup key for edit operations.
        scene_object_ids = {scene_object.id for scene_object in self.scene.objects}
        if len(scene_object_ids) != len(self.scene.objects):
            raise ValueError("Scene edit input must contain unique object ids.")
        # Parent and child checks require the graph to describe this exact scene.
        if set(self.scene_graph.node_by_id()) != scene_object_ids:
            raise ValueError("Scene edit plan graph nodes must match scene object ids.")

        existing_object_ids = set(scene_object_ids)
        added_object_ids: set[str] = set()
        # Collect deletion intents first so move and add cannot target them regardless of order.
        deleted_object_ids = {
            operation.object_id
            for operation in self.operations
            if operation.op == "delete"
        }
        if None in deleted_object_ids:
            raise ValueError("Delete operations must identify an existing object.")

        edited_object_ids: set[str] = set()
        # Validate each operation against the unchanged input scene and graph.
        for operation in self.operations:
            self._validate_operation(
                operation=operation,
                existing_object_ids=existing_object_ids,
                deleted_object_ids=deleted_object_ids,
                edited_object_ids=edited_object_ids,
                added_object_ids=added_object_ids,
            )
        # A removed support object must not leave any child objects orphaned.
        self._validate_deleted_subtrees(deleted_object_ids)

    def _validate_operation(
        self,
        *,
        operation: SceneEditOperation,
        existing_object_ids: set[str],
        deleted_object_ids: set[str],
        edited_object_ids: set[str],
        added_object_ids: set[str],
    ) -> None:
        if operation.op == "add":
            self._validate_add_operation(
                operation,
                existing_object_ids,
                deleted_object_ids,
                added_object_ids,
            )
            return
        if operation.op not in {"move", "delete"}:
            raise ValueError(f"Unsupported scene edit operation: {operation.op!r}")
        if operation.object_id not in existing_object_ids:
            raise ValueError(
                "Move and delete operations must reference existing objects."
            )
        if operation.object_id == TABLE_OBJECT_ID:
            raise ValueError("The table cannot be moved or deleted.")
        # Existing objects accept only one move or delete instruction per plan.
        if operation.object_id in edited_object_ids:
            raise ValueError("An existing object may have only one edit operation.")
        edited_object_ids.add(operation.object_id)

        if operation.op == "delete":
            # Delete carries no new metadata or spatial placement.
            if any(
                value is not None
                for value in (
                    operation.target_id,
                    operation.relation,
                    operation.table_region,
                    operation.category,
                    operation.name,
                    operation.description,
                    operation.pose_description,
                )
            ):
                raise ValueError("Delete operations may only specify object_id.")
            return

        if operation.target_id is None or operation.relation is None:
            raise ValueError("Move operations must specify target_id and relation.")
        if operation.pose_description is not None:
            self._validate_pose_description(operation.pose_description)
        self._validate_position_reference(
            operation=operation,
            existing_object_ids=existing_object_ids,
            deleted_object_ids=deleted_object_ids,
        )
        if any(
            value is not None
            for value in (
                operation.category,
                operation.name,
                operation.description,
            )
        ):
            raise ValueError("Move operations must not declare a new object.")

    @staticmethod
    def _validate_add_operation(
        operation: SceneEditOperation,
        existing_object_ids: set[str],
        deleted_object_ids: set[str],
        added_object_ids: set[str],
    ) -> None:
        if not operation.object_id:
            raise ValueError("Add operations must have a generated object_id.")
        # Generated IDs must not collide with the input scene or this add batch.
        if (
            operation.object_id in existing_object_ids
            or operation.object_id in added_object_ids
        ):
            raise ValueError("Add operations must use unique new object ids.")
        added_object_ids.add(operation.object_id)
        if not all(
            isinstance(value, str) and value.strip()
            for value in (operation.category, operation.name, operation.description)
        ):
            raise ValueError("Add operations require category, name, and description.")
        if operation.pose_description is not None:
            SceneEditPlan._validate_pose_description(operation.pose_description)
        SceneEditPlan._validate_position_reference(
            operation=operation,
            existing_object_ids=existing_object_ids,
            deleted_object_ids=deleted_object_ids,
        )

    @staticmethod
    def _validate_pose_description(pose_description: str) -> None:
        if (
            not isinstance(pose_description, str)
            or not pose_description.strip()
            or len(pose_description) > 240
        ):
            raise ValueError("pose_description must be a short non-empty string.")

    @staticmethod
    def _validate_position_reference(
        *,
        operation: SceneEditOperation,
        existing_object_ids: set[str],
        deleted_object_ids: set[str],
    ) -> None:
        if (operation.target_id is None) != (operation.relation is None):
            raise ValueError("target_id and relation must be specified together.")
        if operation.table_region is not None and (
            operation.target_id != TABLE_OBJECT_ID or operation.relation != "on"
        ):
            raise ValueError(
                "table_region requires target_id='table' and relation='on'."
            )
        if operation.target_id is None:
            return
        # Targets come only from the pre-edit scene, so new objects cannot be targets.
        if operation.target_id not in existing_object_ids:
            raise ValueError("Edit targets must reference existing scene objects.")
        # One edit may not position an object relative to a deleted target.
        if operation.target_id in deleted_object_ids:
            raise ValueError("Edit targets must not reference deleted objects.")

    def _validate_deleted_subtrees(self, deleted_object_ids: set[str]) -> None:
        # Index the support graph once before checking every deleted parent.
        children_by_parent: dict[str, list[str]] = {}
        for node in self.scene_graph.nodes:
            if node.parent_id is not None:
                children_by_parent.setdefault(node.parent_id, []).append(node.object_id)

        for object_id in deleted_object_ids:
            descendants = self._descendant_ids(object_id, children_by_parent)
            if not descendants.issubset(deleted_object_ids):
                raise ValueError(
                    "Deleting a parent requires deleting all of its children."
                )

    @staticmethod
    def _descendant_ids(
        object_id: str,
        children_by_parent: dict[str, list[str]],
    ) -> set[str]:
        descendants: set[str] = set()
        # Traverse every support descendant, not only direct children.
        pending = list(children_by_parent.get(object_id, []))
        while pending:
            child_id = pending.pop()
            descendants.add(child_id)
            pending.extend(children_by_parent.get(child_id, []))
        return descendants
