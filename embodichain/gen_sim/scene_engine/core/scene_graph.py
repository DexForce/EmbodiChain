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

__all__ = [
    "GENERATED_SCENE_GRAPH_SCHEMA",
    "GeneratedSceneGraph",
    "GeneratedSceneNode",
    "GeneratedSceneRelation",
    "OrientationState",
    "PlanarRelationType",
    "SceneConstraintType",
    "SceneGraph",
    "SceneGraphNode",
    "SceneGraphRelation",
    "SupportRelationType",
    "TABLE_OBJECT_ID",
    "TABLE_REGIONS",
    "TableRegion",
]

GENERATED_SCENE_GRAPH_SCHEMA = "generated_scene_graph/v1"
TABLE_OBJECT_ID = "table"

# Static type constraint for the nine regions of the tabletop 3x3 grid.
TableRegion = Literal[
    "left_back",
    "back_center",
    "right_back",
    "left_center",
    "center",
    "right_center",
    "left_front",
    "front_center",
    "right_front",
]
# Runtime membership set for validating serialized and user-provided regions.
TABLE_REGIONS = frozenset(
    {
        "left_back",
        "back_center",
        "right_back",
        "left_center",
        "center",
        "right_center",
        "left_front",
        "front_center",
        "right_front",
    }
)

# A on B, then B is the parent node of A.
SupportRelationType = Literal["on"]

# A PlanarRelation with B, then A and B must have the same parent node.
PlanarRelationType = Literal["left_of", "right_of", "in_front_of", "behind"]
SceneConstraintType = SupportRelationType | PlanarRelationType
OrientationState = Literal["standing", "lying"]


def _validate_stable_id(value: str, *, field_name: str) -> None:
    """Reject identifiers whose spelling can change during serialization."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, trimmed string.")


@dataclass
class GeneratedSceneNode:
    """One object node in the authoring-only scene hierarchy.

    ``orientation_state`` is an image-derived placement semantic, rather than
    an edge to the node itself or an exact three-dimensional transform.
    """

    object_id: str
    parent_id: str | None
    parent_relation: SupportRelationType | None = None
    table_region: TableRegion | None = None
    # Preserves image-observed placement semantics for later pose refinement.
    orientation_state: OrientationState | None = None

    def __post_init__(self) -> None:
        """Validate local node fields before graph-level checks."""
        _validate_stable_id(self.object_id, field_name="object_id")
        if self.parent_id is not None:
            _validate_stable_id(self.parent_id, field_name="parent_id")
        if self.table_region not in {None, *TABLE_REGIONS}:
            raise ValueError("table_region is invalid.")
        if self.orientation_state not in {None, "standing", "lying"}:
            raise ValueError("orientation_state is invalid.")
        # If the node is the table.
        if self.object_id == TABLE_OBJECT_ID:
            if self.parent_id is not None:
                raise ValueError("table must not have a parent.")
            if self.parent_relation is not None:
                raise ValueError("table must not have a parent relation.")
            if self.orientation_state is not None:
                raise ValueError("table must not have an orientation state.")
        # If the node is not the table.
        elif self.parent_id is None:
            raise ValueError("non-table nodes must have a parent.")
        elif self.parent_relation not in {None, "on"}:
            raise ValueError("non-table nodes must be on their parent.")

    def to_dict(self) -> dict[str, object]:
        """Serialize this node for scene graph debugging artifacts."""
        return {
            "object_id": self.object_id,
            "parent_id": self.parent_id,
            "parent_relation": self.parent_relation,
            "table_region": self.table_region,
            "orientation_state": self.orientation_state,
        }


@dataclass
class GeneratedSceneRelation:
    """One authoring-time spatial relation between two non-table objects."""

    source_id: str
    relation: PlanarRelationType
    target_id: str

    def __post_init__(self) -> None:
        """Validate local relation fields before graph-level checks."""
        _validate_stable_id(self.source_id, field_name="source_id")
        _validate_stable_id(self.target_id, field_name="target_id")
        if self.source_id == self.target_id:
            raise ValueError("relation endpoints must be different.")

    def to_dict(self) -> dict[str, object]:
        """Serialize this planar relation for scene graph debugging artifacts."""
        return {
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
        }


@dataclass
class GeneratedSceneGraph:
    """Authoring graph for generation and editing, never live scene state.

    The graph contains provider-free identities and image-derived spatial
    semantics. It deliberately has no simulator handles, pose readers, or
    registration behavior; a later integration layer converts it to the
    canonical runtime scene contracts.
    """

    nodes: list[GeneratedSceneNode] = field(default_factory=list)
    relations: list[GeneratedSceneRelation] = field(default_factory=list)
    validate_on_refresh: bool = True  # Validate after each automatic refresh.

    def __post_init__(self) -> None:
        """Normalize new graphs so downstream stages see canonical constraints."""
        self.refresh()

    def refresh(self) -> None:
        """Normalize the graph and optionally validate semantic constraints."""
        # First normalize then validate (if applicable).
        self.normalize()
        if self.validate_on_refresh:
            self.validate()

    def node_by_id(self) -> dict[str, GeneratedSceneNode]:
        """Return nodes keyed by object id, raising on duplicate ids."""
        nodes_by_id: dict[str, GeneratedSceneNode] = {}
        for node in self.nodes:
            if node.object_id in nodes_by_id:
                raise ValueError(f"Duplicate scene graph node: {node.object_id}")
            nodes_by_id[node.object_id] = node
        return nodes_by_id

    def remove_nodes(self, object_ids: set[str]) -> None:
        """Remove nodes and their incident planar relations, then validate."""
        # If no node to be removed, return directly.
        if not object_ids:
            return
        if TABLE_OBJECT_ID in object_ids:
            raise ValueError("The table cannot be removed from a scene graph.")
        unknown_object_ids = object_ids - set(self.node_by_id())
        if unknown_object_ids:
            raise ValueError(
                f"Cannot remove unknown scene graph nodes: {sorted(unknown_object_ids)}"
            )

        # Removing every incident relation prevents dangling planar endpoints.
        self.nodes = [node for node in self.nodes if node.object_id not in object_ids]
        self.relations = [
            relation
            for relation in self.relations
            if relation.source_id not in object_ids
            and relation.target_id not in object_ids
        ]
        # Refresh.
        self.refresh()

    def add_node(self, node: GeneratedSceneNode) -> None:
        """Add one node and validate the resulting graph."""
        if node.object_id in self.node_by_id():
            raise ValueError(f"Duplicate scene graph node: {node.object_id}")
        self.nodes.append(node)
        self.refresh()

    def apply_updates(
        self,
        *,
        deleted_object_ids: set[str],
        added_object_ids: list[str],
        added_orientation_states_by_id: dict[str, OrientationState | None],
        on_parent_updates: list[tuple[str, str, TableRegion | None]],
        planar_relation_updates: list[tuple[str, PlanarRelationType, str]],
    ) -> None:
        """Apply one atomic batch of node and relationship updates."""
        if TABLE_OBJECT_ID in deleted_object_ids:
            raise ValueError("The table cannot be removed from a scene graph.")

        existing_object_ids = set(self.node_by_id())
        unknown_object_ids = deleted_object_ids - existing_object_ids
        if unknown_object_ids:
            raise ValueError(
                f"Cannot remove unknown scene graph nodes: {sorted(unknown_object_ids)}"
            )

        # Delete all requested nodes before resolving new parents and relations.
        self.nodes = [
            node for node in self.nodes if node.object_id not in deleted_object_ids
        ]
        self.relations = [
            relation
            for relation in self.relations
            if relation.source_id not in deleted_object_ids
            and relation.target_id not in deleted_object_ids
        ]

        remaining_object_ids = set(self.node_by_id())
        if len(added_object_ids) != len(set(added_object_ids)):
            raise ValueError("Added scene graph node ids must be unique.")
        duplicate_object_ids = set(added_object_ids) & remaining_object_ids
        if duplicate_object_ids:
            raise ValueError(
                f"Duplicate scene graph nodes: {sorted(duplicate_object_ids)}"
            )
        if set(added_orientation_states_by_id) != set(added_object_ids):
            raise ValueError("Added orientation states must match added node ids.")

        # New nodes default to the table; later updates replace that parent when needed.
        self.nodes.extend(
            GeneratedSceneNode(
                object_id=object_id,
                parent_id=TABLE_OBJECT_ID,
                parent_relation="on",
                orientation_state=added_orientation_states_by_id[object_id],
            )
            for object_id in added_object_ids
        )

        # Apply support-parent changes before planar updates need the final parent.
        for object_id, parent_id, table_region in on_parent_updates:
            self._set_on_parent(
                object_id=object_id,
                parent_id=parent_id,
                table_region=table_region,
            )

        # Resolve chained planar parent inheritance before adding final relations.
        self._resolve_planar_parent_updates(planar_relation_updates)
        # Clear stale relations before appending this batch so chained updates
        # cannot remove a relation that an earlier update just requested.
        for source_id in {source_id for source_id, _, _ in planar_relation_updates}:
            self._clear_incident_planar_relations(source_id)
        for source_id, relation, target_id in planar_relation_updates:
            self.relations.append(
                GeneratedSceneRelation(
                    source_id=source_id,
                    relation=relation,
                    target_id=target_id,
                )
            )

        # Normalize inverse relations and reject invalid final graph constraints.
        self.refresh()

    def _set_on_parent(
        self,
        *,
        object_id: str,
        parent_id: str,
        table_region: TableRegion | None = None,
    ) -> None:
        """Replace one node's support parent and stale planar constraints."""
        nodes_by_id = self.node_by_id()
        if object_id == TABLE_OBJECT_ID:
            raise ValueError("The table cannot be moved onto another object.")
        if object_id not in nodes_by_id or parent_id not in nodes_by_id:
            raise ValueError(
                "Parent updates must reference existing scene graph nodes."
            )
        if object_id == parent_id:
            raise ValueError("A scene graph node cannot be its own parent.")

        node = nodes_by_id[object_id]
        node.parent_id = parent_id
        node.parent_relation = "on"
        node.table_region = table_region
        self._clear_incident_planar_relations(object_id)

    def _resolve_planar_parent_updates(
        self,
        planar_relation_updates: list[tuple[str, PlanarRelationType, str]],
    ) -> None:
        """Make every planar source share its target's final support parent."""
        for _ in range(len(planar_relation_updates)):
            changed = False
            for source_id, _, target_id in planar_relation_updates:
                nodes_by_id = self.node_by_id()
                if source_id == TABLE_OBJECT_ID:
                    raise ValueError("The table cannot have a planar relation.")
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise ValueError(
                        "Planar updates must reference existing scene graph nodes."
                    )
                target_parent_id = nodes_by_id[target_id].parent_id
                if target_parent_id is None:
                    raise ValueError("Planar relation targets must have a parent.")
                source = nodes_by_id[source_id]
                if source.parent_id != target_parent_id:
                    source.parent_id = target_parent_id
                    source.parent_relation = "on"
                    source.table_region = None
                    changed = True
            if not changed:
                return

    def _clear_incident_planar_relations(self, object_id: str) -> None:
        """Remove planar constraints invalidated when one node changes parent."""
        self.relations = [
            relation
            for relation in self.relations
            if relation.source_id != object_id and relation.target_id != object_id
        ]

    def normalize(self) -> None:
        """Materialize inverse planar relations and remove duplicates."""
        self._materialize_inverse_planar_relations()
        self._deduplicate_relations()

    def layer_by_id(self) -> dict[str, int]:
        """Return the layer depth of each node inferred from parent links."""
        # Build fast lookup tables before walking the table-rooted tree.
        nodes_by_id = self.node_by_id()
        children_by_parent = self._children_by_parent()
        layers: dict[str, int] = {}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str, layer: int) -> None:
            # A node already on the recursion path means the parent chain loops.
            if node_id in visiting:
                raise ValueError(f"Parent cycle detected at node: {node_id}")
            if node_id in visited:
                return
            # Missing parent nodes cannot contribute a valid table-rooted layer.
            if node_id not in nodes_by_id:
                raise ValueError(f"Parent node does not exist: {node_id}")

            visiting.add(node_id)
            layers[node_id] = layer
            # Children are exactly one support level above their parent.
            for child in children_by_parent.get(node_id, []):
                visit(child.object_id, layer + 1)
            visiting.remove(node_id)
            visited.add(node_id)

        if TABLE_OBJECT_ID not in nodes_by_id:
            raise ValueError("Scene graph must contain a table node.")
        visit(TABLE_OBJECT_ID, 0)
        return layers

    def derive_constraints(self) -> list[dict[str, str]]:
        """Return support constraints plus materialized planar relations."""
        self.refresh()
        constraints: list[dict[str, str]] = []
        for node in self.nodes:
            if node.parent_id is None:
                continue
            # Parent links become direct support constraints.
            constraints.append(
                self._constraint_dict(
                    source_id=node.object_id,
                    relation=node.parent_relation,
                    target_id=node.parent_id,
                ),
            )
        for relation in self.relations:
            # Inverse planar relations are already stored during normalization.
            constraints.append(
                self._constraint_dict(
                    source_id=relation.source_id,
                    relation=relation.relation,
                    target_id=relation.target_id,
                ),
            )
        return self._deduplicate_constraints(constraints)

    def validate(self) -> None:
        """Validate hierarchy, table regions, and planar relation constraints."""
        # id -> Node mapping.
        nodes_by_id = self.node_by_id()
        # Table node must exist.
        if TABLE_OBJECT_ID not in nodes_by_id:
            raise ValueError("Scene graph must contain a table node.")

        # Validate the table-rooted support tree before checking sibling relations.
        for node in self.nodes:
            if node.object_id == TABLE_OBJECT_ID:
                if node.table_region is not None:
                    raise ValueError("table must not have a table_region.")
                continue
            parent = nodes_by_id.get(node.parent_id)
            # Parent must exist, except for the table. (root node)
            if parent is None:
                raise ValueError(f"Parent node does not exist: {node.parent_id}")
            if node.parent_relation is None:
                raise ValueError(
                    f"Node {node.object_id} must define its parent relation."
                )
            # Table regions are only valid for objects directly on the table.
            if node.table_region is not None and node.parent_id != TABLE_OBJECT_ID:
                raise ValueError("table_region is only valid for objects on the table.")
        # Get id -> layer mapping.
        layers = self.layer_by_id()
        if len(layers) != len(nodes_by_id):
            raise ValueError("All scene graph nodes must be reachable from the table.")
        # Validate planar relations between nodes with the same parent.
        for relation in self.relations:
            source = nodes_by_id.get(relation.source_id)
            target = nodes_by_id.get(relation.target_id)
            if source is None or target is None:
                raise ValueError("Planar relation endpoint does not exist.")
            if source.parent_id != target.parent_id:
                raise ValueError("Planar relation endpoints must share one parent.")
            if source.parent_relation != "on" or target.parent_relation != "on":
                raise ValueError("Planar relation endpoints must be on their parent.")
        # Validate and imply planar relation.
        self._validate_planar_relation_conflicts()

    def to_dict(self) -> dict[str, object]:
        """Serialize the normalized, versioned authoring artifact."""
        self.refresh()
        return {
            "schema_version": GENERATED_SCENE_GRAPH_SCHEMA,
            "artifact_kind": "scene_authoring",
            "nodes": [node.to_dict() for node in self.nodes],
            "relations": [relation.to_dict() for relation in self.relations],
        }

    def _children_by_parent(self) -> dict[str, list[GeneratedSceneNode]]:
        children_by_parent: dict[str, list[GeneratedSceneNode]] = {}
        for node in self.nodes:
            if node.parent_id is not None:
                children_by_parent.setdefault(node.parent_id, []).append(node)
        return children_by_parent

    def _deduplicate_relations(self) -> None:
        """Remove duplicate planar relations while preserving the first occurrence."""
        deduplicated: list[GeneratedSceneRelation] = []
        seen: set[tuple[str, PlanarRelationType, str]] = set()
        for relation in self.relations:
            key = (relation.source_id, relation.relation, relation.target_id)
            # Only identical triples are duplicates; inverse relations are both retained.
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(relation)
        self.relations = deduplicated

    def _materialize_inverse_planar_relations(self) -> None:
        """Add the inverse of every planar relation to the graph."""
        inverse_relations = [
            GeneratedSceneRelation(
                source_id=relation.target_id,
                relation=self._inverse_planar_relation(relation.relation),
                target_id=relation.source_id,
            )
            for relation in self.relations
        ]
        self.relations.extend(inverse_relations)

    def _deduplicate_constraints(
        self,
        constraints: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        deduplicated: list[dict[str, str]] = []
        seen: set[tuple[str, SceneConstraintType, str]] = set()
        for constraint in constraints:
            key = (
                constraint["source_id"],
                constraint["relation"],
                constraint["target_id"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(constraint)
        return deduplicated

    def _constraint_dict(
        self,
        *,
        source_id: str,
        relation: SceneConstraintType,
        target_id: str,
    ) -> dict[str, str]:
        return {
            "source_id": source_id,
            "relation": relation,
            "target_id": target_id,
        }

    def _validate_planar_relation_conflicts(self) -> None:
        implied_relations: dict[tuple[str, str], PlanarRelationType] = {}
        for relation in self.relations:
            self._add_implied_planar_relation(
                implied_relations,
                source_id=relation.source_id,
                relation=relation.relation,
                target_id=relation.target_id,
            )
            self._add_implied_planar_relation(
                implied_relations,
                source_id=relation.target_id,
                relation=self._inverse_planar_relation(relation.relation),
                target_id=relation.source_id,
            )

    def _add_implied_planar_relation(
        self,
        implied_relations: dict[tuple[str, str], PlanarRelationType],
        *,
        source_id: str,
        relation: PlanarRelationType,
        target_id: str,
    ) -> None:
        key = (source_id, target_id)
        existing_relation = implied_relations.get(key)
        if existing_relation is not None and existing_relation != relation:
            raise ValueError(
                f"Conflicting planar relations: {source_id} "
                f"{existing_relation} and {relation} {target_id}"
            )
        implied_relations[key] = relation

    @classmethod
    def _inverse_planar_relation(
        cls,
        relation: PlanarRelationType,
    ) -> PlanarRelationType:
        if relation == "left_of":
            return "right_of"
        if relation == "right_of":
            return "left_of"
        if relation == "in_front_of":
            return "behind"
        return "in_front_of"


# Preserve the pre-stack authoring names for callers already using the Scene
# Engine edit API. The explicit ``Generated*`` names remain canonical for the
# task-first pipeline and distinguish this graph from live simulator state.
SceneGraph = GeneratedSceneGraph
SceneGraphNode = GeneratedSceneNode
SceneGraphRelation = GeneratedSceneRelation
