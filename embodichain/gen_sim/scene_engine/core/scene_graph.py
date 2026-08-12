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

TABLE_OBJECT_ID = "table"

# 9-grid table regions, treat the table as a 3x3 grid.
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

# A on B, then B is the parent node of A.
SupportRelationType = Literal["on"]

# A PlanarRelation with B, then A and B must have the same parent node.
PlanarRelationType = Literal["left_of", "right_of", "in_front_of", "behind"]
SceneConstraintType = SupportRelationType | PlanarRelationType


@dataclass
class SceneGraphNode:
    """One object node in the edit-time scene hierarchy."""

    object_id: str
    parent_id: str | None
    parent_relation: SupportRelationType | None = None
    table_region: TableRegion | None = None

    def __post_init__(self) -> None:
        """Validate local node fields before graph-level checks."""
        if not self.object_id:
            raise ValueError("object_id must be non-empty.")
        # If the node is the table.
        if self.object_id == TABLE_OBJECT_ID:
            if self.parent_id is not None:
                raise ValueError("table must not have a parent.")
            if self.parent_relation is not None:
                raise ValueError("table must not have a parent relation.")
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
        }


@dataclass
class SceneGraphRelation:
    """One edit-time spatial relation between two non-table objects."""

    source_id: str
    relation: PlanarRelationType
    target_id: str

    def __post_init__(self) -> None:
        """Validate local relation fields before graph-level checks."""
        if not self.source_id or not self.target_id:
            raise ValueError("relation endpoints must be non-empty.")
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
class SceneGraph:
    """Layered support graph plus planar relations for scene editing."""

    nodes: list[SceneGraphNode] = field(default_factory=list)
    relations: list[SceneGraphRelation] = field(default_factory=list)
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

    def node_by_id(self) -> dict[str, SceneGraphNode]:
        """Return nodes keyed by object id, raising on duplicate ids."""
        nodes_by_id: dict[str, SceneGraphNode] = {}
        for node in self.nodes:
            if node.object_id in nodes_by_id:
                raise ValueError(f"Duplicate scene graph node: {node.object_id}")
            nodes_by_id[node.object_id] = node
        return nodes_by_id

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
        """Serialize the normalized graph state."""
        self.refresh()
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "relations": [relation.to_dict() for relation in self.relations],
        }

    def _children_by_parent(self) -> dict[str, list[SceneGraphNode]]:
        children_by_parent: dict[str, list[SceneGraphNode]] = {}
        for node in self.nodes:
            if node.parent_id is not None:
                children_by_parent.setdefault(node.parent_id, []).append(node)
        return children_by_parent

    def _deduplicate_relations(self) -> None:
        """Remove duplicate planar relations while preserving the first occurrence."""
        deduplicated: list[SceneGraphRelation] = []
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
            SceneGraphRelation(
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
