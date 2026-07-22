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

__all__ = [
    "GRID_VALUE_LIST",
    "GRID_VALUES",
    "RELATION_VALUE_LIST",
    "RELATION_VALUES",
    "assign_grids_from_anchor_and_orders",
    "derive_relations_from_orders",
    "invert_relation",
    "normalize_relation",
    "transitive_relation_closure",
    "validate_exact_asset_id_coverage",
]

RELATION_VALUE_LIST = ["left_of", "front_of"]
RELATION_VALUES = frozenset(RELATION_VALUE_LIST)
INVERSE_RELATIONS = {
    "left_of": "right_of",
    "right_of": "left_of",
    "front_of": "behind",
    "behind": "front_of",
}

GRID_VALUE_LIST = [
    "center",
    "front",
    "back",
    "left_center",
    "right_center",
    "left_front",
    "right_front",
    "left_back",
    "right_back",
]
GRID_VALUES = frozenset(GRID_VALUE_LIST)


def validate_exact_asset_id_coverage(
    *,
    values: list[str],
    expected_asset_ids: list[str],
    context: str,
) -> None:
    """Validate that values contain every expected asset id exactly once."""
    expected = set(expected_asset_ids)
    actual = set(values)
    duplicates = sorted({asset_id for asset_id in values if values.count(asset_id) > 1})
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if duplicates:
        raise ValueError(f"{context} has duplicate asset ids: {duplicates}.")
    if missing:
        raise ValueError(f"{context} is missing asset ids: {missing}.")
    if unknown:
        raise ValueError(f"{context} has unknown asset ids: {unknown}.")


def assign_grids_from_anchor_and_orders(
    *,
    anchor_asset_id: str,
    anchor_grid: str,
    x_order: list[list[str]],
    y_order: list[list[str]],
    asset_ids: list[str],
) -> dict[str, str]:
    """Assign 9-grid labels from one anchor grid and two object orderings."""
    anchor_x, anchor_y = _split_grid(anchor_grid)
    x_indices = _order_indices(x_order)
    y_indices = _order_indices(y_order)
    anchor_x_index = x_indices[anchor_asset_id]
    anchor_y_index = y_indices[anchor_asset_id]

    grids: dict[str, str] = {}
    for asset_id in asset_ids:
        x_label = _axis_label_from_anchor(
            index=x_indices[asset_id],
            anchor_index=anchor_x_index,
            anchor_label=anchor_x,
            before_label="left",
            after_label="right",
        )
        y_label = _axis_label_from_anchor(
            index=y_indices[asset_id],
            anchor_index=anchor_y_index,
            anchor_label=anchor_y,
            before_label="front",
            after_label="back",
        )
        grids[asset_id] = _join_grid(x_label=x_label, y_label=y_label)
    return grids


def invert_relation(relation: str) -> str:
    """Return the inverse of a supported spatial relation."""
    if relation not in INVERSE_RELATIONS:
        raise ValueError(f"Unsupported spatial relation: {relation!r}.")
    return INVERSE_RELATIONS[relation]


def normalize_relation(
    *,
    subject: str,
    relation: str,
    object_id: str,
) -> tuple[str, str, str]:
    """Normalize a relation into a canonical directional axis edge."""
    if relation == "left_of":
        return subject, "left_of", object_id
    if relation == "right_of":
        return object_id, "left_of", subject
    if relation == "front_of":
        return subject, "front_of", object_id
    if relation == "behind":
        return object_id, "front_of", subject
    raise ValueError(f"Unsupported spatial relation: {relation!r}.")


def transitive_relation_closure(
    relations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Expand canonical left/front relations with transitive closure."""
    direct_edges: dict[str, set[tuple[str, str]]] = {
        "left_of": set(),
        "front_of": set(),
    }
    input_edges: set[tuple[str, str, str]] = set()
    for relation_record in relations:
        subject = relation_record["subject"]
        relation = relation_record["relation"]
        object_id = relation_record["object"]
        canonical_subject, canonical_relation, canonical_object = normalize_relation(
            subject=subject,
            relation=relation,
            object_id=object_id,
        )
        if canonical_subject == canonical_object:
            raise ValueError("Spatial relation cannot reference the same object.")
        edge = (canonical_subject, canonical_object)
        inverse_edge = (canonical_object, canonical_subject)
        if inverse_edge in direct_edges[canonical_relation]:
            raise ValueError(
                "Conflicting spatial relations: "
                f"{canonical_subject!r} {canonical_relation} {canonical_object!r}."
            )
        direct_edges[canonical_relation].add(edge)
        input_edges.add((subject, relation, object_id))

    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for canonical_relation, edges in direct_edges.items():
        for subject, object_id in sorted(_transitive_edges(edges)):
            _append_relation(
                output=output,
                seen=seen,
                subject=subject,
                relation=canonical_relation,
                object_id=object_id,
                source=(
                    "input"
                    if (subject, canonical_relation, object_id) in input_edges
                    else "closure"
                ),
            )
    return output


def derive_relations_from_orders(
    *,
    x_order: list[list[str]],
    y_order: list[list[str]],
) -> list[dict[str, str]]:
    """Derive canonical relations from adjacent order groups."""
    relations: list[dict[str, str]] = []
    relations.extend(_relations_from_order_groups(x_order, relation="left_of"))
    relations.extend(_relations_from_order_groups(y_order, relation="front_of"))
    closed = transitive_relation_closure(relations)
    return [
        {
            **relation,
            "source": "order" if relation["source"] == "input" else relation["source"],
        }
        for relation in closed
    ]


def _order_indices(order: list[list[str]]) -> dict[str, int]:
    return {
        asset_id: group_index
        for group_index, group in enumerate(order)
        for asset_id in group
    }


def _split_grid(grid: str) -> tuple[str, str]:
    if grid == "center":
        return "center", "center"
    if grid in {"front", "back"}:
        return "center", grid
    if grid in {"left_center", "right_center"}:
        return grid.split("_", maxsplit=1)[0], "center"
    x_label, y_label = grid.split("_", maxsplit=1)
    return x_label, y_label


def _axis_label_from_anchor(
    *,
    index: int,
    anchor_index: int,
    anchor_label: str,
    before_label: str,
    after_label: str,
) -> str:
    if index < anchor_index:
        return before_label
    if index > anchor_index:
        return after_label
    return anchor_label


def _join_grid(*, x_label: str, y_label: str) -> str:
    if x_label == "center" and y_label == "center":
        return "center"
    if x_label == "center":
        return y_label
    if y_label == "center":
        return f"{x_label}_center"
    return f"{x_label}_{y_label}"


def _relations_from_order_groups(
    order_groups: list[list[str]],
    *,
    relation: str,
) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    for earlier_group, later_group in zip(order_groups, order_groups[1:]):
        for subject in earlier_group:
            for object_id in later_group:
                relations.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "object": object_id,
                        "source": "input",
                    }
                )
    return relations


def _transitive_edges(
    edges: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    adjacency: dict[str, set[str]] = {}
    for subject, object_id in edges:
        adjacency.setdefault(subject, set()).add(object_id)
        adjacency.setdefault(object_id, set())

    closure: set[tuple[str, str]] = set(edges)
    for start in adjacency:
        stack = list(adjacency[start])
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            closure.add((start, current))
            stack.extend(adjacency.get(current, ()))
    return closure


def _append_relation(
    *,
    output: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    subject: str,
    relation: str,
    object_id: str,
    source: str,
) -> None:
    key = (subject, relation, object_id)
    if key in seen:
        return
    seen.add(key)
    output.append(
        {
            "subject": subject,
            "relation": relation,
            "object": object_id,
            "source": source,
        }
    )
