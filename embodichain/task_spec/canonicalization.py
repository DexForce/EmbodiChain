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

"""Restricted alpha-equivalence, inverse relations and SI task identity."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations, product
from typing import Any

from ._json import dumps, hash_json
from .expressions import _expression
from .registry import registry_snapshot
from .validation import validate_template_structure

__all__ = ["canonical_template", "canonical_role_map", "semantic_hash"]
_ROLE_FIELDS = {
    name: tuple(spec["roles"])
    for name, spec in registry_snapshot()["predicates"].items()
}


def _canonical_expression(
    value: dict[str, Any], labels: dict[str, str]
) -> dict[str, Any]:
    if "op" in value:
        args = [_canonical_expression(arg, labels) for arg in value["args"]]
        if value["op"] in {"and", "or"}:
            args.sort(key=dumps)
        return {"op": value["op"], "args": args}
    result = dict(value)
    for key in _ROLE_FIELDS[result["predicate"]]:
        result[key] = labels[result[key]]
    if result["predicate"] == "relative_position":
        inverse = {"right_of": "left_of", "behind": "in_front_of"}
        if result["relation"] in inverse:
            result["relation"] = inverse[result["relation"]]
            result["object"], result["reference"] = (
                result["reference"],
                result["object"],
            )
    return result


def _canonicalize(template: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    source = validate_template_structure(template)
    roles = {
        name: {
            "kind": role["kind"],
            "capabilities": sorted(role.get("capabilities", [])),
        }
        for name, role in source["roles"].items()
    }
    groups: dict[str, list[str]] = {}
    for name, role in roles.items():
        groups.setdefault(dumps(role), []).append(name)
    ordered_groups = [sorted(groups[key]) for key in sorted(groups)]
    normalized = {
        field: [_expression(expr, source["roles"]) for expr in source[field]]
        for field in ("init", "goal", "invariants")
    }
    temporal = [
        {
            "op": "sequence",
            "args": [_expression(expr, source["roles"]) for expr in item["args"]],
        }
        for item in source.get("temporal", [])
    ]
    best = None
    best_key = None
    best_labels = None
    for ordering in product(*(permutations(group) for group in ordered_groups)):
        names = [name for group in ordering for name in group]
        labels = {name: f"role_{i}" for i, name in enumerate(names)}
        candidate = {
            "schema_version": source["schema_version"],
            "semantic_version": source["semantic_version"],
            "roles": {labels[name]: roles[name] for name in names},
            **{
                field: sorted(
                    (_canonical_expression(expr, labels) for expr in exprs), key=dumps
                )
                for field, exprs in normalized.items()
            },
            "requirements": sorted(
                (
                    dict(item, role=labels[item["role"]])
                    for item in source["requirements"]
                ),
                key=dumps,
            ),
            "temporal": sorted(
                (
                    {
                        "op": "sequence",
                        "args": [
                            _canonical_expression(expr, labels) for expr in item["args"]
                        ],
                    }
                    for item in temporal
                ),
                key=dumps,
            ),
        }
        key = dumps(candidate)
        if best_key is None or key < best_key:
            best, best_key = candidate, key
            best_labels = labels
    assert best is not None and best_labels is not None
    return best, best_labels


def canonical_template(template: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated canonical semantic payload with no self digest.

    Args:
        template: Normative TaskTemplate data, optionally including its hash.

    Returns:
        Canonical roles and predicates. Only role renaming, inverse spatial
        relations, SI scaling and commutative sorting are supported. Temporal
        argument order is preserved. Instance roles bind this payload.

    Raises:
        ValueError: Unsupported or invalid bounded template.
    """
    return _canonicalize(template)[0]


def canonical_role_map(template: Mapping[str, Any]) -> dict[str, str]:
    """Map author role names to the winning canonical template's role IDs.

    Args:
        template: Normative template used to select physical role grounding.

    Returns:
        Mapping used to rekey instance roles before computing content identity.
        Symmetric ties are resolved by author name order. Evaluators consume
        canonical_template with the canonical instance binding, never author
        predicates with canonical IDs.
    """
    return _canonicalize(template)[1]


def semantic_hash(template: Mapping[str, Any]) -> str:
    """Hash canonical task semantics, excluding only the top-level self digest.

    Args:
        template: Valid normative template data; execution fields are rejected.

    Returns:
        SHA-256 of the canonical JSON payload. A present semantic_hash is
        excluded to support authoring; validate_task_template verifies it.
    """
    return hash_json(canonical_template(template))
