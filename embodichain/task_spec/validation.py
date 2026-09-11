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

"""Strict template structure and role-reference checks without simulator imports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._json import digest, fields, items, json_copy, text
from .expressions import _expression
from .registry import registry_snapshot

__all__ = ["validate_template_structure"]


def validate_template_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate normative template data without verifying its optional digest.

    Args:
        value: TaskTemplate JSON; semantic_hash may be absent during authoring.

    Returns:
        Detached JSON preserving author role names and quantity units.

    Raises:
        ValueError: Invalid schema, fields, roles, capabilities or expressions.
    """
    result = fields(
        json_copy(value),
        {
            "schema_version",
            "semantic_version",
            "roles",
            "init",
            "goal",
            "invariants",
            "requirements",
        },
        {"semantic_hash", "temporal"},
        "TaskTemplate",
    )

    def predicate_count(item: object) -> int:
        if isinstance(item, dict):
            return int("predicate" in item) + sum(
                predicate_count(child) for child in item.values()
            )
        if isinstance(item, list):
            return sum(predicate_count(child) for child in item)
        return 0

    if (
        sum(
            predicate_count(result.get(field, []))
            for field in ("init", "goal", "invariants", "temporal")
        )
        > 128
    ):
        raise ValueError("TaskTemplate exceeds the 128 predicate limit")
    snapshot = registry_snapshot()
    if (
        result["schema_version"] != "taskspec/template/v0.1"
        or result["semantic_version"] != snapshot["semantic_version"]
    ):
        raise ValueError("unsupported TaskTemplate schema/semantic version")
    if "semantic_hash" in result:
        digest(result["semantic_hash"], "TaskTemplate.semantic_hash")
    roles = result["roles"]
    if not isinstance(roles, dict) or not 1 <= len(roles) <= snapshot["max_roles"]:
        raise ValueError("TaskTemplate requires between one and six roles")
    for name, role in roles.items():
        text(name, "role name")
        fields(role, {"kind"}, {"capabilities"}, f"roles.{name}")
        if role["kind"] not in snapshot["role_kinds"]:
            raise ValueError(f"unsupported role kind for {name}")
        capabilities = items(role.get("capabilities", []), f"roles.{name}.capabilities")
        for capability in capabilities:
            if capability not in snapshot["capabilities"]:
                raise ValueError(f"unsupported capability {capability!r}")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError(f"duplicate capabilities for role {name}")
    for field in ("init", "goal", "invariants"):
        for expression in items(result[field], field, nonempty=field == "goal"):
            _expression(expression, roles)
    for requirement in items(result["requirements"], "requirements"):
        fields(requirement, {"role", "capability"}, set(), "requirement")
        role = text(requirement["role"], "requirement.role")
        if role not in roles:
            raise ValueError("requirement references unknown role")
        if requirement["capability"] not in snapshot["capabilities"]:
            raise ValueError("unsupported requirement capability")
    for temporal in items(result.get("temporal", []), "temporal"):
        fields(temporal, {"op", "args"}, set(), "temporal")
        if temporal["op"] != "sequence":
            raise ValueError("unsupported temporal operator")
        args = items(temporal["args"], "temporal.args", nonempty=True)
        if len(args) < 2:
            raise ValueError("temporal sequence requires at least two observations")
        for expression in args:
            _expression(expression, roles)
    return result
