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

"""Task-owned asset role requirements, independent of source selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Literal

from embodichain.data.asset_catalog import AssetQuery

__all__ = ["AssetRoleRequirement", "AssetRelationRequirement", "AssetRequirements"]


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric data.")
    return float(value)


def _closed(data: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError("Requirements must be mappings.")
    if set(data) - allowed:
        raise ValueError(f"Unknown requirement fields: {sorted(set(data) - allowed)}")
    return dict(data)


@dataclass(frozen=True, slots=True)
class AssetRoleRequirement:
    """Requirements for a stable Task Program entity.

    Args:
        entity_id: Stable program object identifier retained across asset changes.
        query: Required semantic category, properties and capabilities.
        max_extent_m: Optional upper bound on every object dimension.
    """

    entity_id: str
    query: AssetQuery
    max_extent_m: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, "entity_id")
        if not isinstance(self.query, AssetQuery):
            raise TypeError("query must be AssetQuery.")
        if self.max_extent_m is not None:
            value = _finite(self.max_extent_m, "max_extent_m")
            if value <= 0:
                raise ValueError("max_extent_m must be positive.")
            object.__setattr__(self, "max_extent_m", value)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible role declaration."""
        return {
            "entity_id": self.entity_id,
            "query": self.query.to_dict(),
            "max_extent_m": self.max_extent_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetRoleRequirement:
        """Decode a role declaration, rejecting unknown keys."""
        values = _closed(data, {"entity_id", "query", "max_extent_m"})
        if "query" in values:
            values["query"] = AssetQuery.from_dict(values["query"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class AssetRelationRequirement:
    """Geometric relation between two selected asset roles.

    Args:
        kind: Containment or support relationship in aligned local coordinates.
        object_role: Role of the manipulated object.
        target_role: Role of its container or support.
        clearance_m: Required nonnegative inset on each relevant side.
    """

    kind: Literal["fits_inside", "supported_by"]
    object_role: str
    target_role: str
    clearance_m: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("fits_inside", "supported_by"):
            raise ValueError("Unsupported asset relation kind.")
        _identifier(self.object_role, "object_role")
        _identifier(self.target_role, "target_role")
        if self.object_role == self.target_role:
            raise ValueError("Asset relation must reference distinct roles.")
        value = _finite(self.clearance_m, "clearance_m")
        if value < 0:
            raise ValueError("clearance_m must be non-negative.")
        object.__setattr__(self, "clearance_m", value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible relation declaration."""
        return {
            "kind": self.kind,
            "object_role": self.object_role,
            "target_role": self.target_role,
            "clearance_m": self.clearance_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetRelationRequirement:
        """Decode a closed relation declaration."""
        return cls(
            **_closed(data, {"kind", "object_role", "target_role", "clearance_m"})
        )


@dataclass(frozen=True, slots=True)
class AssetRequirements:
    """Immutable task role contract used by matching and scene materialization.

    Args:
        roles: Named roles, each identifying a distinct program object.
        relations: Required geometric relationships between named roles.
        category_aliases: Explicit alias-to-category edges; cycles are invalid.
    """

    roles: Mapping[str, AssetRoleRequirement]
    relations: tuple[AssetRelationRequirement, ...] = ()
    category_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.roles, Mapping) or not self.roles:
            raise ValueError("roles must be a non-empty mapping.")
        roles = dict(self.roles)
        for name, role in roles.items():
            _identifier(name, "role name")
            if not isinstance(role, AssetRoleRequirement):
                raise TypeError("roles must contain AssetRoleRequirement values.")
        if len({role.entity_id for role in roles.values()}) != len(roles):
            raise ValueError("Each asset role must have a distinct entity_id.")
        relations = tuple(self.relations)
        for relation in relations:
            if not isinstance(relation, AssetRelationRequirement):
                raise TypeError(
                    "relations must contain AssetRelationRequirement values."
                )
            if relation.object_role not in roles or relation.target_role not in roles:
                raise ValueError("Asset relation references an unknown role.")
        if not isinstance(self.category_aliases, Mapping):
            raise TypeError("category_aliases must be a mapping.")
        aliases = dict(self.category_aliases)
        for alias, category in aliases.items():
            _identifier(alias, "category alias")
            _identifier(category, "canonical category")
            seen = set()
            current = alias
            while current in aliases:
                if current in seen:
                    raise ValueError("category_aliases contains a cycle.")
                seen.add(current)
                current = aliases[current]
        object.__setattr__(self, "roles", MappingProxyType(roles))
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "category_aliases", MappingProxyType(aliases))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible requirements snapshot."""
        return {
            "roles": {k: v.to_dict() for k, v in self.roles.items()},
            "relations": [v.to_dict() for v in self.relations],
            "category_aliases": dict(self.category_aliases),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetRequirements:
        """Decode requirements with strict nested role and relation validation."""
        values = _closed(data, {"roles", "relations", "category_aliases"})
        if "roles" in values:
            if not isinstance(values["roles"], Mapping):
                raise TypeError("roles must be a mapping.")
            values["roles"] = {
                k: AssetRoleRequirement.from_dict(v) for k, v in values["roles"].items()
            }
        if "relations" in values:
            values["relations"] = tuple(
                AssetRelationRequirement.from_dict(v) for v in values["relations"]
            )
        return cls(**values)
