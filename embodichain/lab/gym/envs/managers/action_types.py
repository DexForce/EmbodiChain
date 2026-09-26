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

"""Typed descriptors for ordered policy-action layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from embodichain.lab.gym.envs._json import json_safe_copy

__all__ = ["ActionDescriptor", "ActionTermDescriptor"]

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _validate_non_empty_name(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty stripped string.")


@dataclass(frozen=True, slots=True)
class ActionTermDescriptor:
    """Describe one term's raw policy-action representation.

    Args:
        representation: Stable semantic representation identifier.
        action_dim: Number of flat raw policy-action dimensions.
        feature_names: Ordered name for each raw dimension.
        units: Ordered unit label for each raw dimension.
        normalization: Optional normalization identifier.
        joint_names: Ordered joints controlled by the term.
        metadata: Additional JSON-compatible representation metadata.
    """

    representation: str
    action_dim: int
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    normalization: str | None
    joint_names: tuple[str, ...]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_name(self.representation, field_name="representation")
        if type(self.action_dim) is not int or self.action_dim <= 0:
            raise ValueError("action_dim must be a positive integer.")
        if (
            len(self.feature_names) != self.action_dim
            or len(self.units) != self.action_dim
        ):
            raise ValueError("feature_names, units, and action_dim must agree.")
        for index, name in enumerate(self.feature_names):
            _validate_non_empty_name(name, field_name=f"feature_names[{index}]")
        for index, unit in enumerate(self.units):
            _validate_non_empty_name(unit, field_name=f"units[{index}]")
        if self.normalization is not None:
            _validate_non_empty_name(self.normalization, field_name="normalization")
        for index, name in enumerate(self.joint_names):
            _validate_non_empty_name(name, field_name=f"joint_names[{index}]")
        owned_metadata = json_safe_copy(self.metadata, field_name="metadata")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(owned_metadata),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return an owned JSON-compatible representation."""
        return {
            "representation": self.representation,
            "action_dim": self.action_dim,
            "feature_names": list(self.feature_names),
            "units": list(self.units),
            "normalization": self.normalization,
            "joint_names": list(self.joint_names),
            "metadata": json_safe_copy(self.metadata, field_name="metadata"),
        }


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    """Bind one action-term descriptor to a manager-owned flat slice.

    Args:
        name: Stable term name from the ActionManager configuration.
        start: Inclusive flat-action offset.
        stop: Exclusive flat-action offset.
        term: Term-local action semantics.
    """

    name: str
    start: int
    stop: int
    term: ActionTermDescriptor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("name must be non-empty and stripped.")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be non-negative.")
        if type(self.stop) is not int or self.stop - self.start != self.term.action_dim:
            raise ValueError("slice width must match term action_dim.")

    def to_dict(self) -> dict[str, JSONValue]:
        """Return an owned JSON-compatible representation."""
        return {
            "name": self.name,
            "slice": [self.start, self.stop],
            "term": self.term.to_dict(),
        }
