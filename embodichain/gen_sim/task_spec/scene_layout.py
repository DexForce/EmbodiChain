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

"""Normative role-based scene variation requirements, independent of execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import TaskSpec

__all__ = ["SceneLayoutRequirement", "scene_layout_requirement"]


def _roles(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of role IDs.")
    roles = tuple(value)
    if any(
        not isinstance(role, str) or not role or role != role.strip() for role in roles
    ) or len(set(roles)) != len(roles):
        raise ValueError(f"{name} must contain unique nonempty role IDs.")
    return tuple(sorted(roles))


def _bounds(value: object, *, xy: bool) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError("Variation bounds require ordered lower/upper values.")
    if xy:
        if any(
            not isinstance(row, Sequence) or isinstance(row, str) or len(row) != 2
            for row in value
        ):
            raise ValueError("XY variation bounds require two XY corners.")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(float(x))
            for row in value
            for x in row
        ):
            raise ValueError("XY variation bounds must contain finite numeric values.")
        result = (tuple(float(x) for x in value[0]), tuple(float(x) for x in value[1]))
        if any(result[0][axis] > result[1][axis] for axis in range(2)):
            raise ValueError("XY variation bounds must be ordered per axis.")
        return result
    numbers = tuple(float(x) for x in value)
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) for x in value
    ) or not all(math.isfinite(x) for x in numbers):
        raise ValueError("Variation bounds must be finite numeric values.")
    if not xy and numbers[0] > numbers[1]:
        raise ValueError("Yaw variation bounds must be ordered.")
    return numbers


@dataclass(frozen=True)
class SceneLayoutRequirement:
    """Role-based allowed changes; sampling devices, seeds and budgets are external.

    Args:
        movable_roles: Roots of permitted group/subtree movement.
        fixed_roles: Roles whose arena poses must remain unchanged.
        xy_offset_bounds_m: Optional per-role lower/upper XY displacement bounds.
            Group roots use arena axes; members use their parent support frame.
        yaw_delta_range_deg: Optional per-role member yaw bounds; group-root
            rotation is unsupported by the initial layout generator.
        support_role: Optional role naming the already declared main support.
    """

    movable_roles: tuple[str, ...]
    fixed_roles: tuple[str, ...] = ()
    xy_offset_bounds_m: Mapping[str, object] | None = None
    yaw_delta_range_deg: Mapping[str, object] | None = None
    support_role: str | None = None

    def __post_init__(self) -> None:
        moving = _roles(self.movable_roles, name="movable_roles")
        fixed = _roles(self.fixed_roles, name="fixed_roles")
        if not moving or set(moving) & set(fixed):
            raise ValueError(
                "Scene layout requires movable roles disjoint from fixed roles."
            )
        converted = {}
        for name, xy in (("xy_offset_bounds_m", True), ("yaw_delta_range_deg", False)):
            raw = getattr(self, name)
            if raw is None:
                raw = {}
            if not isinstance(raw, Mapping) or set(raw) - set(moving):
                raise ValueError(f"{name} must reference declared movable roles.")
            converted[name] = MappingProxyType(
                {
                    role: _bounds(bounds, xy=xy)
                    for role, bounds in raw.items()
                    if bounds is not None
                }
            )
        if self.support_role is not None and (
            not isinstance(self.support_role, str) or not self.support_role.strip()
        ):
            raise ValueError("support_role must be a role ID.")
        object.__setattr__(self, "movable_roles", moving)
        object.__setattr__(self, "fixed_roles", fixed)
        for name, value in converted.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, value: Mapping) -> SceneLayoutRequirement:
        """Decode a closed role requirement, rejecting execution parameters.

        Args:
            value: JSON-compatible normative scene-layout declaration.

        Returns:
            Validated requirement with owned normalized ranges.
        """
        if not isinstance(value, Mapping):
            raise ValueError("scene_layout must be a structured requirement.")
        allowed = {
            "movable_roles",
            "fixed_roles",
            "xy_offset_bounds_m",
            "yaw_delta_range_deg",
            "support_role",
        }
        if set(value) - allowed:
            raise ValueError(
                f"Unsupported scene_layout fields: {sorted(set(value) - allowed)}; runtime settings belong to the runner."
            )
        return cls(**dict(value))

    def to_dict(self) -> dict:
        """Return the normalized semantic payload, without runtime settings."""
        return {
            "movable_roles": list(self.movable_roles),
            "fixed_roles": list(self.fixed_roles),
            "xy_offset_bounds_m": {
                role: [list(row) for row in bounds]
                for role, bounds in sorted(self.xy_offset_bounds_m.items())
            },
            "yaw_delta_range_deg": {
                role: list(bounds)
                for role, bounds in sorted(self.yaw_delta_range_deg.items())
            },
            "support_role": self.support_role,
        }


def scene_layout_requirement(spec: TaskSpec) -> SceneLayoutRequirement | None:
    """Decode the reserved expansion/scene_layout requirement without a simulator.

    Args:
        spec: Validated TaskSpec; references must name its logical roles.

    Returns:
        One normative declaration, or None when layout expansion is not requested.
    """
    requirements = [
        item
        for item in spec.requirements
        if item.consumer == "expansion" and item.key == "scene_layout"
    ]
    if not requirements:
        return None
    if len(requirements) != 1 or not isinstance(requirements[0].value, Mapping):
        raise ValueError("TaskSpec requires one structured scene_layout declaration.")
    requirement = SceneLayoutRequirement.from_dict(requirements[0].value)
    known = {role.name for role in spec.roles}
    referenced = set(requirement.movable_roles) | set(requirement.fixed_roles)
    if requirement.support_role is not None:
        referenced.add(requirement.support_role)
    if referenced - known:
        raise ValueError(
            f"Scene layout references undeclared task roles: {sorted(referenced - known)}"
        )
    return requirement
