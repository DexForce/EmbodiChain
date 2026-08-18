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

"""Robot-independent resource requirements published by atomic skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

from .control import ControlCommand

JOINT_POSITION_CAPABILITY = "motion.joint_position"
"""Capability for planning and executing joint-position motion."""

CARTESIAN_POSE_CAPABILITY = "motion.cartesian_pose"
"""Capability for planning and executing Cartesian-pose motion."""

FORWARD_KINEMATICS_CAPABILITY = "kinematics.forward"
"""Capability for resolving forward kinematics for an endpoint."""

INVERSE_KINEMATICS_CAPABILITY = "kinematics.inverse"
"""Capability for resolving inverse kinematics for an endpoint."""

BATCH_INVERSE_KINEMATICS_CAPABILITY = "kinematics.batch_inverse"
"""Capability for resolving batched inverse kinematics for an endpoint."""

GRASP_CAPABILITY = "interaction.grasp"
"""Capability for commanding a grasping end effector."""


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Return one strict, whitespace-free identifier."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _normalize_identifiers(
    values: frozenset[str],
    *,
    field_name: str,
) -> frozenset[str]:
    """Validate one immutable identifier set."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string.")
    try:
        normalized = frozenset(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    for value in normalized:
        _validate_identifier(value, field_name=field_name)
    return normalized


@dataclass(frozen=True, slots=True)
class ActionBindingRoute:
    """Lower one generic resource endpoint into the current action core.

    This is deliberately a transition adapter. Robot resources and skill-local
    slots remain generic; only this route names the two maps currently exposed
    by :class:`~embodichain.lab.sim.atomic_actions.ActionBinding`.
    """

    target: Literal["manipulator", "end_effector"]
    """Current core binding namespace."""

    role: str
    """Action-local role within the selected namespace."""

    def __post_init__(self) -> None:
        if self.target not in ("manipulator", "end_effector"):
            raise ValueError(
                "ActionBindingRoute.target must be 'manipulator' or 'end_effector'."
            )
        _validate_identifier(self.role, field_name="ActionBindingRoute.role")

    @property
    def key(self) -> tuple[str, str]:
        """Return the normalized core target key."""
        return self.target, self.role


def _normalize_required_commands(
    values: Mapping[str, type[ControlCommand]],
) -> Mapping[str, type[ControlCommand]]:
    """Validate and freeze endpoint command requirements."""
    if not isinstance(values, Mapping):
        raise TypeError("required_commands must be a mapping.")
    normalized: dict[str, type[ControlCommand]] = {}
    for name, command_type in values.items():
        _validate_identifier(name, field_name="required command names")
        if not isinstance(command_type, type) or not issubclass(
            command_type, ControlCommand
        ):
            raise TypeError(
                "required_commands values must be ControlCommand subclasses."
            )
        normalized[name] = command_type
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class SkillEndpointRequirement:
    """Capabilities and commands required from one slot-local endpoint."""

    endpoint_id: str
    """Endpoint selector local to the containing participant slot."""

    capabilities: frozenset[str] = frozenset()
    """Open, namespaced all-of capability identifiers."""

    required_commands: Mapping[str, type[ControlCommand]] = field(default_factory=dict)
    """Semantic command names and their required typed command contracts."""

    route: ActionBindingRoute | None = None
    """Optional lowering route into the current atomic-action core."""

    def __post_init__(self) -> None:
        _validate_identifier(
            self.endpoint_id,
            field_name="SkillEndpointRequirement.endpoint_id",
        )
        object.__setattr__(
            self,
            "capabilities",
            _normalize_identifiers(
                self.capabilities,
                field_name="SkillEndpointRequirement.capabilities",
            ),
        )
        object.__setattr__(
            self,
            "required_commands",
            _normalize_required_commands(self.required_commands),
        )
        if self.route is not None and not isinstance(self.route, ActionBindingRoute):
            raise TypeError("route must be an ActionBindingRoute or None.")


@dataclass(frozen=True, slots=True)
class DisjointSlotEndpoints:
    """Require selected endpoints within one participant to be disjoint."""

    endpoint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.endpoint_ids, (str, bytes)):
            raise TypeError("endpoint_ids must be an iterable of endpoint IDs.")
        try:
            endpoint_ids = tuple(self.endpoint_ids)
        except TypeError as exc:
            raise TypeError(
                "endpoint_ids must be an iterable of endpoint IDs."
            ) from exc
        if len(endpoint_ids) < 2:
            raise ValueError("DisjointSlotEndpoints requires at least two endpoints.")
        for endpoint_id in endpoint_ids:
            _validate_identifier(
                endpoint_id,
                field_name="DisjointSlotEndpoints.endpoint_ids",
            )
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError("DisjointSlotEndpoints.endpoint_ids must be unique.")
        object.__setattr__(self, "endpoint_ids", endpoint_ids)


@dataclass(frozen=True, slots=True)
class SkillResourceSlot:
    """One skill-local participant selected as an indivisible resource unit."""

    slot_id: str
    """Skill-local participant name, such as ``primary`` or ``source``."""

    endpoints: tuple[SkillEndpointRequirement, ...]
    """Endpoint requirements that the selected robot resource must satisfy."""

    constraints: tuple[DisjointSlotEndpoints, ...] = ()
    """Physical constraints among endpoint views in this participant."""

    def __post_init__(self) -> None:
        _validate_identifier(self.slot_id, field_name="SkillResourceSlot.slot_id")
        if isinstance(self.endpoints, (str, bytes)):
            raise TypeError(
                "SkillResourceSlot.endpoints must be an iterable of endpoint "
                "requirements."
            )
        try:
            endpoints = tuple(self.endpoints)
        except TypeError as exc:
            raise TypeError(
                "SkillResourceSlot.endpoints must be an iterable of endpoint "
                "requirements."
            ) from exc
        if not endpoints or not all(
            isinstance(endpoint, SkillEndpointRequirement) for endpoint in endpoints
        ):
            raise ValueError(
                "SkillResourceSlot.endpoints must contain at least one "
                "SkillEndpointRequirement."
            )
        endpoint_ids = [endpoint.endpoint_id for endpoint in endpoints]
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError(
                f"Skill resource slot {self.slot_id!r} contains duplicate endpoint "
                "identifiers."
            )
        object.__setattr__(self, "endpoints", endpoints)
        if isinstance(self.constraints, (str, bytes)):
            raise TypeError(
                "SkillResourceSlot.constraints must be an iterable of endpoint "
                "constraints."
            )
        try:
            constraints = tuple(self.constraints)
        except TypeError as exc:
            raise TypeError(
                "SkillResourceSlot.constraints must be an iterable of endpoint "
                "constraints."
            ) from exc
        if not all(
            isinstance(constraint, DisjointSlotEndpoints) for constraint in constraints
        ):
            raise TypeError(
                "SkillResourceSlot.constraints values must be "
                "DisjointSlotEndpoints instances."
            )
        known_endpoints = set(endpoint_ids)
        for constraint in constraints:
            unknown = sorted(set(constraint.endpoint_ids) - known_endpoints)
            if unknown:
                raise ValueError(
                    f"Slot {self.slot_id!r} constraint references unknown endpoints "
                    f"{unknown}; known endpoints are {sorted(known_endpoints)}."
                )
        object.__setattr__(self, "constraints", constraints)


@dataclass(frozen=True, slots=True)
class DisjointResourceSlots:
    """Require selected slots to have pairwise-disjoint physical claims."""

    slots: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.slots, (str, bytes)):
            raise TypeError("DisjointResourceSlots.slots must be an iterable.")
        try:
            slots = tuple(self.slots)
        except TypeError as exc:
            raise TypeError("DisjointResourceSlots.slots must be an iterable.") from exc
        if len(slots) < 2:
            raise ValueError("DisjointResourceSlots requires at least two slots.")
        for slot in slots:
            _validate_identifier(slot, field_name="DisjointResourceSlots.slots")
        if len(set(slots)) != len(slots):
            raise ValueError("DisjointResourceSlots.slots must be unique.")
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True, slots=True)
class SkillBindingContract:
    """Complete robot-independent binding contract for one atomic skill.

    ``slots=()`` explicitly declares that a skill consumes no robot resource.
    ``None`` on :class:`~embodichain.lab.sim.atomic_actions.SkillDescriptor`
    instead means that no semantic binding contract was declared.
    """

    slots: tuple[SkillResourceSlot, ...] = ()
    constraints: tuple[DisjointResourceSlots, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.slots, (str, bytes)):
            raise TypeError("slots must be an iterable of SkillResourceSlot values.")
        try:
            slots = tuple(self.slots)
        except TypeError as exc:
            raise TypeError(
                "slots must be an iterable of SkillResourceSlot values."
            ) from exc
        if not all(isinstance(slot, SkillResourceSlot) for slot in slots):
            raise TypeError("slots values must be SkillResourceSlot instances.")
        slot_ids = [slot.slot_id for slot in slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("SkillBindingContract slot identifiers must be unique.")
        if isinstance(self.constraints, (str, bytes)):
            raise TypeError(
                "constraints must be an iterable of DisjointResourceSlots values."
            )
        try:
            constraints = tuple(self.constraints)
        except TypeError as exc:
            raise TypeError(
                "constraints must be an iterable of DisjointResourceSlots values."
            ) from exc
        if not all(
            isinstance(constraint, DisjointResourceSlots) for constraint in constraints
        ):
            raise TypeError(
                "constraints values must be DisjointResourceSlots instances."
            )
        known_slots = set(slot_ids)
        for constraint in constraints:
            unknown = sorted(set(constraint.slots) - known_slots)
            if unknown:
                raise ValueError(
                    f"Resource constraint references unknown slots {unknown}; "
                    f"known slots are {sorted(known_slots)}."
                )
        routes = [
            endpoint.route.key
            for slot in slots
            for endpoint in slot.endpoints
            if endpoint.route is not None
        ]
        if len(set(routes)) != len(routes):
            raise ValueError("Action binding routes must target unique core roles.")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "constraints", constraints)

    @property
    def slot_ids(self) -> tuple[str, ...]:
        """Return required slot identifiers in declaration order."""
        return tuple(slot.slot_id for slot in self.slots)

    def validate_action_roles(
        self,
        *,
        manipulator_roles: tuple[str, ...],
        end_effector_roles: tuple[str, ...],
    ) -> None:
        """Require lowering routes to cover the current core roles exactly."""
        expected = {("manipulator", role) for role in manipulator_roles}
        expected.update(("end_effector", role) for role in end_effector_roles)
        actual = {
            endpoint.route.key
            for slot in self.slots
            for endpoint in slot.endpoints
            if endpoint.route is not None
        }
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "Skill binding routes do not exactly cover the action roles: "
                f"missing={missing}, extra={extra}."
            )


__all__ = [
    "ActionBindingRoute",
    "BATCH_INVERSE_KINEMATICS_CAPABILITY",
    "CARTESIAN_POSE_CAPABILITY",
    "DisjointResourceSlots",
    "DisjointSlotEndpoints",
    "FORWARD_KINEMATICS_CAPABILITY",
    "GRASP_CAPABILITY",
    "INVERSE_KINEMATICS_CAPABILITY",
    "JOINT_POSITION_CAPABILITY",
    "SkillBindingContract",
    "SkillEndpointRequirement",
    "SkillResourceSlot",
]
