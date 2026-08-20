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

"""Engine-owned planning resources shared by atomic actions."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from .bindings import ActionBinding, EndpointBinding, JointPositionTarget
from .control import ActionControlOverrides, ControlPartCommandProfile
from .core import resolve_runtime_device
from .requirements import (
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    SkillBindingContract,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator


class ActionPlanningServices:
    """Planning resources exclusively owned by one atomic-action engine."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_profiles: Mapping[str, ControlPartCommandProfile] | None = None,
    ) -> None:
        self._motion_generator = motion_generator
        self._robot: Robot = motion_generator.robot
        self._device = resolve_runtime_device(motion_generator.device)
        self._binding_owner_id = uuid4().hex
        self._control_profiles = self._snapshot_control_profiles(
            {} if control_profiles is None else control_profiles
        )

    @property
    def motion_generator(self) -> MotionGenerator:
        """Return the single motion generator owned by the engine."""
        return self._motion_generator

    @property
    def robot(self) -> Robot:
        """Return the robot planned by this service set."""
        return self._robot

    @property
    def device(self) -> torch.device:
        """Return the concrete device used for planning."""
        return self._device

    @property
    def binding_owner_id(self) -> str:
        """Return the opaque identity required by this engine's bindings."""
        return self._binding_owner_id

    @property
    def control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Return owned direct-core command profiles by control-part name."""
        return MappingProxyType(
            {
                name: profile.snapshot()
                for name, profile in self._control_profiles.items()
            }
        )

    @property
    def planner_name(self) -> str:
        """Return the configured planner backend name."""
        planner_cfg = getattr(
            getattr(self._motion_generator, "planner", None), "cfg", None
        )
        planner_name = getattr(planner_cfg, "planner_type", None)
        return "unknown" if planner_name is None else str(planner_name)

    def bind_control_parts(
        self,
        contract: SkillBindingContract,
        endpoints: Mapping[str, Mapping[str, str]],
    ) -> ActionBinding:
        """Build a generic binding from explicit robot control-part names.

        This is the advanced direct-core construction path. Profile-backed
        callers obtain the same :class:`ActionBinding` from
        ``BoundRobotSkillProfile.resolve()``.
        """
        if not isinstance(contract, SkillBindingContract):
            raise TypeError("contract must be a SkillBindingContract.")
        if not isinstance(endpoints, Mapping):
            raise TypeError("endpoints must be a slot-to-endpoint mapping.")
        expected = {
            (slot.slot_id, requirement.endpoint_id): requirement
            for slot in contract.slots
            for requirement in slot.endpoints
        }
        supplied: dict[tuple[str, str], str] = {}
        for slot_id, slot_endpoints in endpoints.items():
            if not isinstance(slot_id, str) or not slot_id.strip():
                raise ValueError("Binding slot IDs must be non-empty strings.")
            if not isinstance(slot_endpoints, Mapping):
                raise TypeError(f"Binding slot {slot_id!r} must contain a mapping.")
            for endpoint_id, control_part in slot_endpoints.items():
                key = (slot_id, endpoint_id)
                if key in supplied:
                    raise ValueError(
                        f"Binding endpoint {slot_id}.{endpoint_id} repeats."
                    )
                if not isinstance(endpoint_id, str) or not endpoint_id.strip():
                    raise ValueError("Binding endpoint IDs must be non-empty strings.")
                if not isinstance(control_part, str) or not control_part.strip():
                    raise ValueError("Control-part names must be non-empty strings.")
                supplied[key] = control_part
        if set(supplied) != set(expected):
            missing = sorted(set(expected) - set(supplied))
            extra = sorted(set(supplied) - set(expected))
            raise ValueError(
                "Direct binding must cover the skill contract exactly: "
                f"missing={missing}, extra={extra}."
            )
        if not expected:
            binding = ActionBinding(owner_id=self.binding_owner_id)
            self.validate_binding(binding, contract)
            return binding

        control_parts = getattr(self.robot, "control_parts", None)
        if not isinstance(control_parts, Mapping):
            raise TypeError("Direct control-part binding requires Robot.control_parts.")
        available = sorted(str(name) for name in control_parts)
        resolved: list[EndpointBinding] = []
        for key, requirement in expected.items():
            slot_id, endpoint_id = key
            control_part = supplied[key]
            if control_part not in control_parts:
                raise ValueError(
                    f"Endpoint {slot_id}.{endpoint_id} references control part "
                    f"{control_part!r}, but Robot.control_parts contains {available}."
                )
            joint_ids = tuple(self.robot.get_joint_ids(name=control_part))
            if not joint_ids:
                raise ValueError(f"Control part {control_part!r} contains no joints.")
            profile = self._control_profiles.get(control_part)
            commands = {} if profile is None else profile.commands
            for name, command_type in requirement.required_commands.items():
                command = commands.get(name)
                if not isinstance(command, command_type):
                    raise ValueError(
                        f"Endpoint {slot_id}.{endpoint_id} requires command {name!r} "
                        f"of type {command_type.__name__}."
                    )
            resolved.append(
                EndpointBinding(
                    slot_id=slot_id,
                    endpoint_id=endpoint_id,
                    resource_id=f"direct.{slot_id}",
                    adapter_id="control_part",
                    target=JointPositionTarget(control_part, joint_ids),
                    capabilities=requirement.capabilities,
                    commands=commands,
                    claim_tokens=frozenset({f"robot.control_part:{control_part}"}),
                    joint_ids=joint_ids,
                )
            )
        binding = ActionBinding(
            owner_id=self.binding_owner_id,
            endpoints=tuple(resolved),
        )
        self.validate_binding(binding, contract)
        return binding

    def validate_binding(
        self,
        binding: ActionBinding,
        contract: SkillBindingContract,
    ) -> None:
        """Validate endpoint coverage, ownership, capabilities, and claims."""
        if not isinstance(binding, ActionBinding):
            raise TypeError("binding must be an ActionBinding.")
        if binding.owner_id != self.binding_owner_id:
            raise ValueError("ActionBinding belongs to another engine instance.")
        expected = {
            (slot.slot_id, requirement.endpoint_id): requirement
            for slot in contract.slots
            for requirement in slot.endpoints
        }
        if set(binding.endpoint_keys) != set(expected):
            missing = sorted(set(expected) - set(binding.endpoint_keys))
            extra = sorted(set(binding.endpoint_keys) - set(expected))
            raise ValueError(
                "ActionBinding must cover the skill contract exactly: "
                f"missing={missing}, extra={extra}."
            )
        for key, requirement in expected.items():
            endpoint = binding.endpoint(*key)
            missing_capabilities = requirement.capabilities - endpoint.capabilities
            if missing_capabilities:
                raise ValueError(
                    f"Endpoint {key[0]}.{key[1]} is missing capabilities "
                    f"{sorted(missing_capabilities)}."
                )
            for name, command_type in requirement.required_commands.items():
                command = endpoint.commands.get(name)
                if not isinstance(command, command_type):
                    raise ValueError(
                        f"Endpoint {key[0]}.{key[1]} requires command {name!r} "
                        f"of type {command_type.__name__}."
                    )
        for slot in contract.slots:
            for constraint in slot.constraints:
                if not isinstance(constraint, DisjointSlotEndpoints):
                    continue
                selected = [
                    binding.endpoint(slot.slot_id, endpoint_id)
                    for endpoint_id in constraint.endpoint_ids
                ]
                self._validate_disjoint(selected, label=f"slot {slot.slot_id!r}")
        for constraint in contract.constraints:
            if not isinstance(constraint, DisjointResourceSlots):
                continue
            for index, left_slot in enumerate(constraint.slots):
                left = [
                    endpoint
                    for endpoint in binding.endpoints
                    if endpoint.slot_id == left_slot
                ]
                for right_slot in constraint.slots[index + 1 :]:
                    right = [
                        endpoint
                        for endpoint in binding.endpoints
                        if endpoint.slot_id == right_slot
                    ]
                    self._validate_disjoint(
                        left + right,
                        label=f"slots {left_slot!r} and {right_slot!r}",
                        only_across=len(left),
                    )

    def apply_command_overrides(
        self,
        binding: ActionBinding,
        overrides: ActionControlOverrides,
    ) -> ActionBinding:
        """Apply endpoint-scoped commands to an owned validated binding."""
        if not isinstance(overrides, ActionControlOverrides):
            raise TypeError("overrides must be an ActionControlOverrides.")
        if overrides.is_empty:
            return ActionBinding(binding.owner_id, binding.endpoints)
        return binding.with_command_overrides(overrides.as_flat_mapping())

    @staticmethod
    def _validate_disjoint(
        endpoints: list[EndpointBinding],
        *,
        label: str,
        only_across: int | None = None,
    ) -> None:
        """Reject overlapping destination, claim-token, or joint ownership."""
        pairs = (
            (
                (left, right)
                for left in endpoints[:only_across]
                for right in endpoints[only_across:]
            )
            if only_across is not None
            else (
                (left, right)
                for index, left in enumerate(endpoints)
                for right in endpoints[index + 1 :]
            )
        )
        for left, right in pairs:
            same_destination = left.destination_key == right.destination_key
            overlapping_tokens = left.claim_tokens & right.claim_tokens
            left_joints = set(left.joint_ids)
            right_joints = set(right.joint_ids)
            if same_destination or overlapping_tokens or left_joints & right_joints:
                raise ValueError(
                    f"ActionBinding violates disjoint constraint for {label}: "
                    f"{left.key} conflicts with {right.key}."
                )

    def _snapshot_control_profiles(
        self,
        profiles: Mapping[str, ControlPartCommandProfile],
    ) -> Mapping[str, ControlPartCommandProfile]:
        """Validate control-part profile ownership and freeze snapshots."""
        if not isinstance(profiles, Mapping):
            raise TypeError("control_profiles must be a mapping.")
        control_parts = getattr(self.robot, "control_parts", None)
        if not isinstance(control_parts, Mapping):
            if profiles:
                raise TypeError(
                    "Control-part command profiles require Robot.control_parts."
                )
            control_parts = {}
        snapshots: dict[str, ControlPartCommandProfile] = {}
        available = sorted(str(name) for name in control_parts)
        for name, profile in profiles.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "control_profiles keys must be non-empty control-part names."
                )
            if name not in control_parts:
                raise ValueError(
                    f"Control profile references unknown control part {name!r}; "
                    f"Robot.control_parts contains {available}."
                )
            if not isinstance(profile, ControlPartCommandProfile):
                raise TypeError(
                    "control_profiles values must be "
                    "ControlPartCommandProfile instances."
                )
            snapshots[name] = profile.snapshot()
        return MappingProxyType(snapshots)


__all__ = ["ActionPlanningServices"]
