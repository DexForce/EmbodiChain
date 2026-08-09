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

import torch

from .bindings import ActionBinding, ResolvedActionBinding, ResolvedControlPart
from .control import (
    ActionControlOverrides,
    ControlCommand,
    ControlPartCommandProfile,
)
from .core import resolve_runtime_device
from .trajectory import TrajectoryBuilder

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.planners import MotionGenerator


class ActionPlanningServices:
    """Planning resources exclusively owned by one atomic-action engine.

    An action may borrow these resources after the engine binds it, but callers
    never pass a motion generator to individual actions. Keeping the generator
    and trajectory builder here gives one engine a single planner backend,
    robot, device, cache, and collision-world owner.

    Args:
        motion_generator: Motion generator owned by the engine.
        control_profiles: Semantic command profiles keyed by names from the
            owned robot's ``control_parts`` mapping.
    """

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_profiles: Mapping[str, ControlPartCommandProfile] | None = None,
    ) -> None:
        self._motion_generator = motion_generator
        self._robot: Robot = motion_generator.robot
        self._device = resolve_runtime_device(motion_generator.device)
        self._trajectory_builder = TrajectoryBuilder(motion_generator)
        self._control_profiles = self._snapshot_control_profiles(
            {} if control_profiles is None else control_profiles
        )
        self._binding_cache: dict[
            tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
            ResolvedActionBinding,
        ] = {}

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
    def trajectory_builder(self) -> TrajectoryBuilder:
        """Return the shared stateless trajectory builder."""
        return self._trajectory_builder

    @property
    def control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Return owned semantic command profiles keyed by control-part name."""
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

    def resolve_binding(
        self,
        binding: ActionBinding,
        control_overrides: ActionControlOverrides | None = None,
    ) -> ResolvedActionBinding:
        """Resolve binding names against the owned robot's control parts.

        ``ActionBinding`` deliberately carries stable string references only.
        This method establishes that every reference is a key in
        ``Robot.control_parts`` and resolves its full-robot joint indices.

        Args:
            binding: Semantic-role mapping to validate and resolve.
            control_overrides: Optional per-role command replacements for this
                invocation revision.

        Returns:
            Immutable runtime resources for action planning.

        Raises:
            TypeError: If ``binding`` or ``Robot.control_parts`` is invalid.
            ValueError: If a referenced control part is unknown or empty.
        """
        if not isinstance(binding, ActionBinding):
            raise TypeError("binding must be an ActionBinding.")
        cache_key = (
            tuple(sorted(binding.manipulators.items())),
            tuple(sorted(binding.end_effectors.items())),
        )
        resolved = self._binding_cache.get(cache_key)
        if resolved is None:
            control_parts = getattr(self.robot, "control_parts", None)
            if not isinstance(control_parts, Mapping):
                if binding.manipulators or binding.end_effectors:
                    raise TypeError(
                        "ActionBinding resources must come from "
                        "Robot.control_parts, but the engine robot does not "
                        "define a control-parts mapping."
                    )
                control_parts = {}

            resolved = ResolvedActionBinding(
                manipulators=self._resolve_resource_map(
                    binding.manipulators,
                    control_parts=control_parts,
                    resource_kind="manipulator",
                ),
                end_effectors=self._resolve_resource_map(
                    binding.end_effectors,
                    control_parts=control_parts,
                    resource_kind="end effector",
                ),
            )
            self._binding_cache[cache_key] = resolved

        if control_overrides is None:
            return resolved
        if not isinstance(control_overrides, ActionControlOverrides):
            raise TypeError("control_overrides must be an ActionControlOverrides.")
        if control_overrides.is_empty:
            return resolved
        return ResolvedActionBinding(
            manipulators=self._apply_command_overrides(
                resolved.manipulators,
                control_overrides.manipulators,
                resource_kind="manipulator",
            ),
            end_effectors=self._apply_command_overrides(
                resolved.end_effectors,
                control_overrides.end_effectors,
                resource_kind="end effector",
            ),
        )

    def _resolve_resource_map(
        self,
        resources: Mapping[str, str],
        *,
        control_parts: Mapping[str, object],
        resource_kind: str,
    ) -> dict[str, ResolvedControlPart]:
        """Resolve one role map through ``Robot.control_parts``."""
        available = sorted(str(name) for name in control_parts)
        resolved: dict[str, ResolvedControlPart] = {}
        for role, name in resources.items():
            if name not in control_parts:
                raise ValueError(
                    f"ActionBinding {resource_kind} role {role!r} references "
                    f"control part {name!r}, but Robot.control_parts contains "
                    f"{available}."
                )
            joint_ids = tuple(self.robot.get_joint_ids(name=name))
            if not joint_ids:
                raise ValueError(
                    f"Robot control part {name!r} bound to {resource_kind} role "
                    f"{role!r} contains no joints."
                )
            profile = self._control_profiles.get(name)
            resolved[role] = ResolvedControlPart(
                name=name,
                joint_ids=joint_ids,
                commands={} if profile is None else profile.commands,
            )
        return resolved

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

    @staticmethod
    def _apply_command_overrides(
        resources: Mapping[str, ResolvedControlPart],
        overrides: Mapping[str, Mapping[str, ControlCommand]],
        *,
        resource_kind: str,
    ) -> dict[str, ResolvedControlPart]:
        """Apply role-scoped commands to already resolved control parts."""
        unknown_roles = sorted(set(overrides) - set(resources))
        if unknown_roles:
            raise KeyError(
                f"Command overrides reference unbound {resource_kind} roles "
                f"{unknown_roles}; bound roles are {sorted(resources)}."
            )
        return {
            role: resource.with_command_overrides(overrides.get(role, {}))
            for role, resource in resources.items()
        }


__all__ = ["ActionPlanningServices"]
