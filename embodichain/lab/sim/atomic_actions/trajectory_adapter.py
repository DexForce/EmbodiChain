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

"""Same-grid conversion between Atomic Action plans and expansion templates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from embodichain.lab.sim.motion.expansion import (
    TrajectoryPhase,
    TrajectoryTemplate,
)

from .plans import ActionPlan

__all__ = ["ActionPlanTemplateAdapter"]


@dataclass(frozen=True)
class ActionPlanTemplateAdapter:
    """Export one explicit Atomic Action plan as a qpos template.

    The first integration deliberately preserves the original control grid.
    Every plan segment must declare its editable operators and phase kind;
    unlisted operators are rejected instead of being inferred from a skill name.
    """

    source_id: str
    source_revision: str
    template_id: str
    joint_names: tuple[str, ...]
    phase_permissions: Mapping[str, Sequence[str]]
    phase_kinds: Mapping[str, str] | None = None
    controlled_joint_indices: tuple[int, ...] | None = None
    validator_id: str = "default"

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "source_revision",
            "template_id",
            "validator_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be non-empty and unique")
        object.__setattr__(self, "joint_names", names)
        permissions = {
            str(name): tuple(values) for name, values in self.phase_permissions.items()
        }
        object.__setattr__(self, "phase_permissions", permissions)
        kinds = {} if self.phase_kinds is None else dict(self.phase_kinds)
        if any(kind not in ("free", "contact", "hold") for kind in kinds.values()):
            raise ValueError("phase_kinds values must be free, contact, or hold")
        object.__setattr__(self, "phase_kinds", kinds)
        controlled = (
            tuple(range(len(names)))
            if self.controlled_joint_indices is None
            else tuple(self.controlled_joint_indices)
        )
        if not controlled or len(set(controlled)) != len(controlled):
            raise ValueError("controlled_joint_indices must be non-empty and unique")
        if any(
            type(index) is not int or not 0 <= index < len(names)
            for index in controlled
        ):
            raise ValueError("controlled_joint_indices must index joint_names")
        object.__setattr__(self, "controlled_joint_indices", controlled)

    def export(self, plan: ActionPlan) -> TrajectoryTemplate:
        """Export a single-row, explicitly permissioned ActionPlan.

        Args:
            plan: Validated plan with one successful row and a retained qpos
                trajectory.

        Returns:
            A detached :class:`TrajectoryTemplate` on the CPU/device of the
            source plan.
        """
        if not isinstance(plan, ActionPlan):
            raise TypeError("plan must be an ActionPlan.")
        trajectory = plan.joint_trajectory
        if trajectory is None:
            raise ValueError("ActionPlan must retain a joint trajectory.")
        if trajectory.batch_size != 1 or not plan.success_all:
            raise ValueError("The initial adapter supports one successful row.")
        if trajectory.robot_dof != len(self.joint_names):
            raise ValueError("joint_names must cover the complete plan trajectory.")
        phases = []
        for segment in plan.segments:
            if segment.name not in self.phase_permissions:
                raise ValueError(
                    f"Missing explicit phase permissions for {segment.name!r}."
                )
            phases.append(
                TrajectoryPhase(
                    segment.name,
                    segment.start,
                    segment.stop,
                    kind=self.phase_kinds.get(segment.name, "free"),
                    allowed_operators=tuple(self.phase_permissions[segment.name]),
                )
            )
        return TrajectoryTemplate(
            source_id=self.source_id,
            source_revision=self.source_revision,
            template_id=self.template_id,
            joint_names=self.joint_names,
            positions=trajectory.positions[0],
            dt=trajectory.dt[0],
            phases=tuple(phases),
            allowed_operators=tuple(
                sorted(
                    {
                        operator
                        for values in self.phase_permissions.values()
                        for operator in values
                    }
                )
            ),
            validator_id=self.validator_id,
            controlled_joint_indices=self.controlled_joint_indices,
        )
