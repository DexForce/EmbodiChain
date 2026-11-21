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

"""Same-grid conversion between Atomic Skill plans and expansion templates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from embodichain.lab.sim.motion.expansion import (
    SourceContext,
    TrajectoryPhase,
    TrajectoryTemplate,
)

from .plans import SkillPlan
from ._expansion_phases import _canonical_samples, _free_joint_indices

__all__ = ["SkillPlanTemplateAdapter"]


@dataclass(frozen=True)
class SkillPlanTemplateAdapter:
    """Export one explicit Atomic Skill plan as a qpos template.

    Canonical expansion removes zero-time joins; restoration retains the source grid.
    Every plan segment must declare its editable operators and phase kind;
    unlisted operators are rejected instead of being inferred from a skill name.
    """

    joint_names: tuple[str, ...]
    phase_permissions: Mapping[str, Sequence[str]]
    phase_kinds: Mapping[str, str]
    controlled_joint_indices: tuple[int, ...] | None = None
    validator_id: str = "default"
    kind: ClassVar[str] = "atomic_skill"

    def __post_init__(self) -> None:
        if type(self.validator_id) is not str or not self.validator_id.strip():
            raise ValueError("validator_id must be a non-empty string")
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be non-empty and unique")
        object.__setattr__(self, "joint_names", names)
        permissions = {
            str(name): tuple(values) for name, values in self.phase_permissions.items()
        }
        object.__setattr__(self, "phase_permissions", permissions)
        kinds = dict(self.phase_kinds)
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

    def export_template(
        self,
        source: SkillPlan,
        *,
        context: SourceContext,
        row_index: int | None = None,
    ) -> TrajectoryTemplate:
        """Export a single-row, explicitly permissioned SkillPlan.

        Args:
            source: Validated plan retaining a joint trajectory.
            context: Coordinator-owned source and scene identity.
            row_index: Explicit batch row. Omission requires a single row.

        Returns:
            A detached :class:`TrajectoryTemplate` on the CPU/device of the
            source plan.
        """
        if not isinstance(source, SkillPlan):
            raise TypeError("source must be an SkillPlan")
        if not isinstance(context, SourceContext):
            raise TypeError("context must be a SourceContext")
        trajectory = source.joint_trajectory
        if trajectory is None:
            raise ValueError("SkillPlan must retain a joint trajectory")
        if row_index is None:
            if trajectory.batch_size != 1:
                raise ValueError("select one successful row with row_index")
            row_index = 0
        if type(row_index) is not int or not 0 <= row_index < trajectory.batch_size:
            raise ValueError("row_index must select a plan trajectory row")
        if not bool(source.plan_success[row_index]):
            raise ValueError("select one successful row with row_index")
        if trajectory.robot_dof != len(self.joint_names):
            raise ValueError("joint_names must cover the complete plan trajectory")
        declared = {segment.name for segment in source.segments}
        if declared != set(self.phase_permissions) or declared != set(self.phase_kinds):
            raise ValueError(
                "every SkillPlan segment requires an exact phase declaration"
            )
        positions, dt, boundaries, _ = _canonical_samples(
            trajectory.positions[row_index], trajectory.dt[row_index]
        )
        phases = []
        for segment in source.segments:
            start, stop = int(boundaries[segment.start]), int(boundaries[segment.stop])
            if stop <= start:
                # A terminal padding/zero-time phase has no editable interior.
                continue
            phases.append(
                TrajectoryPhase(
                    segment.name,
                    start,
                    stop,
                    kind=self.phase_kinds[segment.name],
                    allowed_operators=(
                        tuple(self.phase_permissions[segment.name])
                        if stop - start >= 3
                        and self.phase_kinds[segment.name] == "free"
                        else ()
                    ),
                )
            )
        active = _free_joint_indices(
            positions, tuple(phases), self.controlled_joint_indices
        )
        if not active:
            phases = [replace(phase, allowed_operators=()) for phase in phases]
        return TrajectoryTemplate(
            source_id=context.source_id,
            source_revision=context.source_revision,
            template_id=context.unit_id,
            joint_names=self.joint_names,
            positions=positions,
            dt=dt,
            phases=tuple(phases),
            allowed_operators=tuple(
                dict.fromkeys(
                    operator for phase in phases for operator in phase.allowed_operators
                )
            ),
            validator_id=self.validator_id,
            controlled_joint_indices=active or self.controlled_joint_indices,
        )

    def restore_positions(
        self, source: SkillPlan, positions: torch.Tensor, *, row_index: int = 0
    ) -> torch.Tensor:
        """Restore canonical samples to the source grid and exact protected values.

        Args:
            source: Original plan used by :meth:`export_template`.
            positions: Expanded canonical full-joint positions, shape ``(N, D)``.
            row_index: Source batch row.

        Returns:
            Owned source-grid positions on the original dtype and device.
        """
        trajectory = source.joint_trajectory
        if trajectory is None:
            raise ValueError("SkillPlan must retain a joint trajectory")
        original = trajectory.positions[row_index]
        canonical, _, _, restore = _canonical_samples(
            original, trajectory.dt[row_index]
        )
        if positions.shape != canonical.shape:
            raise ValueError("expanded positions must preserve the canonical grid")
        restored = positions[restore.to(positions.device)].to(original).clone()
        # Only joints moving in editable phases may change. This also protects
        # the stationary arm of a dual-arm action without control-part naming.
        free = tuple(
            TrajectoryPhase(
                segment.name,
                segment.start,
                segment.stop,
                allowed_operators=tuple(self.phase_permissions[segment.name]),
            )
            for segment in source.segments
            if self.phase_permissions[segment.name]
            and self.phase_kinds[segment.name] == "free"
        )
        active = _free_joint_indices(
            original.detach().cpu(), free, self.controlled_joint_indices
        )
        protected = [
            index for index in range(original.shape[-1]) if index not in active
        ]
        restored[:, protected] = original[:, protected]
        for segment in source.segments:
            restored[segment.start] = original[segment.start]
            restored[segment.stop - 1] = original[segment.stop - 1]
            if (
                not self.phase_permissions[segment.name]
                or self.phase_kinds[segment.name] != "free"
            ):
                restored[segment.start : segment.stop] = original[
                    segment.start : segment.stop
                ]
        return restored
