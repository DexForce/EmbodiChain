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

"""MoveJoints atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from ..core import AtomicAction
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..state import PlanningContext
from ..trajectory_ops import (
    build_joint_plan_states,
    resolve_joint_target,
    to_full_robot_trajectory,
)


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Explicit or named joint-space goal for a bound robot resource."""

    goal_kind: ClassVar[str] = "joint_position"

    target: torch.Tensor | str
    """Joint qpos/waypoints or a named control-part profile command."""

    def __post_init__(self) -> None:
        if isinstance(self.target, str):
            if not self.target.strip():
                raise ValueError("Named joint-position target must not be empty.")
            return
        if not isinstance(self.target, torch.Tensor):
            raise TypeError(
                "target must be a torch.Tensor or str, "
                f"got {type(self.target).__name__}."
            )
        if self.target.dim() not in (1, 2, 3) or self.target.shape[-1] == 0:
            raise ValueError(
                "Tensor target must have shape (control_dof,), "
                "(n_envs, control_dof), "
                "or (n_envs, n_waypoint, control_dof), "
                f"got {tuple(self.target.shape)}."
            )


@dataclass(frozen=True, slots=True, eq=False)
class MoveJointsOptions(ActionOptions):
    """Per-invocation behavior for :class:`MoveJoints`."""


class MoveJoints(AtomicAction[JointPositionGoal, MoveJointsOptions]):
    """Plan joint motion from the observed state to one or more waypoints."""

    skill_id: ClassVar[str] = "move_joints"
    GoalType: ClassVar[type] = JointPositionGoal
    OptionsType: ClassVar[type] = MoveJointsOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    agent_visible: ClassVar[bool] = False

    def __init__(
        self,
        default_options: MoveJointsOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a joint-space goal without mutating the robot or task state."""
        goal = self.require_goal(request)
        manipulator = request.binding.manipulator("primary")
        control_part = manipulator.name
        joint_ids = list(manipulator.joint_ids)
        joint_dof = manipulator.dof
        target_qpos = resolve_joint_target(
            self._resolve_target_qpos(
                goal,
                request=request,
                context=context,
            ),
            n_envs=context.batch_size,
            joint_dof=joint_dof,
            control_part=control_part,
            device=self.device,
        )
        start_qpos = context.robot.qpos[:, joint_ids]
        result = self.motion_generator.generate(
            build_joint_plan_states(target_qpos),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
            ),
        )
        success, trajectory = to_full_robot_trajectory(
            result,
            base_qpos=context.robot.qpos,
            joint_ids=joint_ids,
            env_ids=context.env_ids,
            control_dt=request.motion_policy.control_dt,
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=trajectory,
        )

    def _resolve_target_qpos(
        self,
        goal: JointPositionGoal,
        *,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> torch.Tensor:
        """Resolve an explicit or named joint goal to a tensor."""
        if isinstance(goal.target, torch.Tensor):
            return goal.target
        return request.binding.manipulator("primary").joint_positions(
            goal.target,
            n_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )


__all__ = [
    "JointPositionGoal",
    "MoveJoints",
    "MoveJointsOptions",
]
