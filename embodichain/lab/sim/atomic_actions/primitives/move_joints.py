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

from embodichain.utils import configclass, logger

from ..core import ActionCfg, AtomicAction
from ..invocation import ActionInvocation
from ..plans import ActionPlan, CompletionConditionKind
from ..state import PlanningContext


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Explicit or named joint-space goal for a bound robot resource."""

    goal_kind: ClassVar[str] = "joint_position"

    target: torch.Tensor | str
    """Joint qpos/waypoints or a name in ``MoveJointsCfg.named_joint_positions``."""

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


@configclass
class MoveJointsCfg(ActionCfg):
    """Skill-specific MoveJoints configuration."""

    name: str = "move_joints"
    named_joint_positions: dict[str, torch.Tensor] | None = None
    """Optional named joint-position targets. Motion settings belong to ``MotionPolicy``."""


class MoveJoints(AtomicAction[JointPositionGoal]):
    """Plan joint motion from the observed state to one or more waypoints."""

    skill_id: ClassVar[str] = "move_joints"
    GoalType: ClassVar[type] = JointPositionGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    agent_visible: ClassVar[bool] = False

    def __init__(
        self,
        cfg: MoveJointsCfg | None = None,
    ) -> None:
        super().__init__(cfg or MoveJointsCfg())
        self.named_joint_positions = self.cfg.named_joint_positions or {}

    def plan(
        self,
        invocation: ActionInvocation[JointPositionGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a joint-space goal without mutating the robot or task state."""
        goal = self.require_goal(invocation)
        control_part = invocation.binding.manipulator("primary")
        joint_ids = self.robot.get_joint_ids(name=control_part)
        joint_dof = len(joint_ids)
        target_qpos = self.builder.resolve_joint_target(
            self._resolve_target_qpos(goal),
            n_envs=context.batch_size,
            joint_dof=joint_dof,
            control_part=control_part,
        )
        start_qpos = self.builder.resolve_start_qpos(
            context.robot.qpos[:, joint_ids],
            n_envs=context.batch_size,
            arm_dof=joint_dof,
            control_part=control_part,
        )
        result = self.builder.generate_joint_plan(
            start_qpos,
            target_qpos,
            invocation.motion_policy.sample_count,
            control_part=control_part,
            arm_dof=joint_dof,
            cfg=invocation.motion_policy,
        )
        success, trajectory = self.builder.to_full_robot_trajectory(
            result,
            base_qpos=context.robot.qpos,
            joint_ids=joint_ids,
            env_ids=context.env_ids,
            control_dt=invocation.motion_policy.control_dt,
        )
        return self.build_plan(
            invocation,
            context,
            success=success,
            trajectory=trajectory,
            completion_kind=CompletionConditionKind.JOINT_GOAL_REACHED,
        )

    def _resolve_target_qpos(
        self,
        goal: JointPositionGoal,
    ) -> torch.Tensor:
        """Resolve an explicit or named joint goal to a tensor."""
        if isinstance(goal.target, torch.Tensor):
            return goal.target
        if goal.target not in self.named_joint_positions:
            logger.log_error(
                f"Unknown named joint-position goal {goal.target!r}. Available "
                f"goals: {sorted(self.named_joint_positions)}",
                KeyError,
            )
        return self.named_joint_positions[goal.target]


__all__ = [
    "JointPositionGoal",
    "MoveJoints",
    "MoveJointsCfg",
]
