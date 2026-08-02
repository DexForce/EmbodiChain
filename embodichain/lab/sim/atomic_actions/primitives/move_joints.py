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
from ..trajectory import TrajectoryBuilder


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Joint-space goal for a bound robot control resource."""

    goal_kind: ClassVar[str] = "joint_position"

    qpos: torch.Tensor
    """One joint waypoint or a batched sequence of joint waypoints."""

    def __post_init__(self) -> None:
        if not isinstance(self.qpos, torch.Tensor):
            raise TypeError(
                f"qpos must be a torch.Tensor, got {type(self.qpos).__name__}."
            )
        if self.qpos.dim() not in (1, 2, 3) or self.qpos.shape[-1] == 0:
            raise ValueError(
                "qpos must have shape (control_dof,), (n_envs, control_dof), "
                "or (n_envs, n_waypoint, control_dof), "
                f"got {tuple(self.qpos.shape)}."
            )


@dataclass(frozen=True, slots=True, eq=False)
class NamedJointPositionGoal:
    """Named joint-space goal resolved from :class:`MoveJointsCfg`."""

    goal_kind: ClassVar[str] = "named_joint_position"

    name: str
    """Name in ``MoveJointsCfg.named_joint_positions``."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, got {type(self.name).__name__}.")
        if not self.name.strip():
            raise ValueError("name must not be empty.")


@configclass
class MoveJointsCfg(ActionCfg):
    """Skill-specific MoveJoints configuration."""

    name: str = "move_joints"
    named_joint_positions: dict[str, torch.Tensor] | None = None
    """Optional named goals. Motion settings belong to ``MotionPolicy``."""


class MoveJoints(AtomicAction[JointPositionGoal | NamedJointPositionGoal]):
    """Plan joint motion from the observed state to one or more waypoints."""

    skill_id: ClassVar[str] = "move_joints"
    GoalType: ClassVar[tuple[type, ...]] = (
        JointPositionGoal,
        NamedJointPositionGoal,
    )
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    agent_visible: ClassVar[bool] = False

    def __init__(
        self,
        motion_generator,
        cfg: MoveJointsCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveJointsCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.named_joint_positions = self.cfg.named_joint_positions or {}

    def _plan(
        self,
        invocation: ActionInvocation[JointPositionGoal | NamedJointPositionGoal],
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
        goal: JointPositionGoal | NamedJointPositionGoal,
    ) -> torch.Tensor:
        """Resolve an explicit or named joint goal to a tensor."""
        if isinstance(goal, JointPositionGoal):
            return goal.qpos
        if goal.name not in self.named_joint_positions:
            logger.log_error(
                f"Unknown named joint-position goal {goal.name!r}. Available "
                f"goals: {sorted(self.named_joint_positions)}",
                KeyError,
            )
        return self.named_joint_positions[goal.name]


__all__ = [
    "JointPositionGoal",
    "MoveJoints",
    "MoveJointsCfg",
    "NamedJointPositionGoal",
]
