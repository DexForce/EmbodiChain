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

"""MoveEndEffector atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass

from ..core import ActionCfg, AtomicAction
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionInvocation
from ..plans import ActionPlan, CompletionConditionKind
from ..state import PlanningContext
from ..trajectory import TrajectoryBuilder


@dataclass(frozen=True, slots=True, eq=False)
class EndEffectorPoseGoal:
    """End-effector pose goal with optional batched intermediate waypoints."""

    goal_kind: ClassVar[str] = "end_effector_pose"

    xpos: PoseGoalValue
    """Homogeneous pose with shape ``(4,4)``, ``(B,4,4)`` or ``(B,N,4,4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=True)


@configclass
class MoveEndEffectorCfg(ActionCfg):
    """Skill-specific MoveEndEffector configuration."""

    name: str = "move_end_effector"


class MoveEndEffector(AtomicAction[EndEffectorPoseGoal]):
    """Plan a free-space move for a bound manipulator."""

    skill_id: ClassVar[str] = "move_end_effector"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        motion_generator,
        cfg: MoveEndEffectorCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveEndEffectorCfg())
        self.builder = TrajectoryBuilder(motion_generator)

    def plan(
        self,
        invocation: ActionInvocation[EndEffectorPoseGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan an end-effector pose goal from the observed joint state."""
        goal = self.require_goal(invocation)
        control_part = invocation.binding.manipulator("primary")
        joint_ids = self.robot.get_joint_ids(name=control_part)
        arm_dof = len(joint_ids)
        move_xpos = self.builder.resolve_pose_target(
            resolve_pose_goal(goal.xpos, context, name="xpos"),
            n_envs=context.batch_size,
        )
        start_qpos = self.builder.resolve_start_qpos(
            context.robot.qpos[:, joint_ids],
            n_envs=context.batch_size,
            arm_dof=arm_dof,
            control_part=control_part,
        )
        target_states = self._build_target_states(move_xpos, context.batch_size)
        result = self.builder.generate_arm_plan(
            target_states,
            start_qpos,
            invocation.motion_policy.sample_count,
            control_part=control_part,
            arm_dof=arm_dof,
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
            completion_kind=CompletionConditionKind.EEF_GOAL_REACHED,
        )

    @staticmethod
    def _build_target_states(
        move_xpos: torch.Tensor,
        batch_size: int,
    ) -> list[list[PlanState]]:
        """Build per-environment planner states for pose waypoints."""
        if move_xpos.dim() == 3:
            move_xpos = move_xpos.unsqueeze(1)
        return [
            [
                PlanState(xpos=move_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(move_xpos.shape[1])
            ]
            for i in range(batch_size)
        ]


__all__ = ["EndEffectorPoseGoal", "MoveEndEffector", "MoveEndEffectorCfg"]
