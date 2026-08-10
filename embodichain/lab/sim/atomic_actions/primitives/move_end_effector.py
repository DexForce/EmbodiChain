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

from ..core import AtomicAction
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..requirements import (
    ActionBindingRoute,
    CARTESIAN_POSE_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from ..state import PlanningContext
from ..trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
    to_full_robot_trajectory,
)


@dataclass(frozen=True, slots=True, eq=False)
class EndEffectorPoseGoal:
    """End-effector pose goal with optional batched intermediate waypoints."""

    goal_kind: ClassVar[str] = "end_effector_pose"

    xpos: PoseGoalValue
    """Homogeneous pose with shape ``(4,4)``, ``(B,4,4)`` or ``(B,N,4,4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=True)


@dataclass(frozen=True, slots=True, eq=False)
class MoveEndEffectorOptions(ActionOptions):
    """Per-invocation behavior for :class:`MoveEndEffector`."""


class MoveEndEffector(AtomicAction[EndEffectorPoseGoal, MoveEndEffectorOptions]):
    """Plan a free-space move for a bound manipulator."""

    skill_id: ClassVar[str] = "move_end_effector"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                        route=ActionBindingRoute("manipulator", "primary"),
                    ),
                ),
            ),
        ),
    )
    OptionsType: ClassVar[type] = MoveEndEffectorOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        default_options: MoveEndEffectorOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, MoveEndEffectorOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan an end-effector pose goal from the observed joint state."""
        goal = self.require_goal(request)
        manipulator = request.binding.manipulator("primary")
        control_part = manipulator.name
        joint_ids = list(manipulator.joint_ids)
        move_xpos = resolve_pose_target(
            resolve_pose_goal(goal.xpos, context, name="xpos"),
            n_envs=context.batch_size,
            device=self.device,
        )
        start_qpos = context.robot.qpos[:, joint_ids]
        result = self.motion_generator.generate(
            build_pose_plan_states(move_xpos),
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


__all__ = [
    "EndEffectorPoseGoal",
    "MoveEndEffector",
    "MoveEndEffectorOptions",
]
