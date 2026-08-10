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

"""Press atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import logger

from ._helpers import arm_qpos_from_state
from ..control import GRASP_COMMAND
from ..core import AtomicAction
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..state import PlanningContext


@dataclass(frozen=True, slots=True, eq=False)
class PressGoal:
    """Single end-effector contact pose used by :class:`Press`."""

    goal_kind: ClassVar[str] = "press_pose"

    xpos: PoseGoalValue
    """Contact pose, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class PressOptions(ActionOptions):
    """Per-invocation press behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints for closing the gripper before pressing."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")


class Press(AtomicAction[PressGoal, PressOptions]):
    """Close the gripper, press down to a target pose, then return."""

    skill_id: ClassVar[str] = "press"
    GoalType: ClassVar[type] = PressGoal
    OptionsType: ClassVar[type] = PressOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        default_options: PressOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Resolve engine-wide resources from the owning engine."""
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _plan(
        self,
        request: ResolvedActionRequest[PressGoal, PressOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a close, press, and retract sequence."""
        target = self.require_goal(request)
        options = request.skill_options
        binding = request.binding
        manipulator = binding.manipulator()
        end_effector = binding.end_effector()
        control_part = manipulator.name
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        hand_close_qpos = end_effector.joint_positions(
            GRASP_COMMAND,
            n_envs=self.n_envs,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        state = context
        press_xpos = self.builder.resolve_pose_target(
            resolve_pose_goal(target.xpos, context, name="xpos"),
            n_envs=self.n_envs,
        )
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=manipulator.dof,
            control_part=control_part,
        )
        start_hand_qpos = state.last_qpos[:, hand_joint_ids]

        n_close, n_down, n_back = self._compute_phase_waypoints(
            request.motion_policy.sample_count, options
        )

        hand_close_path = self.builder.interpolate_hand_qpos(
            start_hand_qpos,
            hand_close_qpos,
            n_waypoints=n_close,
        )

        target_states_list = [
            [PlanState(xpos=press_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        down_success, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=control_part,
            arm_dof=manipulator.dof,
            cfg=request.motion_policy,
        )

        press_arm_qpos = down_arm[:, -1, :]
        back_success, back_arm = self.builder.plan_joint_motion(
            press_arm_qpos,
            start_arm_qpos,
            n_back,
            control_part=control_part,
            arm_dof=manipulator.dof,
            cfg=request.motion_policy,
        )
        success = down_success & back_success

        # Allocate from the actually-returned phase lengths so collision-aware
        # planners (which preserve their own sample count) are accommodated.
        n_down_actual = down_arm.shape[1]
        n_back_actual = back_arm.shape[1]
        full = torch.empty(
            (self.n_envs, n_close + n_down_actual + n_back_actual, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_close, arm_joint_ids] = start_arm_qpos.unsqueeze(1)
        full[:, :n_close, hand_joint_ids] = hand_close_path
        full[:, n_close : n_close + n_down_actual, arm_joint_ids] = down_arm
        full[:, n_close : n_close + n_down_actual, hand_joint_ids] = (
            hand_close_qpos.unsqueeze(1)
        )
        full[:, n_close + n_down_actual :, arm_joint_ids] = back_arm
        full[:, n_close + n_down_actual :, hand_joint_ids] = hand_close_qpos.unsqueeze(
            1
        )

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=full,
            phase_name="press",
        )

    def _compute_phase_waypoints(
        self, sample_count: int, options: PressOptions
    ) -> tuple[int, int, int]:
        """Split the invocation sample budget across press phases."""
        n_close = options.hand_interp_steps

        motion_waypoints = sample_count - n_close
        n_down = motion_waypoints // 2
        n_back = motion_waypoints - n_down
        if n_down < 2 or n_back < 2:
            logger.log_error(
                "Not enough waypoints for press trajectory. Increase "
                "MotionPolicy.sample_count or decrease hand_interp_steps.",
                ValueError,
            )
        return n_close, n_down, n_back


__all__ = ["Press", "PressGoal", "PressOptions"]
