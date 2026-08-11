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

from embodichain.utils import logger

from ._helpers import arm_qpos_from_state
from ..control import GRASP_COMMAND
from ..core import AtomicAction
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..state import PlanningContext
from ..trajectory_ops import (
    build_joint_plan_states,
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
)


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
        press_xpos = resolve_pose_target(
            resolve_pose_goal(target.xpos, context, name="xpos"),
            n_envs=self.n_envs,
            device=self.device,
        )
        start_arm_qpos = arm_qpos_from_state(state, arm_joint_ids)
        start_hand_qpos = state.last_qpos[:, hand_joint_ids]

        n_close, n_down, n_back = self._compute_segment_waypoints(
            request.motion_policy.sample_count, options
        )

        hand_close_path = interpolate_hand_qpos(
            start_hand_qpos,
            hand_close_qpos,
            n_waypoints=n_close,
        )

        down_result = self.motion_generator.generate(
            build_pose_plan_states(press_xpos),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=control_part,
                sample_count=n_down,
            ),
        )
        assert isinstance(down_result.success, torch.Tensor)
        assert down_result.positions is not None
        down_success = down_result.success
        down_arm = down_result.positions

        press_arm_qpos = down_arm[:, -1, :]
        back_result = self.motion_generator.generate(
            build_joint_plan_states(start_arm_qpos),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=press_arm_qpos,
                control_part=control_part,
                sample_count=n_back,
            ),
        )
        assert isinstance(back_result.success, torch.Tensor)
        assert back_result.positions is not None
        back_success = back_result.success
        back_arm = back_result.positions
        success = down_success & back_success

        # Allocate from the actually returned segment lengths so collision-aware
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
            segment_lengths={
                "close": n_close,
                "press": n_down_actual,
                "retract": n_back_actual,
            },
        )

    def _compute_segment_waypoints(
        self, sample_count: int, options: PressOptions
    ) -> tuple[int, int, int]:
        """Split the invocation sample budget across press segments."""
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
